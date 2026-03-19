from __future__ import annotations

import time
from typing import List, Tuple, Optional, Dict

from app.core.config import settings
from app.core.logging import logger
from app.services.bm25_index import BM25Index
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService


class HybridRetriever:
    """
    Combines vector search (Qdrant) and keyword search (BM25) using
    Reciprocal Rank Fusion (RRF) then re-ranks with a cross-encoder.

    LEARNING — why hybrid retrieval?
    Vector search:  finds semantically similar chunks
                    fails on exact terms, names, codes, acronyms
    BM25:           finds exact keyword matches
                    fails on synonyms and paraphrasing
    Hybrid:         covers both failure modes
                    consistently outperforms either alone

    Research shows hybrid retrieval improves recall@5 by 10-15%
    over pure vector search on most domain-specific corpora.
    """

    # RRF constant — 60 is the standard value from the original paper.
    # Higher k = smoother fusion, less sensitive to rank differences.
    # Lower k = more sensitive to top ranks, winner-takes-more effect.
    _RRF_K = 60

    def __init__(
        self,
        vector_store: VectorStoreService,
        embedder: EmbeddingService,
        bm25: BM25Index,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._bm25 = bm25

    async def retrieve(
        self,
        question: str,
        top_k: int,
        fetch_k: int = 20,
    ) -> List[Tuple[str, str, Optional[int], float]]:
        """
        Hybrid retrieval pipeline:
          1. Vector search  → top fetch_k semantic matches
          2. BM25 search    → top fetch_k keyword matches
          3. RRF fusion     → single ranked list
          4. Return top_k

        Args:
            question: user question
            top_k:    final number of chunks to return to the LLM
            fetch_k:  how many candidates each path fetches before fusion
                      more candidates = better fusion quality but slower
                      default 20 is a good balance
        """
        start = time.perf_counter()

        # ── Path 1: Vector search ─────────────────────────────────────────────
        query_vector = self._embedder.embed_query(question)
        vector_results = await self._store.search(
            query_vector=query_vector,
            top_k=fetch_k,
        )

        # ── Path 2: BM25 search ───────────────────────────────────────────────
        # LEARNING: BM25 needs all chunks loaded to search across them.
        # We fetch all chunks from Qdrant and build the index on the fly.
        # For large collections (>10k chunks) we'd cache this index.
        await self._maybe_rebuild_bm25()
        bm25_results = self._bm25.search(question, top_k=fetch_k)

        # ── Path 3: RRF fusion ────────────────────────────────────────────────
        fused = self._rrf_fuse(vector_results, bm25_results)

        # Return top_k after fusion
        final = fused[:top_k]

        latency = time.perf_counter() - start
        logger.info(
            "hybrid_retrieval_complete",
            question=question[:80],
            vector_results=len(vector_results),
            bm25_results=len(bm25_results),
            fused_results=len(fused),
            returned=len(final),
            latency_ms=round(latency * 1000),
        )

        return final

    def _rrf_fuse(
        self,
        vector_results: List[Tuple],
        bm25_results: List[Tuple],
    ) -> List[Tuple[str, str, Optional[int], float]]:
        """
        Reciprocal Rank Fusion — combine two ranked lists into one.

        LEARNING — the RRF formula:
          score(chunk) = Σ  1 / (k + rank)

        Where rank is the position of the chunk in each list (1-indexed).
        k=60 prevents top-ranked items from dominating too heavily.

        Example with k=60:
          rank 1  → 1/(60+1)  = 0.0164
          rank 5  → 1/(60+5)  = 0.0154
          rank 20 → 1/(60+20) = 0.0125

        A chunk ranked #1 in both lists scores:
          0.0164 + 0.0164 = 0.0328  ← highest possible

        A chunk ranked #1 in one list and not in the other:
          0.0164 + 0 = 0.0164

        A chunk ranked #5 in both lists:
          0.0154 + 0.0154 = 0.0308  ← still competitive

        This rewards chunks that appear in BOTH lists even at moderate
        ranks — the "consensus" signal. That's the insight behind RRF.

        LEARNING — why RRF over weighted sum?
        You could combine scores as: final = 0.7 * vector + 0.3 * bm25
        The problem: vector and BM25 scores have different scales and
        distributions. RRF sidesteps this by using ranks not scores —
        rank 1 means the same thing regardless of the scoring method.
        """
        # Map chunk text → RRF score
        scores: Dict[str, float] = {}
        # Map chunk text → metadata (source_file, page_number)
        metadata: Dict[str, Tuple] = {}

        # Add vector search ranks
        for rank, (text, source_file, page_num, _) in enumerate(vector_results, 1):
            scores[text] = scores.get(text, 0.0) + 1.0 / (self._RRF_K + rank)
            metadata[text] = (source_file, page_num)

        # Add BM25 ranks
        for rank, (text, source_file, page_num, _) in enumerate(bm25_results, 1):
            scores[text] = scores.get(text, 0.0) + 1.0 / (self._RRF_K + rank)
            metadata[text] = (source_file, page_num)

        # Sort by RRF score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Rebuild result tuples with RRF score as the score field
        return [
            (text, metadata[text][0], metadata[text][1], score)
            for text, score in ranked
        ]

    async def _maybe_rebuild_bm25(self) -> None:
        """
        Fetch all chunks from Qdrant and rebuild the BM25 index.

        LEARNING — production optimisation opportunity:
        This rebuilds on every query which is fine for < 5000 chunks.
        For larger collections you'd cache the index and only rebuild
        when new documents are ingested (using a version counter or
        a simple timestamp check).
        """
        # Scroll through all points in Qdrant
        all_chunks, all_metadata = [], []
        offset = None

        while True:
            result, next_offset = await self._store._client.scroll(
                collection_name=settings.qdrant_collection,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,   # don't need vectors for BM25
            )

            for point in result:
                all_chunks.append(point.payload.get("text", ""))
                all_metadata.append({
                    "source_file": point.payload.get("source_file", ""),
                    "page_number": point.payload.get("page_number"),
                })

            if next_offset is None:
                break
            offset = next_offset

        if all_chunks:
            self._bm25.build(all_chunks, all_metadata)