
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
    Combines vector search and BM25 using Reciprocal Rank Fusion.

    LEARNING — RRF formula:
      score(chunk) = Σ  1 / (k + rank)

    k=60 is the standard constant from the original RRF paper.
    rank is the position of the chunk in each result list (1-indexed).

    A chunk ranked #1 in BOTH lists:
      1/(60+1) + 1/(60+1) = 0.0328  ← highest possible

    A chunk ranked #1 in ONE list only:
      1/(60+1) + 0 = 0.0164

    A chunk ranked #5 in BOTH lists:
      1/(60+5) + 1/(60+5) = 0.0308  ← still competitive

    This rewards chunks appearing in both lists — the consensus signal.
    That is the core insight behind RRF.

    LEARNING — why RRF over weighted score combination?
    You could do: final = 0.7 * vector_score + 0.3 * bm25_score
    Problem: vector scores and BM25 scores have different scales.
    A BM25 score of 3.2 and a vector score of 0.85 are not comparable.
    RRF uses ranks not scores — rank 1 means the same thing regardless
    of the scoring method. No calibration needed.
    """

    _RRF_K = 60

    def __init__(
        self,
        vector_store: VectorStoreService,
        embedder: EmbeddingService,
        bm25: BM25Index,
    ) -> None:
        self._store   = vector_store
        self._embedder = embedder
        self._bm25    = bm25

    async def retrieve(
        self,
        question: str,
        top_k: int,
        fetch_k: int = 20,
    ) -> List[Tuple[str, str, Optional[int], float]]:
        """
        Hybrid retrieval: vector + BM25 → RRF → top_k results.

        Args:
            question: user question
            top_k:    final results to return
            fetch_k:  candidates each path fetches before fusion
        """
        start = time.perf_counter()

        # Path 1 — vector search
        query_vector   = self._embedder.embed_query(question)
        vector_results = await self._store.search(
            query_vector=query_vector,
            top_k=fetch_k,
        )

        # Path 2 — BM25 keyword search
        await self._rebuild_bm25()
        bm25_results = self._bm25.search(question, top_k=fetch_k)

        # RRF fusion
        fused = self._rrf_fuse(vector_results, bm25_results)
        final = fused[:top_k]

        logger.info(
            "hybrid_retrieval_complete",
            question=question[:80],
            vector_results=len(vector_results),
            bm25_results=len(bm25_results),
            fused_results=len(fused),
            returned=len(final),
            latency_ms=round((time.perf_counter() - start) * 1000),
        )

        return final

    def _rrf_fuse(
        self,
        vector_results: List[Tuple],
        bm25_results: List[Tuple],
    ) -> List[Tuple[str, str, Optional[int], float]]:
        """Fuse two ranked lists using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        metadata: Dict[str, Tuple] = {}

        for rank, (text, source_file, page_num, _) in enumerate(vector_results, 1):
            scores[text]   = scores.get(text, 0.0) + 1.0 / (self._RRF_K + rank)
            metadata[text] = (source_file, page_num)

        for rank, (text, source_file, page_num, _) in enumerate(bm25_results, 1):
            scores[text]   = scores.get(text, 0.0) + 1.0 / (self._RRF_K + rank)
            metadata[text] = (source_file, page_num)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            (text, metadata[text][0], metadata[text][1], score)
            for text, score in ranked
        ]

    async def _rebuild_bm25(self) -> None:
        """Fetch all chunks from Qdrant and rebuild BM25 index."""
        all_chunks, all_metadata = [], []
        offset = None

        while True:
            result, next_offset = await self._store._client.scroll(
                collection_name=settings.qdrant_collection,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
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
