from __future__ import annotations

import re
import time
from typing import List, Tuple, Optional

from rank_bm25 import BM25Okapi

from app.core.logging import logger


class BM25Index:
    """
    In-memory BM25 keyword index built from Qdrant chunks at query time.

    LEARNING — why BM25 (Best Match 25)?
    BM25 is a probabilistic ranking function that scores documents
    based on term frequency (TF) and inverse document frequency (IDF).

    TF:  how often the query term appears in a chunk
         → chunks that mention "p-value" 5 times score higher than 1 time
    IDF: how rare the term is across all chunks
         → "the" appears everywhere so it gets low weight
         → "heteroscedasticity" is rare so it gets high weight

    The "25" refers to the 25th iteration of the BM25 formula — it has
    two tuning parameters (k1=1.5, b=0.75) that are well-calibrated
    for most English text.

    LEARNING — why in-memory vs persistent?
    BM25 is fast to build (< 1s for 1000 chunks) and cheap to store.
    Building it fresh at query time from the Qdrant payload means it
    always reflects the current collection state without any sync logic.
    In Phase 3 we can cache it if query latency becomes a concern.
    """

    def __init__(self) -> None:
        self._index: Optional[BM25Okapi] = None
        self._chunks: List[str] = []
        self._metadata: List[dict] = []

    def build(
        self,
        chunks: List[str],
        metadata: List[dict],
    ) -> None:
        """
        Build the BM25 index from a list of chunk texts.

        LEARNING — tokenisation matters:
        BM25 operates on tokens (words). We lowercase and split on
        non-alphanumeric characters. Better tokenisation (stemming,
        stopword removal) would improve precision but adds complexity.
        For Phase 2 this simple tokeniser is sufficient.

        Args:
            chunks:   list of raw chunk texts
            metadata: list of dicts with source_file, page_number per chunk
        """
        start = time.perf_counter()

        self._chunks = chunks
        self._metadata = metadata
        tokenised = [self._tokenise(chunk) for chunk in chunks]
        self._index = BM25Okapi(tokenised)

        logger.info(
            "bm25_index_built",
            num_chunks=len(chunks),
            latency_ms=round((time.perf_counter() - start) * 1000),
        )

    def search(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[str, str, Optional[int], float]]:
        """
        Search the BM25 index for the top-k matching chunks.

        Returns list of (text, source_file, page_number, normalised_score).

        LEARNING — score normalisation:
        BM25 scores are not bounded (can be 0 to infinity depending on
        the corpus). We normalise by dividing by the max score in the
        result set so scores are comparable with Qdrant cosine similarity
        scores (0 to 1) when we fuse them in RRF.
        """
        if self._index is None or not self._chunks:
            return []

        tokenised_query = self._tokenise(query)
        scores = self._index.get_scores(tokenised_query)

        # Get top-k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        # Normalise scores to 0-1 range
        max_score = scores[top_indices[0]] if scores[top_indices[0]] > 0 else 1.0

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue    # skip zero-score chunks — not relevant
            meta = self._metadata[idx]
            normalised = float(scores[idx]) / max_score
            results.append((
                self._chunks[idx],
                meta.get("source_file", "unknown"),
                meta.get("page_number"),
                normalised,
            ))

        return results

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """
        Simple tokeniser: lowercase + split on non-alphanumeric chars.

        LEARNING — tokenisation options from simple to complex:
          1. split()                    → splits on whitespace only
          2. re.split                   → splits on non-word chars (we use this)
          3. nltk word_tokenize         → handles contractions, punctuation
          4. spacy                      → full NLP pipeline, overkill for RAG

        Option 2 is the right balance for Phase 2.
        """
        return [
            token for token in re.split(r'\W+', text.lower())
            if len(token) > 1    # skip single chars like "a", "i"
        ]