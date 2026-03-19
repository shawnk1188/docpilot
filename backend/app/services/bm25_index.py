
from __future__ import annotations

import re
import time
from typing import List, Tuple, Optional

from rank_bm25 import BM25Okapi

from app.core.logging import logger


class BM25Index:
    """
    In-memory BM25 keyword index.

    LEARNING — why BM25 alongside vector search:
    Vector search finds semantically similar chunks.
    BM25 finds exact keyword matches.

    Example where BM25 wins:
      Query: "p-value < 0.05"
      Vector: returns chunks about statistics generally
      BM25:   returns the exact chunk mentioning p-value threshold

    Example where vector wins:
      Query: "What causes variation in measurements?"
      BM25:   needs exact words — misses "variance" and "standard deviation"
      Vector: finds all semantically related chunks

    Together they cover each other's blind spots.
    """

    def __init__(self) -> None:
        self._index = None
        self._chunks: List[str] = []
        self._metadata: List[dict] = []

    def build(self, chunks: List[str], metadata: List[dict]) -> None:
        """
        Build the BM25 index from chunk texts.

        LEARNING — tokenisation:
        We lowercase and split on non-alphanumeric chars.
        "What is a p-value?" becomes ["what", "is", "p", "value"]
        Single chars are skipped — they add noise not signal.
        """
        start = time.perf_counter()
        self._chunks   = chunks
        self._metadata = metadata
        tokenised      = [self._tokenise(c) for c in chunks]
        self._index    = BM25Okapi(tokenised)
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
        Search BM25 index. Returns (text, source_file, page_num, score).

        LEARNING — score normalisation:
        BM25 scores are unbounded (0 to infinity).
        We normalise by dividing by max score so BM25 scores are
        comparable with Qdrant cosine similarity scores (0 to 1)
        when RRF fuses the two result lists.
        """
        if self._index is None or not self._chunks:
            return []

        scores      = self._index.get_scores(self._tokenise(query))
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        max_score = scores[top_indices[0]] if scores[top_indices[0]] > 0 else 1.0
        results   = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self._metadata[idx]
            results.append((
                self._chunks[idx],
                meta.get("source_file", "unknown"),
                meta.get("page_number"),
                float(scores[idx]) / max_score,
            ))
        return results

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return [
            t for t in re.split(r"\W+", text.lower())
            if len(t) > 1
        ]
