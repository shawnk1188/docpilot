from __future__ import annotations

import time
from functools import lru_cache
from typing import List, Tuple, Optional

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import logger


@lru_cache(maxsize=1)
def _load_reranker() -> CrossEncoder:
    """
    Load the cross-encoder model once and cache it.

    LEARNING — cross-encoder vs bi-encoder:

    Bi-encoder (what we use for embeddings):
      - Encodes question and chunk SEPARATELY into vectors
      - Similarity = dot product of vectors
      - Fast: encode once, compare many
      - Used for: initial retrieval over large collections

    Cross-encoder (what we use for re-ranking):
      - Encodes question and chunk TOGETHER in one forward pass
      - Sees both texts simultaneously — can model interactions
      - Score = single relevance score from the model
      - Slow: must run once per (question, chunk) pair
      - Much more accurate than bi-encoder
      - Used for: re-scoring a small candidate set (top-20)

    This two-stage approach (fast retrieval + accurate re-ranking) is
    the industry standard. It's the same pattern used by:
      - Google (BM25 first pass + neural re-ranking)
      - Bing, Elasticsearch neural search
      - Most production RAG systems

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - Trained on MS MARCO passage ranking dataset
      - 22M parameters — fast on CPU
      - ~80MB download
    """
    model_name = getattr(
        settings,
        "reranker_model",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    return CrossEncoder(model_name)


class RerankerService:
    """
    Re-scores a candidate set of chunks using a cross-encoder model.

    LEARNING — why re-rank at all?
    Vector search and BM25 both score question and chunk independently.
    A cross-encoder looks at both together — it can tell that
    "What causes rain?" and "Precipitation occurs when water vapor
    condenses" are highly relevant even though they share no words and
    vector similarity is moderate.

    The cost: cross-encoding 20 pairs takes ~200ms on CPU.
    The benefit: top-5 precision typically improves by 15-25%.
    This is always worth it for a user-facing Q&A system.
    """

    def rerank(
        self,
        question: str,
        candidates: List[Tuple[str, str, Optional[int], float]],
        top_k: int,
    ) -> List[Tuple[str, str, Optional[int], float]]:
        """
        Re-score candidates with the cross-encoder and return top_k.

        Args:
            question:   the user question
            candidates: list of (text, source_file, page_num, score)
            top_k:      how many to return after re-ranking

        Returns:
            top_k candidates re-ordered by cross-encoder score.
        """
        if not candidates:
            return []

        start = time.perf_counter()
        model = _load_reranker()

        # Build (question, chunk) pairs for the cross-encoder
        pairs = [(question, chunk[0]) for chunk in candidates]

        # Score all pairs in one batch call
        # LEARNING: predict() returns raw logit scores (not 0-1)
        # Higher score = more relevant. We normalise below.
        scores = model.predict(pairs)

        # Attach cross-encoder scores to candidates and sort
        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Normalise scores to 0-1 using sigmoid
        # LEARNING: sigmoid(x) = 1 / (1 + e^-x)
        # This maps raw logits to probabilities
        import math
        result = []
        for (text, source_file, page_num, _), raw_score in scored[:top_k]:
            normalised = 1.0 / (1.0 + math.exp(-float(raw_score)))
            result.append((text, source_file, page_num, round(normalised, 3)))

        latency = time.perf_counter() - start
        logger.info(
            "reranker_complete",
            question=question[:80],
            candidates_in=len(candidates),
            returned=len(result),
            top_score=result[0][3] if result else 0,
            latency_ms=round(latency * 1000),
        )

        return result