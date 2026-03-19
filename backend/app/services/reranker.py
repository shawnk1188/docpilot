
from __future__ import annotations

import math
import time
from functools import lru_cache
from typing import List, Tuple, Optional

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import logger


@lru_cache(maxsize=1)
def _load_reranker() -> CrossEncoder:
    """
    Load cross-encoder model once and cache it.

    LEARNING — cross-encoder vs bi-encoder:

    Bi-encoder (used for embeddings):
      Encodes question and chunk SEPARATELY → vectors
      Similarity = dot product
      Fast — encode once, compare many
      Used for: initial retrieval over large collections

    Cross-encoder (used here for re-ranking):
      Encodes question and chunk TOGETHER in one pass
      Sees both texts simultaneously — models interactions
      Returns a single relevance score
      Slow — must run once per (question, chunk) pair
      Much more accurate than bi-encoder
      Used for: re-scoring a small candidate set (top 20)

    This two-stage pattern (fast retrieval + accurate re-ranking)
    is the industry standard — same approach used by Google, Bing,
    and most production search systems.
    """
    return CrossEncoder(settings.reranker_model)


class RerankerService:
    """Re-scores candidates using a cross-encoder model."""

    def rerank(
        self,
        question: str,
        candidates: List[Tuple[str, str, Optional[int], float]],
        top_k: int,
    ) -> List[Tuple[str, str, Optional[int], float]]:
        """
        Re-score candidates and return top_k.

        LEARNING — sigmoid normalisation:
        Cross-encoder returns raw logit scores (unbounded).
        sigmoid(x) = 1 / (1 + e^-x) maps them to 0-1 range
        so scores are comparable with retrieval scores.

        Typical improvement: 15-25% precision over bi-encoder alone.
        Cost: ~200ms on CPU for 20 candidates. Worth it.
        """
        if not candidates:
            return []

        start  = time.perf_counter()
        model  = _load_reranker()
        pairs  = [(question, c[0]) for c in candidates]
        scores = model.predict(pairs)

        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for (text, source_file, page_num, _), raw in scored[:top_k]:
            normalised = 1.0 / (1.0 + math.exp(-float(raw)))
            result.append((text, source_file, page_num, round(normalised, 3)))

        logger.info(
            "reranker_complete",
            question=question[:80],
            candidates_in=len(candidates),
            returned=len(result),
            top_score=result[0][3] if result else 0,
            latency_ms=round((time.perf_counter() - start) * 1000),
        )

        return result
