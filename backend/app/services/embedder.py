# backend/app/services/embedder.py

from __future__ import annotations
from functools import lru_cache
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import settings


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """
    Load the embedding model exactly once and cache it in memory.

    LEARNING — lru_cache(maxsize=1) means:
      - First call: loads the model from disk (~80MB), takes ~2 seconds
      - Every subsequent call: returns the already-loaded model instantly
      - Without this: every request would reload the model — catastrophic
    """
    return SentenceTransformer(settings.embedding_model)


class EmbeddingService:
    """
    Converts text into vectors using a local sentence-transformer model.

    LEARNING — what an embedding actually is:
    The model reads a chunk of text and outputs a list of 384 numbers.
    These numbers encode the *meaning* of the text geometrically —
    chunks with similar meaning produce vectors that point in similar
    directions in 384-dimensional space.

    "The patient showed elevated blood pressure" and
    "The doctor noted hypertension in the exam" will produce
    vectors very close together, even though they share no words.
    This is the magic that makes semantic search work.
    """

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts in a single model call.

        LEARNING — always batch, never loop:
        The model processes a batch as one matrix operation.
        Embedding 100 texts in one call takes roughly the same
        time as embedding 10 texts in one call. Calling embed()
        100 times in a loop is ~10x slower. Always pass a list.
        """
        model = _load_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,   # makes cosine sim == dot product
            show_progress_bar=False,
            batch_size=32,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string at retrieval time.

        Note: we call embed_texts() with a list of one item —
        no special handling needed, batching works for size 1 too.
        """
        return self.embed_texts([query])[0]