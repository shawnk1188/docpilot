from __future__ import annotations
import uuid
from typing import List, Tuple, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from app.core.config import settings


class VectorStoreService:
    """
    Wraps Qdrant. Responsible for storing and searching vectors.

    LEARNING — a Qdrant "point" has exactly three parts:
      1. id      → a UUID string that uniquely identifies this chunk
      2. vector  → the embedding (list of 384 floats for all-MiniLM-L6-v2)
      3. payload → arbitrary JSON metadata we store alongside the vector
                   (the original text, filename, page number)

    At query time: we embed the question → find nearest vectors →
    return their payloads. The payload is how we get the text and
    citation info back out.
    """

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing = await self._client.get_collections()
        names = [c.name for c in existing.collections]

        if settings.qdrant_collection not in names:
            await self._client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=settings.vector_dim,
                    # LEARNING — Distance.COSINE measures the angle between
                    # two vectors. It is the standard for text embeddings
                    # because it ignores vector magnitude (document length)
                    # and only cares about direction (semantic meaning).
                    distance=Distance.COSINE,
                ),
            )

    async def upsert_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        source_file: str,
        page_numbers: List[Optional[int]],
    ) -> int:
        """Store chunks + their vectors in Qdrant. Returns count stored."""
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source_file": source_file,
                    "page_number": page_num,
                },
            )
            for chunk, embedding, page_num
            in zip(chunks, embeddings, page_numbers)
        ]

        await self._client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )
        return len(points)

    async def search(
        self,
        query_vector: List[float],
        top_k: int,
        source_file: Optional[str] = None,
    ) -> List[Tuple[str, str, Optional[int], float]]:
        """
        Find the top_k most similar chunks to the query vector.
        Returns list of (text, source_file, page_number, score).

        LEARNING — the score is cosine similarity ranging 0 to 1:
          > 0.75  strong match — almost certainly relevant
          0.5–0.75  moderate match — probably relevant
          < 0.4   weak match — likely noise, consider filtering these out
        """
        query_filter = None
        if source_file:
            query_filter = Filter(
                must=[FieldCondition(
                    key="source_file",
                    match=MatchValue(value=source_file),
                )]
            )

        results = await self._client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            (
                r.payload["text"],
                r.payload["source_file"],
                r.payload.get("page_number"),
                r.score,
            )
            for r in results
        ]

    async def delete_by_source(self, source_file: str) -> None:
        """Remove all chunks from one document. Used before re-ingestion."""
        await self._client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=Filter(
                must=[FieldCondition(
                    key="source_file",
                    match=MatchValue(value=source_file),
                )]
            ),
        )

    async def count(self) -> int:
        """Total chunks currently stored."""
        result = await self._client.count(
            collection_name=settings.qdrant_collection
        )
        return result.count