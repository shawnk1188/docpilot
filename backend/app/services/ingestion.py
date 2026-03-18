from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Optional

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from app.core.config import settings
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService

import time
from app.core.logging import logger
from app.core.metrics import (
    DOCS_INGESTED, CHUNKS_STORED,
    INGESTION_LATENCY, TOTAL_CHUNKS_IN_STORE
)



class IngestionService:
    """
    Runs the offline ingestion pipeline: load → chunk → embed → store.

    LEARNING — why this is offline (not per-request):
    Ingesting a 100-page PDF takes 30–60 seconds. You run it once
    when a document is uploaded, store everything in Qdrant, and
    then every query is fast (< 1 second) because the heavy work
    is already done.
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
        embedder: EmbeddingService,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder

    async def ingest_file(self, file_path: str) -> Tuple[int, str]:
        source_filename = Path(file_path).name
        start = time.perf_counter()

        logger.info("ingestion_started", file=source_filename)

        try:
            docs = self._load_documents(file_path)
            if not docs:
                raise ValueError(f"No text extracted from {source_filename}")

            chunks, page_numbers = self._chunk(docs)
            embeddings = self._embedder.embed_texts(chunks)

            await self._store.delete_by_source(source_filename)
            await self._store.ensure_collection()
            stored = await self._store.upsert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                source_file=source_filename,
                page_numbers=page_numbers,
            )

            latency = time.perf_counter() - start

            # Record metrics
            DOCS_INGESTED.labels(status="success").inc()
            CHUNKS_STORED.observe(stored)
            INGESTION_LATENCY.observe(latency)
            TOTAL_CHUNKS_IN_STORE.set(await self._store.count())

            logger.info(
                "ingestion_complete",
                file=source_filename,
                chunks=stored,
                latency_ms=round(latency * 1000),
            )

            return stored, source_filename

        except Exception as e:
            DOCS_INGESTED.labels(status="failure").inc()
            logger.error("ingestion_failed", file=source_filename, error=str(e))
            raise

    def _chunk(self, docs) -> Tuple[List[str], List[Optional[int]]]:
        """
        Split documents into overlapping chunks.

        LEARNING — SentenceSplitter vs character splitting:
        SentenceSplitter respects sentence boundaries. It will never
        cut a sentence in half. This produces cleaner chunks than
        splitting every N characters, which can cut mid-word.

        The chunk_size here is in TOKENS (not characters).
        1 token ≈ 0.75 words in English.
        600 tokens ≈ 450 words ≈ roughly 1.5 paragraphs.

        The overlap means:
          Chunk 1: tokens 0–600
          Chunk 2: tokens 500–1100   ← 100 tokens repeated
          Chunk 3: tokens 1000–1600  ← 100 tokens repeated

        This means any answer that sits near a chunk boundary
        is fully contained in at least one of the two adjacent chunks.
        """
        splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        nodes = splitter.get_nodes_from_documents(docs)

        chunks, page_numbers = [], []
        for node in nodes:
            chunks.append(node.get_content())
            raw_page = (
                node.metadata.get("page_label")
                or node.metadata.get("page")
            )
            try:
                page_numbers.append(int(raw_page) if raw_page else None)
            except (ValueError, TypeError):
                page_numbers.append(None)

        return chunks, page_numbers