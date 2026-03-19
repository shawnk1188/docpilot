
from __future__ import annotations
import time
from typing import List

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.core.metrics import (
    QUERY_LATENCY, RETRIEVAL_SCORE, LOW_CONFIDENCE_QUERIES,
)
from app.models.schemas import QueryResponse, Source
from app.services.bm25_index import BM25Index
from app.services.embedder import EmbeddingService
from app.services.hybrid_retriever import HybridRetriever
from app.services.reranker import RerankerService
from app.services.vector_store import VectorStoreService


_SYSTEM_PROMPT = """\
You are a precise document assistant for docpilot.
Answer the user's question using ONLY the context chunks provided below.

Rules you must follow:
1. Use ONLY information from the provided context. Never use outside knowledge.
2. If the answer is not in the context, respond with exactly:
   "I don't have enough information in the provided documents to answer this."
3. Always reference the source when you use it, e.g. "According to [filename]..."
4. Be concise and direct. Do not pad your answer.
5. If multiple chunks support the answer, synthesize them into one clear response.

Context chunks:
{context}
"""


class RetrievalService:

    def __init__(
        self,
        vector_store: VectorStoreService,
        embedder: EmbeddingService,
    ) -> None:
        self._store    = vector_store
        self._embedder = embedder
        self._bm25     = BM25Index()
        self._hybrid   = HybridRetriever(
            vector_store=vector_store,
            embedder=embedder,
            bm25=self._bm25,
        )
        self._reranker = RerankerService()

    async def query(self, question: str, top_k: int) -> QueryResponse:
        """Phase 2 RAG: hybrid retrieve → rerank → generate → cite."""
        start = time.perf_counter()

        logger.info(
            "query_started",
            question=question[:80],
            top_k=top_k,
            provider=settings.llm_provider,
            model=settings.llm_model,
        )

        # Step 1 — Hybrid retrieval (vector + BM25 + RRF)
        candidates = await self._hybrid.retrieve(
            question=question,
            top_k=top_k * 2,
            fetch_k=settings.fetch_k,
        )

        if not candidates:
            return QueryResponse(
                answer="No documents have been ingested yet. "
                       "Please upload a document first.",
                sources=[],
                model=settings.llm_model,
            )

        # Step 2 — Re-rank with cross-encoder
        if settings.reranker_enabled:
            results = self._reranker.rerank(
                question=question,
                candidates=candidates,
                top_k=top_k,
            )
        else:
            results = candidates[:top_k]

        # Step 3 — Record metrics
        scores = [r[3] for r in results]
        for score in scores:
            RETRIEVAL_SCORE.observe(score)

        if scores[0] < 0.5:
            LOW_CONFIDENCE_QUERIES.inc()
            logger.warning(
                "low_confidence_retrieval",
                question=question[:80],
                top_score=round(scores[0], 3),
            )

        # Step 4 — Generate answer
        context = self._build_context(results)
        answer  = await self._call_llm(question, context)

        latency = time.perf_counter() - start
        QUERY_LATENCY.observe(latency)

        logger.info(
            "query_complete",
            question=question[:80],
            provider=settings.llm_provider,
            model=settings.llm_model,
            top_score=round(scores[0], 3),
            mean_score=round(sum(scores) / len(scores), 3),
            latency_ms=round(latency * 1000),
            chunks_used=len(results),
            reranker=settings.reranker_enabled,
        )

        sources = [
            Source(
                text=text[:500] + "..." if len(text) > 500 else text,
                source_file=source_file,
                page_number=page_num,
                score=round(score, 3),
            )
            for text, source_file, page_num, score in results
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            model=settings.llm_model,
        )

    def _build_context(self, results) -> str:
        parts = []
        for i, (text, source_file, page_num, score) in enumerate(results, 1):
            page_info = f" · page {page_num}" if page_num else ""
            parts.append(
                f"[Chunk {i} — {source_file}{page_info} "
                f"(relevance: {score:.2f})]\n{text}"
            )
        return "\n\n---\n\n".join(parts)

    async def _call_llm(self, question: str, context: str) -> str:
        if settings.llm_provider == "groq":
            return await self._call_openai_compatible(question, context)
        return await self._call_ollama_native(question, context)

    async def _call_openai_compatible(self, question: str, context: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.llm_api_key}",
        }
        async with httpx.AsyncClient(
            base_url=settings.llm_base_url,
            headers=headers,
            timeout=60.0,
        ) as client:
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": settings.llm_model,
                    "messages": [
                        {"role": "system",
                         "content": _SYSTEM_PROMPT.format(context=context)},
                        {"role": "user", "content": question},
                    ],
                    "temperature": 0.1,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _call_ollama_native(self, question: str, context: str) -> str:
        async with httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=120.0,
        ) as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": [
                        {"role": "system",
                         "content": _SYSTEM_PROMPT.format(context=context)},
                        {"role": "user", "content": question},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
