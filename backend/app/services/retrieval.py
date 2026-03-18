import time
from app.core.logging import logger
from app.core.metrics import (
    QUERY_LATENCY, RETRIEVAL_SCORE, LOW_CONFIDENCE_QUERIES
)

async def query(self, question: str, top_k: int) -> QueryResponse:
    start = time.perf_counter()

    logger.info("query_started", question=question[:80], top_k=top_k)

    query_vector = self._embedder.embed_query(question)
    results = await self._qdrant.search(query_vector=query_vector, top_k=top_k)

    if not results:
        return QueryResponse(
            answer="No documents ingested yet.",
            sources=[],
            model=settings.ollama_model,
        )

    # Record retrieval scores — this is the key RAG health signal
    scores = [r[3] for r in results]
    for score in scores:
        RETRIEVAL_SCORE.observe(score)

    # Flag low confidence queries — top chunk below 0.5 is a warning sign
    if scores[0] < 0.5:
        LOW_CONFIDENCE_QUERIES.inc()
        logger.warning(
            "low_confidence_retrieval",
            question=question[:80],
            top_score=round(scores[0], 3),
        )

    context = self._build_context(results)
    answer = await self._call_ollama(question, context)

    latency = time.perf_counter() - start
    QUERY_LATENCY.observe(latency)

    logger.info(
        "query_complete",
        question=question[:80],
        top_score=round(scores[0], 3),
        mean_score=round(sum(scores) / len(scores), 3),
        latency_ms=round(latency * 1000),
        chunks_used=len(results),
    )

    sources = [
        Source(
            text=text[:500],
            source_file=source_file,
            page_number=page_num,
            score=round(score, 3),
        )
        for text, source_file, page_num, score in results
    ]

    return QueryResponse(answer=answer, sources=sources, model=settings.ollama_model)