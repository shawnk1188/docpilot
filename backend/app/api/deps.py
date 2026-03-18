from fastapi import Request
from app.services.vector_store import VectorStoreService
from app.services.embedder import EmbeddingService
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService


def get_vector_store(request: Request) -> VectorStoreService:
    return VectorStoreService(request.app.state.qdrant)


def get_embedder(request: Request) -> EmbeddingService:
    return request.app.state.embedder


def get_ingestion_service(request: Request) -> IngestionService:
    return IngestionService(
        vector_store=get_vector_store(request),
        embedder=get_embedder(request),
    )


def get_retrieval_service(request: Request) -> RetrievalService:
    return RetrievalService(
        vector_store=get_vector_store(request),
        embedder=get_embedder(request),
    )