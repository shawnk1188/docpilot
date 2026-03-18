# backend/app/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from app.api.routes import router
from app.core.config import settings
from app.services.embedder import EmbeddingService, _load_model
from app.services.vector_store import VectorStoreService
from prometheus_fastapi_instrumentator import Instrumentator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: connect to Qdrant, pre-load embedding model.
    Shutdown: close Qdrant connection cleanly.

    LEARNING — why lifespan instead of @app.on_event:
    lifespan() is the modern FastAPI pattern. Everything before
    `yield` runs at startup, everything after runs at shutdown.
    Resources on app.state are shared across all requests.
    """
    # Connect to Qdrant
    app.state.qdrant = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    # Ensure our collection exists
    store = VectorStoreService(app.state.qdrant)
    await store.ensure_collection()

    # Pre-load the embedding model once at startup (~2s, then cached)
    _load_model()
    app.state.embedder = EmbeddingService()

    yield  # app is running here

    await app.state.qdrant.close()


app = FastAPI(
    title="docpilot",
    description="Production RAG — ask your documents, get cited answers",
    version="0.1.0",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["ops"])
async def health():
    try:
        await app.state.qdrant.get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    return {
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": qdrant_ok,
        "embedding_model": settings.embedding_model,
        "llm": settings.ollama_model,
    }