import os
import shutil
import tempfile
from pathlib import Path
import httpx
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from app.core.config import settings
from app.models.schemas import (
    IngestResponse, QueryRequest, QueryResponse, StatsResponse
)
from app.api.deps import (
    get_ingestion_service, get_retrieval_service, get_vector_store
)
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/api", tags=["docpilot"])

ALLOWED = {".pdf", ".txt", ".md", ".docx"}


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest(
    file: UploadFile = File(...),
    svc: IngestionService = Depends(get_ingestion_service),
    store: VectorStoreService = Depends(get_vector_store),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported type '{ext}'. Allowed: {ALLOWED}",
        )

    # Ensure collection exists before ingesting
    # Handles the case where it was manually deleted via Qdrant API
    await store.ensure_collection()

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        count, source = await svc.ingest_file(tmp_path)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return IngestResponse(
        message=f"Ingested '{source}' successfully",
        chunks_stored=count,
        source_file=source,
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    svc: RetrievalService = Depends(get_retrieval_service),
):
    """
    Ask a question. Returns the answer with source citations.

    LEARNING — the response always includes sources[].
    Each source has: text (the chunk), source_file, page_number, score.
    The score is cosine similarity — higher means more relevant.
    Show these to users so they can verify the answer themselves.
    """
    try:
        return await svc.query(
            question=body.question,
            top_k=body.top_k,
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not reachable. Is the container running?",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stats", response_model=StatsResponse)
async def stats(store: VectorStoreService = Depends(get_vector_store)):
    """Collection stats — chunks stored and current settings."""
    return StatsResponse(
        collection=settings.qdrant_collection,
        total_chunks=await store.count(),
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

@router.delete("/clear", status_code=status.HTTP_204_NO_CONTENT)
async def clear_collection(
    store: VectorStoreService = Depends(get_vector_store)
):
    """Delete and recreate the collection — removes all ingested documents."""
    await store._client.delete_collection(settings.qdrant_collection)
    await store.ensure_collection()