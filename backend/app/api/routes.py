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
):
    """Upload and ingest a document into the vector store."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported type '{ext}'. Allowed: {ALLOWED}",
        )

    # FIX from step 1b: preserve original filename in the temp file
    # so citations show the real name, not tmpmcvlww22.pdf
    original_name = Path(file.filename).stem   # e.g. "my_manual"
    with tempfile.NamedTemporaryFile(
        delete=False,
        prefix=f"{original_name}_",
        suffix=ext,
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        count, source = await svc.ingest_file(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

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