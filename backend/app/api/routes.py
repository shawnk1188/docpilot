# backend/app/api/routes.py

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from app.core.config import settings
from app.models.schemas import IngestResponse, StatsResponse
from app.api.deps import get_ingestion_service, get_vector_store
from app.services.ingestion import IngestionService
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

    # Write upload to a temp file — LlamaIndex needs a real file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
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


@router.get("/stats", response_model=StatsResponse)
async def stats(store: VectorStoreService = Depends(get_vector_store)):
    """How many chunks are stored and what settings were used."""
    return StatsResponse(
        collection=settings.qdrant_collection,
        total_chunks=await store.count(),
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )