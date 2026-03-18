# backend/app/models/schemas.py

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    One citation — a single chunk used to generate the answer.

    LEARNING — this is what separates RAG from a hallucinating chatbot.
    Every answer comes with the exact chunks that were retrieved.
    The user can verify the answer by reading the source text themselves.
    """
    text: str
    source_file: str
    page_number: Optional[int] = None
    score: float                     # cosine similarity 0–1


class IngestResponse(BaseModel):
    message: str
    chunks_stored: int
    source_file: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=15)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    model: str


class StatsResponse(BaseModel):
    collection: str
    total_chunks: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int