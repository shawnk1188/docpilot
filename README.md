# docpilot

> Production-grade Retrieval Augmented Generation (RAG) system.
> Ask questions about your documents and get cited answers.

## Phases
- [x] Phase 1 · Fundamentals — ingestion, chunking, vector search, citations
- [ ] Phase 2 · Production — hybrid BM25 + vector retrieval, cross-encoder reranking
- [ ] Phase 3 · Evaluation — Ragas metrics, golden dataset, CI/CD pipeline

## Stack
| Component | Tool |
|-----------|------|
| Embedding | `all-MiniLM-L6-v2` (local) |
| Vector store | Qdrant |
| LLM | Ollama `llama3` (local) |
| API | FastAPI |
| UI | Streamlit |
| Containers | Podman Compose |

## Quick start
_Coming in commit 2 once the ingestion pipeline is built._
