# docpilot

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.17-DC382D?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Groq](https://img.shields.io/badge/Groq-LLM-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![Ollama](https://img.shields.io/badge/Ollama-local-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge)](https://docs.astral.sh/uv)
[![Podman](https://img.shields.io/badge/Podman-containers-892CA0?style=for-the-badge&logo=podman&logoColor=white)](https://podman.io)
[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io)
[![Grafana](https://img.shields.io/badge/Grafana-dashboards-F46800?style=for-the-badge&logo=grafana&logoColor=white)](https://grafana.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**Production-grade Retrieval Augmented Generation (RAG) system.**
Ask questions about your documents and get cited answers — built in three phases from fundamentals to CI-evaluated production RAG.

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [Phases](#-phases) · [API Reference](#-api-reference) · [Observability](#-observability) · [Contributing](#-contributing)

</div>

---

## What is docpilot?

docpilot is a domain-specific "ask my docs" system. Upload any PDF, Word document, or text file and ask questions about it in plain English. Every answer comes with citations — the exact source chunks used to generate it, with filename, page number, and relevance score.

Unlike a plain chatbot, docpilot is **grounded** — the LLM can only answer from retrieved document content. If the answer isn't in your documents, it says so.

---

## ✨ Features

- 📄 **Multi-format ingestion** — PDF, DOCX, TXT, Markdown
- 🔍 **Semantic search** — `all-MiniLM-L6-v2` embeddings stored in Qdrant
- 🤖 **Dual LLM support** — Groq (cloud, fast, free tier) or Ollama (local, offline)
- 📎 **Citations** — every answer shows source file, page number, and relevance score
- 🎯 **Grounded answers** — LLM constrained to retrieved context, abstains when unsure
- 📊 **RAG-specific metrics** — retrieval scores, low confidence detection, latency
- 📈 **Grafana dashboards** — live observability out of the box
- 🐳 **Fully containerised** — runs locally with one command via Podman Compose
- ⚡ **uv package manager** — 10–100x faster than pip, reproducible lockfiles
- 🧪 **Tested** — 15 tests covering pipeline, citations, provider routing

---

## 🏗️ Architecture

```
                        User / browser
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Streamlit  :8501   │
                   │  Chat UI + citations│
                   └──────────┬──────────┘
                              │ HTTP POST /api/query
                              │ HTTP POST /api/ingest
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Podman network: docpilot-net                    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI  :8000                            │   │
│  │                                                              │   │
│  │  ┌─────────────────┐      ┌──────────────────────────────┐  │   │
│  │  │IngestionService │      │     RetrievalService         │  │   │
│  │  │                 │      │                              │  │   │
│  │  │ load → chunk    │      │ embed → search → generate    │  │   │
│  │  │ embed → store   │      │ → cite                       │  │   │
│  │  └────────┬────────┘      └──────────┬───────────────────┘  │   │
│  │           │                          │                       │   │
│  │           ▼                          ▼                       │   │
│  │  ┌─────────────────┐      ┌──────────────────────────────┐  │   │
│  │  │  EmbeddingService│      │       LLM Provider           │  │   │
│  │  │ all-MiniLM-L6-v2│      │  Groq (default) │  Ollama   │  │   │
│  │  │  384 dimensions  │      │  llama-3.1-8b   │  local    │  │   │
│  │  └────────┬─────────┘      └──────────────────────────────┘  │   │
│  └───────────┼──────────────────────────────────────────────────┘   │
│              │                                                       │
│              ▼                                                       │
│  ┌───────────────────────┐   ┌──────────────┐   ┌───────────────┐  │
│  │    Qdrant  :6333      │   │ Prometheus   │   │   Grafana     │  │
│  │    Vector store       │   │   :9090      │   │   :3000       │  │
│  │    1415+ chunks       │   │   metrics    │   │  dashboards   │  │
│  └───────────────────────┘   └──────────────┘   └───────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Container overview

| Container | Image | Port | Role |
|-----------|-------|------|------|
| `streamlit-app` | python:3.12-slim | 8501 | Chat UI with citations |
| `fastapi-app` | python:3.12-slim | 8000 | REST API + RAG pipeline |
| `qdrant` | qdrant/qdrant | 6333 | Vector store |
| `prometheus` | prom/prometheus | 9090 | Metrics scraping |
| `grafana` | grafana/grafana | 3000 | Dashboards |
| `ollama` _(optional)_ | ollama/ollama | 11434 | Local LLM (profile: local) |

---

## 🚀 Getting Started

### Prerequisites

- [Podman](https://podman.io/getting-started/installation) + [podman-compose](https://github.com/containers/podman-compose)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — `brew install uv`
- A free [Groq API key](https://console.groq.com) — or run Ollama locally

### 1. Clone and configure

```bash
git clone https://github.com/shawnk1188/docpilot.git
cd docpilot

cp .env.example .env
# edit .env → set GROQ_API_KEY=gsk_...
```

### 2. Generate lockfiles

```bash
make lock
```

### 3. Start all containers

```bash
make up
```

### 4. Open the app

| Service | URL | Credentials |
|---------|-----|-------------|
| 💬 Chat UI | http://localhost:8501 | — |
| 📖 API Docs | http://localhost:8000/docs | — |
| 📊 Grafana | http://localhost:3000 | admin / docpilot |
| 🔍 Qdrant | http://localhost:6333/dashboard | — |
| 📈 Prometheus | http://localhost:9090 | — |

### 5. Upload a document and ask questions

1. Open http://localhost:8501
2. Upload a PDF using the sidebar uploader
3. Click **Ingest** — wait for the chunk count
4. Ask any question in the chat input
5. Expand **📎 N source(s)** to verify citations

---

## 📋 Phases

### ✅ Phase 1 — Fundamentals (complete)

**Goal:** end-to-end document Q&A with citations running locally in containers.

#### Step 1a — Repository skeleton
- Three-phase project structure established
- `Makefile` with all dev commands
- `.gitignore`, `README.md`, `podman-compose.yml` stubs

#### Step 1b — Ingestion pipeline
Built the **offline** pipeline that runs once per document:

```
PDF / DOCX / TXT / MD
        │
        ▼
SimpleDirectoryReader (LlamaIndex)
        │  extracts text + page numbers per page
        ▼
SentenceSplitter
        │  chunk_size=600 tokens, chunk_overlap=100 tokens
        │  respects sentence boundaries, no mid-sentence cuts
        ▼
EmbeddingService (all-MiniLM-L6-v2)
        │  384-dimensional vectors, batched, lru_cache model load
        ▼
VectorStoreService (Qdrant)
        │  upsert points: {id, vector, payload{text, file, page}}
        ▼
Qdrant collection "docpilot"
```

Key files: `ingestion.py` · `embedder.py` · `vector_store.py` · `config.py`

#### Step 1c — Query pipeline + citations
Built the **online** pipeline that runs on every user question:

```
User question
        │
        ▼
EmbeddingService.embed_query()
        │  same model as ingestion — vectors in same space
        ▼
VectorStoreService.search()
        │  cosine similarity, top-k=5 chunks
        │  score > 0.75 = strong · 0.5–0.75 = moderate · < 0.5 = low
        ▼
RetrievalService._build_context()
        │  [Chunk N — filename · page X (relevance: 0.87)]
        │  --- separator between chunks
        ▼
LLM generation (Groq / Ollama)
        │  system prompt: answer ONLY from context, abstain if unsure
        │  temperature=0.1 for factual accuracy
        ▼
QueryResponse { answer, sources[], model }
        │  sources: text, source_file, page_number, score
        ▼
Streamlit chat UI
        │  answer rendered in markdown
        │  citations in expandable panel, colour-coded by score
```

Key files: `retrieval.py` · `routes.py` · `frontend/app.py`

#### Observability (phase 1)
Full observability stack running alongside the RAG pipeline:

**Structured logging** via `structlog` — every ingestion and query emits a JSON log line:
```json
{
  "event": "query_complete",
  "question": "what is this document about",
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "top_score": 0.847,
  "mean_score": 0.731,
  "latency_ms": 1243,
  "chunks_used": 5
}
```

**Prometheus metrics** — RAG-specific signals:

| Metric | Type | What it measures |
|--------|------|-----------------|
| `docpilot_documents_ingested_total` | Counter | ingestions by status (success/failure) |
| `docpilot_chunks_per_document` | Histogram | chunk count distribution |
| `docpilot_ingestion_latency_seconds` | Histogram | end-to-end ingest time |
| `docpilot_query_latency_seconds` | Histogram | end-to-end query time |
| `docpilot_retrieval_score` | Histogram | cosine similarity distribution |
| `docpilot_low_confidence_queries_total` | Counter | queries with top score < 0.5 |
| `docpilot_total_chunks` | Gauge | current chunks in Qdrant |

**Grafana dashboard** — auto-provisioned, no manual setup:
- Stat panels: total chunks, low confidence queries, avg latency, avg score
- Time series: query latency p95, retrieval score distribution, ingest rate

#### LLM provider abstraction
Switch between Groq and Ollama with a single `.env` change:

```env
# Groq (default — cloud, fast, free tier, no RAM issues)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant

# Ollama (local, offline, no API key)
LLM_PROVIDER=ollama
OLLAMA_MODEL=tinyllama
```

Zero code changes — the provider abstraction in `config.py` resolves the correct base URL, model, and API key automatically.

---

### 🔲 Phase 2 — Production retrieval (upcoming)

- BM25 keyword index alongside vector search
- Reciprocal Rank Fusion (RRF) to combine both result sets
- Cross-encoder re-ranker on top-20 candidates
- Score threshold filtering

---

### 🔲 Phase 3 — Continuous evaluation (upcoming)

- Golden QA dataset (50–200 question-answer pairs)
- Ragas evaluation: faithfulness, answer relevancy, context recall
- GitHub Actions CI/CD — fail pipeline if faithfulness drops below 0.75
- MLflow experiment tracking across chunking configurations

---

## 📁 Project Structure

```
docpilot/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, lifespan, health
│   │   ├── core/
│   │   │   ├── config.py            # all settings + provider resolution
│   │   │   ├── logging.py           # structlog JSON setup
│   │   │   └── metrics.py           # Prometheus RAG metrics
│   │   ├── api/
│   │   │   ├── routes.py            # POST /ingest, POST /query, GET /stats
│   │   │   └── deps.py              # FastAPI dependency injection
│   │   ├── models/
│   │   │   └── schemas.py           # Source, QueryRequest, QueryResponse
│   │   └── services/
│   │       ├── ingestion.py         # load → chunk → embed → store
│   │       ├── embedder.py          # all-MiniLM-L6-v2, lru_cache
│   │       ├── retrieval.py         # embed → search → generate → cite
│   │       └── vector_store.py      # async Qdrant client
│   ├── tests/
│   │   ├── test_ingestion.py        # chunking + embedding tests
│   │   └── test_retrieval.py        # 15 tests — pipeline, citations, providers
│   ├── pyproject.toml               # uv dependencies
│   ├── uv.lock                      # exact pinned versions
│   └── Dockerfile
├── frontend/
│   ├── app.py                       # Streamlit chat UI + citation panel
│   ├── pyproject.toml
│   ├── uv.lock
│   └── Dockerfile
├── infra/
│   ├── prometheus/
│   │   └── prometheus.yml           # scrape config
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/         # auto-connects Prometheus
│       │   └── dashboards/          # auto-loads dashboard JSON
│       └── dashboards/
│           └── docpilot.json        # RAG metrics dashboard
├── notebooks/
│   ├── 01_chunking.ipynb            # experiment with chunk_size/overlap
│   ├── 02_embeddings.ipynb          # visualise embedding similarity
│   ├── 03_retrieval_debug.ipynb     # inspect retrieved chunks per query
│   └── 04_ragas_eval.ipynb          # Phase 3 evaluation (coming)
├── docs/sample/                     # drop documents here
├── podman-compose.yml
├── Makefile
├── .env.example
└── README.md
```

---

## 💬 API Reference

Full interactive docs at **http://localhost:8000/docs**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ingest` | Upload and ingest a document |
| `POST` | `/api/query` | Ask a question, get answer + citations |
| `GET` | `/api/stats` | Collection stats and current settings |
| `GET` | `/health` | Health check — Qdrant + provider status |
| `GET` | `/metrics` | Prometheus scrape endpoint |

### Query request

```json
{
  "question": "What are the key findings?",
  "top_k": 5
}
```

### Query response

```json
{
  "answer": "According to the report, the key findings include...",
  "sources": [
    {
      "text": "The analysis revealed three primary findings...",
      "source_file": "annual_report.pdf",
      "page_number": 12,
      "score": 0.891
    }
  ],
  "model": "llama-3.1-8b-instant"
}
```

---

## 🛠️ Makefile Commands

```bash
# ── Setup ──────────────────────────────────────────────────────────────
make up              # start all containers (Groq mode, default)
make up-local        # start with Ollama local LLM
make setup-local     # start + pull tinyllama model
make rebuild         # full no-cache image rebuild
make lock            # regenerate uv.lock files

# ── Daily use ──────────────────────────────────────────────────────────
make ingest          # prompted file path → ingest document
make ask             # prompted question → query via terminal
make status          # health check all services
make test-groq       # verify Groq API connection

# ── Logs ───────────────────────────────────────────────────────────────
make logs            # tail fastapi-app logs
make logs-ui         # tail streamlit-app logs
make logs-all        # tail all container logs

# ── Dependencies ────────────────────────────────────────────────────────
make deps            # install deps locally via uv sync
make add             # add a package to a service
make remove          # remove a package from a service

# ── Testing ─────────────────────────────────────────────────────────────
make test            # run pytest suite (15 tests)
make test-watch      # run pytest in watch mode

# ── Cleanup ─────────────────────────────────────────────────────────────
make clean           # remove containers + prune networks/volumes
make down            # stop all containers
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `GROQ_API_KEY` | — | **Required for Groq.** Get at console.groq.com |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama container URL |
| `OLLAMA_MODEL` | `tinyllama` | Ollama model name |
| `QDRANT_HOST` | `qdrant` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `docpilot` | Collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `VECTOR_DIM` | `384` | Must match embedding model output |
| `CHUNK_SIZE` | `600` | Tokens per chunk (tune: 400–800) |
| `CHUNK_OVERLAP` | `100` | Overlap tokens between chunks (~15% of chunk_size) |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per query (tune: 3–10) |

---

## 📊 Observability

### Grafana dashboard

Open http://localhost:3000 (admin / docpilot)

The docpilot dashboard auto-provisions on startup. Panels:

- **Total chunks** — how much content is indexed
- **Low confidence queries** — queries where top retrieval score < 0.5
- **Avg query latency** — end-to-end response time
- **Avg retrieval score** — mean cosine similarity across queries
- **Query latency over time** — p50 and p95 trends
- **Score distribution** — mean score and p10 worst score
- **Ingest rate** — documents ingested over time

### Reading retrieval scores

| Score | Meaning | Action |
|-------|---------|--------|
| > 0.75 | Strong match | Answer is well-grounded |
| 0.5–0.75 | Moderate match | Answer may be partially relevant |
| < 0.5 | Weak match | Low confidence — answer may be unreliable |

A sustained drop in mean retrieval score signals that users are asking questions outside the scope of ingested documents, or that chunk size needs tuning.

---

## 🐛 Troubleshooting

**`streamlit not found in $PATH`**
Missing `ENV PATH="/app/.venv/bin:$PATH"` in frontend Dockerfile. Add it and run `make rebuild`.

**`ModuleNotFoundError: No module named 'sentence_transformers'`**
Use `uv run python` not bare `python` in the Dockerfile pre-bake step.

**`model requires more system memory than is available`**
Switch to a smaller model. Update `.env`: `OLLAMA_MODEL=tinyllama` and `make rebuild`.

**`error: No pyproject.toml found`**
You ran `uv add` from the wrong directory. `cd backend` first, then `uv add <package>`.

**Citations show `tmpmcvlww22.pdf`**
Old ingested data from before the filename fix. Re-ingest the document.

**`401 Unauthorized` on Groq queries**
`GROQ_API_KEY` in `.env` is missing or wrong. Check at console.groq.com.

**Low confidence queries spiking in Grafana**
Users are asking about topics not in ingested documents, or `CHUNK_SIZE` needs tuning. Try lowering to 400 tokens.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit: `git commit -m 'feat: description'`
4. Push and open a PR

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
