# docpilot

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.17-DC382D?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Groq](https://img.shields.io/badge/Groq-LLM-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge)](https://docs.astral.sh/uv)
[![Podman](https://img.shields.io/badge/Podman-containers-892CA0?style=for-the-badge&logo=podman&logoColor=white)](https://podman.io)
[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io)
[![Grafana](https://img.shields.io/badge/Grafana-dashboards-F46800?style=for-the-badge&logo=grafana&logoColor=white)](https://grafana.com)
[![Open In Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/shawnk1188/docpilot/blob/main/notebooks/00_colab_setup.ipynb)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**Production-grade Retrieval Augmented Generation (RAG) system.**  
Ask questions about your documents and get cited answers.  
Built in three phases — from fundamentals to CI-evaluated production RAG.

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [Results](#-evaluation-results) · [Phases](#-phases) · [API](#-api-reference) · [Observability](#-observability)

</div>

---

## What is docpilot?

docpilot is a domain-specific "ask my docs" system. Upload any PDF, Word document, or text file and ask questions in plain English. Every answer comes with citations — the exact source chunks used, with filename, page number, and relevance score.

Unlike a plain chatbot, docpilot is **grounded** — the LLM can only answer from retrieved document content. If the answer is not in your documents, it says so.

---

## ✨ Features

- 📄 **Multi-format ingestion** — PDF, DOCX, TXT, Markdown
- 🔍 **Hybrid retrieval** — BM25 keyword + semantic vector search fused with RRF
- 🎯 **Cross-encoder reranking** — 15–25% precision improvement over vector-only
- 🤖 **Dual LLM support** — Groq (cloud, fast, free tier) or Ollama (local, offline)
- 📎 **Citations** — every answer shows source file, page number, and relevance score
- 🎯 **Grounded answers** — LLM constrained to retrieved context, abstains when unsure
- 📊 **RAG-specific metrics** — retrieval scores, low confidence detection, latency tracking
- 📈 **Grafana dashboards** — live observability auto-provisioned on startup
- 🧪 **Continuous evaluation** — LLM-scored faithfulness, relevancy, and precision metrics
- 🐳 **Fully containerised** — Podman Compose or Google Colab, one command to run
- ⚡ **uv package manager** — 10–100x faster than pip, reproducible lockfiles

---

## 📊 Evaluation Results

Evaluated against a golden dataset of 10 QA pairs generated from thinkstats2.pdf.

### Phase comparison — retrieval quality

| Phase | Method | Top Score | Mean Score |
|-------|--------|-----------|------------|
| Phase 1 | Vector search only | 0.465 🔴 | 0.465 |
| Phase 2 | Hybrid BM25 + vector + RRF + reranker | **0.992** 🟢 | **0.958** |

**2x improvement in retrieval precision** after adding hybrid search and cross-encoder reranking.

### LLM quality metrics

| Metric | llama-3.1-8b | llama-3.3-70b | Threshold | Status |
|--------|-------------|--------------|-----------|--------|
| Faithfulness | 0.610 | **0.745** | 0.70 | PASS ✅ |
| Answer relevancy | 0.630 | **0.740** | 0.70 | PASS ✅ |
| Context precision | 0.800 | **0.840** | 0.65 | PASS ✅ |

**Key finding:** retrieval quality (context precision 0.84) is strong. The LLM is the bottleneck — switching from 8B to 70B parameters improves faithfulness by 22% and relevancy by 17%.

### What the metrics mean

| Metric | Question answered |
|--------|------------------|
| **Faithfulness** | Does the answer use only retrieved context? (no hallucination) |
| **Answer relevancy** | Does the answer actually address the question? |
| **Context precision** | Are the retrieved chunks actually useful for answering? |

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
                              │ HTTP
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                   Podman network: docpilot-net                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    FastAPI  :8000                        │  │
│  │                                                          │  │
│  │  ┌───────────────────┐   ┌──────────────────────────┐   │  │
│  │  │IngestionService   │   │    RetrievalService       │   │  │
│  │  │ load→chunk→embed  │   │ hybrid→rerank→generate    │   │  │
│  │  └────────┬──────────┘   └──────────┬────────────────┘   │  │
│  │           │                         │                     │  │
│  │           ▼                         ▼                     │  │
│  │  ┌────────────────┐   ┌─────────────────────────────┐    │  │
│  │  │EmbeddingService│   │      HybridRetriever         │    │  │
│  │  │all-MiniLM-L6-v2│   │  Vector + BM25 + RRF        │    │  │
│  │  │  384 dims      │   │  + CrossEncoder reranker     │    │  │
│  │  └────────────────┘   └──────────────────────────────┘   │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────┐  ┌────────▼──────┐  ┌──────┐  ┌──────────┐  │
│  │  Prometheus  │  │    Qdrant     │  │Groq  │  │ Grafana  │  │
│  │    :9090     │  │    :6333      │  │ API  │  │  :3000   │  │
│  └──────────────┘  └───────────────┘  └──────┘  └──────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Container overview

| Container | Image | Port | Role |
|-----------|-------|------|------|
| `streamlit-app` | python:3.12-slim | 8501 | Chat UI with citations |
| `fastapi-app` | python:3.12-slim | 8000 | REST API + RAG pipeline |
| `qdrant` | qdrant/qdrant | 6333 | Vector store |
| `prometheus` | prom/prometheus | 9090 | Metrics scraping |
| `grafana` | grafana/grafana | 3000 | Dashboards |
| `ollama` _(optional)_ | ollama/ollama | 11434 | Local LLM |

---

## 🚀 Getting Started

### Option A — Local (Podman)

**Prerequisites:** [Podman](https://podman.io) + [podman-compose](https://github.com/containers/podman-compose) + [uv](https://docs.astral.sh/uv) + free [Groq API key](https://console.groq.com)

```bash
git clone https://github.com/shawnk1188/docpilot.git
cd docpilot

cp .env.example .env
# edit .env → set GROQ_API_KEY=gsk_...

cd backend && uv lock && cd ../frontend && uv lock && cd ..

make up
```

### Option B — Google Colab (no local setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shawnk1188/docpilot/blob/main/notebooks/00_colab_setup.ipynb)

1. Click the badge above
2. Set runtime to **T4 GPU**
3. Add secrets: `GROQ_API_KEY` and `NGROK_TOKEN`
4. Run all cells

### Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| 💬 Chat UI | http://localhost:8501 | — |
| 📖 API Docs | http://localhost:8000/docs | — |
| 📊 Grafana | http://localhost:3000 | admin / docpilot |
| 🔍 Qdrant | http://localhost:6333/dashboard | — |
| 📈 Prometheus | http://localhost:9090 | — |

---

## 📋 Phases

### ✅ Phase 1 — Fundamentals

**Ingestion pipeline** (offline, runs once per document):
```
PDF / DOCX / TXT / MD
  → SimpleDirectoryReader   (LlamaIndex — extracts text + page numbers)
  → SentenceSplitter        (600 tokens, 100 overlap, sentence-aware)
  → all-MiniLM-L6-v2        (384-dim embeddings, local, no API key)
  → Qdrant                  (cosine similarity, persistent storage)
```

**Query pipeline** (online, runs per question):
```
User question
  → embed query             (same model as ingestion)
  → cosine search Qdrant    (top-k chunks)
  → Groq / Ollama           (grounded answer with citations)
  → answer + sources[]      (text, filename, page, score)
```

**Observability:**
- Structured JSON logging via `structlog`
- 7 Prometheus metrics including RAG-specific signals
- Grafana dashboard auto-provisioned on startup

### ✅ Phase 2 — Production Retrieval

**Problem solved:** vector search alone scores 0.465. BM25 covers exact keyword matches that semantic search misses.

**Hybrid retrieval pipeline:**
```
User question
  ├── Vector search (Qdrant)     semantic similarity, top-20
  └── BM25 search (rank-bm25)   keyword matching, top-20
        │
        ▼
  RRF fusion                     score = Σ 1/(k+rank), k=60
        │                        ranks not scores — no calibration needed
        ▼
  Cross-encoder reranker         ms-marco-MiniLM-L-6-v2
        │                        scores (question, chunk) pairs together
        ▼
  top-5 results → LLM            score improvement: 0.465 → 0.992
```

**Why RRF over weighted sum:**
Vector and BM25 scores have different scales and distributions. RRF uses ranks not scores — rank 1 means the same thing regardless of the scoring method.

### ✅ Phase 3 — Continuous Evaluation

**Golden dataset:** 50 QA pairs generated from thinkstats2.pdf covering factual, conceptual, and comparative questions.

**Evaluation metrics:**

| Metric | What it measures | 8B score | 70B score |
|--------|-----------------|----------|-----------|
| Faithfulness | Answer uses only retrieved context | 0.610 | 0.745 |
| Answer relevancy | Answer addresses the question | 0.630 | 0.740 |
| Context precision | Retrieved chunks are useful | 0.800 | 0.840 |

**Key insight from evaluation:**
Context precision (0.84) confirms retrieval is strong. Faithfulness and relevancy are LLM-dependent — the 70B model scores 22% higher on faithfulness than the 8B model. This is the expected pattern for RAG systems: retrieval quality determines the ceiling, LLM quality determines how close you get to it.

---

## 📁 Project Structure

```
docpilot/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app + lifespan
│   │   ├── core/
│   │   │   ├── config.py            # all settings + provider resolution
│   │   │   ├── logging.py           # structlog JSON setup
│   │   │   └── metrics.py           # 7 Prometheus RAG metrics
│   │   ├── api/
│   │   │   ├── routes.py            # /ingest /query /stats /clear
│   │   │   └── deps.py              # dependency injection
│   │   ├── models/
│   │   │   └── schemas.py           # Source, QueryRequest, QueryResponse
│   │   └── services/
│   │       ├── ingestion.py         # load → chunk → embed → store
│   │       ├── embedder.py          # all-MiniLM-L6-v2, lru_cache
│   │       ├── bm25_index.py        # in-memory BM25Okapi index
│   │       ├── hybrid_retriever.py  # vector + BM25 + RRF fusion
│   │       ├── reranker.py          # cross-encoder ms-marco reranker
│   │       ├── retrieval.py         # full query pipeline
│   │       └── vector_store.py      # async Qdrant client
│   ├── tests/
│   │   ├── test_ingestion.py        # chunking + embedding tests
│   │   └── test_retrieval.py        # 15 tests — pipeline, citations, providers
│   ├── pyproject.toml               # uv dependencies
│   └── uv.lock
├── frontend/
│   ├── app.py                       # Streamlit chat UI + citation panel
│   ├── pyproject.toml
│   └── uv.lock
├── evaluation/
│   ├── golden_dataset.json          # 50 QA pairs ground truth
│   ├── baseline_results.json        # evaluation scores by model
│   └── run_eval.py                  # CI evaluation runner
├── infra/
│   ├── prometheus/prometheus.yml
│   └── grafana/
│       ├── provisioning/            # auto-connect datasource + dashboard
│       └── dashboards/docpilot.json # RAG metrics dashboard
├── notebooks/
│   ├── 00_colab_setup.ipynb         # one-click Colab setup
│   ├── 01_chunking.ipynb            # chunk size exploration
│   ├── 02_embeddings.ipynb          # embedding visualisation
│   └── 03_retrieval_debug.ipynb     # retrieval comparison
├── podman-compose.yml
├── Makefile
└── .env.example
```

---

## 💬 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ingest` | Upload and ingest a document |
| `POST` | `/api/query` | Ask a question, get answer + citations |
| `GET` | `/api/stats` | Collection stats and current settings |
| `DELETE` | `/api/clear` | Clear all ingested documents |
| `GET` | `/health` | Health check — Qdrant + LLM status |
| `GET` | `/metrics` | Prometheus scrape endpoint |

**Query request:**
```json
{
  "question": "What is a probability mass function?",
  "top_k": 5
}
```

**Query response:**
```json
{
  "answer": "A probability mass function (PMF) maps each value...",
  "sources": [
    {
      "text": "The PMF is a function that gives the probability...",
      "source_file": "thinkstats2.pdf",
      "page_number": 24,
      "score": 0.992
    }
  ],
  "model": "llama-3.3-70b-versatile"
}
```

---

## 📊 Observability

### Prometheus metrics

| Metric | Type | What it measures |
|--------|------|-----------------|
| `docpilot_documents_ingested_total` | Counter | ingestions by status |
| `docpilot_chunks_per_document` | Histogram | chunks per document |
| `docpilot_ingestion_latency_seconds` | Histogram | ingest time |
| `docpilot_query_latency_seconds` | Histogram | end-to-end query time |
| `docpilot_retrieval_score` | Histogram | cosine similarity distribution |
| `docpilot_low_confidence_queries_total` | Counter | queries with score < 0.5 |
| `docpilot_total_chunks` | Gauge | chunks currently in Qdrant |

### Grafana dashboard

Open http://localhost:3000 (admin / docpilot). Auto-provisioned — no manual setup.

### Reading retrieval scores

| Score | Meaning |
|-------|---------|
| > 0.75 🟢 | Strong match — answer well-grounded |
| 0.50–0.75 🟡 | Moderate — answer probably relevant |
| < 0.50 🔴 | Weak — low confidence, may be unreliable |

---

## 🛠️ Makefile Commands

```bash
make up              # start all containers (Groq mode)
make up-local        # start with Ollama local LLM
make rebuild         # full no-cache image rebuild
make status          # health check all services
make ingest          # upload and ingest a document
make ask             # ask a question via terminal
make test            # run 15 pytest tests
make logs            # tail fastapi logs
make logs-all        # tail all containers
make lock            # regenerate uv.lock files
make add             # add a package to a service
make clean           # remove containers + prune
make help            # show all commands
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `GROQ_API_KEY` | — | **Required.** Get free at console.groq.com |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `OLLAMA_MODEL` | `tinyllama` | Ollama model for local mode |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `600` | Tokens per chunk (tune: 400–800) |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Chunks returned per query |
| `FETCH_K` | `20` | Candidates per retrieval path before RRF |
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranking |

---

## 🐛 Troubleshooting

| Error | Fix |
|-------|-----|
| `streamlit not found` | Add `ENV PATH="/app/.venv/bin:$PATH"` to frontend Dockerfile |
| `ModuleNotFoundError: sentence_transformers` | Use `uv run python` not bare `python` in Dockerfile |
| `SessionNotFoundError` | Click "New session" in Streamlit sidebar |
| `model requires more system memory` | Switch to `tinyllama` or use Groq instead |
| `401 Unauthorized` | Check `GROQ_API_KEY` in `.env` |
| `No such file: /content/qdrant` | Download binary — it's cleared on Colab restart |
| `numpy dtype size changed` | Restart Colab runtime, install numpy first |
| `Collection doesn't exist` | Call `ensure_collection()` — handled automatically on ingest |

---

## 🤝 Contributing

```bash
git checkout -b feat/my-feature
git commit -m 'feat: description'
git push && open PR
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
