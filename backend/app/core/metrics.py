from prometheus_client import Counter, Histogram, Gauge

# ── Ingestion metrics ─────────────────────────────────────────────────────────
DOCS_INGESTED = Counter(
    "docpilot_documents_ingested_total",
    "Total documents ingested",
    ["status"],           # labels: success / failure
)

CHUNKS_STORED = Histogram(
    "docpilot_chunks_per_document",
    "Number of chunks produced per document",
    buckets=[10, 50, 100, 200, 500, 1000],
)

INGESTION_LATENCY = Histogram(
    "docpilot_ingestion_latency_seconds",
    "Time to ingest one document end to end",
    buckets=[1, 5, 10, 30, 60, 120],
)

# ── Retrieval metrics — RAG specific ─────────────────────────────────────────
# LEARNING: these are the metrics you won't find in generic web tutorials.
# They measure the quality of the AI pipeline, not just the HTTP layer.

QUERY_LATENCY = Histogram(
    "docpilot_query_latency_seconds",
    "End to end query latency (embed + retrieve + generate)",
    buckets=[0.5, 1, 2, 5, 10, 30],
)

RETRIEVAL_SCORE = Histogram(
    "docpilot_retrieval_score",
    "Cosine similarity scores of retrieved chunks",
    buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

LOW_CONFIDENCE_QUERIES = Counter(
    "docpilot_low_confidence_queries_total",
    "Queries where top chunk score was below 0.5",
)

TOTAL_CHUNKS_IN_STORE = Gauge(
    "docpilot_total_chunks",
    "Current number of chunks in Qdrant",
)