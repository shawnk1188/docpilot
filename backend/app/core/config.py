from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Ollama runs inside a container — no API key needed.
    # The model name must match what you pulled: `ollama pull llama3`
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3"

    # ── Vector store ──────────────────────────────────────────────────────────
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "docpilot"

    # ── Embedding ─────────────────────────────────────────────────────────────
    # all-MiniLM-L6-v2: free, local, 384 dimensions, good English quality.
    # To upgrade to OpenAI: change to "text-embedding-3-small" and add
    # OPENAI_API_KEY — you'll also need to update VECTOR_DIM in vector_store.py
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dim: int = 384          # must match embedding model output size

    # ── Chunking ──────────────────────────────────────────────────────────────
    # LEARNING: these two numbers are the most important RAG tuning knobs.
    #
    # chunk_size: how many tokens per chunk.
    #   Too small (< 200) → not enough context, LLM gives vague answers.
    #   Too large (> 1000) → chunk spans multiple topics, retrieval gets noisy.
    #   Sweet spot: 400–800. We start at 600.
    #
    # chunk_overlap: how many tokens repeat between adjacent chunks.
    #   Prevents an answer that straddles a chunk boundary from being missed.
    #   Rule of thumb: ~15% of chunk_size. For 600 tokens → 100 overlap.
    chunk_size: int = 600
    chunk_overlap: int = 100

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # How many chunks to pass to the LLM as context.
    # More = richer context but also more noise injected into the prompt.
    # Start at 5. If answers are too vague, try 7. If too noisy, try 3.
    retrieval_top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Single shared instance — import this everywhere
settings = Settings()