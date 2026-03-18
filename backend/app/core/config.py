from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ── LLM Provider ─────────────────────────────────────────────────────────
    # Switch between "ollama" and "groq" via LLM_PROVIDER in .env
    # All other LLM settings adapt automatically based on the provider
    llm_provider: str = "groq"          # "ollama" | "groq"

    # Ollama settings (used when llm_provider=ollama)
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "tinyllama"

    # Groq settings (used when llm_provider=groq)
    # Models: llama-3.1-8b-instant | llama-3.3-70b-versatile | mixtral-8x7b-32768
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.1-8b-instant"

    # ── Resolved LLM settings ────────────────────────────────────────────────
    # These properties resolve the correct values based on llm_provider
    # Use these everywhere instead of checking the provider manually
    @property
    def llm_base_url(self) -> str:
        return (
            self.groq_base_url
            if self.llm_provider == "groq"
            else self.ollama_base_url
        )

    @property
    def llm_model(self) -> str:
        return (
            self.groq_model
            if self.llm_provider == "groq"
            else self.ollama_model
        )

    @property
    def llm_api_key(self) -> str | None:
        return (
            self.groq_api_key
            if self.llm_provider == "groq"
            else None
        )

    # ── Vector store ──────────────────────────────────────────────────────────
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "docpilot"

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dim: int = 384

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 600
    chunk_overlap: int = 100

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()