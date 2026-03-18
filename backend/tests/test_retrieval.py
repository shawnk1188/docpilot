# backend/tests/test_retrieval.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.retrieval import RetrievalService
from app.models.schemas import QueryResponse


# ── Query pipeline tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_returns_answer_with_sources():
    """
    Full pipeline returns an answer and source citations.

    LEARNING: this test verifies the happy path end to end —
    chunks are retrieved, LLM generates an answer, citations are built.
    """
    mock_store = AsyncMock()
    mock_store.search.return_value = [
        ("Llama3 is a large language model by Meta.", "llama_docs.pdf", 1, 0.91),
        ("It supports 8B and 70B parameter variants.", "llama_docs.pdf", 2, 0.85),
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    svc = RetrievalService(vector_store=mock_store, embedder=mock_embedder)

    with patch.object(
        svc, "_call_llm",
        new=AsyncMock(return_value="Llama3 is a large language model by Meta.")
    ):
        result = await svc.query("What is Llama3?", top_k=2)

    assert isinstance(result, QueryResponse)
    assert "Llama3" in result.answer
    assert len(result.sources) == 2
    assert result.sources[0].score == 0.91
    assert result.sources[0].page_number == 1
    assert result.sources[0].source_file == "llama_docs.pdf"


@pytest.mark.asyncio
async def test_query_no_documents_returns_helpful_message():
    """
    When nothing is ingested, return a helpful message not an error.

    LEARNING: an empty vector store should degrade gracefully —
    a 500 error here would be a bad user experience.
    """
    mock_store = AsyncMock()
    mock_store.search.return_value = []

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    svc = RetrievalService(vector_store=mock_store, embedder=mock_embedder)
    result = await svc.query("What is Llama3?", top_k=5)

    assert "No documents" in result.answer
    assert result.sources == []


@pytest.mark.asyncio
async def test_query_truncates_long_source_text():
    """
    Source text longer than 500 chars should be truncated with ellipsis.

    LEARNING: the UI renders source text — very long chunks would
    break the citation panel layout in Streamlit.
    """
    long_text = "word " * 200     # 1000 chars

    mock_store = AsyncMock()
    mock_store.search.return_value = [
        (long_text, "long_doc.pdf", 1, 0.82),
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    svc = RetrievalService(vector_store=mock_store, embedder=mock_embedder)

    with patch.object(svc, "_call_llm", new=AsyncMock(return_value="answer")):
        result = await svc.query("question", top_k=1)

    assert len(result.sources[0].text) <= 503   # 500 + "..."
    assert result.sources[0].text.endswith("...")


@pytest.mark.asyncio
async def test_query_score_is_rounded_to_3_decimal_places():
    """Source scores should be rounded to 3 decimal places."""
    mock_store = AsyncMock()
    mock_store.search.return_value = [
        ("Some text.", "doc.pdf", 1, 0.876543219),
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    svc = RetrievalService(vector_store=mock_store, embedder=mock_embedder)

    with patch.object(svc, "_call_llm", new=AsyncMock(return_value="answer")):
        result = await svc.query("question", top_k=1)

    assert result.sources[0].score == 0.877


# ── Low confidence detection ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_low_confidence_query_increments_counter():
    """
    Queries where the top chunk score is below 0.5 should
    increment the LOW_CONFIDENCE_QUERIES counter.

    LEARNING: this metric is your early warning system.
    A spike in low confidence queries means either:
      1. Users are asking about topics not in your documents
      2. Your chunk size is wrong for this document type
      3. The embedding model isn't well-suited to your domain
    """
    mock_store = AsyncMock()
    mock_store.search.return_value = [
        ("Loosely related text.", "random.pdf", 1, 0.31),
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    svc = RetrievalService(vector_store=mock_store, embedder=mock_embedder)

    with patch.object(
        svc, "_call_llm",
        new=AsyncMock(return_value="I don't have enough information.")
    ):
        with patch("app.services.retrieval.LOW_CONFIDENCE_QUERIES") as mock_counter:
            await svc.query("What is quantum entanglement?", top_k=1)
            mock_counter.inc.assert_called_once()


@pytest.mark.asyncio
async def test_high_confidence_query_does_not_increment_counter():
    """Queries with top score above 0.5 should NOT trigger low confidence."""
    mock_store = AsyncMock()
    mock_store.search.return_value = [
        ("Highly relevant text.", "good_doc.pdf", 1, 0.91),
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    svc = RetrievalService(vector_store=mock_store, embedder=mock_embedder)

    with patch.object(
        svc, "_call_llm",
        new=AsyncMock(return_value="A clear answer.")
    ):
        with patch("app.services.retrieval.LOW_CONFIDENCE_QUERIES") as mock_counter:
            await svc.query("A relevant question?", top_k=1)
            mock_counter.inc.assert_not_called()


# ── Context building ──────────────────────────────────────────────────────────

def test_build_context_includes_source_and_page():
    """
    Context string must include filename and page number for each chunk.

    LEARNING: the system prompt instructs the LLM to cite sources.
    If the context doesn't include source info, the LLM can't cite correctly.
    The format [Chunk N — filename · page X] is what drives citations.
    """
    svc = RetrievalService(
        vector_store=MagicMock(),
        embedder=MagicMock(),
    )

    results = [
        ("Some text about AI.", "ai_guide.pdf", 3, 0.88),
        ("More text about ML.", "ai_guide.pdf", 7, 0.75),
    ]

    context = svc._build_context(results)

    assert "ai_guide.pdf" in context
    assert "page 3" in context
    assert "page 7" in context
    assert "0.88" in context
    assert "0.75" in context
    assert "Chunk 1" in context
    assert "Chunk 2" in context


def test_build_context_handles_missing_page_number():
    """
    Chunks without a page number (TXT, MD files) should not
    show 'page None' in the context string.
    """
    svc = RetrievalService(
        vector_store=MagicMock(),
        embedder=MagicMock(),
    )

    results = [
        ("Text from a plain text file.", "notes.txt", None, 0.80),
    ]

    context = svc._build_context(results)

    assert "None" not in context
    assert "notes.txt" in context


def test_build_context_chunks_separated_by_divider():
    """Multiple chunks should be clearly separated in the context."""
    svc = RetrievalService(
        vector_store=MagicMock(),
        embedder=MagicMock(),
    )

    results = [
        ("First chunk.", "doc.pdf", 1, 0.90),
        ("Second chunk.", "doc.pdf", 2, 0.80),
        ("Third chunk.", "doc.pdf", 3, 0.70),
    ]

    context = svc._build_context(results)

    # Each chunk is separated by ---
    assert context.count("---") == 2


# ── Ollama native response parsing ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_ollama_native_parses_response():
    """
    _call_ollama_native should parse Ollama's /api/chat response shape.

    LEARNING — Ollama /api/chat response:
      { "message": { "role": "assistant", "content": "answer" } }

    Different from OpenAI format:
      { "choices": [{ "message": { "content": "answer" } }] }
    """
    svc = RetrievalService(
        vector_store=MagicMock(),
        embedder=MagicMock(),
    )

    ollama_response = {
        "model": "tinyllama",
        "message": {
            "role": "assistant",
            "content": "This is the Ollama answer.",
        },
        "done": True,
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.json.return_value = ollama_response
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await svc._call_ollama_native("question", "context")

    assert result == "This is the Ollama answer."


@pytest.mark.asyncio
async def test_call_ollama_native_uses_correct_endpoint():
    """
    _call_ollama_native must call /api/chat not /v1/chat/completions.

    LEARNING: this is a contract test — it will catch a regression
    immediately if someone accidentally switches back to the OpenAI
    endpoint format which Ollama doesn't support without extensions.
    """
    svc = RetrievalService(
        vector_store=MagicMock(),
        embedder=MagicMock(),
    )

    ollama_response = {
        "message": {"role": "assistant", "content": "test"},
        "done": True,
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.json.return_value = ollama_response
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        await svc._call_ollama_native("question", "context")

        endpoint = mock_client.post.call_args[0][0]
        assert endpoint == "/api/chat", (
            f"Expected /api/chat but got {endpoint}. "
            "Do not use the OpenAI-compatible endpoint for Ollama."
        )


# ── Groq / OpenAI-compatible response parsing ─────────────────────────────────

@pytest.mark.asyncio
async def test_call_openai_compatible_parses_response():
    """
    _call_openai_compatible should parse the OpenAI/Groq response shape.

    LEARNING — Groq and OpenAI response format:
      { "choices": [{ "message": { "content": "answer" } }] }
    """
    svc = RetrievalService(
        vector_store=MagicMock(),
        embedder=MagicMock(),
    )

    groq_response = {
        "model": "llama-3.1-8b-instant",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "This is the Groq answer.",
            }
        }]
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.json.return_value = groq_response
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await svc._call_openai_compatible("question", "context")

    assert result == "This is the Groq answer."


@pytest.mark.asyncio
async def test_call_openai_compatible_sends_auth_header():
    """
    Groq requests must include Authorization: Bearer <key> header.
    Without this the request returns 401.
    """
    svc = RetrievalService(
        vector_store=MagicMock(),
        embedder=MagicMock(),
    )

    groq_response = {
        "choices": [{"message": {"content": "answer"}}]
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.json.return_value = groq_response
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        await svc._call_openai_compatible("question", "context")

        # Verify Authorization header was passed to AsyncClient
        call_kwargs = mock_client_class.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")


# ── Provider routing ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_llm_routes_to_groq_when_configured(monkeypatch):
    """
    When LLM_PROVIDER=groq, _call_llm routes to _call_openai_compatible.

    LEARNING: the provider abstraction means the rest of the codebase
    never checks the provider — only _call_llm does. This keeps the
    routing logic in one place and makes it easy to add new providers.
    """
    svc = RetrievalService(
        vector_store=MagicMock(),
        embedder=MagicMock(),
    )

    monkeypatch.setattr(
        "app.services.retrieval.settings.llm_provider", "groq"
    )

    with patch.object(
        svc, "_call_openai_compatible",
        new=AsyncMock(return_value="groq answer")
    ) as mock_groq:
        result = await svc._call_llm("question", "context")
        mock_groq.assert_called_once_with("question", "context")
        assert result == "groq answer"


@pytest.mark.asyncio
async def test_call_llm_routes_to_ollama_when_configured(monkeypatch):
    """
    When LLM_PROVIDER=ollama, _call_llm routes to _call_ollama_native.
    """
    svc = RetrievalService(
        vector_store=MagicMock(),
        embedder=MagicMock(),
    )

    monkeypatch.setattr(
        "app.services.retrieval.settings.llm_provider", "ollama"
    )

    with patch.object(
        svc, "_call_ollama_native",
        new=AsyncMock(return_value="ollama answer")
    ) as mock_ollama:
        result = await svc._call_llm("question", "context")
        mock_ollama.assert_called_once_with("question", "context")
        assert result == "ollama answer"