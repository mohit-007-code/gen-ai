"""Integration tests for third-party LLM and Embedding providers.

All tests require external credentials or running local servers and will
skip cleanly when credentials or connectivity are absent.
Credentials must NEVER be hardcoded or logged.
"""

import os
import urllib.error
import urllib.request

import pytest

from gen_ai.adapters.embeddings.gemini import GeminiEmbeddings
from gen_ai.adapters.embeddings.ollama import OllamaEmbeddings
from gen_ai.adapters.embeddings.openai import OpenAIEmbeddings
from gen_ai.adapters.llms.gemini import GeminiLLM
from gen_ai.adapters.llms.groq import GroqLLM
from gen_ai.adapters.llms.ollama import OllamaLLM
from gen_ai.adapters.llms.openai import OpenAILLM


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable is not set",
)
def test_openai_provider_integration():
    """Verify live OpenAI embedding and generation when OPENAI_API_KEY is supplied."""
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    vec = embedder.embed_query("Integration test query")
    assert len(vec) == 1536
    assert all(isinstance(v, float) for v in vec)

    llm = OpenAILLM(model="gpt-4o-mini", max_tokens=10)
    answer = llm.generate("Say 'pass'")
    assert answer and len(answer.strip()) > 0


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set",
)
def test_gemini_provider_integration():
    """Verify live Gemini embedding and generation when credentials are supplied."""
    embedder = GeminiEmbeddings()
    vec = embedder.embed_query("Integration test query")
    assert len(vec) > 0
    assert all(isinstance(v, float) for v in vec)

    llm = GeminiLLM()
    answer = llm.generate("Say 'pass'")
    assert answer and len(answer.strip()) > 0


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY environment variable is not set",
)
def test_groq_provider_integration():
    """Verify live Groq generation when GROQ_API_KEY is supplied."""
    llm = GroqLLM(model="llama3-8b-8192")
    answer = llm.generate("Say 'pass'")
    assert answer and len(answer.strip()) > 0


def _is_ollama_server_reachable() -> bool:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1.0) as resp:
            return resp.status == 200
    except (OSError, urllib.error.URLError):
        return False


@pytest.mark.skipif(
    not _is_ollama_server_reachable(),
    reason="Local Ollama daemon is not reachable at OLLAMA_HOST or http://localhost:11434",
)
def test_ollama_provider_integration():
    """Verify local Ollama embedding and generation when Ollama daemon is reachable."""
    try:
        embedder = OllamaEmbeddings(model="nomic-embed-text")
        vec = embedder.embed_query("Test query")
        assert len(vec) > 0

        llm = OllamaLLM(model="llama3")
        ans = llm.generate("Say 'pass'")
        assert len(ans.strip()) > 0
    except (RuntimeError, OSError, ConnectionError, ImportError) as exc:
        pytest.skip(f"Ollama integration test skipped due to model error: {exc}")
