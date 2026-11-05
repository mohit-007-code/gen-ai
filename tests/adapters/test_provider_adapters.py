import sys
from unittest.mock import MagicMock, patch

import pytest

from gen_ai.adapters.embeddings.gemini import GeminiEmbeddings
from gen_ai.adapters.embeddings.ollama import OllamaEmbeddings
from gen_ai.adapters.embeddings.openai import OpenAIEmbeddings
from gen_ai.adapters.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from gen_ai.adapters.llms.gemini import GeminiLLM
from gen_ai.adapters.llms.groq import GroqLLM
from gen_ai.adapters.llms.ollama import OllamaLLM
from gen_ai.adapters.llms.openai import OpenAILLM
from gen_ai.core.exceptions import DependencyError, EmbeddingError, LLMError


# --- OpenAI Embeddings Tests ---
def test_openai_embeddings_success():
    mock_client = MagicMock()
    mock_item1 = MagicMock()
    mock_item1.embedding = [0.1, 0.2, 0.3]
    mock_item2 = MagicMock()
    mock_item2.embedding = [0.4, 0.5, 0.6]
    mock_client.embeddings.create.return_value = MagicMock(
        data=[mock_item1, mock_item2]
    )

    adapter = OpenAIEmbeddings(client=mock_client, model="text-embedding-3-small")
    docs = adapter.embed_documents(["doc1", "doc2"])

    assert len(docs) == 2
    assert docs[0] == [0.1, 0.2, 0.3]
    assert docs[1] == [0.4, 0.5, 0.6]

    query = adapter.embed_query("query")
    assert query == [0.1, 0.2, 0.3]


def test_openai_embeddings_empty_documents():
    mock_client = MagicMock()
    adapter = OpenAIEmbeddings(client=mock_client)
    assert adapter.embed_documents([]) == []
    mock_client.embeddings.create.assert_not_called()


def test_openai_embeddings_error_wrapping():
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = RuntimeError("Rate limit")

    adapter = OpenAIEmbeddings(client=mock_client)
    with pytest.raises(EmbeddingError, match="OpenAI embedding generation failed"):
        adapter.embed_documents(["test"])


def test_openai_embeddings_missing_dependency():
    with patch.dict(sys.modules, {"openai": None}):
        with pytest.raises(DependencyError) as exc_info:
            OpenAIEmbeddings()
        assert exc_info.value.package_name == "openai"


# --- Sentence Transformers Embeddings Tests ---
def test_sentence_transformer_embeddings_success():
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, **kwargs: (
        [[0.1, 0.2]] if isinstance(texts, list) else [0.1, 0.2]
    )

    adapter = SentenceTransformerEmbeddings(model=mock_model)
    docs = adapter.embed_documents(["hello"])
    assert docs == [[0.1, 0.2]]

    query = adapter.embed_query("hello")
    assert query == [0.1, 0.2]


def test_sentence_transformer_missing_dependency():
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        with pytest.raises(DependencyError) as exc_info:
            SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
        assert exc_info.value.package_name == "sentence-transformers"


# --- Gemini Embeddings Tests ---
def test_gemini_embeddings_success():
    mock_client = MagicMock()
    mock_client.embed_content.return_value = {"embedding": [[0.1, 0.2]]}

    adapter = GeminiEmbeddings(client=mock_client)
    docs = adapter.embed_documents(["doc"])
    assert docs == [[0.1, 0.2]]

    mock_client.embed_content.return_value = {"embedding": [0.3, 0.4]}
    query = adapter.embed_query("q")
    assert query == [0.3, 0.4]


def test_gemini_embeddings_missing_dependency():
    with patch.dict(sys.modules, {"google.generativeai": None}):
        with pytest.raises(DependencyError) as exc_info:
            GeminiEmbeddings()
        assert exc_info.value.package_name == "google-generativeai"


# --- Ollama Embeddings Tests ---
def test_ollama_embeddings_success():
    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    adapter = OllamaEmbeddings(client=mock_client)
    docs = adapter.embed_documents(["a", "b"])
    assert docs == [[0.1, 0.2], [0.3, 0.4]]


def test_ollama_embeddings_missing_dependency():
    with patch.dict(sys.modules, {"ollama": None}):
        with pytest.raises(DependencyError) as exc_info:
            OllamaEmbeddings()
        assert exc_info.value.package_name == "ollama"


# --- OpenAI LLM Tests ---
def test_openai_llm_success():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "OpenAI answer"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    llm = OpenAILLM(client=mock_client)
    result = llm.generate("Hello")
    assert result == "OpenAI answer"


def test_openai_llm_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("Quota exceeded")

    llm = OpenAILLM(client=mock_client)
    with pytest.raises(LLMError, match="OpenAI generation failed"):
        llm.generate("Hello")


def test_openai_llm_missing_dependency():
    with patch.dict(sys.modules, {"openai": None}):
        with pytest.raises(DependencyError) as exc_info:
            OpenAILLM()
        assert exc_info.value.package_name == "openai"


# --- Gemini LLM Tests ---
def test_gemini_llm_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Gemini answer"
    mock_client.generate_content.return_value = mock_response

    llm = GeminiLLM(client=mock_client)
    assert llm.generate("Prompt") == "Gemini answer"


def test_gemini_llm_missing_dependency():
    with patch.dict(sys.modules, {"google.generativeai": None}):
        with pytest.raises(DependencyError) as exc_info:
            GeminiLLM()
        assert exc_info.value.package_name == "google-generativeai"


# --- Groq LLM Tests ---
def test_groq_llm_success():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Groq answer"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    llm = GroqLLM(client=mock_client)
    assert llm.generate("Prompt") == "Groq answer"


def test_groq_llm_missing_dependency():
    with patch.dict(sys.modules, {"groq": None}):
        with pytest.raises(DependencyError) as exc_info:
            GroqLLM()
        assert exc_info.value.package_name == "groq"


# --- Ollama LLM Tests ---
def test_ollama_llm_success():
    mock_client = MagicMock()
    mock_client.generate.return_value = {"response": "Ollama answer"}

    llm = OllamaLLM(client=mock_client)
    assert llm.generate("Prompt") == "Ollama answer"


def test_ollama_llm_missing_dependency():
    with patch.dict(sys.modules, {"ollama": None}):
        with pytest.raises(DependencyError) as exc_info:
            OllamaLLM()
        assert exc_info.value.package_name == "ollama"
