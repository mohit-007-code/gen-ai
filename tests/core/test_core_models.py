import pytest
from pydantic import ValidationError

from gen_ai.core import (
    ConfigurationError,
    DependencyError,
    Document,
    GenAIError,
    RAGConfig,
    SearchResult,
)


def test_search_result_creation():
    doc = Document(content="Sample text", metadata={"source": "test.txt"})
    result = SearchResult(document=doc, score=0.95, score_type="similarity")

    assert result.document.content == "Sample text"
    assert result.score == 0.95
    assert result.score_type == "similarity"
    assert result.metadata == {}


def test_search_result_unavailable_score():
    doc = Document(content="Doc without score")
    result = SearchResult(document=doc, score=None, score_type="unavailable")

    assert result.document.content == "Doc without score"
    assert result.score is None
    assert result.score_type == "unavailable"


def test_search_result_rejects_extra_fields():
    doc = Document(content="Sample text")
    with pytest.raises(ValidationError):
        SearchResult(document=doc, score=0.5, unknown_field="invalid")


def test_rag_config_defaults():
    config = RAGConfig()
    assert config.chunk_size == 500
    assert config.chunk_overlap is None
    assert config.top_k == 3
    assert config.score_threshold is None
    assert config.candidate_multiplier == 3
    assert config.system_prompt is None
    assert config.prompt_template is None


def test_rag_config_validation():
    with pytest.raises(ValidationError):
        RAGConfig(chunk_size=0)

    with pytest.raises(ValidationError):
        RAGConfig(top_k=-1)

    config = RAGConfig(chunk_size=1000, chunk_overlap=100, top_k=5)
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 100
    assert config.top_k == 5


def test_dependency_error_message():
    err = DependencyError(
        package_name="openai",
        extra_name="openai",
        feature_name="OpenAI LLM and Embeddings",
    )
    assert "Optional dependency 'openai' is required" in str(err)
    assert 'pip install "gen-ai[openai]"' in str(err)
    assert err.package_name == "openai"
    assert err.extra_name == "openai"


def test_gen_ai_error_hierarchy():
    err = ConfigurationError("Bad config")
    assert isinstance(err, GenAIError)
    assert str(err) == "Bad config"
