import pytest

from gen_ai.adapters.embeddings.callable import CallableEmbeddings
from gen_ai.core.exceptions import EmbeddingError
from gen_ai.interfaces.embeddings import BaseEmbeddings


def test_callable_embeddings_subclasses_base_embeddings():
    adapter = CallableEmbeddings(
        embed_documents=lambda texts: [[0.1, 0.2] for _ in texts],
        embed_query=lambda text: [0.1, 0.2],
    )
    assert isinstance(adapter, BaseEmbeddings)


def test_callable_embeddings_embed_documents():
    def mock_embed(texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), float(len(t) * 2)] for t in texts]

    adapter = CallableEmbeddings(embed_documents=mock_embed)
    result = adapter.embed_documents(["hello", "world!"])

    assert result == [[5.0, 10.0], [6.0, 12.0]]


def test_callable_embeddings_embed_query_default_fallback():
    # If embed_query is omitted, it should default to embed_documents([query])[0]
    def mock_embed(texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0] for t in texts]

    adapter = CallableEmbeddings(embed_documents=mock_embed)
    query_vec = adapter.embed_query("test")

    assert query_vec == [4.0, 0.0]


def test_callable_embeddings_explicit_embed_query():
    adapter = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0] for _ in texts],
        embed_query=lambda text: [99.0],
    )
    assert adapter.embed_query("anything") == [99.0]


def test_callable_embeddings_empty_documents_input():
    called = False

    def mock_embed(texts: list[str]) -> list[list[float]]:
        nonlocal called
        called = True
        return []

    adapter = CallableEmbeddings(embed_documents=mock_embed)
    result = adapter.embed_documents([])

    assert result == []
    assert not called  # Short-circuit empty lists


def test_callable_embeddings_invalid_count_mismatch():
    # Return 1 embedding for 2 input texts
    def bad_embed(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2]]

    adapter = CallableEmbeddings(embed_documents=bad_embed)
    with pytest.raises(EmbeddingError, match="Expected 2 embeddings, received 1"):
        adapter.embed_documents(["doc1", "doc2"])


def test_callable_embeddings_invalid_return_type():
    def bad_embed(texts: list[str]) -> str:
        return "not a list"  # type: ignore

    adapter = CallableEmbeddings(embed_documents=bad_embed)  # type: ignore
    with pytest.raises(EmbeddingError, match="must return a list"):
        adapter.embed_documents(["doc1"])


def test_callable_embeddings_invalid_vector_element_type():
    def bad_embed(texts: list[str]) -> list[object]:
        return ["not a vector"]

    adapter = CallableEmbeddings(embed_documents=bad_embed)  # type: ignore
    with pytest.raises(EmbeddingError, match="must be a list of numbers"):
        adapter.embed_documents(["doc1"])


def test_callable_embeddings_invalid_query_return_type():
    adapter = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0]],
        embed_query=lambda text: "invalid",  # type: ignore
    )
    with pytest.raises(EmbeddingError, match="must return a list of numbers"):
        adapter.embed_query("query")


def test_callable_embeddings_exception_propagation():
    def failing_embed(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("API timeout")

    adapter = CallableEmbeddings(embed_documents=failing_embed)
    with pytest.raises(EmbeddingError, match="Embedding generation failed"):
        adapter.embed_documents(["doc1"])


def test_callable_embeddings_non_callable_init():
    with pytest.raises(TypeError, match="embed_documents must be callable"):
        CallableEmbeddings(embed_documents="not_a_function")  # type: ignore

    with pytest.raises(TypeError, match="embed_query must be callable"):
        CallableEmbeddings(
            embed_documents=lambda t: [[1.0]],
            embed_query="not_a_function",  # type: ignore
        )
