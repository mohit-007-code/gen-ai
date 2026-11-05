from unittest.mock import MagicMock

import pytest

from gen_ai.adapters.embeddings.callable import CallableEmbeddings
from gen_ai.core.document import Document
from gen_ai.core.exceptions import RetrievalError
from gen_ai.core.search_result import SearchResult
from gen_ai.retrieval.retriever import Retriever
from gen_ai.vectorstores.in_memory import InMemoryVectorStore


def test_retriever_basic():
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0, 0.0] for _ in texts],
        embed_query=lambda text: [1.0, 0.0],
    )
    store = InMemoryVectorStore()
    doc1 = Document(content="Doc 1", metadata={"tag": "python"})
    doc2 = Document(content="Doc 2", metadata={"tag": "rust"})
    store.add_documents([doc1, doc2], [[1.0, 0.0], [0.0, 1.0]])

    retriever = Retriever(embeddings=embeddings, vector_store=store, top_k=2)

    results = retriever.retrieve("Tell me about Python")
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].document.content == "Doc 1"
    assert pytest.approx(results[0].score, 0.01) == 1.0


def test_retriever_retrieve_documents():
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0, 0.0] for _ in texts],
        embed_query=lambda text: [1.0, 0.0],
    )
    store = InMemoryVectorStore()
    doc = Document(content="Doc 1")
    store.add_documents([doc], [[1.0, 0.0]])

    retriever = Retriever(embeddings=embeddings, vector_store=store)
    docs = retriever.retrieve_documents("query")
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].content == "Doc 1"


def test_retriever_score_threshold_similarity():
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0, 0.0] for _ in texts],
        embed_query=lambda text: [1.0, 0.0],
    )
    store = InMemoryVectorStore()
    doc1 = Document(content="Close doc")
    doc2 = Document(content="Far doc")
    store.add_documents([doc1, doc2], [[1.0, 0.0], [0.0, 1.0]])

    retriever = Retriever(
        embeddings=embeddings, vector_store=store, score_threshold=0.5
    )
    results = retriever.retrieve("query")
    assert len(results) == 1
    assert results[0].document.content == "Close doc"


def test_retriever_score_threshold_distance():
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0]],
        embed_query=lambda text: [1.0],
    )
    mock_store = MagicMock()
    doc1 = Document(content="Close doc")
    doc2 = Document(content="Far doc")
    mock_store.search_with_scores.return_value = [
        SearchResult(document=doc1, score=0.2, score_type="distance"),
        SearchResult(document=doc2, score=0.8, score_type="distance"),
    ]

    retriever = Retriever(
        embeddings=embeddings, vector_store=mock_store, score_threshold=0.5
    )
    results = retriever.retrieve("query")
    assert len(results) == 1
    assert results[0].document.content == "Close doc"


def test_retriever_metadata_filter():
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0, 0.0] for _ in texts],
        embed_query=lambda text: [1.0, 0.0],
    )
    store = InMemoryVectorStore()
    doc1 = Document(content="Python tutorial", metadata={"lang": "python"})
    doc2 = Document(content="Python in Rust", metadata={"lang": "rust"})
    store.add_documents([doc1, doc2], [[1.0, 0.0], [0.99, 0.01]])

    retriever = Retriever(embeddings=embeddings, vector_store=store)
    results = retriever.retrieve("Python", metadata_filter={"lang": "python"})

    assert len(results) == 1
    assert results[0].document.metadata["lang"] == "python"


def test_retriever_empty_query():
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [],
        embed_query=lambda text: [1.0],
    )
    store = InMemoryVectorStore()
    retriever = Retriever(embeddings=embeddings, vector_store=store)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve("   ")


def test_retriever_invalid_top_k():
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [],
        embed_query=lambda text: [1.0],
    )
    store = InMemoryVectorStore()
    retriever = Retriever(embeddings=embeddings, vector_store=store)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        retriever.retrieve("query", top_k=0)


def test_retriever_embedding_error():
    embeddings = CallableEmbeddings(
        embed_documents=lambda t: [],
        embed_query=lambda t: 123,  # type: ignore
    )
    store = InMemoryVectorStore()
    retriever = Retriever(embeddings=embeddings, vector_store=store)

    with pytest.raises(RetrievalError, match="Failed to generate query embedding"):
        retriever.retrieve("query")


def test_retriever_store_error():
    embeddings = CallableEmbeddings(
        embed_documents=lambda t: [[1.0]],
        embed_query=lambda t: [1.0],
    )
    mock_store = MagicMock()
    mock_store.search_with_scores.side_effect = RuntimeError("DB down")

    retriever = Retriever(embeddings=embeddings, vector_store=mock_store)
    with pytest.raises(RetrievalError, match="Vector store search failed"):
        retriever.retrieve("query")
