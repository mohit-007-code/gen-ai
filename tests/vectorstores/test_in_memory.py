import pytest

from gen_ai.core.document import Document
from gen_ai.core.exceptions import VectorStoreError
from gen_ai.interfaces.vectorstore import BaseVectorStore
from gen_ai.vectorstores.in_memory import InMemoryVectorStore, _cosine_similarity


def test_in_memory_implements_interface():
    store = InMemoryVectorStore()
    assert isinstance(store, BaseVectorStore)


def test_cosine_similarity_zero_norm():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0


def test_in_memory_add_and_search():
    store = InMemoryVectorStore()

    doc1 = Document(content="Python programming", metadata={"topic": "code"})
    doc2 = Document(content="Baking bread", metadata={"topic": "food"})
    doc3 = Document(content="Machine learning with Python", metadata={"topic": "ai"})

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],
    ]

    store.add_documents([doc1, doc2, doc3], embeddings)

    query = [1.0, 0.0, 0.0]
    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].content == "Python programming"
    assert results[1].content == "Machine learning with Python"

    scored_results = store.search_with_scores(query, top_k=2)
    assert len(scored_results) == 2
    assert scored_results[0].document.content == "Python programming"
    assert pytest.approx(scored_results[0].score, 0.01) == 1.0
    assert scored_results[0].score_type == "similarity"


def test_in_memory_delete_and_clear():
    store = InMemoryVectorStore()
    doc1 = Document(content="A")
    doc2 = Document(content="B")

    store.add_documents([doc1, doc2], [[1.0, 0.0], [0.0, 1.0]])
    assert len(store) == 2
    assert store.get_document(doc1.id) is not None

    store.delete([doc1.id])
    assert len(store) == 1
    assert store.get_document(doc1.id) is None

    store.clear()
    assert len(store) == 0


def test_in_memory_empty_search():
    store = InMemoryVectorStore()
    assert store.search([1.0, 2.0], top_k=3) == []
    assert store.search_with_scores([1.0, 2.0], top_k=3) == []


def test_in_memory_dimension_mismatch():
    store = InMemoryVectorStore()
    doc1 = Document(content="A")
    store.add_documents([doc1], [[1.0, 0.0]])

    doc2 = Document(content="B")
    with pytest.raises(VectorStoreError, match="Vector dimension mismatch"):
        store.add_documents([doc2], [[1.0, 0.0, 0.0]])


def test_in_memory_count_mismatch():
    store = InMemoryVectorStore()
    with pytest.raises(
        VectorStoreError, match="Number of documents .* and embeddings .* must match"
    ):
        store.add_documents([Document(content="A")], [[1.0], [2.0]])


def test_in_memory_invalid_top_k():
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        store.search([1.0], top_k=0)


def test_in_memory_empty_documents():
    store = InMemoryVectorStore()
    store.add_documents([], [])
    assert len(store) == 0


def test_in_memory_negative_cosine_similarity():
    store = InMemoryVectorStore()
    doc_pos = Document(content="Positive")
    doc_neg = Document(content="Opposite")
    doc_orth = Document(content="Orthogonal")

    store.add_documents(
        [doc_pos, doc_neg, doc_orth],
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
    )

    results = store.search_with_scores([1.0, 0.0], top_k=3)
    assert len(results) == 3
    # Check ranking order: highest similarity to lowest
    assert results[0].document.content == "Positive"
    assert pytest.approx(results[0].score, 0.001) == 1.0

    assert results[1].document.content == "Orthogonal"
    assert pytest.approx(results[1].score, 0.001) == 0.0

    assert results[2].document.content == "Opposite"
    assert pytest.approx(results[2].score, 0.001) == -1.0


def test_in_memory_zero_vector_query():
    store = InMemoryVectorStore()
    doc = Document(content="Sample")
    store.add_documents([doc], [[1.0, 2.0]])

    results = store.search_with_scores([0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].score == 0.0
