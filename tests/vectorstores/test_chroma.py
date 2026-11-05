import sys
from unittest.mock import MagicMock, patch

import pytest

from gen_ai.core.document import Document
from gen_ai.core.exceptions import DependencyError, VectorStoreError
from gen_ai.interfaces.vectorstore import BaseVectorStore
from gen_ai.vectorstores.chroma import ChromaVectorStore, _clean_metadata_for_chroma


def test_clean_metadata_for_chroma():
    meta = {
        "str_val": "hello",
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "none_val": None,
        "dict_val": {"nested": "value"},
    }
    cleaned = _clean_metadata_for_chroma(meta)
    assert cleaned["str_val"] == "hello"
    assert cleaned["int_val"] == 42
    assert cleaned["float_val"] == 3.14
    assert cleaned["bool_val"] is True
    assert "none_val" not in cleaned
    assert cleaned["dict_val"] == '{"nested": "value"}'


def test_chroma_missing_dependency():
    with patch.dict(sys.modules, {"chromadb": None}):
        with pytest.raises(DependencyError) as exc_info:
            ChromaVectorStore()
        assert exc_info.value.package_name == "chromadb"


def test_chroma_implements_interface():
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    store = ChromaVectorStore(client=mock_client)
    assert isinstance(store, BaseVectorStore)


def test_chroma_add_documents():
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    store = ChromaVectorStore(client=mock_client)
    docs = [
        Document(content="Doc 1", metadata={"key": "val"}),
        Document(content="Doc 2"),
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    store.add_documents(docs, embeddings)

    mock_collection.upsert.assert_called_once()
    call_kwargs = mock_collection.upsert.call_args[1]
    assert call_kwargs["ids"] == [docs[0].id, docs[1].id]
    assert call_kwargs["documents"] == ["Doc 1", "Doc 2"]
    assert call_kwargs["embeddings"] == embeddings


def test_chroma_add_empty_documents():
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    store = ChromaVectorStore(client=mock_client)
    store.add_documents([], [])
    mock_collection.upsert.assert_not_called()


def test_chroma_search_with_scores():
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    doc_id = "test-id"
    mock_collection.query.return_value = {
        "ids": [[doc_id]],
        "documents": [["Hello world"]],
        "metadatas": [[{"source": "test.txt"}]],
        "distances": [[0.2]],
    }

    store = ChromaVectorStore(client=mock_client)
    results = store.search_with_scores([0.1, 0.2], top_k=1)

    assert len(results) == 1
    assert results[0].document.id == doc_id
    assert results[0].document.content == "Hello world"
    assert results[0].document.metadata["source"] == "test.txt"
    assert pytest.approx(results[0].score, 0.01) == 0.8
    assert results[0].score_type == "similarity"

    docs = store.search([0.1, 0.2], top_k=1)
    assert len(docs) == 1
    assert docs[0].content == "Hello world"


def test_chroma_delete():
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    store = ChromaVectorStore(client=mock_client)
    store.delete(["id1", "id2"])

    mock_collection.delete.assert_called_once_with(ids=["id1", "id2"])

    # Empty delete should short-circuit
    mock_collection.delete.reset_mock()
    store.delete([])
    mock_collection.delete.assert_not_called()


def test_chroma_length_mismatch():
    mock_client = MagicMock()
    store = ChromaVectorStore(client=mock_client)
    with pytest.raises(VectorStoreError):
        store.add_documents([Document(content="A")], [[0.1], [0.2]])


def test_chroma_negative_cosine_similarity():
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_collection.query.return_value = {
        "ids": [["doc-opp"]],
        "documents": [["Opposite meaning"]],
        "metadatas": [[{}]],
        "distances": [[2.0]],  # Opposite vector in cosine distance space
    }

    store = ChromaVectorStore(client=mock_client)
    results = store.search_with_scores([0.1, 0.2], top_k=1)

    assert len(results) == 1
    # 1.0 - 2.0 = -1.0; must not be clamped to 0.0
    assert pytest.approx(results[0].score, 0.001) == -1.0
    assert results[0].score_type == "similarity"


def test_chroma_search_with_none_distance():
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_collection.query.return_value = {
        "ids": [["doc-1"]],
        "documents": [["Doc 1"]],
        "metadatas": [[{}]],
        "distances": [[None]],
    }

    store = ChromaVectorStore(client=mock_client)
    results = store.search_with_scores([0.1, 0.2], top_k=1)

    assert len(results) == 1
    assert results[0].score is None
    assert results[0].score_type == "unavailable"


def test_chroma_nested_metadata_restoration():
    from gen_ai.vectorstores.chroma import _restore_metadata_from_chroma

    stored = {
        "title": "Paper",
        "nested_dict": '{"section": 1, "topic": "rag"}',
        "nested_list": '["a", "b", "c"]',
        "plain_string": "hello",
    }
    restored = _restore_metadata_from_chroma(stored)
    assert restored["title"] == "Paper"
    assert restored["nested_dict"] == {"section": 1, "topic": "rag"}
    assert restored["nested_list"] == ["a", "b", "c"]
    assert restored["plain_string"] == "hello"


def test_chroma_where_filtering():
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_collection.query.return_value = {
        "ids": [["doc-filtered"]],
        "documents": [["Filtered doc"]],
        "metadatas": [[{"category": "tech"}]],
        "distances": [[0.1]],
    }

    store = ChromaVectorStore(client=mock_client)
    results = store.search_with_scores([0.1, 0.2], top_k=1, where={"category": "tech"})

    mock_collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2]],
        n_results=1,
        include=["documents", "metadatas", "distances"],
        where={"category": "tech"},
    )
    assert len(results) == 1
    assert results[0].document.metadata["category"] == "tech"


def test_chroma_invalid_top_k():
    mock_client = MagicMock()
    store = ChromaVectorStore(client=mock_client)
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        store.search([0.1], top_k=-1)
