import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gen_ai.core.document import Document
from gen_ai.core.exceptions import DependencyError, VectorStoreError
from gen_ai.interfaces.vectorstore import BaseVectorStore
from gen_ai.vectorstores.faiss import FAISSVectorStore, _normalize_vector


def test_normalize_vector_zero_norm():
    vec = [0.0, 0.0]
    assert _normalize_vector(vec) == [0.0, 0.0]


def test_faiss_missing_dependency():
    with patch.dict(sys.modules, {"faiss": None}):
        with pytest.raises(DependencyError) as exc_info:
            FAISSVectorStore(dimension=128)
        assert exc_info.value.package_name == "faiss-cpu"


def test_faiss_implements_interface():
    mock_faiss = MagicMock()
    mock_index = MagicMock()
    mock_faiss.IndexFlatIP.return_value = mock_index

    store = FAISSVectorStore(dimension=3, faiss_module=mock_faiss)
    assert isinstance(store, BaseVectorStore)


def test_faiss_add_and_search():
    mock_faiss = MagicMock()
    mock_index = MagicMock()
    mock_index.ntotal = 2
    mock_faiss.IndexFlatIP.return_value = mock_index
    # D: similarities, I: indices
    mock_index.search.return_value = ([[0.95, 0.70]], [[0, 1]])

    store = FAISSVectorStore(dimension=2, faiss_module=mock_faiss)

    doc1 = Document(content="Doc 1", metadata={"tag": "tech"})
    doc2 = Document(content="Doc 2", metadata={"tag": "news"})

    store.add_documents([doc1, doc2], [[1.0, 0.0], [0.0, 1.0]])
    assert len(store) == 2

    results = store.search_with_scores([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].document.content == "Doc 1"
    assert pytest.approx(results[0].score, 0.01) == 0.95
    assert results[0].score_type == "similarity"
    assert results[1].document.content == "Doc 2"

    docs_only = store.search([1.0, 0.0], top_k=2)
    assert len(docs_only) == 2
    assert docs_only[0].content == "Doc 1"


def test_faiss_delete():
    mock_faiss = MagicMock()
    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_faiss.IndexFlatIP.return_value = mock_index

    store = FAISSVectorStore(dimension=2, faiss_module=mock_faiss)
    doc1 = Document(content="Doc 1")
    store.add_documents([doc1], [[1.0, 0.0]])

    assert doc1.id in store._id_to_doc
    store.delete([doc1.id])
    assert doc1.id not in store._id_to_doc


def test_faiss_dimension_mismatch():
    mock_faiss = MagicMock()
    store = FAISSVectorStore(dimension=4, faiss_module=mock_faiss)
    with pytest.raises(VectorStoreError, match="Vector dimension mismatch"):
        store.add_documents([Document(content="A")], [[1.0, 2.0]])


def test_faiss_save_and_load_local(tmp_path: Path):
    mock_faiss = MagicMock()
    mock_index = MagicMock()
    mock_faiss.IndexFlatIP.return_value = mock_index
    mock_faiss.read_index.return_value = mock_index

    # When mock write_index is called, create the dummy index file
    save_dir = tmp_path / "faiss_index"

    def side_effect_write(index, path):
        Path(path).write_bytes(b"dummy_index_bytes")

    mock_faiss.write_index.side_effect = side_effect_write

    store = FAISSVectorStore(dimension=2, faiss_module=mock_faiss)
    doc = Document(content="Test content", metadata={"source": "test.txt"})
    store.add_documents([doc], [[1.0, 0.0]])

    store.save_local(save_dir)

    assert (save_dir / "documents.json").exists()
    assert (save_dir / "index.faiss").exists()
    mock_faiss.write_index.assert_called_once()

    loaded_store = FAISSVectorStore.load_local(save_dir, faiss_module=mock_faiss)
    assert loaded_store.dimension == 2
    assert len(loaded_store) == 1
    assert doc.id in loaded_store._id_to_doc
    assert loaded_store._id_to_doc[doc.id].content == "Test content"


def test_faiss_load_local_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        FAISSVectorStore.load_local(tmp_path / "nonexistent")
