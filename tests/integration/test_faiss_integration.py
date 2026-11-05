"""Integration tests for FAISSVectorStore using the real faiss-cpu library."""

import importlib.util

import pytest

HAS_FAISS = importlib.util.find_spec("faiss") is not None

from gen_ai.core.document import Document
from gen_ai.vectorstores.faiss import FAISSVectorStore


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu is not installed")
def test_real_faiss_lifecycle_and_scores():
    """Verify real FAISS IndexFlatIP with normalized vectors and cosine scoring."""
    store = FAISSVectorStore(dimension=3, normalize=True)

    doc_pos = Document(id="doc-1", content="Match positive", metadata={"group": "A"})
    doc_neg = Document(id="doc-2", content="Match opposite", metadata={"group": "B"})
    doc_orth = Document(id="doc-3", content="Match orthogonal", metadata={"group": "C"})

    embeddings = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_documents([doc_pos, doc_neg, doc_orth], embeddings)
    assert len(store) == 3

    results = store.search_with_scores([1.0, 0.0, 0.0], top_k=3)
    assert len(results) == 3

    # Ranking: doc-1 should have score 1.0
    assert results[0].document.id == "doc-1"
    assert pytest.approx(results[0].score, 0.001) == 1.0
    assert results[0].score_type == "similarity"

    # doc-3 is orthogonal: score 0.0
    assert results[1].document.id == "doc-3"
    assert pytest.approx(results[1].score, 0.001) == 0.0

    # doc-2 is opposite: score -1.0
    assert results[2].document.id == "doc-2"
    assert pytest.approx(results[2].score, 0.001) == -1.0


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu is not installed")
def test_real_faiss_duplicate_id_and_delete():
    """Verify duplicate document ID updates and index rebuilding on delete."""
    store = FAISSVectorStore(dimension=2, normalize=True)

    doc = Document(id="doc-x", content="Version 1", metadata={"v": 1})
    store.add_documents([doc], [[1.0, 0.0]])
    assert len(store) == 1

    # Update with duplicate ID
    doc_updated = Document(id="doc-x", content="Version 2", metadata={"v": 2})
    store.add_documents([doc_updated], [[0.0, 1.0]])
    assert len(store) == 1
    assert store._id_to_doc["doc-x"].content == "Version 2"

    # Search confirms updated vector
    results = store.search_with_scores([0.0, 1.0], top_k=1)
    assert results[0].document.id == "doc-x"
    assert pytest.approx(results[0].score, 0.001) == 1.0

    # Delete
    store.delete(["doc-x"])
    assert len(store) == 0
    assert store.search([0.0, 1.0], top_k=1) == []


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu is not installed")
def test_real_faiss_save_and_reload(tmp_path):
    """Verify FAISS index save_local and load_local with real binary index files."""
    save_folder = tmp_path / "real_faiss_dir"
    store1 = FAISSVectorStore(dimension=2, normalize=True)

    doc1 = Document(id="doc-1", content="Text 1", metadata={"tag": "first"})
    doc2 = Document(id="doc-2", content="Text 2", metadata={"tag": "second"})
    store1.add_documents([doc1, doc2], [[1.0, 0.0], [0.0, 1.0]])

    store1.save_local(save_folder)
    assert (save_folder / "index.faiss").exists()
    assert (save_folder / "documents.json").exists()

    # Load in new instance
    store2 = FAISSVectorStore.load_local(save_folder)
    assert len(store2) == 2
    assert store2.dimension == 2
    assert store2.normalize is True

    res = store2.search_with_scores([1.0, 0.0], top_k=1)
    assert len(res) == 1
    assert res[0].document.id == "doc-1"
    assert res[0].document.metadata["tag"] == "first"
    assert pytest.approx(res[0].score, 0.001) == 1.0
