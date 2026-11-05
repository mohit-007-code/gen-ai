"""Integration tests for ChromaVectorStore using the real chromadb library."""

import pytest

try:
    import chromadb

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

from gen_ai.core.document import Document
from gen_ai.vectorstores.chroma import ChromaVectorStore


@pytest.mark.skipif(
    not HAS_CHROMADB, reason="chromadb optional dependency is not installed"
)
def test_real_chroma_lifecycle_and_negative_cosine(tmp_path):
    """Test real ChromaDB end-to-end including negative cosine similarity and empty metadata."""
    client = chromadb.Client()
    store = ChromaVectorStore(
        collection_name="test_real_lifecycle",
        client=client,
    )

    doc_pos = Document(
        id="doc-1", content="Antigravity framework", metadata={"cat": "rag"}
    )
    doc_neg = Document(
        id="doc-2", content="Baking sourdough bread", metadata={}
    )  # empty metadata!
    doc_nested = Document(
        id="doc-3",
        content="Deep learning models",
        metadata={"stats": {"layers": 12}, "tags": ["nlp", "ai"]},
    )

    # 3-dim vectors: doc1 and doc2 are opposing vectors on first dimension
    embeddings = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_documents([doc_pos, doc_neg, doc_nested], embeddings)

    # Search with positive query
    results = store.search_with_scores([1.0, 0.0, 0.0], top_k=3)
    assert len(results) == 3

    # Ranking: doc-1 should have similarity ~1.0
    assert results[0].document.id == "doc-1"
    assert pytest.approx(results[0].score, 0.01) == 1.0

    # Orthogonal doc-3 should have similarity ~0.0
    assert results[1].document.id == "doc-3"
    assert pytest.approx(results[1].score, 0.01) == 0.0
    # Nested metadata must be restored as dict / list
    assert results[1].document.metadata["stats"] == {"layers": 12}
    assert results[1].document.metadata["tags"] == ["nlp", "ai"]

    # Opposing doc-2 must preserve negative similarity ~ -1.0 (not clamped to 0.0)
    assert results[2].document.id == "doc-2"
    assert pytest.approx(results[2].score, 0.01) == -1.0
    # Empty metadata must be restored as {}
    assert results[2].document.metadata == {}


@pytest.mark.skipif(
    not HAS_CHROMADB, reason="chromadb optional dependency is not installed"
)
def test_real_chroma_persistence_restart(tmp_path):
    """Test persistent ChromaDB storage surviving client restart."""
    db_path = str(tmp_path / "chroma_db")

    # Step 1: Write documents using a persistent store
    store1 = ChromaVectorStore(
        collection_name="persisted_collection",
        persist_directory=db_path,
    )
    doc1 = Document(
        id="doc-persist-1", content="Persistent doc content", metadata={"saved": True}
    )
    store1.add_documents([doc1], [[0.5, 0.5]])

    # Step 2: Simulate restart with a new store instance pointing to same directory
    del store1

    store2 = ChromaVectorStore(
        collection_name="persisted_collection",
        persist_directory=db_path,
    )
    results = store2.search_with_scores([0.5, 0.5], top_k=1)
    assert len(results) == 1
    assert results[0].document.id == "doc-persist-1"
    assert results[0].document.content == "Persistent doc content"
    assert results[0].document.metadata["saved"] is True

    # Step 3: Delete and verify
    store2.delete(["doc-persist-1"])
    remaining = store2.search([0.5, 0.5], top_k=1)
    assert len(remaining) == 0
