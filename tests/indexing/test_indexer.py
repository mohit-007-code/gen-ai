from pathlib import Path

import pytest

from gen_ai.adapters.embeddings.callable import CallableEmbeddings
from gen_ai.core.document import Document
from gen_ai.core.exceptions import EmbeddingError, LoaderError
from gen_ai.indexing.indexer import Indexer, IndexingStats
from gen_ai.ingestion.chunkers.text import TextChunker
from gen_ai.vectorstores.in_memory import InMemoryVectorStore


def test_indexer_index_documents():
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[float(len(t)), 0.0] for t in texts],
    )
    store = InMemoryVectorStore()
    chunker = TextChunker(chunk_size=10, chunk_overlap=0)

    indexer = Indexer(
        embeddings=embeddings,
        vector_store=store,
        chunker=chunker,
        batch_size=2,
    )

    doc1 = Document(content="0123456789ABCDE", metadata={"source": "doc1"})
    doc2 = Document(content="0123456789", metadata={"source": "doc2"})

    stats = indexer.index_documents([doc1, doc2])

    assert isinstance(stats, IndexingStats)
    assert stats.total_documents == 2
    assert stats.total_chunks == 3
    assert stats.total_batches == 2
    assert len(store) == 3


def test_indexer_empty_documents():
    embeddings = CallableEmbeddings(embed_documents=lambda texts: [])
    store = InMemoryVectorStore()
    indexer = Indexer(embeddings=embeddings, vector_store=store)

    stats = indexer.index_documents([])
    assert stats.total_documents == 0
    assert stats.total_chunks == 0
    assert stats.total_batches == 0
    assert len(store) == 0


def test_indexer_invalid_batch_size():
    embeddings = CallableEmbeddings(embed_documents=lambda texts: [])
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        Indexer(embeddings=embeddings, vector_store=store, batch_size=0)


def test_indexer_index_source_txt(tmp_path: Path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello world from a text file!", encoding="utf-8")

    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0] for _ in texts],
    )
    store = InMemoryVectorStore()
    indexer = Indexer(embeddings=embeddings, vector_store=store)

    stats = indexer.index_source(txt_file)
    assert stats.total_documents == 1
    assert stats.total_chunks >= 1
    assert len(store) >= 1


def test_indexer_index_source_md(tmp_path: Path):
    md_file = tmp_path / "sample.md"
    md_file.write_text("# Markdown file content", encoding="utf-8")

    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0] for _ in texts],
    )
    store = InMemoryVectorStore()
    indexer = Indexer(embeddings=embeddings, vector_store=store)

    stats = indexer.index_source(md_file)
    assert stats.total_documents == 1
    assert stats.total_chunks == 1


def test_indexer_index_source_unsupported_file(tmp_path: Path):
    unknown_file = tmp_path / "data.xyz"
    unknown_file.write_text("Binary data", encoding="utf-8")

    embeddings = CallableEmbeddings(embed_documents=lambda texts: [])
    store = InMemoryVectorStore()
    indexer = Indexer(embeddings=embeddings, vector_store=store)

    with pytest.raises(LoaderError, match="Unsupported file format"):
        indexer.index_source(unknown_file)


def test_indexer_index_source_directory(tmp_path: Path):
    file1 = tmp_path / "f1.txt"
    file1.write_text("Content file 1", encoding="utf-8")
    file2 = tmp_path / "f2.txt"
    file2.write_text("Content file 2", encoding="utf-8")

    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0] for _ in texts],
    )
    store = InMemoryVectorStore()
    indexer = Indexer(embeddings=embeddings, vector_store=store)

    stats = indexer.index_source(tmp_path)
    assert stats.total_documents == 2
    assert stats.total_chunks == 2
    assert len(store) == 2


def test_indexer_embedding_error_propagation():
    def failing_embed(texts):
        raise RuntimeError("API dead")

    embeddings = CallableEmbeddings(embed_documents=failing_embed)
    store = InMemoryVectorStore()
    indexer = Indexer(embeddings=embeddings, vector_store=store)

    doc = Document(content="test")
    with pytest.raises(EmbeddingError):
        indexer.index_documents([doc])
