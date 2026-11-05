from pathlib import Path

import pytest

from gen_ai import (
    RAG,
    CallableEmbeddings,
    CallableLLM,
    Document,
    InMemoryVectorStore,
    RAGConfig,
    RAGResponse,
    SearchResult,
)
from gen_ai.indexing.indexer import IndexingStats


def test_public_imports():
    assert RAG is not None
    assert Document is not None
    assert SearchResult is not None
    assert RAGResponse is not None
    assert RAGConfig is not None
    assert InMemoryVectorStore is not None
    assert CallableEmbeddings is not None
    assert CallableLLM is not None


def test_rag_end_to_end_quickstart(tmp_path: Path):
    doc_path = tmp_path / "info.txt"
    doc_path.write_text("Antigravity is an agentic AI assistant.", encoding="utf-8")

    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0, 0.0] for _ in texts],
        embed_query=lambda text: [1.0, 0.0],
    )
    llm = CallableLLM(generate=lambda prompt: "Antigravity is an agentic AI.")

    rag = RAG(embeddings=embeddings, llm=llm)

    stats = rag.add_documents(doc_path)
    assert isinstance(stats, IndexingStats)
    assert stats.total_documents == 1
    assert stats.total_chunks >= 1

    response = rag.ask("What is Antigravity?")
    assert isinstance(response, RAGResponse)
    assert response.answer == "Antigravity is an agentic AI."
    assert len(response.sources) >= 1
    assert "Antigravity is an agentic AI assistant." in response.sources[0].content


def test_rag_add_document_objects():
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0] for _ in texts],
        embed_query=lambda text: [1.0],
    )
    llm = CallableLLM(generate=lambda prompt: "Generated answer")

    rag = RAG(embeddings=embeddings, llm=llm)
    docs = [Document(content="Custom document object")]
    stats = rag.add_documents(docs)

    assert stats.total_documents == 1
    assert len(rag.vector_store) == 1

    response = rag.ask("Question")
    assert response.answer == "Generated answer"


def test_rag_add_single_document_object():
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0] for _ in texts],
        embed_query=lambda text: [1.0],
    )
    llm = CallableLLM(generate=lambda prompt: "Answer")

    rag = RAG(embeddings=embeddings, llm=llm)
    stats = rag.add_documents(Document(content="Single doc"))
    assert stats.total_documents == 1
    assert len(rag.vector_store) == 1


def test_rag_add_invalid_input_type():
    embeddings = CallableEmbeddings(
        embed_documents=lambda t: [], embed_query=lambda t: []
    )
    llm = CallableLLM(generate=lambda p: "")
    rag = RAG(embeddings=embeddings, llm=llm)

    with pytest.raises(
        TypeError, match="Expected Document, list of Documents, or path"
    ):
        rag.add_documents(12345)  # type: ignore


def test_rag_search_only():
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0] for _ in texts],
        embed_query=lambda text: [1.0],
    )
    llm = CallableLLM(generate=lambda prompt: "")

    rag = RAG(embeddings=embeddings, llm=llm)
    rag.add_documents([Document(content="Searchable document")])

    results = rag.search("Searchable", top_k=1)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].document.content == "Searchable document"

    docs = rag.search_documents("Searchable", top_k=1)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)


def test_rag_custom_config():
    config = RAGConfig(chunk_size=200, chunk_overlap=20, top_k=5)
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0] for _ in texts],
        embed_query=lambda text: [1.0],
    )
    llm = CallableLLM(generate=lambda p: "")
    store = InMemoryVectorStore()

    rag = RAG(
        embeddings=embeddings,
        llm=llm,
        vector_store=store,
        config=config,
    )

    assert rag.config.chunk_size == 200
    assert rag.config.top_k == 5
    assert rag.vector_store is store
