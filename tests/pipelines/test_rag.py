from unittest.mock import MagicMock

import pytest

from gen_ai.adapters.embeddings.callable import CallableEmbeddings
from gen_ai.adapters.llms.callable import CallableLLM
from gen_ai.core.document import Document
from gen_ai.core.exceptions import RAGError
from gen_ai.core.search_result import SearchResult
from gen_ai.pipelines.rag import RAGPipeline, RAGResponse
from gen_ai.retrieval.retriever import Retriever
from gen_ai.vectorstores.in_memory import InMemoryVectorStore


def test_rag_pipeline_end_to_end():
    store = InMemoryVectorStore()
    doc1 = Document(
        content="Antigravity is designed by Google DeepMind.",
        metadata={"source": "deepmind.txt"},
    )
    doc2 = Document(
        content="Paris is the capital of France.",
        metadata={"source": "geo.txt"},
    )
    store.add_documents([doc1, doc2], [[1.0, 0.0], [0.0, 1.0]])

    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0, 0.0] for _ in texts],
        embed_query=lambda text: [1.0, 0.0],
    )
    retriever = Retriever(embeddings=embeddings, vector_store=store, top_k=1)

    generated_prompts = []

    def mock_generate(prompt: str) -> str:
        generated_prompts.append(prompt)
        return "Answer based on context"

    llm = CallableLLM(generate=mock_generate)
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    response = pipeline.run("Who designed Antigravity?")

    assert isinstance(response, RAGResponse)
    assert response.query == "Who designed Antigravity?"
    assert response.answer == "Answer based on context"
    assert len(response.sources) == 1
    assert response.sources[0].content == "Antigravity is designed by Google DeepMind."
    assert response.sources[0].metadata["source"] == "deepmind.txt"
    assert len(response.results) == 1
    assert "deepmind.txt" in generated_prompts[0]
    assert "Who designed Antigravity?" in generated_prompts[0]


def test_rag_pipeline_custom_prompt_template():
    store = InMemoryVectorStore()
    store.add_documents([Document(content="Information")], [[1.0]])
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0] for _ in texts],
        embed_query=lambda text: [1.0],
    )
    retriever = Retriever(embeddings=embeddings, vector_store=store)

    received_prompt = None

    def mock_generate(prompt: str) -> str:
        nonlocal received_prompt
        received_prompt = prompt
        return "Done"

    llm = CallableLLM(generate=mock_generate)
    custom_template = "CTX: {context}\nQ: {query}\nA:"
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        prompt_template=custom_template,
        system_prompt="SYS_RULE",
    )

    pipeline.run("My Question")
    assert "SYS_RULE" in received_prompt
    assert "CTX: [1] (Source: Doc-1)\nInformation" in received_prompt
    assert "Q: My Question" in received_prompt


def test_rag_pipeline_invalid_prompt_template():
    store = InMemoryVectorStore()
    embeddings = CallableEmbeddings(
        embed_documents=lambda t: [[1.0]], embed_query=lambda t: [1.0]
    )
    retriever = Retriever(embeddings=embeddings, vector_store=store)
    llm = CallableLLM(generate=lambda p: "")

    with pytest.raises(
        ValueError, match="Prompt template must contain.*context.*and.*query"
    ):
        RAGPipeline(
            retriever=retriever, llm=llm, prompt_template="Missing placeholders"
        )


def test_rag_pipeline_empty_query():
    store = InMemoryVectorStore()
    embeddings = CallableEmbeddings(
        embed_documents=lambda t: [[1.0]], embed_query=lambda t: [1.0]
    )
    retriever = Retriever(embeddings=embeddings, vector_store=store)
    llm = CallableLLM(generate=lambda p: "")
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        pipeline.run("   ")


def test_rag_pipeline_llm_failure():
    store = InMemoryVectorStore()
    embeddings = CallableEmbeddings(
        embed_documents=lambda t: [[1.0]], embed_query=lambda t: [1.0]
    )
    retriever = Retriever(embeddings=embeddings, vector_store=store)

    def failing_llm(p):
        raise RuntimeError("LLM exploded")

    llm = CallableLLM(generate=failing_llm)
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    with pytest.raises(RAGError, match="RAG generation failed"):
        pipeline.run("Question")


def test_rag_pipeline_empty_retrieval_grounding():
    store = InMemoryVectorStore()
    embeddings = CallableEmbeddings(
        embed_documents=lambda t: [], embed_query=lambda t: [1.0]
    )
    retriever = Retriever(embeddings=embeddings, vector_store=store)

    received_prompts = []
    llm = CallableLLM(
        generate=lambda p: (
            received_prompts.append(p) or "I do not have enough information to answer."
        )
    )
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    response = pipeline.run("What is quantum computing?")
    assert response.sources == []
    assert response.results == []
    assert response.metadata["has_context"] is False
    assert "No relevant context found." in received_prompts[0]


def test_rag_pipeline_deduplicates_duplicate_sources():
    doc1 = Document(
        id="dup-1", content="Content from chunk 1", metadata={"source": "book.pdf"}
    )
    # Same doc id returned twice by retrieval
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        SearchResult(document=doc1, score=0.9, score_type="similarity"),
        SearchResult(document=doc1, score=0.85, score_type="similarity"),
    ]

    llm = CallableLLM(generate=lambda p: "Answer")
    pipeline = RAGPipeline(retriever=mock_retriever, llm=llm)

    response = pipeline.run("Query")
    assert len(response.sources) == 1
    assert len(response.results) == 1
    assert response.metadata["retrieved_count"] == 2
    assert response.metadata["unique_sources_count"] == 1


def test_rag_pipeline_missing_citations_fallback():
    doc_no_meta = Document(content="Uncited fact", metadata={})
    pipeline = RAGPipeline(
        retriever=MagicMock(),
        llm=CallableLLM(generate=lambda p: "Done"),
    )
    context = pipeline.build_context([doc_no_meta])
    assert "[1] (Source: Doc-1)" in context
    assert "Uncited fact" in context


def test_rag_pipeline_multiple_sources_ordering():
    doc1 = Document(content="Fact 1", metadata={"source": "a.txt"})
    doc2 = Document(content="Fact 2", metadata={"file_name": "b.txt"})
    pipeline = RAGPipeline(
        retriever=MagicMock(),
        llm=CallableLLM(generate=lambda p: "Done"),
    )
    context = pipeline.build_context([doc1, doc2])
    assert "[1] (Source: a.txt)\nFact 1" in context
    assert "[2] (Source: b.txt)\nFact 2" in context


def test_rag_pipeline_invalid_template_placeholder():
    store = InMemoryVectorStore()
    store.add_documents([Document(content="Data")], [[1.0]])
    embeddings = CallableEmbeddings(
        embed_documents=lambda t: [[1.0]], embed_query=lambda t: [1.0]
    )
    retriever = Retriever(embeddings=embeddings, vector_store=store)
    llm = CallableLLM(generate=lambda p: "Done")

    # Template has valid {context} and {query}, but also unknown {unknown_param}
    template_with_unknown = "Context: {context}\nQuery: {query}\nExtra: {unknown_param}"
    pipeline = RAGPipeline(
        retriever=retriever, llm=llm, prompt_template=template_with_unknown
    )

    with pytest.raises(
        ValueError, match="Prompt template contains unknown placeholder"
    ):
        pipeline.run("Test query")


def test_rag_pipeline_llm_empty_response(caplog):
    store = InMemoryVectorStore()
    store.add_documents([Document(content="Data")], [[1.0]])
    embeddings = CallableEmbeddings(
        embed_documents=lambda t: [[1.0]], embed_query=lambda t: [1.0]
    )
    retriever = Retriever(embeddings=embeddings, vector_store=store)
    llm = CallableLLM(generate=lambda p: "   ")  # Whitespace only
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    response = pipeline.run("Test query")
    assert response.answer == "   "
    assert response.metadata["empty_response"] is True
    assert "LLM returned an empty or whitespace response" in caplog.text
