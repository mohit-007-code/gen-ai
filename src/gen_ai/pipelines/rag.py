"""RAG Pipeline orchestration layer."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gen_ai.core.document import Document
from gen_ai.core.exceptions import RAGError
from gen_ai.core.search_result import SearchResult
from gen_ai.interfaces.llm import BaseLLM
from gen_ai.retrieval.retriever import Retriever
from gen_ai.utils.logging import get_logger

logger = get_logger("pipelines.rag")

DEFAULT_PROMPT_TEMPLATE = """Use the following context to answer the question at the end.
If the context does not contain enough information to answer the question, state that clearly.
Do not make assumptions or fabricate information.

Context:
{context}

Question: {query}

Answer:"""


class RAGResponse(BaseModel):
    """Encapsulates the result of a RAG query."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    query: str = Field(description="The original user query")
    answer: str = Field(description="The generated response from the LLM")
    sources: list[Document] = Field(
        default_factory=list, description="Documents retrieved as context"
    )
    results: list[SearchResult] = Field(
        default_factory=list,
        description="Search results with scores and rankings",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Execution and retrieval metadata"
    )


class RAGPipeline:
    """
    Orchestrates the full Retrieval-Augmented Generation pipeline:
    Query -> Retrieval -> Context Construction -> Prompt Construction -> LLM Generation.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        prompt_template: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt

        template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        if "{context}" not in template or "{query}" not in template:
            raise ValueError(
                "Prompt template must contain '{context}' and '{query}' placeholders."
            )
        self.prompt_template = template

    def build_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into a context string."""
        if not documents:
            return "No relevant context found."

        parts: list[str] = []
        for i, doc in enumerate(documents, start=1):
            source = (
                doc.metadata.get("source")
                or doc.metadata.get("file_name")
                or f"Doc-{i}"
            )
            parts.append(f"[{i}] (Source: {source})\n{doc.content.strip()}")

        return "\n\n".join(parts)

    def build_prompt(self, query: str, context: str) -> str:
        """Construct the full prompt for the LLM."""
        formatted_prompt = self.prompt_template.format(
            context=context,
            query=query,
        )

        if self.system_prompt:
            return f"{self.system_prompt.strip()}\n\n{formatted_prompt.strip()}"
        return formatted_prompt.strip()

    def run(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """
        Execute the RAG workflow for a user query.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only")

        clean_query = query.strip()
        logger.info("Running RAG pipeline for query: '%s'", clean_query)

        try:
            results = self.retriever.retrieve(
                query=clean_query,
                top_k=top_k,
                score_threshold=score_threshold,
                metadata_filter=metadata_filter,
            )
        except Exception as exc:
            raise RAGError(f"RAG retrieval step failed: {exc}") from exc

        sources = [r.document for r in results]
        logger.debug("Retrieved %d sources for query context", len(sources))
        context = self.build_context(sources)
        prompt = self.build_prompt(query=clean_query, context=context)

        try:
            logger.debug(
                "Dispatching prompt to LLM (%s)...", self.llm.__class__.__name__
            )
            answer = self.llm.generate(prompt)
        except Exception as exc:
            raise RAGError(f"RAG generation failed: {exc}") from exc

        logger.info("Successfully generated RAG response")
        return RAGResponse(
            query=clean_query,
            answer=answer,
            sources=sources,
            results=results,
            metadata={
                "retrieved_count": len(results),
                "has_context": bool(sources),
            },
        )
