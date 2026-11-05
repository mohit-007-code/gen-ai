"""Retriever for querying vector stores and filtering results."""

from typing import Any

from gen_ai.core.document import Document
from gen_ai.core.exceptions import RetrievalError
from gen_ai.core.search_result import SearchResult
from gen_ai.interfaces.embeddings import BaseEmbeddings
from gen_ai.interfaces.vectorstore import BaseVectorStore
from gen_ai.utils.logging import get_logger

logger = get_logger("retrieval")


class Retriever:
    """
    Retrieves relevant documents from a vector store for a given user query.
    Supports score threshold filtering and metadata filtering.
    """

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vector_store: BaseVectorStore,
        top_k: int = 3,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        candidate_multiplier: int = 3,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, received {top_k}")
        if candidate_multiplier <= 0:
            raise ValueError(
                f"candidate_multiplier must be greater than 0, received {candidate_multiplier}"
            )

        self.embeddings = embeddings
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.metadata_filter = metadata_filter
        self.candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Embed the query and retrieve top-k matching SearchResult items.

        Note on top_k guarantee:
            When score_threshold or metadata_filter is specified, top_k acts as an
            upper bound, NOT a strict guarantee. If fewer candidates satisfy the
            filters within the candidate retrieval window (candidate_multiplier * top_k),
            fewer results will be returned.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only")

        effective_top_k = top_k if top_k is not None else self.top_k
        if effective_top_k <= 0:
            raise ValueError(
                f"top_k must be greater than 0, received {effective_top_k}"
            )

        effective_threshold = (
            score_threshold if score_threshold is not None else self.score_threshold
        )
        effective_filter = (
            metadata_filter if metadata_filter is not None else self.metadata_filter
        )

        logger.debug(
            "Retrieving documents (query='%s', top_k=%d, threshold=%s)",
            query,
            effective_top_k,
            effective_threshold,
        )

        try:
            query_vector = self.embeddings.embed_query(query.strip())
        except Exception as exc:
            raise RetrievalError(f"Failed to generate query embedding: {exc}") from exc

        try:
            candidate_expansion = (
                effective_filter is not None or effective_threshold is not None
            )
            fetch_k = (
                effective_top_k * self.candidate_multiplier
                if candidate_expansion
                else effective_top_k
            )
            results = self.vector_store.search_with_scores(query_vector, top_k=fetch_k)
        except Exception as exc:
            raise RetrievalError(f"Vector store search failed: {exc}") from exc

        filtered_results: list[SearchResult] = []
        for result in results:
            # Score threshold filter
            if effective_threshold is not None:
                if result.score is None or result.score_type == "unavailable":
                    logger.warning(
                        "Score threshold %s specified but result for doc '%s' has unavailable score. Retaining candidate.",
                        effective_threshold,
                        result.document.id,
                    )
                elif (
                    result.score_type == "similarity"
                    and result.score < effective_threshold
                ) or (
                    result.score_type == "distance"
                    and result.score > effective_threshold
                ):
                    continue

            # Metadata filter
            if effective_filter:
                match = True
                for key, expected_val in effective_filter.items():
                    if result.document.metadata.get(key) != expected_val:
                        match = False
                        break
                if not match:
                    continue

            filtered_results.append(result)
            if len(filtered_results) >= effective_top_k:
                break

        logger.debug(
            "Retrieved %d candidates, returning %d matches",
            len(results),
            len(filtered_results),
        )
        return filtered_results

    def retrieve_documents(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieve matching Document objects directly.
        """
        results = self.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )
        return [r.document for r in results]
