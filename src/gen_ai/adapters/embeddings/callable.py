"""Callable embedding adapter."""

from collections.abc import Callable

from gen_ai.core.exceptions import EmbeddingError
from gen_ai.interfaces.embeddings import BaseEmbeddings


class CallableEmbeddings(BaseEmbeddings):
    """
    Adapter enabling custom user functions to act as embedding providers.
    """

    def __init__(
        self,
        embed_documents: Callable[[list[str]], list[list[float]]],
        embed_query: Callable[[str], list[float]] | None = None,
    ) -> None:
        if not callable(embed_documents):
            raise TypeError("embed_documents must be callable")
        if embed_query is not None and not callable(embed_query):
            raise TypeError("embed_query must be callable")

        self._embed_documents = embed_documents
        self._embed_query = embed_query

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            results = self._embed_documents(texts)
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

        if not isinstance(results, list):
            raise EmbeddingError(
                f"embed_documents must return a list, received {type(results).__name__}"
            )

        if len(results) != len(texts):
            raise EmbeddingError(
                f"Expected {len(texts)} embeddings, received {len(results)}"
            )

        for i, vec in enumerate(results):
            if not isinstance(vec, list) or not all(
                isinstance(v, (int, float)) for v in vec
            ):
                raise EmbeddingError(
                    f"Embedding at index {i} must be a list of numbers"
                )

        return results

    def embed_query(self, text: str) -> list[float]:
        if self._embed_query is not None:
            try:
                result = self._embed_query(text)
            except Exception as exc:
                raise EmbeddingError(
                    f"Query embedding generation failed: {exc}"
                ) from exc
        else:
            docs = self.embed_documents([text])
            if not docs:
                raise EmbeddingError("Failed to generate embedding for query")
            result = docs[0]

        if not isinstance(result, list) or not all(
            isinstance(v, (int, float)) for v in result
        ):
            raise EmbeddingError(
                f"embed_query must return a list of numbers, received {type(result).__name__}"
            )

        return result
