"""OpenAI Embeddings adapter."""

from typing import Any

from gen_ai.core.exceptions import DependencyError, EmbeddingError
from gen_ai.interfaces.embeddings import BaseEmbeddings


class OpenAIEmbeddings(BaseEmbeddings):
    """Adapter for OpenAI embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.kwargs = kwargs

        if client is not None:
            self._client = client
        else:
            try:
                import openai
            except ImportError as exc:
                raise DependencyError(
                    package_name="openai",
                    extra_name="openai",
                    feature_name="OpenAI Embeddings",
                ) from exc
            self._client = openai.OpenAI(api_key=api_key, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        params: dict[str, Any] = {"input": texts, "model": self.model}
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions

        try:
            response = self._client.embeddings.create(**params)
            return [list(item.embedding) for item in response.data]
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding generation failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        results = self.embed_documents([text])
        if not results:
            raise EmbeddingError("Failed to generate query embedding from OpenAI")
        return results[0]
