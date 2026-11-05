"""Ollama Embeddings adapter."""

from typing import Any

from gen_ai.core.exceptions import DependencyError, EmbeddingError
from gen_ai.interfaces.embeddings import BaseEmbeddings


class OllamaEmbeddings(BaseEmbeddings):
    """Adapter for Ollama local embeddings."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str | None = None,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.kwargs = kwargs

        if client is not None:
            self._client = client
        else:
            try:
                import ollama
            except ImportError as exc:
                raise DependencyError(
                    package_name="ollama",
                    extra_name="ollama",
                    feature_name="Ollama Embeddings",
                ) from exc
            self._client = ollama.Client(host=host, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            # Modern ollama supports client.embed(model=..., input=...)
            if hasattr(self._client, "embed"):
                response = self._client.embed(
                    model=self.model, input=texts, **self.kwargs
                )
                embeddings = response.get("embeddings") or response.embeddings
                return [list(map(float, vec)) for vec in embeddings]

            # Fallback to per-text embeddings
            results = []
            for text in texts:
                resp = self._client.embeddings(
                    model=self.model, prompt=text, **self.kwargs
                )
                vec = resp.get("embedding") or resp.embedding
                results.append(list(map(float, vec)))
            return results
        except Exception as exc:
            raise EmbeddingError(f"Ollama embedding generation failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        results = self.embed_documents([text])
        if not results:
            raise EmbeddingError("Failed to generate query embedding from Ollama")
        return results[0]
