"""Sentence Transformers local embeddings adapter."""

from typing import Any

from gen_ai.core.exceptions import DependencyError, EmbeddingError
from gen_ai.interfaces.embeddings import BaseEmbeddings


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Adapter for Sentence Transformers local embeddings."""

    def __init__(
        self,
        model: Any = "all-MiniLM-L6-v2",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.device = device
        self.kwargs = kwargs

        if isinstance(model, str):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise DependencyError(
                    package_name="sentence-transformers",
                    extra_name="local",
                    feature_name="Sentence Transformers Embeddings",
                ) from exc
            self._model = SentenceTransformer(model, device=device, **kwargs)
        else:
            self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return [list(map(float, vec)) for vec in embeddings]
        except Exception as exc:
            raise EmbeddingError(
                f"SentenceTransformer embedding generation failed: {exc}"
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return list(map(float, embedding))
        except Exception as exc:
            raise EmbeddingError(
                f"SentenceTransformer query embedding failed: {exc}"
            ) from exc
