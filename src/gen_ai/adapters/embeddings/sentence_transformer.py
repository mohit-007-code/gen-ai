"""Sentence Transformers local embeddings adapter."""

from typing import Any

from gen_ai.core.exceptions import DependencyError, EmbeddingError
from gen_ai.interfaces.embeddings import BaseEmbeddings

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Adapter for Sentence Transformers local embeddings."""

    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(
        self,
        model: Any = None,
        device: str | None = None,
        *,
        model_name: Any = None,
        **kwargs: Any,
    ) -> None:
        if model is not None and model_name is not None and model != model_name:
            raise ValueError(
                f"Conflicting parameters provided: model={model!r} and model_name={model_name!r}. "
                "Please specify only one or ensure they have identical values."
            )

        resolved_model = (
            model
            if model is not None
            else (model_name if model_name is not None else self.DEFAULT_MODEL)
        )

        self.model = resolved_model
        self.model_name = resolved_model if isinstance(resolved_model, str) else None
        self.device = device
        self.kwargs = kwargs

        if isinstance(resolved_model, str):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise DependencyError(
                    package_name="sentence-transformers",
                    extra_name="local",
                    feature_name="Sentence Transformers Embeddings",
                ) from exc

            clean_kwargs = dict(kwargs)
            clean_kwargs.pop("model_name", None)
            clean_kwargs.pop("model", None)

            self._model = SentenceTransformer(
                resolved_model, device=device, **clean_kwargs
            )
        else:
            self._model = resolved_model

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
