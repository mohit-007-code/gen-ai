"""Google Gemini Embeddings adapter."""

from typing import Any

from gen_ai.core.exceptions import DependencyError, EmbeddingError
from gen_ai.interfaces.embeddings import BaseEmbeddings


class GeminiEmbeddings(BaseEmbeddings):
    """Adapter for Google Gemini embeddings API."""

    def __init__(
        self,
        model: str = "models/text-embedding-004",
        api_key: str | None = None,
        client: Any = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.task_type = task_type
        self.kwargs = kwargs

        if client is not None:
            self._client = client
        else:
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise DependencyError(
                    package_name="google-generativeai",
                    extra_name="gemini",
                    feature_name="Gemini Embeddings",
                ) from exc
            if api_key:
                genai.configure(api_key=api_key)
            self._client = genai

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = self._client.embed_content(
                model=self.model,
                content=texts,
                task_type=self.task_type,
                **self.kwargs,
            )
            raw = response["embedding"]
            # raw can be a list of lists or a single list if one text was passed
            if texts and isinstance(raw[0], (int, float)):
                return [list(map(float, raw))]
            return [list(map(float, vec)) for vec in raw]
        except Exception as exc:
            raise EmbeddingError(f"Gemini embedding generation failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            response = self._client.embed_content(
                model=self.model,
                content=text,
                task_type="RETRIEVAL_QUERY",
                **self.kwargs,
            )
            raw = response["embedding"]
            return list(map(float, raw))
        except Exception as exc:
            raise EmbeddingError(f"Gemini query embedding failed: {exc}") from exc
