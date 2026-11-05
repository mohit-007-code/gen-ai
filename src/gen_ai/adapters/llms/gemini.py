"""Google Gemini LLM adapter."""

from typing import Any

from gen_ai.core.exceptions import DependencyError, LLMError
from gen_ai.interfaces.llm import BaseLLM


class GeminiLLM(BaseLLM):
    """Adapter for Google Gemini models."""

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: str | None = None,
        generation_config: dict[str, Any] | None = None,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.generation_config = generation_config
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
                    feature_name="Gemini LLM",
                ) from exc
            if api_key:
                genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(model_name=self.model, **kwargs)

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.generate_content(
                prompt,
                generation_config=self.generation_config,
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as exc:
            raise LLMError(f"Gemini generation failed: {exc}") from exc
