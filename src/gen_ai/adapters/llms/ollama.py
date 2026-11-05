"""Ollama LLM adapter."""

from typing import Any

from gen_ai.core.exceptions import DependencyError, LLMError
from gen_ai.interfaces.llm import BaseLLM


class OllamaLLM(BaseLLM):
    """Adapter for Ollama local LLM models."""

    def __init__(
        self,
        model: str = "llama3",
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
                    feature_name="Ollama LLM",
                ) from exc
            self._client = ollama.Client(host=host, **kwargs)

    def generate(self, prompt: str) -> str:
        try:
            # Modern ollama supports generate(model=..., prompt=...)
            if hasattr(self._client, "generate"):
                response = self._client.generate(
                    model=self.model, prompt=prompt, **self.kwargs
                )
                if isinstance(response, dict):
                    return str(response.get("response", ""))
                return str(getattr(response, "response", ""))

            # Fallback to chat
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **self.kwargs,
            )
            if isinstance(response, dict):
                return str(response.get("message", {}).get("content", ""))
            return str(response.message.content)
        except Exception as exc:
            raise LLMError(f"Ollama generation failed: {exc}") from exc
