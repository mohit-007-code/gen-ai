"""Groq LLM adapter."""

from typing import Any

from gen_ai.core.exceptions import DependencyError, LLMError
from gen_ai.interfaces.llm import BaseLLM


class GroqLLM(BaseLLM):
    """Adapter for Groq Cloud LLM API."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        if client is not None:
            self._client = client
        else:
            try:
                import groq
            except ImportError as exc:
                raise DependencyError(
                    package_name="groq",
                    extra_name="groq",
                    feature_name="Groq LLM",
                ) from exc
            self._client = groq.Groq(api_key=api_key, **kwargs)

    def generate(self, prompt: str) -> str:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            **self.kwargs,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        try:
            response = self._client.chat.completions.create(**params)
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as exc:
            raise LLMError(f"Groq generation failed: {exc}") from exc
