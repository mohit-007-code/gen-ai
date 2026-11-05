"""OpenAI LLM adapter."""

from typing import Any

from gen_ai.core.exceptions import DependencyError, LLMError
from gen_ai.interfaces.llm import BaseLLM


class OpenAILLM(BaseLLM):
    """Adapter for OpenAI Chat Completion models."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
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
                import openai
            except ImportError as exc:
                raise DependencyError(
                    package_name="openai",
                    extra_name="openai",
                    feature_name="OpenAI LLM",
                ) from exc
            self._client = openai.OpenAI(api_key=api_key, **kwargs)

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
            raise LLMError(f"OpenAI generation failed: {exc}") from exc
