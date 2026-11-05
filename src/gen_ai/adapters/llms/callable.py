"""Callable LLM adapter."""

from collections.abc import Callable

from gen_ai.core.exceptions import LLMError
from gen_ai.interfaces.llm import BaseLLM


class CallableLLM(BaseLLM):
    """
    Adapter enabling custom user functions to act as language models.
    """

    def __init__(self, generate: Callable[[str], str]) -> None:
        if not callable(generate):
            raise TypeError("generate must be callable")
        self._generate = generate

    def generate(self, prompt: str) -> str:
        try:
            result = self._generate(prompt)
        except Exception as exc:
            raise LLMError(f"LLM generation failed: {exc}") from exc

        if not isinstance(result, str):
            raise LLMError(
                f"generate must return a string, received {type(result).__name__}"
            )

        return result
