from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract interface for all LLM providers.

    Every LLM implementation must provide a generate()
    method that accepts a prompt and returns text.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the language model.

        Args:
            prompt: Input prompt for the LLM.

        Returns:
            Generated text response.
        """
        ...
