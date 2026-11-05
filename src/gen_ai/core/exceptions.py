"""Core exception hierarchy for the gen-ai framework."""

from typing import Any


class GenAIError(Exception):
    """Base exception for all errors in the gen-ai framework."""

    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args)
        self.message = message


class DependencyError(GenAIError):
    """Raised when an optional third-party dependency is missing."""

    def __init__(
        self,
        package_name: str,
        extra_name: str | None = None,
        feature_name: str | None = None,
    ) -> None:
        install_target = f"gen-ai[{extra_name}]" if extra_name else package_name
        feature_desc = f" for {feature_name}" if feature_name else ""
        msg = (
            f"Optional dependency '{package_name}' is required{feature_desc}. "
            f'Install it using: pip install "{install_target}"'
        )
        super().__init__(msg)
        self.package_name = package_name
        self.extra_name = extra_name
        self.feature_name = feature_name


class ConfigurationError(GenAIError):
    """Raised when framework or component configuration is invalid."""


class DocumentError(GenAIError):
    """Raised when document creation or validation fails."""


class LoaderError(GenAIError):
    """Raised when loading documents from a source fails."""


class ChunkerError(GenAIError):
    """Raised when text splitting or chunking fails."""


class EmbeddingError(GenAIError):
    """Raised when embedding generation fails."""


class VectorStoreError(GenAIError):
    """Raised when vector database operations fail."""


class RetrievalError(GenAIError):
    """Raised when retrieval operations fail."""


class LLMError(GenAIError):
    """Raised when LLM text generation fails."""


class RAGError(GenAIError):
    """Raised when RAG orchestration fails."""
