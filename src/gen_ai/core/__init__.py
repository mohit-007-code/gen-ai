"""Core data structures and exceptions for gen-ai."""

from gen_ai.core.config import RAGConfig
from gen_ai.core.document import Document
from gen_ai.core.exceptions import (
    ChunkerError,
    ConfigurationError,
    DependencyError,
    DocumentError,
    EmbeddingError,
    GenAIError,
    LLMError,
    LoaderError,
    RAGError,
    RetrievalError,
    VectorStoreError,
)
from gen_ai.core.search_result import SearchResult

__all__ = [
    "ChunkerError",
    "ConfigurationError",
    "DependencyError",
    "Document",
    "DocumentError",
    "EmbeddingError",
    "GenAIError",
    "LLMError",
    "LoaderError",
    "RAGConfig",
    "RAGError",
    "RetrievalError",
    "SearchResult",
    "VectorStoreError",
]
