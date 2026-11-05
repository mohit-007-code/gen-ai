"""gen-ai: A lightweight, extensible RAG framework."""

from gen_ai.adapters.embeddings.callable import CallableEmbeddings
from gen_ai.adapters.llms.callable import CallableLLM
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
from gen_ai.indexing.indexer import Indexer, IndexingStats
from gen_ai.ingestion.chunkers.base import BaseChunker
from gen_ai.ingestion.chunkers.recursive import RecursiveCharacterChunker
from gen_ai.ingestion.chunkers.text import TextChunker
from gen_ai.ingestion.loaders.base import BaseLoader
from gen_ai.ingestion.loaders.directory import DirectoryLoader
from gen_ai.ingestion.loaders.markdown import MarkdownLoader
from gen_ai.ingestion.loaders.pdf import PDFLoader
from gen_ai.ingestion.loaders.text import TextLoader
from gen_ai.interfaces.embeddings import BaseEmbeddings
from gen_ai.interfaces.llm import BaseLLM
from gen_ai.interfaces.vectorstore import BaseVectorStore
from gen_ai.pipelines.rag import RAGPipeline, RAGResponse
from gen_ai.rag import RAG
from gen_ai.retrieval.retriever import Retriever
from gen_ai.vectorstores.in_memory import InMemoryVectorStore

__version__ = "0.1.0"

__all__ = [
    "RAG",
    "BaseChunker",
    "BaseEmbeddings",
    "BaseLLM",
    "BaseLoader",
    "BaseVectorStore",
    "CallableEmbeddings",
    "CallableLLM",
    "ChunkerError",
    "ConfigurationError",
    "DependencyError",
    "DirectoryLoader",
    "Document",
    "DocumentError",
    "EmbeddingError",
    "GenAIError",
    "InMemoryVectorStore",
    "Indexer",
    "IndexingStats",
    "LLMError",
    "LoaderError",
    "MarkdownLoader",
    "PDFLoader",
    "RAGConfig",
    "RAGError",
    "RAGPipeline",
    "RAGResponse",
    "RecursiveCharacterChunker",
    "RetrievalError",
    "Retriever",
    "SearchResult",
    "TextChunker",
    "TextLoader",
    "VectorStoreError",
    "__version__",
]
