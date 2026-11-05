"""Abstract interfaces defining the core extension points of gen-ai."""

from gen_ai.interfaces.embeddings import BaseEmbeddings
from gen_ai.interfaces.llm import BaseLLM
from gen_ai.interfaces.vectorstore import BaseVectorStore

__all__ = ["BaseEmbeddings", "BaseLLM", "BaseVectorStore"]
