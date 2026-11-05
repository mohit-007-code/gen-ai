"""Embedding adapters for various providers and custom callables."""

from gen_ai.adapters.embeddings.callable import CallableEmbeddings
from gen_ai.adapters.embeddings.gemini import GeminiEmbeddings
from gen_ai.adapters.embeddings.ollama import OllamaEmbeddings
from gen_ai.adapters.embeddings.openai import OpenAIEmbeddings
from gen_ai.adapters.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

__all__ = [
    "CallableEmbeddings",
    "GeminiEmbeddings",
    "OllamaEmbeddings",
    "OpenAIEmbeddings",
    "SentenceTransformerEmbeddings",
]
