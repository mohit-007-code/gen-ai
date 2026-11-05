from abc import ABC, abstractmethod


class BaseEmbeddings(ABC):
    """
    Abstract interface for embedding providers.

    Every embedding implementation must provide
    document and query embedding methods.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of document texts.

        Returns:
            List of embedding vectors.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single query.

        Args:
            text: User query.

        Returns:
            Embedding vector.
        """
        ...
