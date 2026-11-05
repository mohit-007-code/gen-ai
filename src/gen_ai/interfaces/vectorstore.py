from abc import ABC, abstractmethod

from gen_ai.core.document import Document
from gen_ai.core.search_result import SearchResult


class BaseVectorStore(ABC):
    """
    Abstract interface for vector database integrations.

    Implementations must support document insertion,
    similarity search, and document deletion.
    """

    @abstractmethod
    def add_documents(
        self, documents: list[Document], embeddings: list[list[float]]
    ) -> None:
        """
        Store documents and their corresponding embeddings.

        Args:
            documents: Documents to store.
            embeddings: Embedding vectors corresponding to documents.
        """
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Document]:
        """
        Retrieve the most similar documents.

        Args:
            query_embedding: Vector representation of the query.
            top_k: Maximum number of documents to retrieve.

        Returns:
            Relevant documents ordered by similarity.
        """
        ...

    def search_with_scores(
        self, query_embedding: list[float], top_k: int = 3
    ) -> list[SearchResult]:
        """
        Retrieve the most similar documents with similarity or distance scores.

        Default implementation falls back to calling search() and returns
        results with score=None and score_type="unavailable" rather than
        assigning arbitrary or fake relevance scores.

        Args:
            query_embedding: Vector representation of the query.
            top_k: Maximum number of documents to retrieve.

        Returns:
            List of SearchResult objects containing document and score (if available).
        """
        documents = self.search(query_embedding, top_k=top_k)
        return [
            SearchResult(document=doc, score=None, score_type="unavailable")
            for doc in documents
        ]

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """
        Delete documents using their unique IDs.

        Args:
            ids: Document identifiers to delete.
        """
        ...
