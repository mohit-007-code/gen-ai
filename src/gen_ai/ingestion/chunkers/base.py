from abc import ABC, abstractmethod

from gen_ai.core.document import Document


class BaseChunker(ABC):
    """
    Base interface for document chunkers.
    """

    @abstractmethod
    def split_documents(self, documents: list[Document]) -> list[Document]: ...
