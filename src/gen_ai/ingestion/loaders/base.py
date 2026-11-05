from abc import ABC, abstractmethod
from pathlib import Path

from gen_ai.core.document import Document


class BaseLoader(ABC):
    """
    Abstract interface for document loaders.

    Every loader must implement a load method that
    converts a source into Document objects.
    """

    @abstractmethod
    def load(self, source: str | Path) -> list[Document]:
        """
        Load documents from a given source.

        Args:
            source: File path or supported source.

        Returns:
            List of loaded Document objects.
        """
        ...
