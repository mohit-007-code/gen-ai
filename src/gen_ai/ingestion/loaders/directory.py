from pathlib import Path

from gen_ai.core.document import Document
from gen_ai.ingestion.loaders.base import BaseLoader
from gen_ai.ingestion.loaders.markdown import MarkdownLoader
from gen_ai.ingestion.loaders.pdf import PDFLoader
from gen_ai.ingestion.loaders.text import TextLoader


class DirectoryLoader(BaseLoader):
    """
    Loads supported documents from a directory.
    """

    def __init__(self, recursive: bool = False) -> None:
        self.recursive = recursive

        self.loaders: dict[str, BaseLoader] = {
            ".txt": TextLoader(),
            ".pdf": PDFLoader(),
            ".md": MarkdownLoader(),
            ".markdown": MarkdownLoader(),
        }

    def load(self, source: str | Path) -> list[Document]:
        directory = Path(source)

        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {directory}")

        documents: list[Document] = []

        files = directory.rglob("*") if self.recursive else directory.iterdir()

        for file_path in sorted(files):
            if not file_path.is_file():
                continue

            loader = self.loaders.get(file_path.suffix.lower())

            if loader is None:
                continue

            loaded_documents = loader.load(file_path)
            documents.extend(loaded_documents)

        return documents
