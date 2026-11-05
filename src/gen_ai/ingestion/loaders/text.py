from pathlib import Path

from gen_ai.core.document import Document
from gen_ai.ingestion.loaders.base import BaseLoader


class TextLoader(BaseLoader):
    """
    Load plain text files into Document objects.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load(self, source: str | Path) -> list[Document]:
        """
        Read a text file and convert it into a Document.

        Args:
            source: Path to the text file.

        Returns:
            A list containing the loaded Document.
        """

        file_path = Path(source)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise IsADirectoryError(
                f"Expected a file, received a directory: {file_path}"
            )

        content = file_path.read_text(encoding=self.encoding)

        if not content.strip():
            raise ValueError(f"Text file is empty: {file_path}")

        document = Document(
            content=content,
            metadata={
                "source": str(file_path.resolve()),
                "file_name": file_path.name,
                "file_type": "txt",
            },
        )

        return [document]
