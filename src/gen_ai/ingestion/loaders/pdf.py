from pathlib import Path

import pymupdf

from gen_ai.core.document import Document
from gen_ai.ingestion.loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    """
    Load PDF files into Document objects.

    Each PDF page is represented as a separate Document.
    """

    def load(self, source: str | Path) -> list[Document]:

        file_path = Path(source)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.is_file():
            raise IsADirectoryError(f"Expected a PDF file, received: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, received: {file_path.suffix}")

        documents = []

        try:
            with pymupdf.open(file_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    content = page.get_text().strip()

                    if not content:
                        continue

                    document = Document(
                        content=content,
                        metadata={
                            "source": str(file_path.resolve()),
                            "file_name": file_path.name,
                            "file_type": "pdf",
                            "page": page_number,
                        },
                    )

                    documents.append(document)

        except pymupdf.FileDataError as exc:
            raise ValueError(f"Unable to read PDF file: {file_path}") from exc

        if not documents:
            raise ValueError(f"No extractable text found in PDF: {file_path}")

        return documents
