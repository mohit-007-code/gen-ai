from gen_ai.core.document import Document
from gen_ai.ingestion.chunkers.base import BaseChunker


class TextChunker(BaseChunker):
    """
    Splits documents into overlapping character-based chunks.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int | None = None,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if chunk_overlap is None:
            chunk_overlap = min(50, chunk_size // 5)

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:

        chunks: list[Document] = []

        for document in documents:
            content = document.content
            start = 0
            chunk_index = 0

            while start < len(content):
                end = start + self.chunk_size
                chunk_content = content[start:end]

                chunks.append(
                    Document(
                        content=chunk_content,
                        metadata={
                            **document.metadata,
                            "source_document_id": document.id,
                            "chunk_index": chunk_index,
                        },
                    )
                )

                chunk_index += 1
                start += self.chunk_size - self.chunk_overlap

        return chunks
