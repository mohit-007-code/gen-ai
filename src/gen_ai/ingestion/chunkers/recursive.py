"""Recursive character text chunker."""

from gen_ai.core.document import Document
from gen_ai.ingestion.chunkers.base import BaseChunker


class RecursiveCharacterChunker(BaseChunker):
    """
    Splits text recursively using a hierarchy of separators
    (paragraphs, lines, words, characters) to maintain semantic cohesion.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
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
        self.separators = (
            separators if separators is not None else ["\n\n", "\n", " ", ""]
        )

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        separator = separators[-1]
        new_separators: list[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits: list[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_chunks = self._split_text(s, new_separators)
                    good_splits.extend(sub_chunks)
                else:
                    # Hard split
                    for idx in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        good_splits.append(s[idx : idx + self.chunk_size])

        # Merge splits with separator up to chunk_size
        return self._merge_splits(good_splits, separator)

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        merged: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for piece in splits:
            piece_len = len(piece)
            sep_len = len(separator) if current_chunk else 0

            if current_length + sep_len + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_length += sep_len + piece_len
            else:
                if current_chunk:
                    chunk_str = separator.join(current_chunk).strip()
                    if chunk_str:
                        merged.append(chunk_str)

                    # Build overlap from end of current chunk
                    overlap_pieces: list[str] = []
                    overlap_len = 0
                    for prev in reversed(current_chunk):
                        if (
                            overlap_len + len(prev) + len(separator)
                            <= self.chunk_overlap
                        ):
                            overlap_pieces.insert(0, prev)
                            overlap_len += len(prev) + len(separator)
                        else:
                            break

                    current_chunk = overlap_pieces + [piece]
                    current_length = sum(len(p) for p in current_chunk) + (
                        len(separator) * max(0, len(current_chunk) - 1)
                    )
                else:
                    merged.append(piece)

        if current_chunk:
            chunk_str = separator.join(current_chunk).strip()
            if chunk_str:
                merged.append(chunk_str)

        return merged

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []

        for document in documents:
            text = document.content
            if not text:
                continue

            text_chunks = self._split_text(text, self.separators)
            for chunk_index, chunk_content in enumerate(text_chunks):
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

        return chunks
