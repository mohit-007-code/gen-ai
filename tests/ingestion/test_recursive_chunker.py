import pytest

from gen_ai.core.document import Document
from gen_ai.ingestion.chunkers.base import BaseChunker
from gen_ai.ingestion.chunkers.recursive import RecursiveCharacterChunker


def test_recursive_chunker_implements_interface():
    chunker = RecursiveCharacterChunker(chunk_size=100)
    assert isinstance(chunker, BaseChunker)


def test_recursive_chunker_paragraph_split():
    text = "Paragraph one is short.\n\nParagraph two has some more content.\n\nParagraph three is the final one."
    doc = Document(content=text, metadata={"topic": "general"})

    # Set chunk_size small enough so paragraphs don't all fit in one chunk,
    # but large enough so each paragraph fits individually
    chunker = RecursiveCharacterChunker(chunk_size=40, chunk_overlap=0)
    chunks = chunker.split_documents([doc])

    assert len(chunks) == 3
    assert chunks[0].content == "Paragraph one is short."
    assert chunks[1].content == "Paragraph two has some more content."
    assert chunks[2].content == "Paragraph three is the final one."
    assert chunks[0].metadata["source_document_id"] == doc.id
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1
    assert chunks[2].metadata["chunk_index"] == 2
    assert chunks[0].metadata["topic"] == "general"


def test_recursive_chunker_with_overlap():
    text = "Sentence A. Sentence B. Sentence C. Sentence D."
    doc = Document(content=text)

    chunker = RecursiveCharacterChunker(
        chunk_size=25, chunk_overlap=10, separators=[". "]
    )
    chunks = chunker.split_documents([doc])

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.content) <= 35


def test_recursive_chunker_validation():
    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        RecursiveCharacterChunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=-5)

    with pytest.raises(
        ValueError, match="chunk_overlap must be smaller than chunk_size"
    ):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)


def test_recursive_chunker_empty_documents():
    chunker = RecursiveCharacterChunker()
    assert chunker.split_documents([]) == []
