import pytest

from gen_ai.core.document import Document
from gen_ai.ingestion.chunkers.text import TextChunker


def test_split_single_document():
    document = Document(content="A" * 1000)

    chunker = TextChunker(chunk_size=500, chunk_overlap=50)

    chunks = chunker.split_documents([document])

    assert len(chunks) == 3
    assert len(chunks[0].content) == 500
    assert len(chunks[1].content) == 500
    assert len(chunks[2].content) == 100


def test_chunk_overlap():
    document = Document(content="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    chunker = TextChunker(chunk_size=10, chunk_overlap=2)

    chunks = chunker.split_documents([document])

    assert chunks[0].content == "ABCDEFGHIJ"
    assert chunks[1].content == "IJKLMNOPQR"
    assert chunks[0].content[-2:] == chunks[1].content[:2]


def test_preserve_metadata():
    document = Document(
        content="A" * 100,
        metadata={"source": "sample.pdf", "page": 1},
    )

    chunks = TextChunker(chunk_size=50).split_documents([document])

    assert chunks[0].metadata["source"] == "sample.pdf"
    assert chunks[0].metadata["page"] == 1


def test_source_document_id():
    document = Document(content="A" * 100)

    chunks = TextChunker(chunk_size=50).split_documents([document])

    assert all(chunk.metadata["source_document_id"] == document.id for chunk in chunks)


def test_chunk_indexes():
    document = Document(content="A" * 100)

    # Explicitly disable overlap to test chunk indexes.
    chunks = TextChunker(
        chunk_size=40,
        chunk_overlap=0,
    ).split_documents([document])

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [
        0,
        1,
        2,
    ]


def test_multiple_documents():
    documents = [
        Document(content="A" * 100),
        Document(content="B" * 100),
    ]

    # Explicitly disable overlap to test the expected chunk count.
    chunks = TextChunker(
        chunk_size=50,
        chunk_overlap=0,
    ).split_documents(documents)

    assert len(chunks) == 4


def test_empty_document_list():
    chunks = TextChunker().split_documents([])

    assert chunks == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_negative_overlap():
    with pytest.raises(ValueError):
        TextChunker(chunk_overlap=-1)


def test_empty_document_content():
    document = Document(content="   ")

    chunks = TextChunker().split_documents([document])

    assert len(chunks) == 1
    assert chunks[0].content == "   "
