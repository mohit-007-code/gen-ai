import pytest
from pydantic import ValidationError

from gen_ai.core.document import Document


def test_document_creation():

    doc = Document(
        content="Python is a programming language.",
        metadata={"source": "python.txt"},
    )

    assert doc.content == "Python is a programming language."
    assert doc.metadata["source"] == "python.txt"


def test_document_generates_unique_ids():

    doc1 = Document(content="First document")
    doc2 = Document(content="Second document")

    assert doc1.id != doc2.id


def test_document_rejects_empty_content():

    with pytest.raises(ValidationError):
        Document(content="")


def test_document_default_metadata():

    doc = Document(content="Hello world")

    assert doc.metadata == {}


def test_document_rejects_unknown_fields():

    with pytest.raises(ValidationError):
        Document(
            content="Hello",
            unknown_field="unexpected",
        )
