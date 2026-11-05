from pathlib import Path

import pytest

from gen_ai.core.document import Document
from gen_ai.ingestion.loaders.text import TextLoader


def test_load_text_file(tmp_path):

    file_path = tmp_path / "python.txt"

    file_path.write_text(
        "Python is a programming language.",
        encoding="utf-8",
    )

    loader = TextLoader()
    documents = loader.load(file_path)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)

    assert documents[0].content == ("Python is a programming language.")


def test_load_text_file_with_string_path(tmp_path):

    file_path = tmp_path / "fastapi.txt"

    file_path.write_text(
        "FastAPI is a Python framework.",
        encoding="utf-8",
    )

    loader = TextLoader()
    documents = loader.load(str(file_path))

    assert documents[0].content == ("FastAPI is a Python framework.")


def test_document_metadata(tmp_path):

    file_path = tmp_path / "django.txt"

    file_path.write_text(
        "Django is a Python framework.",
        encoding="utf-8",
    )

    loader = TextLoader()
    documents = loader.load(file_path)

    metadata = documents[0].metadata

    assert metadata["file_name"] == "django.txt"
    assert metadata["file_type"] == "txt"
    assert metadata["source"] == str(file_path.resolve())


def test_file_not_found():

    loader = TextLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("missing.txt")


def test_directory_instead_of_file(tmp_path):

    loader = TextLoader()

    with pytest.raises(IsADirectoryError):
        loader.load(tmp_path)


def test_empty_file(tmp_path):

    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    loader = TextLoader()

    with pytest.raises(ValueError):
        loader.load(file_path)


def test_unicode_text(tmp_path):

    file_path = tmp_path / "unicode.txt"

    file_path.write_text(
        "Python is awesome. नमस्ते दुनिया!",
        encoding="utf-8",
    )

    loader = TextLoader()
    documents = loader.load(file_path)

    assert "नमस्ते दुनिया!" in documents[0].content


def test_loader_accepts_path_object(tmp_path):

    file_path = Path(tmp_path) / "example.txt"

    file_path.write_text(
        "Testing Path support.",
        encoding="utf-8",
    )

    loader = TextLoader()
    documents = loader.load(file_path)

    assert documents[0].metadata["file_name"] == "example.txt"
