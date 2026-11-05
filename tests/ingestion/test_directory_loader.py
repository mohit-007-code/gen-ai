import pymupdf
import pytest

from gen_ai.ingestion.loaders.directory import DirectoryLoader


def create_pdf(path, content: str) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), content)
    pdf.save(path)
    pdf.close()


def test_load_txt_files(tmp_path):
    (tmp_path / "first.txt").write_text("First document")
    (tmp_path / "second.txt").write_text("Second document")

    documents = DirectoryLoader().load(tmp_path)

    assert len(documents) == 2


def test_load_pdf_files(tmp_path):
    create_pdf(tmp_path / "first.pdf", "First PDF")
    create_pdf(tmp_path / "second.pdf", "Second PDF")

    documents = DirectoryLoader().load(tmp_path)

    assert len(documents) == 2


def test_load_mixed_files(tmp_path):
    (tmp_path / "sample.txt").write_text("Text document")
    create_pdf(tmp_path / "sample.pdf", "PDF document")

    documents = DirectoryLoader().load(tmp_path)

    assert len(documents) == 2


def test_skip_unsupported_files(tmp_path):
    (tmp_path / "sample.txt").write_text("Supported")
    (tmp_path / "sample.docx").write_text("Unsupported")

    documents = DirectoryLoader().load(tmp_path)

    assert len(documents) == 1


def test_recursive_loading(tmp_path):
    subdirectory = tmp_path / "nested"
    subdirectory.mkdir()

    (tmp_path / "first.txt").write_text("Root document")
    (subdirectory / "second.txt").write_text("Nested document")

    documents = DirectoryLoader(recursive=True).load(tmp_path)

    assert len(documents) == 2


def test_non_recursive_loading(tmp_path):
    subdirectory = tmp_path / "nested"
    subdirectory.mkdir()

    (tmp_path / "first.txt").write_text("Root document")
    (subdirectory / "second.txt").write_text("Nested document")

    documents = DirectoryLoader(recursive=False).load(tmp_path)

    assert len(documents) == 1


def test_directory_not_found():
    with pytest.raises(FileNotFoundError):
        DirectoryLoader().load("missing_directory")


def test_file_instead_of_directory(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello")

    with pytest.raises(NotADirectoryError):
        DirectoryLoader().load(file_path)


def test_empty_directory(tmp_path):
    documents = DirectoryLoader().load(tmp_path)

    assert documents == []
