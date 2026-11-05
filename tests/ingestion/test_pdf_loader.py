import pymupdf
import pytest

from gen_ai.ingestion.loaders.pdf import PDFLoader


def create_pdf(path, pages: list[str]) -> None:
    pdf = pymupdf.open()

    for text in pages:
        page = pdf.new_page()
        page.insert_text((72, 72), text)

    pdf.save(path)
    pdf.close()


def test_load_single_page_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_pdf(pdf_path, ["Hello from PDF"])

    loader = PDFLoader()
    documents = loader.load(pdf_path)

    assert len(documents) == 1
    assert "Hello from PDF" in documents[0].content


def test_load_multiple_page_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    create_pdf(
        pdf_path,
        [
            "This is page one",
            "This is page two",
            "This is page three",
        ],
    )

    loader = PDFLoader()
    documents = loader.load(pdf_path)

    assert len(documents) == 3
    assert "page one" in documents[0].content
    assert "page two" in documents[1].content
    assert "page three" in documents[2].content


def test_pdf_page_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_pdf(pdf_path, ["Page one", "Page two"])

    documents = PDFLoader().load(pdf_path)

    assert documents[0].metadata["file_type"] == "pdf"
    assert documents[0].metadata["file_name"] == "sample.pdf"
    assert documents[0].metadata["page"] == 1
    assert documents[1].metadata["page"] == 2
    assert documents[0].metadata["source"] == str(pdf_path.resolve())


def test_pdf_file_not_found():
    loader = PDFLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("missing.pdf")


def test_directory_instead_of_pdf(tmp_path):
    loader = PDFLoader()

    with pytest.raises(IsADirectoryError):
        loader.load(tmp_path)


def test_invalid_file_extension(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is not a PDF")

    loader = PDFLoader()

    with pytest.raises(ValueError, match="Expected a PDF file"):
        loader.load(file_path)


def test_invalid_pdf_content(tmp_path):
    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_text("This is not a valid PDF")

    loader = PDFLoader()

    with pytest.raises(ValueError, match="Unable to read PDF"):
        loader.load(pdf_path)


def test_pdf_without_extractable_text(tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    create_pdf(pdf_path, [""])

    loader = PDFLoader()

    with pytest.raises(ValueError, match="No extractable text"):
        loader.load(pdf_path)
