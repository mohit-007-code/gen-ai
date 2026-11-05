from pathlib import Path

import pytest

from gen_ai.core.document import Document
from gen_ai.ingestion.loaders.base import BaseLoader
from gen_ai.ingestion.loaders.markdown import MarkdownLoader


def test_markdown_loader_implements_interface():
    loader = MarkdownLoader()
    assert isinstance(loader, BaseLoader)


def test_markdown_loader_loads_file(tmp_path: Path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Title\n\nParagraph text here.", encoding="utf-8")

    loader = MarkdownLoader()
    docs = loader.load(md_file)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].content == "# Title\n\nParagraph text here."
    assert docs[0].metadata["file_type"] == "markdown"
    assert docs[0].metadata["file_name"] == "test.md"


def test_markdown_loader_with_frontmatter(tmp_path: Path):
    content = """---
title: Introduction to RAG
author: DeepMind
---
# Main Content
This is the body.
"""
    md_file = tmp_path / "frontmatter.md"
    md_file.write_text(content, encoding="utf-8")

    loader = MarkdownLoader(parse_frontmatter=True)
    docs = loader.load(md_file)

    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Introduction to RAG"
    assert docs[0].metadata["author"] == "DeepMind"
    assert "# Main Content" in docs[0].content
    assert "---" not in docs[0].content


def test_markdown_loader_file_not_found():
    loader = MarkdownLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("non_existent_file.md")


def test_markdown_loader_directory_error(tmp_path: Path):
    loader = MarkdownLoader()
    with pytest.raises(IsADirectoryError):
        loader.load(tmp_path)


def test_markdown_loader_invalid_extension(tmp_path: Path):
    txt_file = tmp_path / "doc.txt"
    txt_file.write_text("text", encoding="utf-8")

    loader = MarkdownLoader()
    with pytest.raises(ValueError, match="Expected a Markdown file"):
        loader.load(txt_file)


def test_markdown_loader_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("   \n  ", encoding="utf-8")

    loader = MarkdownLoader()
    with pytest.raises(ValueError, match="Markdown file is empty"):
        loader.load(empty_file)
