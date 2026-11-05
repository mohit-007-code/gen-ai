"""Markdown document loader."""

from pathlib import Path
from typing import Any

from gen_ai.core.document import Document
from gen_ai.ingestion.loaders.base import BaseLoader


def _parse_simple_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML-style frontmatter without requiring PyYAML."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    frontmatter_lines: list[str] = []
    body_lines: list[str] = []
    closing_found = False

    for line in lines[1:]:
        if not closing_found:
            if line.strip() == "---":
                closing_found = True
            else:
                frontmatter_lines.append(line)
        else:
            body_lines.append(line)

    if not closing_found:
        return {}, text

    metadata: dict[str, Any] = {}
    for fm_line in frontmatter_lines:
        if ":" in fm_line:
            key, val = fm_line.split(":", 1)
            metadata[key.strip()] = val.strip().strip("\"'")

    body = "\n".join(body_lines).strip()
    return metadata, body


class MarkdownLoader(BaseLoader):
    """
    Load Markdown (.md, .markdown) files into Document objects.
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        parse_frontmatter: bool = True,
    ) -> None:
        self.encoding = encoding
        self.parse_frontmatter = parse_frontmatter

    def load(self, source: str | Path) -> list[Document]:
        file_path = Path(source)

        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        if not file_path.is_file():
            raise IsADirectoryError(
                f"Expected a file, received a directory: {file_path}"
            )

        if file_path.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError(
                f"Expected a Markdown file (.md, .markdown), received: {file_path.suffix}"
            )

        raw_content = file_path.read_text(encoding=self.encoding)
        if not raw_content.strip():
            raise ValueError(f"Markdown file is empty: {file_path}")

        metadata: dict[str, Any] = {
            "source": str(file_path.resolve()),
            "file_name": file_path.name,
            "file_type": "markdown",
        }

        if self.parse_frontmatter:
            fm_meta, body = _parse_simple_frontmatter(raw_content)
            metadata.update(fm_meta)
            content = body
        else:
            content = raw_content

        if not content.strip():
            raise ValueError(
                f"Markdown file content is empty after frontmatter: {file_path}"
            )

        document = Document(
            content=content,
            metadata=metadata,
        )

        return [document]
