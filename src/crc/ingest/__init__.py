"""Ingestion: any input -> a chunked ``Document``.

    from crc.ingest import ingest
    doc = ingest("paper.pdf")          # PDF, DOCX, MD, TXT path, or a raw string
    doc.doc_type                       # DocType.PAPER / ARTICLE / PARAGRAPH
    doc.classifiable_chunks()          # section-labelled chunks, refs stripped
"""
from __future__ import annotations

import re
from pathlib import Path

from .chunkers import (
    DEFAULT_CONFIG,
    ChunkConfig,
    chunk_article,
    chunk_document,
    chunk_paper,
    chunk_paragraph,
    sections_from_markdown,
)
from .doctype import DocTypeVerdict, detect
from .parsers import parse_docx, parse_markdown, parse_pdf, parse_text
from .schema import (
    NOISE_SECTIONS,
    SECTION_PRIOR,
    Chunk,
    Document,
    DocType,
    ParsedSection,
    Section,
    normalise_heading,
)

_SUFFIX_PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
}
_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text"}
SUPPORTED = sorted({*_SUFFIX_PARSERS, *_TEXT_SUFFIXES})

#: A one-line string ending in a file extension, with a path separator or no
#: spaces: "C:/papers/x.pdf", "paper.pdf". A sentence ending "in Node.js" has
#: spaces and no separator, so it stays text.
_PATH_LIKE = re.compile(r"\.[A-Za-z][A-Za-z0-9]{1,4}$")


class IngestError(ValueError):
    """The input cannot be turned into text to classify. The message says why."""


def _looks_like_path(s: str) -> bool:
    s = s.strip()
    return bool(_PATH_LIKE.search(s)) and ("/" in s or "\\" in s or " " not in s)


def ingest(source: str | Path, *, title: str | None = None,
           config: ChunkConfig = DEFAULT_CONFIG,
           force_type: DocType | None = None) -> Document:
    """Ingest a file path or a raw string and return a chunked ``Document``.

    A string that names an existing file is parsed as that file; anything else
    is treated as the document text itself -- except a string that is plainly a
    file path, which is an error when the file is missing rather than being
    classified as if the path were a paper.

    Raises ``FileNotFoundError`` for a missing file and ``IngestError`` for an
    unsupported file type or a document with no extractable text (a scanned PDF,
    an empty string), so no caller can present a classification of nothing.
    """
    doc: Document
    path: Path | None = None
    if isinstance(source, Path):
        path = source
    elif isinstance(source, str) and len(source) < 400 and "\n" not in source:
        candidate = Path(source.strip())
        try:
            if candidate.exists() and candidate.is_file():
                path = candidate
        except OSError:
            path = None
        if path is None and _looks_like_path(source):
            raise FileNotFoundError(f"No such file: {source.strip()}")

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        suffix = path.suffix.lower()
        if suffix in _SUFFIX_PARSERS:
            doc = _SUFFIX_PARSERS[suffix](path)
        elif suffix in _TEXT_SUFFIXES or suffix == "":
            raw = path.read_text(encoding="utf-8", errors="replace")
            doc = (parse_markdown(raw, source=str(path))
                   if suffix in {".md", ".markdown"}
                   else parse_text(raw, title=title, source=str(path)))
        else:
            raise IngestError(
                f"Unsupported file type '{suffix}'. Supported: {', '.join(SUPPORTED)}, "
                f"or paste the text.")
    else:
        doc = parse_text(str(source), title=title)

    if not doc.text.split():
        if path is not None and path.suffix.lower() == ".pdf":
            raise IngestError(
                "No extractable text in this PDF. It is probably a scanned image; "
                "run OCR on it first, or paste the abstract.")
        raise IngestError("The input contains no text to classify.")

    if force_type is not None:
        doc.doc_type = force_type
    return chunk_document(doc, config)


__all__ = [
    "ingest",
    "IngestError",
    "SUPPORTED",
    "Document",
    "DocType",
    "Section",
    "Chunk",
    "ParsedSection",
    "ChunkConfig",
    "DEFAULT_CONFIG",
    "NOISE_SECTIONS",
    "SECTION_PRIOR",
    "normalise_heading",
    "detect",
    "DocTypeVerdict",
    "chunk_document",
    "chunk_paper",
    "chunk_article",
    "chunk_paragraph",
    "sections_from_markdown",
    "parse_pdf",
    "parse_docx",
    "parse_markdown",
    "parse_text",
]
