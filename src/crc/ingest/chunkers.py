"""Document-type-aware chunking.

The system must accept papers, articles and bare paragraphs, and those need
different treatment. Chunking a 200-word pasted abstract with a sliding window
produces one degenerate chunk plus noise; chunking a 30-page paper as a single
unit throws away the section structure that Agent 3 depends on. So the router
picks a strategy per document type:

    PAPER      section-aware windows that never straddle a section boundary,
               each carrying its canonical section label
    ARTICLE    paragraph-packed sliding windows with overlap, labelled BODY
               (or by any markdown headings the source happens to carry)
    PARAGRAPH  exactly one chunk, no splitting

All three emit the same ``Chunk`` type, so downstream aggregation is uniform.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Chunk, Document, DocType, ParsedSection, Section, normalise_heading


@dataclass(frozen=True)
class ChunkConfig:
    """Window geometry, in words (a word is ~1.3 subword tokens for BERT-family)."""

    target_words: int = 200
    overlap_words: int = 40
    min_words: int = 25
    #: A section shorter than this is emitted whole rather than windowed.
    max_single_section_words: int = 260
    #: Hard ceiling on chunks per document, to bound latency on long inputs.
    max_chunks: int = 64


DEFAULT_CONFIG = ChunkConfig()

_PARA_SPLIT = re.compile(r"\n\s*\n+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def _window(words: list[str], cfg: ChunkConfig) -> list[tuple[int, int]]:
    """Index spans for overlapping windows over a word list."""
    n = len(words)
    if n <= cfg.target_words:
        return [(0, n)]
    stride = max(1, cfg.target_words - cfg.overlap_words)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + cfg.target_words, n)
        spans.append((start, end))
        if end >= n:
            break
        start += stride
    # Fold a runt tail into its predecessor.
    if len(spans) >= 2 and (spans[-1][1] - spans[-1][0]) < cfg.min_words:
        prev = spans[-2]
        spans[-2] = (prev[0], spans[-1][1])
        spans.pop()
    return spans


def _emit(text: str, section: Section, heading_raw: str, cfg: ChunkConfig,
          out: list[Chunk]) -> None:
    """Split one section's text into windows and append them to ``out``."""
    text = text.strip()
    if not text:
        return
    words = text.split()
    if len(words) < cfg.min_words and out:
        # Too small to stand alone: append to the previous chunk of the same section.
        if out[-1].section == section:
            out[-1].text = f"{out[-1].text} {text}".strip()
            out[-1].n_words = len(out[-1].text.split())
            return
    for start, end in _window(words, cfg):
        piece = " ".join(words[start:end])
        out.append(Chunk(
            text=piece,
            section=section,
            index=len(out),
            n_words=end - start,
            section_raw=heading_raw,
        ))
        if len(out) >= cfg.max_chunks:
            return


def chunk_paper(sections: list[ParsedSection], title: str | None = None,
                cfg: ChunkConfig = DEFAULT_CONFIG) -> list[Chunk]:
    """Section-aware chunking: windows never cross a section boundary."""
    chunks: list[Chunk] = []
    if title:
        chunks.append(Chunk(text=title.strip(), section=Section.TITLE,
                            index=0, n_words=len(title.split()),
                            section_raw="title"))
    for sec in sections:
        if len(chunks) >= cfg.max_chunks:
            break
        body = sec.text.strip()
        if not body:
            continue
        # Short sections stay whole — windowing them only fragments the signal.
        if len(body.split()) <= cfg.max_single_section_words:
            chunks.append(Chunk(text=body, section=sec.section, index=len(chunks),
                                n_words=len(body.split()),
                                section_raw=sec.heading_raw))
            continue
        _emit(body, sec.section, sec.heading_raw, cfg, chunks)
    for i, c in enumerate(chunks):
        c.index = i
    return chunks[: cfg.max_chunks]


def chunk_article(text: str, title: str | None = None,
                  cfg: ChunkConfig = DEFAULT_CONFIG) -> list[Chunk]:
    """Paragraph-packed sliding windows for prose without formal sections.

    Paragraphs are packed greedily up to ``target_words`` so a window rarely cuts
    mid-idea; the overlap is re-seeded from the tail of the previous window so a
    concept spanning a paragraph break is still seen intact by some chunk.
    """
    chunks: list[Chunk] = []
    if title:
        chunks.append(Chunk(text=title.strip(), section=Section.TITLE, index=0,
                            n_words=len(title.split()), section_raw="title"))

    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    buf: list[str] = []
    buf_words = 0

    def flush() -> None:
        nonlocal buf, buf_words
        if not buf:
            return
        piece = "\n\n".join(buf).strip()
        if piece:
            chunks.append(Chunk(text=piece, section=Section.BODY,
                                index=len(chunks), n_words=len(piece.split()),
                                section_raw=""))
        # Re-seed with the tail of what we just emitted, for overlap.
        tail = piece.split()[-cfg.overlap_words:] if cfg.overlap_words else []
        buf = [" ".join(tail)] if tail else []
        buf_words = len(tail)

    for para in paragraphs:
        pw = len(para.split())
        if pw > cfg.target_words:
            flush()
            _emit(para, Section.BODY, "", cfg, chunks)
            buf, buf_words = [], 0
            if len(chunks) >= cfg.max_chunks:
                break
            continue
        if buf_words + pw > cfg.target_words and buf_words >= cfg.min_words:
            flush()
            if len(chunks) >= cfg.max_chunks:
                break
        buf.append(para)
        buf_words += pw
    if len(chunks) < cfg.max_chunks:
        flush()

    for i, c in enumerate(chunks):
        c.index = i
    return chunks[: cfg.max_chunks]


def chunk_paragraph(text: str, title: str | None = None,
                    cfg: ChunkConfig = DEFAULT_CONFIG) -> list[Chunk]:
    """Short input: one chunk, no splitting.

    Labelled ABSTRACT rather than BODY when it reads like one, because the
    section-weighted aggregator trusts abstracts more and a pasted abstract is
    the single most common interactive input.
    """
    body = text.strip()
    if title and not body.lower().startswith(title.strip().lower()[:40]):
        body = f"{title.strip()}. {body}"
    section = Section.ABSTRACT if _looks_like_abstract(body) else Section.BODY
    return [Chunk(text=body, section=section, index=0,
                  n_words=len(body.split()), section_raw="")]


_ABSTRACT_CUES = re.compile(
    r"\b(we (propose|present|introduce|show|study|investigate|develop)"
    r"|this (paper|work|article|study)|in this (paper|work|study)"
    r"|our (results|approach|method|contribution)"
    r"|the (results|findings) (show|indicate|suggest))\b",
    re.IGNORECASE,
)


def _looks_like_abstract(text: str) -> bool:
    return bool(_ABSTRACT_CUES.search(text))


def chunk_document(doc: Document, cfg: ChunkConfig = DEFAULT_CONFIG) -> Document:
    """Populate ``doc.chunks`` using the strategy for ``doc.doc_type``."""
    if doc.doc_type == DocType.PARAGRAPH:
        doc.chunks = chunk_paragraph(doc.text, doc.title, cfg)
    elif doc.doc_type == DocType.PAPER:
        sections = doc.sections
        if not sections:
            # Structured type but no headings recovered — degrade to article mode.
            doc.chunks = chunk_article(doc.text, doc.title, cfg)
        else:
            doc.chunks = chunk_paper(sections, doc.title, cfg)
    else:
        doc.chunks = chunk_article(doc.text, doc.title, cfg)
    return doc


def sections_from_markdown(md: str) -> tuple[str | None, list[ParsedSection]]:
    """Split markdown (or any '#'-headed text) into canonical sections.

    This is the training-side parser: the arxiver corpus stores LaTeX-derived
    markdown whose headings are already clean ('## 2 Methodology'). The
    inference-side PDF parser targets the same output shape.
    """
    lines = md.split("\n")
    title: str | None = None
    sections: list[ParsedSection] = []
    cur_heading = ""
    cur_section = Section.OTHER
    buf: list[str] = []

    def close() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(ParsedSection(cur_heading, cur_section, body))

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped.lstrip("#").strip()
            if title is None and level == 1 and heading:
                title = heading
                continue
            close()
            buf = []
            cur_heading = heading
            cur_section = normalise_heading(heading)
        else:
            buf.append(line)
    close()
    return title, sections


__all__ = [
    "ChunkConfig",
    "DEFAULT_CONFIG",
    "chunk_paper",
    "chunk_article",
    "chunk_paragraph",
    "chunk_document",
    "sections_from_markdown",
]
