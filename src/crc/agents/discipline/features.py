"""Canonical input formatting for Agent 1.

Training, evaluation and inference must format text identically or the model
sees a different distribution at serve time than it was fitted on. Every path
goes through ``format_chunk`` — there is no second place that builds model input.

The section name is prepended as plain text rather than as an added special
token so the same code works across SciBERT, DeBERTa-v3, ModernBERT and
SPECTER2 without vocabulary surgery. It matters because the model has to handle
a 6-word title chunk and a 200-word methods chunk in the same batch: the tag
tells it which it is looking at, and lets the calibrated confidence differ by
section (a methods paragraph genuinely carries less discipline signal than an
abstract).
"""
from __future__ import annotations

import pandas as pd

from crc.ingest.schema import Chunk, Section

#: Sections whose tag is dropped — for a bare pasted paragraph there is no
#: meaningful section and tagging it would invent structure that isn't there.
_UNTAGGED = {Section.BODY.value, Section.OTHER.value, ""}


def format_chunk(text: str, section: str | Section | None = None,
                 title: str | None = None) -> str:
    """Build the exact string the tokenizer sees."""
    if isinstance(section, Section):
        section = section.value
    text = (text or "").strip()
    parts: list[str] = []
    if section and section not in _UNTAGGED:
        parts.append(f"[{section}]")
    # A title gives context to an otherwise anonymous mid-document chunk.
    if title and section not in (Section.TITLE.value, None):
        t = title.strip()
        if t and not text.lower().startswith(t.lower()[:40]):
            parts.append(f"{t}.")
    parts.append(text)
    return " ".join(parts).strip()


def format_chunk_obj(chunk: Chunk, title: str | None = None) -> str:
    return format_chunk(chunk.text, chunk.section, title)


def format_abstract_row(title: str | None, abstract: str) -> str:
    """Abstract-level corpus rows are formatted as an ABSTRACT-tagged chunk."""
    return format_chunk(abstract, Section.ABSTRACT, title)


def build_text_column(df: pd.DataFrame) -> pd.Series:
    """Vectorised formatting for a corpus DataFrame.

    Handles both corpus shapes: the abstract-level corpus (``title`` +
    ``abstract``) and the chunk corpus (``text`` + ``section``).
    """
    if "section" in df.columns and "text" in df.columns:
        titles = df["title"] if "title" in df.columns else pd.Series(
            [None] * len(df), index=df.index)
        return pd.Series(
            [format_chunk(t, s, ti)
             for t, s, ti in zip(df["text"], df["section"], titles)],
            index=df.index,
        )
    if "abstract" in df.columns:
        titles = df["title"] if "title" in df.columns else pd.Series(
            [None] * len(df), index=df.index)
        return pd.Series(
            [format_abstract_row(ti, a) for ti, a in zip(titles, df["abstract"])],
            index=df.index,
        )
    raise ValueError(
        "DataFrame needs either (text, section) or (abstract) columns; "
        f"got {list(df.columns)}"
    )


__all__ = ["format_chunk", "format_chunk_obj", "format_abstract_row",
           "build_text_column"]
