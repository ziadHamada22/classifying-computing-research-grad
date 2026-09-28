"""Detect which chunking strategy an input needs.

Deliberately heuristic and cheap: this runs before any model, on every input,
and a wrong call degrades gracefully (a paper misread as an article still gets
classified, just without section weighting). The signals are structural rather
than semantic so the router never needs the GPU.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import (
    DocType,
    ParsedSection,
    Section,
    is_strict_heading_name,
    normalise_heading,
)

#: Below this, treat the input as a single paragraph. A typical arXiv abstract
#: is 130-250 words, so the threshold sits just above the abstract range.
PARAGRAPH_MAX_WORDS = 350
#: Above this, a document with no recognisable headings is still long-form.
ARTICLE_MIN_WORDS = 350
#: How many distinct canonical academic sections make something a "paper".
PAPER_MIN_SECTIONS = 3

_STRUCTURAL_SECTIONS = frozenset({
    Section.ABSTRACT, Section.INTRODUCTION, Section.BACKGROUND,
    Section.RELATED_WORK, Section.METHODS, Section.RESULTS,
    Section.DISCUSSION, Section.CONCLUSION, Section.REFERENCES,
})

# Numbered or all-caps headings on their own line, e.g. "3. Methodology",
# "II. RELATED WORK", "## Results", "Methods".
#
# PDF text extraction hard-wraps prose, so *every* wrapped line looks like a
# standalone line to a MULTILINE regex. An earlier permissive version of this
# pattern reported 107 "sections" in a 7.5k-word paper, nearly all of them
# ordinary sentences. `candidate_headings` therefore requires a second,
# independent signal (canonical vocabulary, explicit numbering, or all-caps)
# before accepting a match.
_HEADING_LINE = re.compile(
    r"^\s{0,6}(?:#{1,6}\s*)?"
    r"((?:(?:\d{1,2}(?:\.\d{1,2})*|[IVXLC]{1,6})[\.\)]?\s+)?"
    r"[A-Z][A-Za-z][^\n]{2,70})\s*$",
    re.MULTILINE,
)

#: A heading must start with explicit numbering to qualify on structure alone.
_NUMBERED = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2})*|[IVXLC]{1,6})[\.\)]\s+\S")
#: ALL-CAPS headings ("RELATED WORK") are unambiguous in extracted PDF text.
_ALLCAPS = re.compile(r"^[A-Z][A-Z0-9\s\-&,:]{3,60}$")
#: At least one alphabetic word, to reject bare numbering and footnote markers.
_HAS_WORD = re.compile(r"[A-Za-z]{3,}")
#: Float captions are typeset like headings (short, bold, often all-caps) but
#: are not document structure -- slicing at "TABLE III" splits a section in two.
_FLOAT_CAPTION = re.compile(
    r"^(?:table|figure|fig\.?|alg(?:orithm)?|listing|eq(?:uation)?|scheme|plate|"
    r"appendix\s+table|supplementary\s+(?:table|figure))\b\s*[ivxlcdm\d]*\s*[.:]?\s*$",
    re.IGNORECASE,
)

_CITATION = re.compile(r"\[\d{1,3}(?:\s*[,\-]\s*\d{1,3})*\]"
                       r"|\([A-Z][A-Za-z\-]+(?: et al\.?)?,?\s+\d{4}[a-z]?\)")
_REFERENCES_HEAD = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\d+\.?\s*)?"
                              r"(references|bibliography|works cited)\s*$",
                              re.IGNORECASE | re.MULTILINE)
_ABSTRACT_HEAD = re.compile(r"^\s*(?:#{1,6}\s*)?abstract\s*$",
                            re.IGNORECASE | re.MULTILINE)


@dataclass
class DocTypeVerdict:
    doc_type: DocType
    confidence: float
    reasons: list[str]
    found_sections: list[Section]


def candidate_headings(text: str, limit: int = 400) -> list[str]:
    """Lines that look like headings. Cheap pre-filter for section detection.

    Requires a positive signal beyond "short line starting with a capital",
    because in PDF-extracted text that describes most of the prose. A line
    qualifies only if it names a canonical section, carries explicit section
    numbering, or is set in all caps.
    """
    out: list[str] = []
    for m in _HEADING_LINE.finditer(text):
        head = m.group(1).strip()
        if len(head.split()) > 9 or head.endswith((".", ",", ";", ":")):
            continue
        # A bare "1" or "2.3" is an affiliation marker, footnote or page number,
        # never a section heading -- require at least one real word.
        if not _HAS_WORD.search(head) or _FLOAT_CAPTION.match(head):
            continue
        # Strict test: the line must *be* a section name. Using the permissive
        # `normalise_heading` here matched any prose line containing a content
        # word like "models" or "training", which sliced papers at nonsense
        # points such as "early stopping. We propose a mutual learning frame".
        if not (is_strict_heading_name(head)
                or _NUMBERED.match(head) or _ALLCAPS.match(head)):
            continue
        out.append(head)
        if len(out) >= limit:
            break
    return out


def detect(text: str, sections: list[ParsedSection] | None = None,
           source_hint: str | None = None) -> DocTypeVerdict:
    """Classify the input as PAPER, ARTICLE or PARAGRAPH."""
    reasons: list[str] = []
    words = len(text.split())

    # Prefer structure the parser already recovered; fall back to scanning text.
    if sections:
        found = {s.section for s in sections} & _STRUCTURAL_SECTIONS
        reasons.append(f"parser recovered {len(sections)} sections")
    else:
        heads = candidate_headings(text)
        found = {normalise_heading(h) for h in heads} & _STRUCTURAL_SECTIONS
        reasons.append(f"scanned {len(heads)} candidate headings")

    # Structure beats length. A short conference paper or an extended abstract
    # is still a paper, and collapsing it to one chunk would discard the section
    # labels Agent 3 depends on -- so this check has to precede the word count.
    if len(found) >= PAPER_MIN_SECTIONS:
        return DocTypeVerdict(
            DocType.PAPER, 0.9,
            reasons + [f"{len(found)} canonical sections despite {words} words: "
                       f"{sorted(s.value for s in found)}"],
            sorted(found, key=lambda s: s.value))

    if words <= PARAGRAPH_MAX_WORDS:
        return DocTypeVerdict(
            DocType.PARAGRAPH, 0.95,
            reasons + [f"only {words} words (<= {PARAGRAPH_MAX_WORDS}) "
                       f"and no section structure"], [])

    has_refs = bool(_REFERENCES_HEAD.search(text))
    has_abstract = bool(_ABSTRACT_HEAD.search(text))
    n_citations = len(_CITATION.findall(text))
    cite_density = n_citations / max(1, words / 1000)   # citations per 1k words

    score = 0.0
    if len(found) >= PAPER_MIN_SECTIONS:
        score += 0.5
        reasons.append(f"{len(found)} canonical sections: "
                       f"{sorted(s.value for s in found)}")
    if has_abstract and has_refs:
        score += 0.25
        reasons.append("has both an Abstract and a References heading")
    elif has_refs:
        score += 0.1
        reasons.append("has a References heading")
    if cite_density >= 5:
        score += 0.2
        reasons.append(f"citation density {cite_density:.1f}/1k words")
    if words > 3000:
        score += 0.05
        reasons.append(f"{words} words")
    if source_hint == "pdf":
        score += 0.05
        reasons.append("source is PDF")

    if score >= 0.5:
        return DocTypeVerdict(DocType.PAPER, min(0.99, 0.5 + score / 2),
                              reasons, sorted(found, key=lambda s: s.value))
    return DocTypeVerdict(DocType.ARTICLE, max(0.5, 1.0 - score),
                          reasons + [f"score {score:.2f} below paper threshold"],
                          sorted(found, key=lambda s: s.value))


__all__ = ["detect", "DocTypeVerdict", "candidate_headings",
           "PARAGRAPH_MAX_WORDS", "ARTICLE_MIN_WORDS", "PAPER_MIN_SECTIONS"]
