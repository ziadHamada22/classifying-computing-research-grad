"""Data model shared by ingestion, chunking and the three agents.

The canonical section vocabulary is the contract between *training* and
*inference*. The training chunker (over LaTeX-derived markdown) and the
inference chunker (over parsed PDFs) both emit these labels, so a chunk the
model sees at inference is drawn from the same distribution it was trained on.
Agent 3 additionally relies on the distinction because methodology is declared
in METHODS far more often than in the abstract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class DocType(str, Enum):
    """How the input should be chunked."""

    PAPER = "paper"          # formal academic structure -> section-aware chunking
    ARTICLE = "article"      # long-form prose, no formal sections -> sliding window
    PARAGRAPH = "paragraph"  # short text / abstract / single para -> one chunk


class Section(str, Enum):
    """Canonical section labels."""

    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    RELATED_WORK = "related_work"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    ACKNOWLEDGEMENTS = "acknowledgements"
    APPENDIX = "appendix"
    BODY = "body"            # unstructured prose (articles)
    OTHER = "other"


#: Sections that carry no discipline signal and are dropped before classification.
NOISE_SECTIONS: frozenset[Section] = frozenset(
    {Section.REFERENCES, Section.ACKNOWLEDGEMENTS}
)

#: Rough prior on how informative each section is for the *discipline* task.
#: Used as the initialisation for learned section weights in aggregation;
#: the fitted weights supersede these.
SECTION_PRIOR: dict[Section, float] = {
    Section.TITLE: 1.30,
    Section.ABSTRACT: 1.50,
    Section.INTRODUCTION: 1.20,
    Section.BACKGROUND: 0.90,
    Section.RELATED_WORK: 0.80,
    Section.METHODS: 1.00,
    Section.RESULTS: 0.80,
    Section.DISCUSSION: 0.85,
    Section.CONCLUSION: 1.10,
    Section.APPENDIX: 0.40,
    Section.BODY: 1.00,
    Section.OTHER: 0.60,
    Section.REFERENCES: 0.0,
    Section.ACKNOWLEDGEMENTS: 0.0,
}

# Ordered longest-intent-first: the first pattern that matches a heading wins.
# The vocabulary here was widened after measuring it against 3,000 real papers,
# where an earlier version left 55% of sections unmatched. The biggest gaps were
# plural forms ("Related Works"), and the fact that most method and result
# sections in practice are named after their *content* -- "Datasets",
# "Baselines", "Ablations" -- rather than with the canonical section word.
_HEADING_PATTERNS: list[tuple[re.Pattern[str], Section]] = [
    # Back matter first: "Author Contributions" must not be read as a method,
    # and "Data Availability" must not be read as a dataset section.
    (re.compile(r"\backnowledge?ment(s)?\b|\bfunding\b|\bcompeting\s+interests?\b"
                r"|\bconflicts?\s+of\s+interest\b|\bdeclarations?\b"
                r"|\bauthors?.{0,3}\s+contributions?\b|\bcontributions?\s+statement\b"
                r"|\b(data|code|software)\s+availability\b|\bethics\s+statement\b"
                r"|\bdisclosure\b|\bdeclaration\s+of\b|\bcredit\s+authorship\b"),
     Section.ACKNOWLEDGEMENTS),
    (re.compile(r"\breferences?\b|\bbibliography\b|\bworks\s+cited\b"), Section.REFERENCES),
    (re.compile(r"\bappendix\b|\bsupplement(ary|al)?\b|\bsupporting\s+information\b"
                r"|\bproofs?\b|\bproof\s+of\b|\bderivations?\b"), Section.APPENDIX),

    (re.compile(r"\brelated\s+works?\b|\brelated\s+(literature|research)\b"
                r"|\bprior\s+(works?|art)\b|\bliterature\s+review\b"
                r"|\bstate\s+of\s+the\s+art\b|\bexisting\s+(works?|approaches)\b"),
     Section.RELATED_WORK),
    (re.compile(r"\babstract\b|\bsummary\b(?!\s+of\s+contributions)"), Section.ABSTRACT),
    # Roadmap and contribution headings are introduction content in practice:
    # "Organization of the paper" and "Our contributions" are almost always the
    # closing paragraphs of section 1.
    (re.compile(r"\bintroduction\b|\bmotivation\b|^intro$"
                r"|\bour\s+contributions?\b|^contributions?$"
                r"|\borgani[sz]ation(\s+of\s+the\s+paper)?\b|^outline$"
                r"|\b(structure|plan|roadmap|organi[sz]ation)\s+of\s+(the\s+)?(paper|article|manuscript)\b"),
     Section.INTRODUCTION),
    (re.compile(r"\bbackground\b|\bpreliminar(y|ies)\b|\bnotations?\b"
                r"|\bdefinitions?\b|\bterminology\b|\brelated\s+concepts\b"
                r"|\bproblem\s+(statement|formulation|setting|definition)\b"),
     Section.BACKGROUND),
    (re.compile(r"\bmethod(s|ology|ologies)?\b|\bapproach(es)?\b|\bmodels?\b"
                r"|\barchitectures?\b|\bframeworks?\b|\balgorithms?\b|\bdesign\b"
                r"|\bimplementations?\b|\bsystems?\b|\bmaterials\s+and\s+methods\b"
                r"|\bproposed\b|\bstudy\s+design\b"
                # Named-by-content method sections, by frequency in real papers:
                r"|\bdata\s?sets?\b|\bcorpus\b|\bcorpora\b|\bdata\b"
                r"|\bbaselines?\b|\b(experimental\s+)?set(-|\s)?up\b|\bsettings?\b"
                r"|\btraining\b|\bfine[\s-]?tuning\b|\bpre[\s-]?training\b"
                r"|\binference\b|\bloss\s+functions?\b|\bobjectives?\b"
                r"|\bmetrics?\b|\bmeasures?\b|\bprocedures?\b|\bprotocols?\b"
                r"|\bparticipants?\b|\bapparatus\b|\binstruments?\b"
                r"|\bsampling\b|\bpre[\s-]?processing\b|\bannotations?\b"
                r"|\bconstruction\b|\bcollection\b|\bhyper[\s-]?parameters?\b"),
     Section.METHODS),
    (re.compile(r"\bresults?\b|\bexperiments?\b|\bevaluations?\b|\bfindings\b"
                r"|\banalys[ie]s\b|\bablations?\b|\bbenchmarks?\b|\bperformance\b"
                r"|\bcase\s+stud(y|ies)\b|\bcomparisons?\b|\bobservations?\b"
                r"|\bvalidation\b|\bexamples?\b|\bdemonstrations?\b"),
     Section.RESULTS),
    (re.compile(r"\bdiscussions?\b|\bimplications\b|\blimitations?\b"
                r"|\bthreats\s+to\s+validity\b|\binterpretations?\b"),
     Section.DISCUSSION),
    (re.compile(r"\bconclusions?\b|\bconcluding\b|\bfuture\s+work\b|\boutlook\b"
                r"|\bsummary\s+and\b|\bfinal\s+remarks\b"), Section.CONCLUSION),
]

# Strips markdown hashes/emphasis and leading section numbering from both ends.
# Emphasis matters: LaTeX-derived markdown writes italic headings as `_Dataset_`,
# and leaving the underscores attached defeated every pattern above.
_HEADING_CLEAN = re.compile(r"^[\s#*_]*(?:(?:\d+|[IVXLC]+|[A-Z])[\.\):]?\s*)*")
_HEADING_TRAIL = re.compile(r"[\s#*_]*$")


def normalise_heading(raw: str) -> Section:
    """Map a raw heading string onto the canonical vocabulary.

    Strips markdown hashes and leading numbering ("## 3.1 Proposed Method" ->
    "proposed method") before matching, so numbered and unnumbered papers
    normalise identically.
    """
    if not raw:
        return Section.OTHER
    text = _HEADING_TRAIL.sub("", raw.strip().lower())
    text = _HEADING_CLEAN.sub("", text)
    text = re.sub(r"[^a-z\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return Section.OTHER
    for pattern, section in _HEADING_PATTERNS:
        if pattern.search(text):
            return section
    return Section.OTHER


#: Whether a line *is* a section heading, as opposed to which section a known
#: heading belongs to. These must stay separate tests. `normalise_heading` is
#: deliberately permissive -- it matches content words like "models" or
#: "training" so that content-named sections classify correctly -- but that
#: permissiveness is disastrous as evidence of headingness, because ordinary
#: prose contains those words too. This pattern is anchored: the line must *be*
#: a section name, not merely contain one.
_STRICT_HEADING_NAME = re.compile(
    r"^(?:"
    r"abstract|summary|introduction|background|preliminaries|motivation"
    r"|related works?|related literature|prior works?|literature review"
    r"|state of the art"
    r"|methods?|methodology|methodologies|approach(?:es)?|materials and methods"
    r"|proposed (?:method|approach|model|framework|system)"
    r"|model|models|architecture|framework|system model|problem (?:statement|formulation|definition)"
    r"|experiments?|experimental (?:setup|settings?|results?|evaluation)"
    r"|evaluation|results?|results and discussion|findings|analysis"
    r"|ablation studies?|case stud(?:y|ies)|dataset|datasets|data|baselines?"
    r"|discussion|limitations|threats to validity|implications"
    r"|conclusions?|conclusions? and future work|future work|concluding remarks"
    r"|references|bibliography|works cited"
    r"|acknowledge?ments?|funding|declarations?|ethics statement"
    r"|appendix(?:\s+[a-z0-9]{1,3})?|supplementary(?:\s+\w+)?"
    r")$"
)


def is_strict_heading_name(raw: str) -> bool:
    """True when the line reads as a section heading in its own right."""
    if not raw:
        return False
    text = _HEADING_TRAIL.sub("", raw.strip().lower())
    text = _HEADING_CLEAN.sub("", text)
    text = re.sub(r"[^a-z\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return bool(text) and bool(_STRICT_HEADING_NAME.match(text))


@dataclass
class Chunk:
    """One classifiable unit of text."""

    text: str
    section: Section
    index: int
    n_words: int
    section_raw: str = ""
    char_start: int = 0
    char_end: int = 0

    @property
    def weight_prior(self) -> float:
        return SECTION_PRIOR.get(self.section, 1.0)


@dataclass
class ParsedSection:
    """A contiguous run of text under one heading."""

    heading_raw: str
    section: Section
    text: str


@dataclass
class Document:
    """An ingested document, ready for chunk-level classification."""

    text: str
    doc_type: DocType
    title: str | None = None
    abstract: str | None = None
    sections: list[ParsedSection] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    source: str = "text"
    parser: str = "none"
    meta: dict = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def classifiable_chunks(self) -> list[Chunk]:
        """Chunks excluding reference lists and acknowledgements."""
        keep = [c for c in self.chunks if c.section not in NOISE_SECTIONS]
        return keep or self.chunks

    def section_summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.section.value] = out.get(c.section.value, 0) + 1
        return out


__all__ = [
    "DocType",
    "Section",
    "Chunk",
    "ParsedSection",
    "Document",
    "NOISE_SECTIONS",
    "SECTION_PRIOR",
    "normalise_heading",
    "is_strict_heading_name",
]
