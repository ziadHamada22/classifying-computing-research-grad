"""arXiv subject class -> computing discipline map, version 2.

Why this file exists
--------------------
Version 1 of this project mapped each arXiv category to exactly one discipline
and labelled every paper by its *first listed* category. Measured on the v1
test set that rule was the dominant source of error, not the model:

  * 14.9% of test papers listed categories spanning two or more disciplines,
    and accuracy on them was 0.649 versus 0.881 on unambiguous papers.
  * 23.9% of all "errors" were cases where the predicted discipline WAS listed
    among the paper's own arXiv categories — the model was defensibly right and
    the single-label ground truth called it wrong.
  * Several individual mappings were simply indefensible. Their v1 error rates:
    cs.CY -> CS 100%, cs.AI -> CS 67%, cs.HC -> CS 60%, cs.NE -> IT 38%.

Version 2 changes three things:

  1. Every entry carries an explicit CC2020-grounded ``rationale`` and a
     ``strength`` in (0, 1] recording how canonical the assignment is. A
     contested category such as cs.AI gets a low strength and a ``secondary``
     discipline rather than being silently forced into one bucket.
  2. Labelling votes over *all* of a paper's categories instead of trusting the
     first one, with the author-chosen primary category weighted more heavily
     (see ``label_paper``).
  3. Papers whose category set does not produce a clear winner are flagged
     ``ambiguous`` so they can be held out of training rather than teaching the
     model contradictions.

Categories deliberately excluded (``EXCLUDED``) are ones with no defensible home
in a six-way computing split — general literature, "other", and pure
mathematical statistics.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .disciplines import (
    COMPUTER_ENGINEERING,
    COMPUTER_SCIENCE,
    DATA_SCIENCE,
    DISCIPLINES,
    INFORMATION_SYSTEMS,
    INFORMATION_TECHNOLOGY,
    SOFTWARE_ENGINEERING,
)


@dataclass(frozen=True)
class CategoryMapping:
    """One arXiv category's assignment, with the reasoning kept alongside it."""

    category: str
    discipline: str
    strength: float          # 1.0 = definitional, <=0.6 = genuinely contested
    rationale: str
    secondary: str | None = None
    changed_from_v1: str | None = None   # set when v2 moved this entry


def _m(cat, disc, strength, rationale, secondary=None, changed_from_v1=None):
    return CategoryMapping(cat, disc, strength, rationale, secondary, changed_from_v1)


CATEGORY_MAPPINGS: list[CategoryMapping] = [
    # ---------------- Computer Science ----------------
    _m("cs.CC", COMPUTER_SCIENCE, 1.0,
       "Computational complexity is the definitional core of CC2020's Theory of "
       "Computation knowledge area."),
    _m("cs.DS", COMPUTER_SCIENCE, 1.0,
       "Data structures and algorithms is the canonical Algorithms and Complexity "
       "knowledge area."),
    _m("cs.DM", COMPUTER_SCIENCE, 0.9,
       "Discrete mathematics underwrites algorithm analysis; combinatorics applied "
       "to computation sits in CS rather than in mathematics proper."),
    _m("cs.FL", COMPUTER_SCIENCE, 1.0,
       "Formal languages and automata are Theory of Computation."),
    _m("cs.LO", COMPUTER_SCIENCE, 0.9,
       "Logic in computer science — verification calculi, type theory, model "
       "checking foundations — is CS theory."),
    _m("cs.GT", COMPUTER_SCIENCE, 0.8,
       "Algorithmic game theory is studied in CS as a computational question "
       "(equilibrium computation, mechanism design complexity)."),
    _m("cs.CG", COMPUTER_SCIENCE, 1.0,
       "Computational geometry is an algorithms sub-area. Absent from the v1 map "
       "entirely.",
       changed_from_v1="absent -> CS"),
    _m("cs.SC", COMPUTER_SCIENCE, 0.9,
       "Symbolic computation and computer algebra are algorithmic method work. "
       "Absent from the v1 map.",
       changed_from_v1="absent -> CS"),
    _m("cs.PL", COMPUTER_SCIENCE, 0.75,
       "CC2020 places Programming Languages in CS. Compilers, type systems and "
       "semantics are method-first; the applied program-analysis end shades into SE.",
       secondary=SOFTWARE_ENGINEERING),
    _m("cs.AI", COMPUTER_SCIENCE, 0.55,
       "CC2020 assigns Intelligent Systems to CS, but a large share of modern "
       "arXiv cs.AI is applied machine learning that reads as Data Science. Kept in "
       "CS at low strength so the consensus vote can be overturned by a co-listed "
       "cs.LG or stat.ML. This category had a 67% error rate under v1.",
       secondary=DATA_SCIENCE),
    _m("cs.CL", COMPUTER_SCIENCE, 0.8,
       "Computational linguistics and NLP — a CS knowledge area covering language "
       "as a computational problem.",
       secondary=DATA_SCIENCE),
    _m("cs.CV", COMPUTER_SCIENCE, 0.8,
       "Computer vision is a CS knowledge area; the contribution is normally a "
       "perception method rather than an analysis of a dataset.",
       secondary=DATA_SCIENCE),
    _m("cs.SD", COMPUTER_SCIENCE, 0.7,
       "Sound and audio processing as a perception/method problem. Shares a border "
       "with CE signal processing.",
       secondary=COMPUTER_ENGINEERING),
    _m("cs.MM", COMPUTER_SCIENCE, 0.7,
       "Multimedia systems and coding; method-oriented, adjacent to graphics.",
       secondary=COMPUTER_ENGINEERING),
    _m("cs.GR", COMPUTER_SCIENCE, 0.95,
       "Graphics and Visual Computing is an explicit CS knowledge area."),
    _m("cs.MA", COMPUTER_SCIENCE, 0.8,
       "Multiagent systems is an Intelligent Systems sub-area. Absent from v1.",
       changed_from_v1="absent -> CS"),
    _m("cs.NA", COMPUTER_SCIENCE, 0.6,
       "Numerical analysis is algorithmic method work, with a strong pull toward "
       "computational statistics. Absent from v1.",
       secondary=DATA_SCIENCE, changed_from_v1="absent -> CS"),

    # ---------------- Information Systems ----------------
    _m("cs.DB", INFORMATION_SYSTEMS, 1.0,
       "Databases are the Data and Information Management knowledge area."),
    _m("cs.IR", INFORMATION_SYSTEMS, 1.0,
       "Information retrieval and search are definitional for IS."),
    _m("cs.DL", INFORMATION_SYSTEMS, 1.0,
       "Digital libraries — information curation and scholarly infrastructure."),
    _m("cs.CY", INFORMATION_SYSTEMS, 0.8,
       "Computers and Society is socio-technical and organisational, which is the "
       "IS competency profile, not CS theory. Under v1 this sat in CS and every "
       "single test paper in it was misclassified (7/7).",
       changed_from_v1="CS -> IS"),
    _m("cs.HC", INFORMATION_SYSTEMS, 0.65,
       "Human-computer interaction. CC2020 splits HCI across CS, IS and IT; arXiv "
       "cs.HC is dominated by user studies and socio-technical work, which is the "
       "IS reading. Had a 60% error rate as CS under v1.",
       secondary=COMPUTER_SCIENCE, changed_from_v1="CS -> IS"),
    _m("cs.SI", INFORMATION_SYSTEMS, 0.7,
       "Social and information networks — information structure in a social "
       "context. Absent from v1.",
       secondary=DATA_SCIENCE, changed_from_v1="absent -> IS"),

    # ---------------- Information Technology ----------------
    _m("cs.NI", INFORMATION_TECHNOLOGY, 1.0,
       "Networking and Communications is a definitional IT knowledge area."),
    _m("cs.CR", INFORMATION_TECHNOLOGY, 0.8,
       "Cryptography and security. CC2020 makes Cybersecurity its own discipline; "
       "under this project's six-way split IT carries Security Operations and is "
       "the closest competency profile."),
    _m("cs.DC", INFORMATION_TECHNOLOGY, 0.7,
       "Distributed, parallel and cluster computing as deployed infrastructure. "
       "Distributed-algorithm theory legitimately reads as CS, hence the reduced "
       "strength — v1 forced these into IT and the model kept answering CS.",
       secondary=COMPUTER_SCIENCE),
    _m("cs.OS", INFORMATION_TECHNOLOGY, 0.85,
       "Operating systems as the operating platform layer."),
    _m("cs.PF", INFORMATION_TECHNOLOGY, 0.8,
       "Performance measurement and capacity work is System Performance."),

    # ---------------- Software Engineering ----------------
    _m("cs.SE", SOFTWARE_ENGINEERING, 1.0,
       "Software engineering is a one-to-one match with the SE discipline."),
    _m("cs.MS", SOFTWARE_ENGINEERING, 0.7,
       "Mathematical software — library construction and numerical tooling is "
       "software construction. Absent from v1.",
       secondary=COMPUTER_SCIENCE, changed_from_v1="absent -> SE"),

    # ---------------- Computer Engineering ----------------
    _m("cs.AR", COMPUTER_ENGINEERING, 1.0,
       "Computer architecture is the definitional CE knowledge area."),
    _m("cs.ET", COMPUTER_ENGINEERING, 0.85,
       "Emerging technologies — novel devices and substrates — is hardware work."),
    _m("cs.SY", COMPUTER_ENGINEERING, 0.75,
       "Systems and control. Genuinely an electrical-engineering discipline that CE "
       "carries as Systems and Control Engineering. This was the single largest "
       "error source under v1 (20 errors over 91 test papers) because the corpus "
       "was 62% cs.SY control theory; v2 caps per-category share instead."),
    _m("cs.IT", COMPUTER_ENGINEERING, 0.6,
       "Information theory on arXiv is dominated by coding theory and "
       "communications, which is the CE signal/communications profile rather than "
       "CS theory. Absent from v1.",
       secondary=COMPUTER_SCIENCE, changed_from_v1="absent -> CE"),
    _m("cs.RO", COMPUTER_ENGINEERING, 0.6,
       "Robotics spans CS and CE in CC2020. arXiv cs.RO is heavily control, "
       "embodiment and hardware integration, so v2 reads it as CE; this also "
       "relieves CS of one more grab-bag member.",
       secondary=COMPUTER_SCIENCE, changed_from_v1="CS -> CE"),
    _m("eess.SY", COMPUTER_ENGINEERING, 0.75,
       "Alias of cs.SY under the eess archive."),
    _m("eess.SP", COMPUTER_ENGINEERING, 0.9,
       "Signal processing is a CE knowledge area."),
    _m("eess.IV", COMPUTER_ENGINEERING, 0.7,
       "Image and video processing at the signal level.",
       secondary=COMPUTER_SCIENCE),
    _m("eess.AS", COMPUTER_ENGINEERING, 0.7,
       "Audio and speech processing at the signal level. Absent from v1.",
       secondary=COMPUTER_SCIENCE, changed_from_v1="absent -> CE"),

    # ---------------- Data Science ----------------
    _m("cs.LG", DATA_SCIENCE, 0.7,
       "Machine learning. CC2020 shares ML between CS and DS; v2 anchors it in DS "
       "so that Data Science denotes modern data-driven modelling rather than "
       "classical statistics alone, which is what v1's corpus collapsed to.",
       secondary=COMPUTER_SCIENCE),
    _m("cs.NE", DATA_SCIENCE, 0.8,
       "Neural and evolutionary computing is learning and optimisation "
       "methodology. v1 placed it in Information Technology, apparently for a "
       "'networks' reading of the name; it errored 38% of the time and has no IT "
       "competency content at all.",
       secondary=COMPUTER_SCIENCE, changed_from_v1="IT -> DS"),
    _m("cs.CE", DATA_SCIENCE, 0.6,
       "Computational engineering, finance and science — applied analysis in a "
       "substantive domain. Absent from v1.",
       secondary=COMPUTER_ENGINEERING, changed_from_v1="absent -> DS"),
    _m("stat.ML", DATA_SCIENCE, 0.9,
       "Statistical machine learning."),
    _m("stat.ME", DATA_SCIENCE, 0.9,
       "Statistical methodology and inference."),
    _m("stat.AP", DATA_SCIENCE, 0.9,
       "Applied statistics — analysis answering a substantive question."),
    _m("stat.CO", DATA_SCIENCE, 0.9,
       "Computational statistics."),
]

# No defensible home in a six-way computing split.
EXCLUDED: dict[str, str] = {
    "cs.GL": "General literature — editorials and history, not a research discipline.",
    "cs.OH": "'Other' by construction; carries no discipline signal.",
    "stat.TH": "Mathematical statistics theory; belongs to mathematics, not computing.",
    "stat.OT": "'Other' statistics; no discipline signal.",
}

BY_CATEGORY: dict[str, CategoryMapping] = {m.category: m for m in CATEGORY_MAPPINGS}
ARXIV_CATEGORIES: list[str] = sorted(BY_CATEGORY)

# Weight given to the author-chosen primary (first-listed) category relative to
# the co-listed ones. Authors pick the primary deliberately, so it should count
# for more -- but not so much that v2 degenerates back into first-category-wins.
PRIMARY_WEIGHT = 2.0
SECONDARY_SHARE = 0.5     # a mapping's `secondary` gets this share of its vote
AMBIGUITY_MARGIN = 0.25   # winner must lead by this share of total vote


@dataclass
class PaperLabel:
    """Result of labelling one paper from its arXiv category list."""

    discipline: str | None
    margin: float                       # (top - runner_up) / total_vote
    ambiguous: bool
    votes: dict[str, float] = field(default_factory=dict)
    mapped_categories: list[str] = field(default_factory=list)
    co_listed: list[str] = field(default_factory=list)   # all disciplines with any vote

    @property
    def usable_for_training(self) -> bool:
        return self.discipline is not None and not self.ambiguous


def label_paper(categories: list[str] | tuple[str, ...]) -> PaperLabel:
    """Assign a discipline by weighted vote over *all* of a paper's categories.

    Each mapped category contributes ``strength`` to its discipline and, when a
    ``secondary`` is declared, ``(1 - strength) * SECONDARY_SHARE`` to that one.
    The first-listed (primary) category's contribution is multiplied by
    ``PRIMARY_WEIGHT``.
    """
    votes: dict[str, float] = defaultdict(float)
    mapped: list[str] = []

    for i, cat in enumerate(categories):
        m = BY_CATEGORY.get(cat)
        if m is None:
            continue
        mapped.append(cat)
        w = PRIMARY_WEIGHT if i == 0 else 1.0
        votes[m.discipline] += w * m.strength
        if m.secondary:
            votes[m.secondary] += w * (1.0 - m.strength) * SECONDARY_SHARE

    if not votes:
        return PaperLabel(None, 0.0, True, {}, [], [])

    total = sum(votes.values())
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = (top_score - runner_up) / total if total else 0.0

    return PaperLabel(
        discipline=top_label,
        margin=margin,
        ambiguous=margin < AMBIGUITY_MARGIN,
        votes=dict(votes),
        mapped_categories=mapped,
        co_listed=[d for d, _ in ranked],
    )


def v1_to_v2_changes() -> list[CategoryMapping]:
    """Entries v2 moved or added, for the report's taxonomy-revision table."""
    return [m for m in CATEGORY_MAPPINGS if m.changed_from_v1]


def coverage_summary() -> dict[str, list[str]]:
    """Categories grouped by assigned discipline."""
    out: dict[str, list[str]] = {d: [] for d in DISCIPLINES}
    for m in CATEGORY_MAPPINGS:
        out[m.discipline].append(m.category)
    return {d: sorted(v) for d, v in out.items()}


__all__ = [
    "CategoryMapping",
    "CATEGORY_MAPPINGS",
    "BY_CATEGORY",
    "ARXIV_CATEGORIES",
    "EXCLUDED",
    "PaperLabel",
    "label_paper",
    "v1_to_v2_changes",
    "coverage_summary",
    "PRIMARY_WEIGHT",
    "SECONDARY_SHARE",
    "AMBIGUITY_MARGIN",
]
