"""Map OpenAlex primary-topic *subfields* onto our six CC2020 disciplines.

OpenAlex is the largest open scholarly index (~250M works); its topic classifier
is a fine-tuned mBERT over a Scopus-ASJC-derived hierarchy
(domain -> field -> subfield -> topic). It is the strongest *general* scholarly
classifier to compare Agent 1 against — but it does NOT target our label space,
and measurement shows its taxonomy cuts **orthogonally** to the CC2020
practitioner split:

  * Only ~55% of a balanced computing sample lands in OpenAlex's top-level
    "Computer Science" field at all; the rest scatter to Engineering, Social
    Sciences, Decision Sciences, and the life sciences.
  * A single OpenAlex "Information Systems" subfield absorbs the majority of our
    Software Engineering papers as well as our Information Systems ones; its
    "Artificial Intelligence" subfield absorbs most of our Data Science papers.

So this map is deliberately **conservative and lossy**: a subfield is mapped only
when it has a defensible one-to-one CC2020 analogue. Subfields with no clean
analogue are left UNMAPPED and counted as coverage loss, never as a wrong answer
(the same philosophy as `wos_map.py`). The point of the baseline is not to make
OpenAlex look bad but to show, with real numbers, that no off-the-shelf system
produces the CC2020 discipline label — which is the gap the project fills.
"""
from __future__ import annotations

from crc.taxonomy.disciplines import (
    COMPUTER_ENGINEERING,
    COMPUTER_SCIENCE,
    DATA_SCIENCE,
    INFORMATION_SYSTEMS,
    INFORMATION_TECHNOLOGY,
    SOFTWARE_ENGINEERING,
)

#: OpenAlex subfield display name -> our discipline. Only subfields with a
#: defensible one-to-one analogue appear; everything else is intentionally absent
#: (treated as coverage loss). Rationale is given per line.
SUBFIELD_TO_DISCIPLINE: dict[str, str] = {
    # ---- Computer Science: method/theory-first contributions ----
    "Artificial Intelligence": COMPUTER_SCIENCE,
    "Computational Theory and Mathematics": COMPUTER_SCIENCE,
    "Theoretical Computer Science": COMPUTER_SCIENCE,
    "Computer Vision and Pattern Recognition": COMPUTER_SCIENCE,
    "Computer Graphics and Computer-Aided Design": COMPUTER_SCIENCE,
    # ---- Software Engineering ----
    "Software": SOFTWARE_ENGINEERING,
    # ---- Information Systems: information + organisation + people ----
    "Information Systems": INFORMATION_SYSTEMS,
    "Information Systems and Management": INFORMATION_SYSTEMS,
    "Human-Computer Interaction": INFORMATION_SYSTEMS,   # CC2020 IS: socio-technical
    "Library and Information Sciences": INFORMATION_SYSTEMS,
    # ---- Information Technology: infrastructure / operations ----
    "Computer Networks and Communications": INFORMATION_TECHNOLOGY,
    # ---- Computer Engineering: hardware / device / signal / control ----
    "Hardware and Architecture": COMPUTER_ENGINEERING,
    "Electrical and Electronic Engineering": COMPUTER_ENGINEERING,
    "Signal Processing": COMPUTER_ENGINEERING,
    "Control and Systems Engineering": COMPUTER_ENGINEERING,
    # ---- Data Science: statistics-first ----
    "Statistics and Probability": DATA_SCIENCE,
    "Statistics, Probability and Uncertainty": DATA_SCIENCE,
}

#: Subfields that are genuine judgement calls — plausibly two disciplines. Given
#: a defensible default here but reported separately so the headline never rests
#: on them (mirrors WOS_AMBIGUOUS_AREAS).
AMBIGUOUS_SUBFIELDS: set[str] = {
    "Management Science and Operations Research",  # IS (decision support) vs DS
    "Modeling and Simulation",                     # cross-cuts CS / CE / DS
}
SUBFIELD_TO_DISCIPLINE["Management Science and Operations Research"] = INFORMATION_SYSTEMS
SUBFIELD_TO_DISCIPLINE["Modeling and Simulation"] = COMPUTER_SCIENCE


def map_subfield(subfield: str | None) -> str | None:
    """OpenAlex subfield -> discipline, or None if out of scope / unknown.

    Robust to non-str inputs (NaN/None arrive for papers OpenAlex could not
    resolve), which are treated as unmapped.
    """
    if not isinstance(subfield, str):
        return None
    return SUBFIELD_TO_DISCIPLINE.get(subfield.strip())


def is_ambiguous(subfield: str | None) -> bool:
    if not isinstance(subfield, str):
        return False
    return subfield.strip() in AMBIGUOUS_SUBFIELDS


__all__ = [
    "SUBFIELD_TO_DISCIPLINE",
    "AMBIGUOUS_SUBFIELDS",
    "map_subfield",
    "is_ambiguous",
]
