"""Map WoS-46985 (domain, area) labels onto our disciplines and fields.

WoS is human-labelled but its taxonomy is *flat* and sits at roughly our
**discipline** granularity, not our field granularity (see
`docs/WOS_AUGMENTATION_SCOPE.md`). So the discipline map below is used for the
Agent 1 external-validation (Workstream A); the field map is the clean subset for
a later Agent 2 check (Workstream C) and is intentionally partial — a WoS area is
only mapped to a field when the two genuinely coincide.

Every mapping is a deliberate, auditable choice. Two CS areas are genuine
judgement calls and are flagged AMBIGUOUS; the evaluation reports results both
with and without them so the headline (especially Software Engineering) rests
only on unambiguous labels.
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

#: WoS area -> our discipline. ``None`` means deliberately out of scope (pure
#: electrical engineering that is not a computing discipline for us).
WOS_AREA_TO_DISCIPLINE: dict[str, str | None] = {
    # ---- CS domain (6,514 papers, 17 areas) ----
    "Software engineering": SOFTWARE_ENGINEERING,   # the confound-breaker
    "network security": INFORMATION_TECHNOLOGY,
    "Cryptography": INFORMATION_TECHNOLOGY,
    "Parallel computing": INFORMATION_TECHNOLOGY,
    "Distributed computing": INFORMATION_TECHNOLOGY,
    "Operating systems": INFORMATION_TECHNOLOGY,
    "Computer vision": COMPUTER_SCIENCE,
    "Computer graphics": COMPUTER_SCIENCE,
    "Algorithm design": COMPUTER_SCIENCE,
    "Data structures": COMPUTER_SCIENCE,
    "Machine learning": DATA_SCIENCE,
    "Relational databases": INFORMATION_SYSTEMS,
    "Structured Storage": INFORMATION_SYSTEMS,
    "Image processing": COMPUTER_ENGINEERING,       # signal-level, cf. CS vision
    "Bioinformatics": DATA_SCIENCE,
    # ---- ECE domain: only the computing-relevant areas map; the rest is pure
    #      electrical engineering and is dropped. ----
    "PID controller": COMPUTER_ENGINEERING,
    "Digital control": COMPUTER_ENGINEERING,
    "System identification": COMPUTER_ENGINEERING,
    "Control engineering": COMPUTER_ENGINEERING,
    "State space representation": COMPUTER_ENGINEERING,
    "Signal-flow graph": COMPUTER_ENGINEERING,
    "Analog signal processing": COMPUTER_ENGINEERING,
    "Microcontroller": COMPUTER_ENGINEERING,
    "Electricity": None,
    "Operational amplifier": None,
    "Electrical network": None,
    "Electrical circuits": None,
    "Electric motor": None,
    "Electrical generator": None,
    "Voltage law": None,
    "Lorentz force law": None,
    "Single-phase electric power": None,
    "Satellite radio": None,
}

#: CS areas whose discipline assignment is a genuine judgement call. Kept in the
#: map (with a defensible default) but excluded from the "clean" headline.
WOS_AMBIGUOUS_AREAS: set[str] = {
    "Computer programming",   # SE/Construction vs CS/Programming-Languages
    "Symbolic computation",   # CS/PL-Compilers vs CS/Algorithms
}
# Defaults for the ambiguous pair (used only in the "all" figures).
WOS_AREA_TO_DISCIPLINE["Computer programming"] = SOFTWARE_ENGINEERING
WOS_AREA_TO_DISCIPLINE["Symbolic computation"] = COMPUTER_SCIENCE

#: The clean subset where a WoS area coincides with exactly one of our 38 fields.
#: For Workstream C only; SE sub-fields are absent by construction (WoS has a
#: single "Software engineering" area and cannot sub-divide it).
WOS_AREA_TO_FIELD: dict[str, str] = {
    "Computer vision": "Computer Vision",
    "Computer graphics": "Graphics, Multimedia and Sound",
    "Algorithm design": "Algorithms and Complexity",
    "Data structures": "Algorithms and Complexity",
    "Machine learning": "Machine Learning Methods",
    "Relational databases": "Data and Database Management",
    "Structured Storage": "Data and Database Management",
    "network security": "Cybersecurity and Cryptography",
    "Cryptography": "Cybersecurity and Cryptography",
    "Parallel computing": "Distributed and Cloud Computing",
    "Distributed computing": "Distributed and Cloud Computing",
    "Operating systems": "Operating Systems and Platforms",
    "Image processing": "Image and Video Processing",
    "Analog signal processing": "Signal and Audio Processing",
    # NOTE: "Microcontroller" was deliberately REMOVED from this field map after
    # Workstream C diagnosis (2026-07-27). It is not a research field but a
    # cross-cutting hardware *platform*: WoS's microcontroller papers are
    # overwhelmingly control/instrumentation work (motor control, solar
    # trackers, protection schemes), which Agent 2 correctly and confidently
    # routes to Control Systems. Our "Embedded and Emerging Devices" field is
    # scoped (via CC2020 / arXiv cs.ET) to *emerging substrates* — neuromorphic,
    # molecular, programmable matter — and scores 0.83 on arXiv doing exactly
    # that. WoS has no emerging-substrate area, so it simply cannot validate
    # this field (it joins SE as WoS-unvalidatable). Mapping microcontroller ->
    # Embedded measured Agent 2 against the wrong target; the discipline map
    # still keeps it (-> CE), where Agent 1 places it correctly (0.85).
    "PID controller": "Control Systems and Robotics",
    "Digital control": "Control Systems and Robotics",
    "System identification": "Control Systems and Robotics",
    "Control engineering": "Control Systems and Robotics",
    "State space representation": "Control Systems and Robotics",
    "Signal-flow graph": "Control Systems and Robotics",
}

#: Domains we consider at all; the other five WoS domains are non-computing.
IN_SCOPE_DOMAINS = {"CS", "ECE"}


def map_area(area: str) -> str | None:
    """WoS area -> discipline, or None if out of scope / unknown."""
    return WOS_AREA_TO_DISCIPLINE.get((area or "").strip())


def is_ambiguous(area: str) -> bool:
    return (area or "").strip() in WOS_AMBIGUOUS_AREAS


__all__ = [
    "WOS_AREA_TO_DISCIPLINE",
    "WOS_AMBIGUOUS_AREAS",
    "WOS_AREA_TO_FIELD",
    "IN_SCOPE_DOMAINS",
    "map_area",
    "is_ambiguous",
]
