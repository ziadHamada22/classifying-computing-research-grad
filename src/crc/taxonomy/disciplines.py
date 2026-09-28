"""The six computing disciplines, defined by ACM/IEEE Computing Curricula 2020.

These definitions are the single source of truth for the whole system. They are
consumed in three places:

1. `arxiv_map.py` — to justify every arXiv-category -> discipline assignment.
2. The local-LLM second opinion — the scope text below is injected verbatim into
   the prompt, so the LLM judges by the same rubric the labels were built from.
3. The report — the scope table is reproduced as the taxonomy definition.

CC2020 (Computing Curricula 2020, ACM/IEEE) recognises seven computing
disciplines: CE, CS, CY (Cybersecurity), IS, IT, SE and DS. The project brief
fixes a six-way split that omits Cybersecurity as a top-level discipline, so
security work is assigned to the discipline whose CC2020 competency profile it
sits closest to — Information Technology, which carries the "Security
Operations" and "System Administration" knowledge areas.
"""
from __future__ import annotations

from dataclasses import dataclass

COMPUTER_SCIENCE = "Computer Science"
INFORMATION_SYSTEMS = "Information Systems"
INFORMATION_TECHNOLOGY = "Information Technology"
SOFTWARE_ENGINEERING = "Software Engineering"
COMPUTER_ENGINEERING = "Computer Engineering"
DATA_SCIENCE = "Data Science"

# Canonical order. Every probability vector in the system is in THIS order.
DISCIPLINES: list[str] = [
    COMPUTER_SCIENCE,
    INFORMATION_SYSTEMS,
    INFORMATION_TECHNOLOGY,
    SOFTWARE_ENGINEERING,
    COMPUTER_ENGINEERING,
    DATA_SCIENCE,
]

DISCIPLINE_ABBR: dict[str, str] = {
    COMPUTER_SCIENCE: "CS",
    INFORMATION_SYSTEMS: "IS",
    INFORMATION_TECHNOLOGY: "IT",
    SOFTWARE_ENGINEERING: "SE",
    COMPUTER_ENGINEERING: "CE",
    DATA_SCIENCE: "DS",
}

LABEL2ID: dict[str, int] = {d: i for i, d in enumerate(DISCIPLINES)}
ID2LABEL: dict[int, str] = {i: d for d, i in LABEL2ID.items()}


@dataclass(frozen=True)
class DisciplineScope:
    """CC2020-grounded description of one discipline."""

    name: str
    abbr: str
    one_line: str
    focus: str
    knowledge_areas: tuple[str, ...]
    # What most often gets mistakenly assigned here, and the rule that resolves it.
    contrast: str


SCOPES: dict[str, DisciplineScope] = {
    COMPUTER_SCIENCE: DisciplineScope(
        name=COMPUTER_SCIENCE,
        abbr="CS",
        one_line=(
            "Discovering what can be computed and how efficiently — the theory, "
            "algorithms, languages and computational models underpinning computing."
        ),
        focus=(
            "Advances a computational method, proves a property of one, or studies "
            "the fundamental capabilities and limits of computation. The contribution "
            "is the algorithm, model or proof itself rather than its deployment."
        ),
        knowledge_areas=(
            "Algorithms and Complexity",
            "Theory of Computation",
            "Programming Languages and Compilers",
            "Artificial Intelligence and Intelligent Systems",
            "Computer Vision, Natural Language and Speech",
            "Graphics and Visual Computing",
            "Robotics and Autonomous Systems",
        ),
        contrast=(
            "Against Data Science: CS asks whether a method is correct, expressive or "
            "efficient; DS asks what a specific body of data reveals. A new "
            "architecture or learning algorithm evaluated on benchmarks is CS; a "
            "statistical analysis answering a substantive question about data is DS."
        ),
    ),
    INFORMATION_SYSTEMS: DisciplineScope(
        name=INFORMATION_SYSTEMS,
        abbr="IS",
        one_line=(
            "Applying computing to organisational and societal information needs — "
            "how information is modelled, stored, found, governed and used by people."
        ),
        focus=(
            "The unit of concern is information in an organisational or social "
            "context: data management, retrieval and curation, enterprise and "
            "business processes, digital libraries, and the socio-technical "
            "consequences of computing."
        ),
        knowledge_areas=(
            "Data and Information Management",
            "Information Retrieval and Digital Libraries",
            "Enterprise Systems and Business Intelligence",
            "IS Strategy, Management and Governance",
            "Computing, Ethics and Society",
            "Human-Centred and Socio-Technical Computing",
        ),
        contrast=(
            "Against Information Technology: IS is concerned with the information "
            "and the organisation around it; IT is concerned with the infrastructure "
            "that carries it. A study of how a firm adopts a database is IS; tuning "
            "that database server's throughput is IT."
        ),
    ),
    INFORMATION_TECHNOLOGY: DisciplineScope(
        name=INFORMATION_TECHNOLOGY,
        abbr="IT",
        one_line=(
            "Selecting, deploying, integrating, operating and securing computing "
            "infrastructure so that it works reliably for its users."
        ),
        focus=(
            "The contribution concerns running systems rather than inventing methods: "
            "networks, distributed and cloud infrastructure, operating systems, "
            "system and performance administration, and security operations."
        ),
        knowledge_areas=(
            "Networking and Communications",
            "Distributed, Parallel and Cloud Infrastructure",
            "Systems Administration and Operating Platforms",
            "Cybersecurity and Security Operations",
            "System Performance and Capacity",
            "Service and User Support",
        ),
        contrast=(
            "Against Computer Science: a complexity result about a distributed "
            "protocol is CS; measuring, deploying or hardening that protocol in a "
            "real network is IT. Under this project's six-way split IT also absorbs "
            "cybersecurity, which CC2020 treats as its own discipline."
        ),
    ),
    SOFTWARE_ENGINEERING: DisciplineScope(
        name=SOFTWARE_ENGINEERING,
        abbr="SE",
        one_line=(
            "Engineering large software systems systematically across their whole "
            "lifecycle, and studying the processes and people that build them."
        ),
        focus=(
            "Requirements, architecture and design, construction, verification, "
            "testing, maintenance and evolution, developer tooling, software "
            "process and quality, and empirical study of practitioners."
        ),
        knowledge_areas=(
            "Requirements Engineering",
            "Software Architecture and Design",
            "Software Construction and Tooling",
            "Verification, Validation and Testing",
            "Software Maintenance and Evolution",
            "Software Process, Quality and Management",
            "Empirical Software Engineering",
        ),
        contrast=(
            "Against Computer Science: a type system's metatheory is CS; an empirical "
            "study of how developers use that type system in production code is SE. "
            "Bug finding, program repair and developer studies are SE."
        ),
    ),
    COMPUTER_ENGINEERING: DisciplineScope(
        name=COMPUTER_ENGINEERING,
        abbr="CE",
        one_line=(
            "Designing and building computing hardware and the tightly coupled "
            "hardware-software systems that run on it."
        ),
        focus=(
            "Physical and embedded computing: processor and accelerator "
            "architecture, digital and analogue circuits, embedded and real-time "
            "systems, control systems, and signal and image processing at the "
            "device level."
        ),
        knowledge_areas=(
            "Computer Architecture and Organisation",
            "Digital Design, Circuits and VLSI",
            "Embedded and Real-Time Systems",
            "Systems and Control Engineering",
            "Signal and Image Processing",
            "Emerging Hardware and Devices",
        ),
        contrast=(
            "Against Information Technology: CE builds the device or the silicon; IT "
            "operates the estate assembled from devices. A cache-replacement policy "
            "realised in an architecture is CE; capacity-planning a server fleet is IT."
        ),
    ),
    DATA_SCIENCE: DisciplineScope(
        name=DATA_SCIENCE,
        abbr="DS",
        one_line=(
            "Extracting defensible knowledge from data through statistical "
            "modelling, inference, mining and analysis."
        ),
        focus=(
            "The contribution is about the data and what can validly be concluded "
            "from it: statistical methodology and inference, experimental design, "
            "data mining, applied analytics in a substantive domain, and the "
            "communication of results."
        ),
        knowledge_areas=(
            "Statistical Methodology and Inference",
            "Applied Statistics and Analytics",
            "Data Mining and Knowledge Discovery",
            "Computational Statistics",
            "Data Engineering for Analysis",
            "Data Visualisation and Communication",
        ),
        contrast=(
            "Against Computer Science: DS is data-first and conclusion-oriented; CS "
            "is method-first. A new statistical estimator or an applied analysis of a "
            "real dataset is DS; a new neural architecture benchmarked for accuracy "
            "is CS."
        ),
    ),
}


def scope_block(abbr_only: bool = False) -> str:
    """Render the six scopes as prompt text for the local-LLM second opinion."""
    parts = []
    for d in DISCIPLINES:
        s = SCOPES[d]
        head = s.abbr if abbr_only else f"{s.name} ({s.abbr})"
        parts.append(
            f"{head}\n"
            f"  Definition: {s.one_line}\n"
            f"  Counts as this when: {s.focus}\n"
            f"  Boundary rule: {s.contrast}"
        )
    return "\n\n".join(parts)


__all__ = [
    "DISCIPLINES",
    "DISCIPLINE_ABBR",
    "LABEL2ID",
    "ID2LABEL",
    "SCOPES",
    "DisciplineScope",
    "scope_block",
    "COMPUTER_SCIENCE",
    "INFORMATION_SYSTEMS",
    "INFORMATION_TECHNOLOGY",
    "SOFTWARE_ENGINEERING",
    "COMPUTER_ENGINEERING",
    "DATA_SCIENCE",
]
