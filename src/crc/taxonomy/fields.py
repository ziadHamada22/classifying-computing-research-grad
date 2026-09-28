"""Field taxonomy — the second level, conditioned on discipline.

Agent 2 predicts a field *within* the discipline Agent 1 chose, so there is one
label space per discipline and they have different sizes. The taxonomy is
derived from the ACM/IEEE Computing Curricula 2020 knowledge areas — the same
source as the discipline level in `disciplines.py` — so the two levels nest
properly rather than being two unrelated schemes.

Why not just use arXiv subcategories as fields
----------------------------------------------
Measured on the 513,663-paper modern computing pool, arXiv categories cannot
supply this level:

  * **Software Engineering would have one field.** All 12,227 SE papers are
    `cs.SE` (11,790) or `cs.MS` (437). SE is precisely where practitioners care
    most about field structure — requirements vs testing vs maintenance — and
    arXiv offers nothing.
  * **Computer Science would be a vision detector.** `cs.CV` is 47% and `cs.CL`
    24% of CS, so 71% of the class is two categories.
  * **Categories are not fields.** `cs.AI` is a grab-bag; `cs.CV` spans several
    genuine fields.

So each field instead carries *two* independent kinds of evidence, and the
labeller combines them:

  ``categories``  arXiv categories that strongly imply this field. High
                  precision where it applies, absent for SE.
  ``keywords``    phrases that appear in the title/abstract of papers in this
                  field. Carries the disciplines that categories cannot split.

Keeping both on the field object means the labelling rule is auditable — every
weak label can be traced to the evidence that produced it — and the same
descriptions feed the local-LLM prompt later.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

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
class Field:
    """One field within a discipline."""

    name: str
    discipline: str
    description: str
    #: arXiv categories that imply this field (may be empty — see SE).
    categories: tuple[str, ...] = ()
    #: Distinctive phrases, lowercased, matched against title + abstract.
    keywords: tuple[str, ...] = ()
    #: Fields most often confused with this one, for the report's error analysis.
    contrast: str = ""

    @property
    def key(self) -> str:
        """Stable identifier, e.g. 'SE/testing-and-verification'."""
        from .disciplines import DISCIPLINE_ABBR

        slug = self.name.lower().replace(" ", "-").replace(",", "")
        return f"{DISCIPLINE_ABBR[self.discipline]}/{slug}"


def _f(name, disc, desc, cats=(), kws=(), contrast=""):
    return Field(name, disc, desc, tuple(cats), tuple(kws), contrast)


# --------------------------------------------------------------------------
# Computer Science — method-first work: algorithms, theory, and the perception
# and language problems CC2020 places under Intelligent Systems.
# --------------------------------------------------------------------------
_CS = [
    _f("Algorithms and Complexity", COMPUTER_SCIENCE,
       "Design and analysis of algorithms, data structures, and the complexity "
       "of computational problems.",
       ["cs.DS", "cs.CC", "cs.CG", "cs.DM"],
       ["approximation algorithm", "np-hard", "np-complete", "time complexity",
        "lower bound", "data structure", "graph algorithm", "combinatorial",
        "polynomial time", "competitive ratio", "randomized algorithm"],
       "Against Theory of Computation: algorithms build and analyse procedures; "
       "theory characterises what is computable at all."),
    _f("Theory of Computation and Logic", COMPUTER_SCIENCE,
       "Formal languages, automata, computability, type theory, and logic "
       "applied to computation.",
       ["cs.LO", "cs.FL"],
       ["automata", "formal language", "type system", "lambda calculus",
        "model checking", "temporal logic", "decidability", "proof assistant",
        "semantics", "theorem proving", "satisfiability"],
       "Against Programming Languages: logic studies the formal system; PL "
       "applies it to language design and implementation."),
    _f("Artificial Intelligence and Agents", COMPUTER_SCIENCE,
       "Knowledge representation, search, planning, reasoning, and multi-agent "
       "systems.",
       ["cs.AI", "cs.MA"],
       ["knowledge representation", "automated planning", "multi-agent",
        "constraint satisfaction", "heuristic search", "reasoning",
        "ontology", "expert system", "game playing", "belief revision"],
       "Against Machine Learning (Data Science): AI here is symbolic reasoning "
       "and search; learning from data is DS."),
    _f("Computer Vision", COMPUTER_SCIENCE,
       "Extracting information from images and video: recognition, detection, "
       "segmentation, reconstruction and generation.",
       ["cs.CV"],
       ["image classification", "object detection", "segmentation",
        "convolutional", "visual", "3d reconstruction", "pose estimation",
        "video understanding", "image generation", "optical flow", "diffusion model"],
       "Against Image and Video Processing (CE): vision infers scene content; "
       "CE processing operates on the signal itself."),
    _f("Natural Language Processing", COMPUTER_SCIENCE,
       "Computational processing of human language: understanding, generation, "
       "translation and dialogue.",
       ["cs.CL"],
       ["language model", "natural language", "machine translation",
        "question answering", "named entity", "sentiment analysis", "parsing",
        "text generation", "summarization", "dialogue", "tokeniz", "llm",
        "prompt", "word embedding"],
       "Against Information Retrieval (IS): NLP models language; IR finds "
       "documents that satisfy a need."),
    _f("Programming Languages and Compilers", COMPUTER_SCIENCE,
       "Language design, semantics, compilation, and program transformation.",
       ["cs.PL", "cs.SC"],
       ["compiler", "programming language", "static analysis", "type checking",
        "code generation", "interpreter", "abstract interpretation",
        "symbolic execution", "program synthesis", "domain-specific language"],
       "Against Construction and Tooling (SE): PL contributes language theory "
       "or a compiler technique; SE studies developers using such tools."),
    _f("Graphics, Multimedia and Sound", COMPUTER_SCIENCE,
       "Rendering, geometric modelling, animation, audio and multimedia "
       "systems.",
       ["cs.GR", "cs.MM", "cs.SD"],
       ["rendering", "ray tracing", "mesh", "animation", "texture",
        "geometry processing", "audio synthesis", "music", "multimedia",
        "shader", "point cloud"],
       "Against Computer Vision: graphics synthesises imagery; vision "
       "interprets it."),
]

# --------------------------------------------------------------------------
# Information Systems — information in an organisational and social context.
# --------------------------------------------------------------------------
_IS = [
    _f("Information Retrieval and Search", INFORMATION_SYSTEMS,
       "Finding relevant information: ranking, indexing, recommendation and "
       "retrieval-augmented systems.",
       ["cs.IR"],
       ["information retrieval", "search engine", "ranking", "relevance",
        "recommendation", "recommender", "query", "retrieval-augmented",
        "collaborative filtering", "index", "click", "vector search"],
       "Against Natural Language Processing (CS): IR satisfies an information "
       "need; NLP models the language itself."),
    _f("Data and Database Management", INFORMATION_SYSTEMS,
       "Database systems, query processing, data integration, and data "
       "governance.",
       ["cs.DB"],
       ["database", "query optimization", "sql", "transaction", "schema",
        "data integration", "data warehouse", "olap", "indexing structure",
        "knowledge graph", "data quality", "provenance"],
       "Against Distributed Computing (IT): DB concerns the data model and "
       "query semantics; IT concerns the infrastructure running it."),
    _f("Digital Libraries and Scholarly Information", INFORMATION_SYSTEMS,
       "Curation, metadata, bibliometrics and scholarly communication "
       "infrastructure.",
       ["cs.DL"],
       ["digital library", "bibliometric", "citation analysis", "metadata",
        "scholarly", "open access", "repository", "preservation",
        "scientometric", "peer review"],
       "Against Information Retrieval: digital libraries curate and describe "
       "collections; IR searches them."),
    _f("Human-Computer Interaction", INFORMATION_SYSTEMS,
       "Interface design, usability, user studies and interaction techniques.",
       ["cs.HC"],
       ["user study", "usability", "user interface", "participants",
        "interaction design", "user experience", "think-aloud", "questionnaire",
        "accessibility", "augmented reality", "virtual reality", "wearable"],
       "Against Empirical Software Engineering (SE): HCI studies end users of "
       "interfaces; empirical SE studies developers building software."),
    _f("Social and Information Networks", INFORMATION_SYSTEMS,
       "Structure and dynamics of social and information networks, and "
       "computational social science.",
       ["cs.SI"],
       ["social network", "network analysis", "community detection",
        "information diffusion", "misinformation", "social media", "twitter",
        "influence", "centrality", "graph mining"],
       "Against Data Science: network work here is about social structure; "
       "generic graph learning methods are DS."),
    _f("Computing, Ethics and Society", INFORMATION_SYSTEMS,
       "Societal, ethical, legal and policy dimensions of computing.",
       ["cs.CY"],
       ["ethics", "fairness", "policy", "regulation", "privacy concern",
        "digital divide", "governance", "societal impact", "accountability",
        "education", "curriculum", "law"],
       "Against Cybersecurity (IT): this is the policy and social reading; the "
       "technical mechanism is IT."),
]

# --------------------------------------------------------------------------
# Information Technology — running systems and infrastructure.
# --------------------------------------------------------------------------
_IT = [
    _f("Networking and Communications", INFORMATION_TECHNOLOGY,
       "Network architecture, protocols, wireless and mobile communication, "
       "and network measurement.",
       ["cs.NI"],
       ["network protocol", "wireless", "routing", "tcp", "throughput",
        "bandwidth", "5g", "6g", "sdn", "network measurement", "latency",
        "internet of things", "edge network", "congestion"],
       "Against Distributed Computing: networking moves packets; distributed "
       "computing coordinates computation across machines."),
    _f("Cybersecurity and Cryptography", INFORMATION_TECHNOLOGY,
       "Security mechanisms, threat analysis, privacy technology and applied "
       "cryptography.",
       ["cs.CR"],
       ["security", "attack", "adversary", "vulnerability", "malware",
        "intrusion detection", "encryption", "cryptographic", "authentication",
        "privacy-preserving", "differential privacy", "blockchain",
        "side-channel", "threat model"],
       "Against Computing and Society (IS): the mechanism and threat model are "
       "IT; the policy and ethical debate is IS."),
    _f("Distributed and Cloud Computing", INFORMATION_TECHNOLOGY,
       "Distributed systems, parallel and cluster computing, cloud and "
       "serverless infrastructure.",
       ["cs.DC"],
       ["distributed system", "cloud computing", "consensus", "fault tolerance",
        "microservice", "serverless", "kubernetes", "mapreduce", "replication",
        "load balancing", "parallel computing", "hpc", "federated"],
       "Against Algorithms (CS): a complexity result about a distributed "
       "protocol is CS; deploying and operating it is IT."),
    _f("Operating Systems and Platforms", INFORMATION_TECHNOLOGY,
       "Operating systems, virtualisation, storage systems and system "
       "administration.",
       ["cs.OS"],
       ["operating system", "kernel", "scheduler", "virtualization",
        "hypervisor", "file system", "container", "memory management",
        "system call", "device driver"],
       "Against Computer Architecture (CE): the OS manages the machine; CE "
       "builds it."),
    _f("Performance, Capacity and Reliability", INFORMATION_TECHNOLOGY,
       "Performance measurement, capacity planning, benchmarking and "
       "reliability engineering of deployed systems.",
       ["cs.PF"],
       ["performance evaluation", "benchmark", "throughput", "queueing",
        "capacity planning", "workload characterization", "profiling",
        "scalability", "service level", "reliability", "availability"],
       "Against Empirical Software Engineering: this measures running systems; "
       "empirical SE measures the process of building them."),
]

# --------------------------------------------------------------------------
# Software Engineering — arXiv gives ONE category here, so these fields are
# distinguished entirely by keyword evidence. This is the discipline the field
# level matters most for and the one arXiv categories cannot serve at all.
# --------------------------------------------------------------------------
_SE = [
    _f("Requirements and Specification", SOFTWARE_ENGINEERING,
       "Eliciting, modelling, specifying and validating what software must do.",
       [],
       ["requirement", "elicitation", "specification", "user story",
        "use case", "stakeholder", "traceability", "feature request",
        "acceptance criteria", "goal model"],
       "Against Architecture: requirements say what the system must do; "
       "architecture says how it is structured."),
    _f("Software Architecture and Design", SOFTWARE_ENGINEERING,
       "System structure, design patterns, modularity, and architectural "
       "decisions and their erosion.",
       [],
       ["software architecture", "design pattern", "architectural",
        "modularity", "coupling", "cohesion", "microservice architecture",
        "api design", "component", "reference architecture", "design decision"],
       "Against Construction: architecture is structural intent; construction "
       "is the code that realises it."),
    _f("Construction, Tooling and Program Analysis", SOFTWARE_ENGINEERING,
       "Writing code and the tools that support it: IDEs, code generation, "
       "static and dynamic analysis, program repair, code models.",
       # Deliberately NO category evidence. `cs.MS` (mathematical software) is
       # the only non-cs.SE category in this discipline, but its papers are
       # numerical libraries — "An AD Library for .NET", "Sparse LU
       # Factorization" — not software-engineering research. Giving it category
       # weight flooded this field with them, so SE is labelled by keywords
       # alone and cs.MS papers fall through to unlabelled, which is correct.
       [],
       ["code generation", "code completion", "program repair",
        "automated repair", "program synthesis", "static analysis", "linter",
        "code review", "code smell", "code clone", "bug detection",
        "code search", "developer tool", "build system", "compiler bug",
        "code model", "code llm", "copilot", "programming assistant",
        "code embedding", "source code representation", "memory safety",
        "dynamic analysis", "symbolic execution"],
       "Against Programming Languages (CS): PL contributes language theory; SE "
       "studies the tool in practitioner use."),
    _f("Testing, Verification and Validation", SOFTWARE_ENGINEERING,
       "Establishing that software behaves correctly: test generation, "
       "fuzzing, coverage, formal verification of implementations.",
       [],
       ["test case", "unit test", "test generation", "software testing",
        "testing", "fuzzing", "fuzzer", "fuzz test", "code coverage",
        "regression test", "mutation testing", "flaky test", "test amplif",
        "assertion", "bug report", "fault localization", "debugging",
        "verification", "validation method", "test suite", "test-driven",
        "defect prediction", "defect classifier", "oracle problem",
        "model checking of programs", "runtime monitoring"],
       "Against Theory of Computation (CS): verification here is applied to "
       "real implementations, not to formal systems as objects of study."),
    _f("Maintenance and Evolution", SOFTWARE_ENGINEERING,
       "Changing software after release: refactoring, technical debt, "
       "migration, program comprehension and legacy systems.",
       [],
       ["software maintenance", "maintainability", "software evolution",
        "technical debt", "legacy system", "refactoring", "refactor",
        "program comprehension", "code comprehension", "change impact",
        "co-change", "migration", "deprecat", "versioning",
        "semantic versioning", "commit history", "software aging",
        "reengineering", "modernization", "backward compatib", "regression"],
       "Against Construction: maintenance concerns changing existing systems; "
       "construction concerns building new code."),
    _f("Software Process and Management", SOFTWARE_ENGINEERING,
       "How software work is organised: agile and DevOps practice, CI/CD, "
       "effort estimation, quality management, open-source governance.",
       [],
       ["agile", "scrum", "devops", "continuous integration",
        "continuous delivery", "software process", "effort estimation",
        "project management", "release engineering", "pull request",
        "open source project", "software quality model", "maturity model"],
       "Against Empirical SE: process work proposes or evaluates a way of "
       "working; empirical SE studies people and artefacts scientifically."),
    _f("Empirical and Human Factors in SE", SOFTWARE_ENGINEERING,
       "Scientific study of software engineering: developer behaviour, "
       "mining software repositories, surveys, experiments and replication.",
       [],
       ["empirical study", "developers", "practitioners", "survey of",
        "mining software repositories", "case study", "grounded theory",
        "controlled experiment", "interview", "replication study",
        "questionnaire", "github projects", "productivity", "onboarding"],
       "Against Human-Computer Interaction (IS): the population studied is "
       "software developers, not end users of an interface."),
]

# --------------------------------------------------------------------------
# Computer Engineering — hardware and tightly coupled hardware-software.
# --------------------------------------------------------------------------
_CE = [
    _f("Computer Architecture and Hardware", COMPUTER_ENGINEERING,
       "Processor and accelerator architecture, memory hierarchy, and digital "
       "design.",
       ["cs.AR"],
       ["processor", "microarchitecture", "cache hierarchy", "gpu architecture",
        "accelerator", "fpga", "asic", "vlsi", "rtl", "instruction set",
        "hardware design", "systolic array", "dataflow architecture"],
       "Against Operating Systems (IT): CE designs the hardware; IT operates "
       "the platform built from it."),
    _f("Embedded and Emerging Devices", COMPUTER_ENGINEERING,
       "Embedded and real-time systems, and emerging computing substrates "
       "including neuromorphic and quantum hardware.",
       ["cs.ET"],
       ["embedded system", "real-time system", "microcontroller", "firmware",
        "low-power", "energy efficient hardware", "neuromorphic",
        "quantum computing hardware", "memristor", "sensor node", "rtos"],
       "Against Computer Architecture: emerging devices concern novel "
       "substrates; architecture concerns conventional organisation."),
    _f("Control Systems and Robotics", COMPUTER_ENGINEERING,
       "Control theory, autonomous systems, robot perception, planning and "
       "actuation.",
       ["eess.SY", "cs.SY", "cs.RO"],
       ["control system", "controller design", "model predictive control",
        "feedback control", "robot", "manipulator", "trajectory", "actuator",
        "autonomous vehicle", "state estimation", "kalman", "stability analysis",
        "reinforcement learning control", "motion planning"],
       "Against Artificial Intelligence (CS): control and embodiment are CE; "
       "abstract planning and reasoning are CS."),
    _f("Signal and Audio Processing", COMPUTER_ENGINEERING,
       "Processing of one-dimensional signals: communications, speech and "
       "audio at the signal level.",
       ["eess.SP", "eess.AS"],
       ["signal processing", "beamforming", "spectrum", "modulation",
        "channel estimation", "mimo", "speech recognition", "speech enhancement",
        "acoustic", "filter design", "fourier", "wavelet", "antenna"],
       "Against Natural Language Processing (CS): speech at the waveform level "
       "is CE; language semantics is CS."),
    _f("Image and Video Processing", COMPUTER_ENGINEERING,
       "Image and video at the signal level: acquisition, compression, "
       "restoration, and medical or remote-sensing imaging.",
       ["eess.IV"],
       ["image processing", "image compression", "denoising", "super-resolution",
        "image restoration", "medical imaging", "mri", "ct scan",
        "remote sensing", "hyperspectral", "video coding", "codec"],
       "Against Computer Vision (CS): CE processes and reconstructs the image; "
       "CS infers what is in the scene."),
    _f("Information and Coding Theory", COMPUTER_ENGINEERING,
       "Information theory, channel and source coding, and communication "
       "limits.",
       ["cs.IT"],
       ["information theory", "channel capacity", "coding theory", "ldpc",
        "polar code", "error correcting", "rate distortion", "entropy bound",
        "shannon", "compression bound", "mutual information"],
       "Against Algorithms (CS): coding theory concerns communication limits; "
       "algorithmic complexity concerns computation."),
]

# --------------------------------------------------------------------------
# Data Science — data-first, conclusion-oriented work.
# --------------------------------------------------------------------------
_DS = [
    _f("Machine Learning Methods", DATA_SCIENCE,
       "Learning algorithms and model families, their training and their "
       "empirical behaviour.",
       ["cs.LG"],
       ["neural network", "deep learning", "training", "gradient descent",
        "transformer", "self-supervised", "transfer learning", "fine-tuning",
        "representation learning", "generalization", "overfitting",
        "reinforcement learning", "graph neural network", "attention"],
       "Against Statistical Learning Theory: methods work is empirical and "
       "architectural; theory proves guarantees."),
    _f("Statistical Learning Theory", DATA_SCIENCE,
       "Theoretical foundations of learning: generalisation bounds, "
       "optimisation guarantees and statistical properties of estimators.",
       ["stat.ML"],
       ["generalization bound", "convergence rate", "sample complexity",
        "pac learning", "regret bound", "minimax", "consistency",
        "asymptotic normality", "concentration inequality", "excess risk"],
       "Against Machine Learning Methods: theory proves; methods measure."),
    _f("Statistical Methodology and Inference", DATA_SCIENCE,
       "Development of statistical models and inference procedures: "
       "experimental design, causal inference, Bayesian methods.",
       ["stat.ME"],
       ["statistical model", "bayesian", "posterior", "prior distribution",
        "hypothesis test", "confidence interval", "causal inference",
        "treatment effect", "propensity score", "mixed model", "regression model",
        "experimental design", "missing data", "survival analysis"],
       "Against Applied Statistics: methodology develops the procedure; applied "
       "work uses it to answer a substantive question."),
    _f("Applied Statistics and Analytics", DATA_SCIENCE,
       "Analysis answering a substantive question in a real domain, and the "
       "communication of those findings.",
       ["stat.AP"],
       ["we analyse data", "case study data", "empirical analysis",
        "cohort", "epidemiolog", "clinical", "economic", "forecasting",
        "time series analysis", "spatial analysis", "survey data", "real-world data"],
       "Against Statistical Methodology: the contribution is the substantive "
       "conclusion, not the procedure."),
    _f("Neural and Evolutionary Computation", DATA_SCIENCE,
       "Evolutionary algorithms, swarm and nature-inspired optimisation, and "
       "neural computation as an optimisation paradigm.",
       ["cs.NE"],
       ["genetic algorithm", "evolutionary algorithm", "swarm optimization",
        "particle swarm", "fitness function", "mutation operator",
        "neuroevolution", "metaheuristic", "simulated annealing",
        "multi-objective optimization", "spiking neural"],
       "Against Machine Learning Methods: this is population-based search and "
       "nature-inspired optimisation, not gradient learning."),
    _f("Computational Science and Engineering", DATA_SCIENCE,
       "Computation applied to science, engineering and finance: simulation, "
       "scientific ML and domain modelling.",
       ["cs.CE"],
       ["simulation", "finite element", "computational fluid", "molecular",
        "scientific computing", "physics-informed", "surrogate model",
        "computational biology", "financial modelling", "numerical simulation"],
       "Against Applied Statistics: CSE simulates a mechanistic model; applied "
       "statistics infers from observed data."),
    _f("Computational Statistics", DATA_SCIENCE,
       "Algorithms that make statistical inference computationally feasible: "
       "MCMC, variational inference and resampling.",
       ["stat.CO"],
       ["mcmc", "markov chain monte carlo", "gibbs sampling",
        "variational inference", "importance sampling", "bootstrap",
        "expectation maximization", "sequential monte carlo", "sampler",
        "approximate bayesian computation"],
       "Against Statistical Methodology: the contribution is the computational "
       "procedure, not the statistical model."),
]


FIELDS: list[Field] = [*_CS, *_IS, *_IT, *_SE, *_CE, *_DS]

#: Fields grouped by discipline, in canonical order. Agent 2's label space for a
#: paper is exactly ``FIELDS_BY_DISCIPLINE[predicted_discipline]``.
FIELDS_BY_DISCIPLINE: dict[str, list[Field]] = {
    d: [f for f in FIELDS if f.discipline == d] for d in DISCIPLINES
}

FIELD_NAMES_BY_DISCIPLINE: dict[str, list[str]] = {
    d: [f.name for f in fs] for d, fs in FIELDS_BY_DISCIPLINE.items()
}

BY_KEY: dict[str, Field] = {f.key: f for f in FIELDS}

# ---------------------------------------------------------------------------
# Global (flat) index over all 38 fields.
#
# Agent 2 is a single shared encoder with one 38-way head, masked at inference
# to the fields of the discipline Agent 1 predicted. That gives the conditioned
# behaviour — a CS paper is only ever scored against CS fields — while letting
# Software Engineering, which has just 3,226 labelled papers, benefit from the
# representation learned on the other ~108k. Six separate models would isolate
# SE with its own small corpus and cost 6x the VRAM, which the 6 GB budget
# cannot afford once Agents 1 and 3 are also resident.
# ---------------------------------------------------------------------------
GLOBAL_FIELDS: list[Field] = list(FIELDS)
GLOBAL_LABEL2ID: dict[str, int] = {f.name: i for i, f in enumerate(GLOBAL_FIELDS)}
GLOBAL_ID2LABEL: dict[int, str] = {i: f.name for i, f in enumerate(GLOBAL_FIELDS)}
N_GLOBAL_FIELDS: int = len(GLOBAL_FIELDS)

#: Global indices belonging to each discipline — the mask Agent 2 applies.
DISCIPLINE_FIELD_IDS: dict[str, list[int]] = {
    d: [GLOBAL_LABEL2ID[f.name] for f in FIELDS_BY_DISCIPLINE[d]]
    for d in DISCIPLINES
}


def discipline_mask(discipline: str):
    """Boolean mask over the 38 global fields, True for this discipline's."""
    import numpy as np

    m = np.zeros(N_GLOBAL_FIELDS, dtype=bool)
    m[DISCIPLINE_FIELD_IDS[discipline]] = True
    return m


def label2id(discipline: str) -> dict[str, int]:
    """Field-name -> index, within one discipline's label space."""
    return {f.name: i for i, f in enumerate(FIELDS_BY_DISCIPLINE[discipline])}


def id2label(discipline: str) -> dict[int, str]:
    return {i: f.name for i, f in enumerate(FIELDS_BY_DISCIPLINE[discipline])}


def n_fields(discipline: str) -> int:
    return len(FIELDS_BY_DISCIPLINE[discipline])


def scope_block(discipline: str) -> str:
    """Render one discipline's fields as prompt text for the local LLM."""
    parts = []
    for f in FIELDS_BY_DISCIPLINE[discipline]:
        parts.append(f"{f.name}\n  {f.description}"
                     + (f"\n  Boundary: {f.contrast}" if f.contrast else ""))
    return "\n\n".join(parts)


def summary() -> str:
    lines = [f"{len(FIELDS)} fields across {len(DISCIPLINES)} disciplines"]
    for d in DISCIPLINES:
        fs = FIELDS_BY_DISCIPLINE[d]
        n_cat = sum(1 for f in fs if f.categories)
        lines.append(f"  {d:24s} {len(fs)} fields "
                     f"({n_cat} with arXiv-category evidence)")
    return "\n".join(lines)


__all__ = [
    "Field",
    "FIELDS",
    "FIELDS_BY_DISCIPLINE",
    "FIELD_NAMES_BY_DISCIPLINE",
    "BY_KEY",
    "GLOBAL_FIELDS",
    "GLOBAL_LABEL2ID",
    "GLOBAL_ID2LABEL",
    "N_GLOBAL_FIELDS",
    "DISCIPLINE_FIELD_IDS",
    "discipline_mask",
    "label2id",
    "id2label",
    "n_fields",
    "scope_block",
    "summary",
]
