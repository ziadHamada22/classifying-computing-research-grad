"""Methodology taxonomy — the third level, predicted from the methods section.

Agent 3 answers *how* the research was done, not what it is about. The scheme
follows Pilkington & Pretorius (2015), "A Conceptual Model of the Research
Methodology Domain" (built for computing), and the strategy vocabulary of Oates,
*Researching Information Systems and Computing*. It has three orthogonal axes:

  * **research design**  — the strategy (experiment, design-science, case study,
    …). This is the axis with real textual signal and the one the *primary*
    classifier targets.
  * **research method**  — quantitative / qualitative / mixed / secondary. Mostly
    determined by the design; carried here as a prior.
  * **worldview**        — positivist / interpretivist / pragmatist / critical.
    Almost never stated; carried as a prior and refined only by the local-LLM
    hard-case reviewer.

Why the design axis is primary
------------------------------
Unlike discipline and field, methodology is not signalled by the arXiv category
or the abstract — it is declared in the **methods section** ("we conduct a
controlled experiment", "we implement and evaluate", "systematic literature
review"). So, exactly like Agent 2's field labels, each design carries a set of
``cues`` — distinctive methods-section phrases — used for weak supervision. The
worldview and method priors let a single design prediction populate all three
axes, while the LLM refines the genuinely ambiguous cases (worldview above all).

The same ``description``/``contrast`` text feeds the local-LLM prompt, so the
taxonomy stays the single source of truth across weak labelling and the reviewer.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Axis 3a — philosophical worldview (Creswell / Pilkington & Pretorius).
# Rarely stated; used as a prior and refined by the LLM reviewer.
# ---------------------------------------------------------------------------
WORLDVIEWS: dict[str, str] = {
    "positivist": "Objective reality measured empirically; hypotheses, controls, "
                  "quantification, generalisation.",
    "interpretivist": "Reality is socially constructed; understanding meaning in "
                      "context through qualitative inquiry.",
    "pragmatist": "What works in practice; problem-driven, often building and "
                  "evaluating an artefact (design science).",
    "critical": "Research to change a situation, emancipatory or participatory, "
                "acting on the studied setting.",
}

# ---------------------------------------------------------------------------
# Axis 3b — research method (data type). Mostly implied by the design.
# ---------------------------------------------------------------------------
METHODS: dict[str, str] = {
    "quantitative": "Numeric data and statistical or computational analysis.",
    "qualitative": "Non-numeric data (text, interviews, observation) and "
                   "interpretive analysis.",
    "mixed": "Both quantitative and qualitative data combined.",
    "secondary": "No new data collected; synthesis of existing studies or "
                 "purely analytical/theoretical work.",
}


@dataclass(frozen=True)
class ResearchDesign:
    """One research strategy — the primary label Agent 3 predicts."""

    name: str
    description: str
    #: Distinctive methods-section phrases, lowercased, for weak supervision.
    cues: tuple[str, ...]
    #: Boundary rule against the design it is most often confused with.
    contrast: str
    #: Default worldview and method this design usually implies (a prior; the
    #: LLM reviewer may override for a specific paper).
    worldview: str = "positivist"
    method: str = "quantitative"

    @property
    def key(self) -> str:
        return self.name.lower().replace(" ", "-").replace("/", "-")


def _d(name, desc, cues, contrast, worldview="positivist", method="quantitative"):
    return ResearchDesign(name, desc, tuple(cues), contrast, worldview, method)


# ---------------------------------------------------------------------------
# Axis 3c — research designs. Grounded in Oates' strategies + the CS-heavy
# formal and design-science modes that dominate computing.
# ---------------------------------------------------------------------------
DESIGNS: list[ResearchDesign] = [
    _d("Formal / Theoretical",
       "A claim established by mathematical proof or formal analysis rather than "
       "by observation: theorems, complexity results, formal models.",
       ["we prove", "theorem", "lemma", "corollary", "proof of", "we show that",
        "formal model", "complexity analysis", "np-hard", "upper bound",
        "lower bound", "formal proof", "we derive", "closed-form", "asymptotic",
        "formally define", "soundness", "completeness"],
       "Against Experiment: a theoretical paper proves its claim; an experimental "
       "one measures it on data.",
       worldview="positivist", method="secondary"),

    _d("Experiment / Empirical Evaluation",
       "A claim tested by controlled measurement: benchmarking, controlled "
       "experiments, quantitative empirical evaluation on data.",
       ["we conduct experiments", "experimental setup", "benchmark", "baselines",
        "we evaluate", "evaluation on", "test set", "metrics", "ablation",
        "statistically significant", "controlled experiment", "we measure",
        "empirical evaluation", "accuracy", "f1 score", "outperforms",
        "hyperparameters", "we compare against"],
       "Against Design & Creation: the contribution here is the measured finding; "
       "there it is the artefact that was built.",
       worldview="positivist", method="quantitative"),

    _d("Design & Creation (Design Science)",
       "The contribution is a novel built artefact — a system, algorithm, tool, "
       "framework or method — demonstrated and evaluated.",
       ["we propose", "we present", "we design", "we implement", "we develop",
        "our approach", "our system", "our method", "our framework",
        "our model", "prototype", "we build", "architecture of", "toolchain",
        "design science", "artefact", "we introduce a"],
       "Against Experiment: design & creation's contribution is the artefact; a "
       "pure experiment evaluates existing things without building a new one.",
       worldview="pragmatist", method="quantitative"),

    _d("Case Study",
       "In-depth empirical study of one or a few real-world instances in their "
       "natural context.",
       ["case study", "case studies", "we study the case", "in-depth study",
        "real-world deployment", "at a company", "industrial case", "single case",
        "multiple case", "embedded case", "field setting", "in practice at"],
       "Against Experiment: a case study observes a real instance without "
       "controlled manipulation of variables.",
       worldview="interpretivist", method="mixed"),

    _d("Survey (empirical study of respondents)",
       "Data gathered from a population of people via questionnaire or interview "
       "to characterise opinions, practices or attributes.",
       ["we surveyed", "questionnaire", "respondents",
        "participants completed", "we interviewed", "interviews with",
        "likert", "survey respondents", "practitioners were asked",
        "online survey", "response rate", "developers reported"],
       "Against Systematic Literature Review: this survey collects data from "
       "people; an SLR surveys the published literature.",
       worldview="positivist", method="mixed"),

    _d("Systematic Literature Review / Secondary Study",
       "Synthesis of existing published research following a defined search and "
       "selection protocol.",
       ["systematic literature review", "systematic mapping",
        "inclusion criteria", "exclusion criteria", "we searched",
        "search string", "primary studies", "prisma", "meta-analysis",
        "data extraction", "quality assessment", "selected studies",
        "literature search"],
       "Against Survey: an SLR's data are published papers, not responses from "
       "people.",
       worldview="positivist", method="secondary"),

    _d("Simulation & Modelling",
       "Study of a system through a computational or numerical model of it, "
       "rather than the real system or a proof.",
       ["simulation", "we simulate", "simulated environment", "numerical "
        "experiments", "monte carlo", "we model the", "numerical simulation",
        "simulator", "synthetic environment", "agent-based model",
        "discrete-event", "we simulate the behaviour"],
       "Against Experiment: a simulation studies a model of the system; an "
       "experiment measures the real system or artefact.",
       worldview="positivist", method="quantitative"),

    _d("Qualitative Field Study / Grounded Theory",
       "Interpretive study of behaviour or artefacts through qualitative data — "
       "observation, coding, thematic or grounded-theory analysis.",
       ["grounded theory", "thematic analysis", "we observed", "open coding",
        "axial coding", "ethnography", "ethnographic", "transcribed",
        "qualitative analysis", "we coded", "coding scheme", "saturation",
        "field notes", "semi-structured"],
       "Against Case Study: the emphasis here is the interpretive qualitative "
       "method itself, not the boundedness to one instance.",
       worldview="interpretivist", method="qualitative"),

    _d("Action Research",
       "The researcher intervenes in a real setting in iterative cycles, "
       "changing the situation while studying it.",
       ["action research", "we intervened", "iterative cycles", "action cycles",
        "in collaboration with practitioners", "participatory", "we introduced "
        "the intervention", "cycles of planning", "reflect and act"],
       "Against Case Study: action research deliberately changes the setting; a "
       "case study observes it without intervening.",
       worldview="critical", method="mixed"),
]

DESIGN_NAMES: list[str] = [d.name for d in DESIGNS]

#: The designs the primary encoder is trained on. The other five are genuine but
#: arXiv-starved (Survey/Case-Study/SLR/Qualitative/Action-Research live in IS/SE
#: venues that do not post to arXiv), too thin to learn from ~50-100 examples.
#: They stay in the taxonomy as low-support labels handled by the local-LLM
#: hard-case reviewer — the encoder abstains on them. (Measured 2026-07-28: the
#: four below are 97.8% of weak labels.)
TRAINABLE_DESIGNS: list[str] = [
    "Design & Creation (Design Science)",
    "Experiment / Empirical Evaluation",
    "Formal / Theoretical",
    "Simulation & Modelling",
]
TRAIN_LABEL2ID: dict[str, int] = {n: i for i, n in enumerate(TRAINABLE_DESIGNS)}
TRAIN_ID2LABEL: dict[int, str] = {i: n for i, n in enumerate(TRAINABLE_DESIGNS)}

BY_KEY: dict[str, ResearchDesign] = {d.key: d for d in DESIGNS}
BY_NAME: dict[str, ResearchDesign] = {d.name: d for d in DESIGNS}
LABEL2ID: dict[str, int] = {d.name: i for i, d in enumerate(DESIGNS)}
ID2LABEL: dict[int, str] = {i: d.name for i, d in enumerate(DESIGNS)}
N_DESIGNS: int = len(DESIGNS)


def worldview_of(design: str) -> str:
    d = BY_NAME.get(design)
    return d.worldview if d else "positivist"


def method_of(design: str) -> str:
    d = BY_NAME.get(design)
    return d.method if d else "quantitative"


def scope_block() -> str:
    """Render the designs as prompt text for the local-LLM hard-case reviewer."""
    parts = []
    for d in DESIGNS:
        parts.append(f"{d.name}\n  {d.description}\n  Boundary: {d.contrast}")
    return "\n\n".join(parts)


def summary() -> str:
    lines = [f"{len(DESIGNS)} research designs · {len(WORLDVIEWS)} worldviews · "
             f"{len(METHODS)} methods"]
    for d in DESIGNS:
        lines.append(f"  {d.name:44s} [{d.worldview}/{d.method}] "
                     f"{len(d.cues)} cues")
    return "\n".join(lines)


__all__ = [
    "WORLDVIEWS",
    "METHODS",
    "ResearchDesign",
    "DESIGNS",
    "DESIGN_NAMES",
    "BY_KEY",
    "BY_NAME",
    "LABEL2ID",
    "ID2LABEL",
    "N_DESIGNS",
    "worldview_of",
    "method_of",
    "scope_block",
    "summary",
]
