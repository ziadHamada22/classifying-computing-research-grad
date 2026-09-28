"""Methodological facets — observable properties, from which a design is derived.

Agent 3 originally forced every paper into exactly one of nine research designs.
Two measurements say that is the wrong shape for computing research:

* The Design-&-Creation vs Experiment confusion held at **~52% of all errors**
  through every architecture tried — because most computing papers genuinely
  *are* both: they build an artefact and then measure it. A single label makes
  that a coin flip.
* Two independent weak labellers (methods-section cues vs title self-declaration)
  agree only **43%** of the time, and their disagreements are systematic rather
  than random — "case study" in computing usually means *a worked example*, not a
  Yin-style case study.

So this module replaces the forced choice with **facets**: near-observable
yes/no properties of a paper. "Does this paper prove theorems?" is answerable by
anyone who can read the abstract; "is this design science or a case study?" needs
a methodologist. That difference is the point — facets can be cross-checked by
independent detectors, and their accuracy estimated, *without a hand-labelled gold
set*.

The nine designs are then **derived** from facets by the explicit rules in
:data:`DESIGN_RULES`, grounded in Oates and Pilkington & Pretorius. The derivation
is a transparent lookup, so a reader can audit why a design was assigned instead
of trusting a black box.

One boundary is honestly not derivable from these eight facets alone: Survey
(questionnaire) versus Qualitative Field Study both rest on ``human_data``, and
what separates them is whether the *analysis* is qualitative. Rather than invent a
ninth facet for it, :func:`derive_designs` returns a ranked list of compatible
designs, so an underdetermined case is visible as such.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Facet:
    """One observable methodological property."""

    key: str
    question: str
    #: Phrases whose presence is evidence FOR the facet. Matched case-insensitively
    #: as substrings, so they must be specific enough not to fire on prose.
    cues: tuple[str, ...]
    #: Regex matched against the title only. Title evidence is independent of the
    #: body, which is what lets the two act as separate labelling functions.
    title_pattern: str = ""
    #: Phrases that argue AGAINST the facet where they would otherwise fire.
    anti_cues: tuple[str, ...] = ()

    def title_re(self) -> re.Pattern | None:
        return re.compile(self.title_pattern, re.I) if self.title_pattern else None


FACETS: list[Facet] = [
    Facet(
        key="builds",
        question="Does the paper contribute a new artefact (system, algorithm, tool, framework)?",
        # Deliberately NOT "we propose" / "we present" / "we introduce" / "our
        # approach". Measured: with those included this facet fired on 65% of all
        # papers and was the only one where the trained model LOST to plain cues,
        # because every kind of paper "presents" something -- "we present an
        # empirical study of X" is an Experiment triggering a build cue. The
        # discriminating evidence for an artefact is language about *constructing,
        # implementing or releasing* it, not about proposing it.
        cues=("we implement", "we develop", "we build", "we construct",
              "we devise", "our implementation", "we release",
              "open source", "publicly available", "source code",
              "our system", "our tool", "our framework", "our prototype",
              "our architecture", "our algorithm", "we deploy",
              "prototype implementation", "novel architecture",
              "we designed a", "we design a"),
        title_pattern=r":\s*a (novel |new )?(framework|system|tool|library|platform|toolkit|prototype)\b",
    ),
    Facet(
        key="evaluates",
        question="Does it measure something empirically on data?",
        cues=("we evaluate", "we conduct experiments", "experimental results",
              "experimental setup", "benchmark", "baselines", "evaluation on",
              "we compare", "outperforms", "accuracy of", "we measure",
              "empirical evaluation", "ablation", "test set", "dataset",
              "f1 score", "we report results"),
        title_pattern=r"empirical (study|evaluation|analysis)|benchmark|evaluation of|comparative (study|analysis)",
    ),
    Facet(
        key="proves",
        question="Does it establish claims by mathematical proof or formal analysis?",
        cues=("we prove", "theorem", "lemma", "corollary", "proof of",
              "we show that", "np-hard", "np-complete", "complexity bound",
              "lower bound", "upper bound", "formally", "we derive",
              "proposition", "qed", "asymptotically"),
        title_pattern=r"\b(bounds?|complexity|theorem|proof|optimality|convergence)\b",
    ),
    Facet(
        key="simulates",
        question="Does it study a computational model of a system rather than the system itself?",
        cues=("simulation", "we simulate", "simulated environment",
              "numerical experiments", "monte carlo", "we model the",
              "simulation results", "simulator", "synthetic environment",
              "agent-based model", "discrete-event"),
        title_pattern=r"simulation (study|model)|monte carlo|agent-based (model|simulation)",
        # A "simulated annealing" optimiser is not a simulation study.
        anti_cues=("simulated annealing",),
    ),
    Facet(
        key="human_data",
        question="Does it collect data from people (questionnaire, interview, observation)?",
        cues=("questionnaire", "respondents", "we surveyed", "we interviewed",
              "interviews with", "participants completed", "participants were",
              "study participants", "focus group", "think-aloud",
              "user study", "human subjects", "irb", "informed consent",
              "likert", "we recruited"),
        title_pattern=r"(questionnaire|interview study|user study|practitioner survey|survey of (developers|practitioners|students|users|engineers))",
    ),
    Facet(
        key="field_context",
        question="Is the study situated in a real organisation or deployment?",
        cues=("case study", "in industry", "at a company", "industrial setting",
              "real-world deployment", "in production", "practitioners at",
              "organisation", "organization", "field study",
              "deployed in", "our industrial partner"),
        title_pattern=r"\bcase stud(y|ies)\b|industrial|in practice|field study",
    ),
    Facet(
        key="secondary",
        question="Is the data other published papers rather than new observation?",
        cues=("systematic literature review", "systematic review",
              "systematic mapping", "inclusion criteria", "exclusion criteria",
              "search string", "we searched", "primary studies",
              "literature review", "meta-analysis", "papers were screened",
              "snowballing", "prisma", "bibliometric"),
        title_pattern=r"systematic (literature )?review|systematic mapping|mapping study|meta-analysis|scoping review|:\s*a survey|^(a )?survey of",
    ),
    Facet(
        key="intervenes",
        question="Does the researcher deliberately change the setting while studying it?",
        cues=("action research", "we intervened", "iterative cycles",
              "action cycles", "in collaboration with practitioners",
              "participatory", "we introduced the process",
              "intervention", "we worked with the team"),
        title_pattern=r"action research|participatory",
    ),
]

FACET_KEYS: list[str] = [f.key for f in FACETS]
BY_KEY: dict[str, Facet] = {f.key: f for f in FACETS}

# --------------------------------------------------------------------------
# Vocabulary v2 (2026-09-26): phrasings the substring cues above cannot express.
#
# 18.4% of papers had no facet at all, and the rules then returned Design &
# Creation by default. Reading those abstracts showed three facets whose cue lists
# missed the commonest way computing papers state them:
#   builds    -- a *named* artefact ("We introduce Wasm-Mutate, a ... engine"),
#                "design and implementation";
#   evaluates -- "experiments on/show ...", "empirical results", "evaluated on",
#                and quantified gains ("improves accuracy by 12%");
#   proves    -- "regret bound", "sample complexity", "provably", big-O bounds.
# A broader builds rule ("we propose a <noun> method/model/...") was measured and
# REJECTED: builds prevalence went 20% -> 43%, Design & Creation 37% -> 55%, and
# fewer theory-venue papers came out Formal. Every addition below was judged on the
# label model's estimated accuracy, region-to-region kappa, and arXiv comment /
# journal-ref signals the labellers never read (`docs/AGENT3_COMPLETION_RESULTS.md`).
# `vote_region(..., extended=False)` reproduces the v1 vocabulary exactly, so the
# old yardstick stays available.
# --------------------------------------------------------------------------
VOCABULARY_VERSION = 2

_NAMED_ARTEFACT = re.compile(
    r"\b(?:[Ww]e|[Tt]his (?:paper|work)) (?:propose|present|introduce|develop)s? "
    r"[A-Z][\w-]*[A-Z0-9][\w-]*\s*[,(:]")
_IMPLEMENTATION = ("design and implementation", "designed and implemented",
                   "we implemented", "is implemented in", "is implemented on",
                   "is implemented as")
_EXPERIMENTS = re.compile(
    r"\bexperiments? (?:on|with|show|demonstrate|reveal|indicate|confirm|conducted)"
    r"|extensive experiments|experimental (?:evaluation|study|analysis|validation|comparison)"
    r"|\bempirical(?:ly)? (?:results|evaluation|study|analysis|validat\w+|evidence)"
    r"|evaluated (?:on|using|against|with)"
    r"|state[- ]of[- ]the[- ]art (?:performance|results|accuracy|methods|approaches|baselines)",
    re.I)
_QUANTITY = re.compile(r"\d+(?:\.\d+)?\s?(?:%|x\b|×|times\b|percentage points)")
_GAIN = re.compile(r"improv|reduc|increas|speed|gain|outperform|lower|boost", re.I)
_PROOF = re.compile(
    r"regret bounds?|sample complexity|convergence (?:rate|guarantee)s?|provabl[ey]"
    r"|theoretical guarantees?|we establish|approximation (?:ratio|factor|algorithm|guarantee)"
    r"|hardness|decidab|polynomial[- ]time", re.I)
_BIG_O = re.compile(r"(?<![A-Za-z])[OΘΩ]\((?:n|d|k|t|m|T|N|\\|log|1|sqrt|\d)")


def _builds_v2(text: str, low: str) -> bool:
    return bool(_NAMED_ARTEFACT.search(text)) or any(c in low for c in _IMPLEMENTATION)


def _evaluates_v2(text: str, low: str) -> bool:
    if _EXPERIMENTS.search(text):
        return True
    # a quantified gain: the number, with a gain word shortly before it
    return any(_GAIN.search(text, max(0, m.start() - 70), m.start())
               for m in _QUANTITY.finditer(text))


def _proves_v2(text: str, low: str) -> bool:
    return bool(_PROOF.search(text) or _BIG_O.search(text))


_EXTENDED = {"builds": _builds_v2, "evaluates": _evaluates_v2, "proves": _proves_v2}


def extended_evidence(key: str, text: str, low: str | None = None) -> bool:
    """Vocabulary-v2 evidence for a facet, on the original-case text."""
    fn = _EXTENDED.get(key)
    return bool(fn and fn(text, text.lower() if low is None else low))

#: Ordered rules: the first whose condition holds names the design. Specific,
#: strongly-diagnostic designs are tested before the broad technical ones, the
#: same precedence principle the cue labeller uses -- a paper that intervenes in a
#: real setting is Action Research even though it almost certainly also builds and
#: evaluates something.
#:
#: Each entry is (design name, predicate over the facet dict).
DESIGN_RULES: list[tuple[str, str]] = [
    ("Action Research",
     "intervenes and field_context"),
    ("Systematic Literature Review / Secondary Study",
     "secondary and not human_data"),
    ("Case Study",
     "field_context and not intervenes and not secondary"),
    ("Qualitative Field Study / Grounded Theory",
     "human_data and not evaluates"),
    ("Survey (empirical study of respondents)",
     "human_data"),
    ("Simulation & Modelling",
     "simulates and not builds"),
    ("Formal / Theoretical",
     "proves and not builds"),
    ("Design & Creation (Design Science)",
     "builds"),
    ("Experiment / Empirical Evaluation",
     "evaluates"),
    ("Formal / Theoretical",
     "proves"),
    ("Simulation & Modelling",
     "simulates"),
]


def derive_designs(facets: dict[str, bool]) -> list[tuple[str, str]]:
    """Designs compatible with a facet assignment, most specific first.

    Returns ``[(design, rule_that_fired), ...]``. More than one entry means the
    facets genuinely underdetermine the design -- which is information worth
    surfacing rather than hiding behind an arbitrary tie-break.
    """
    env = {k: bool(facets.get(k, False)) for k in FACET_KEYS}
    out: list[tuple[str, str]] = []
    for design, rule in DESIGN_RULES:
        if eval(rule, {"__builtins__": {}}, env) and design not in [d for d, _ in out]:
            out.append((design, rule))
    if not out:
        # Nothing fired: the paper declares no recognisable method. Design &
        # Creation is the base rate for computing, but say so rather than imply
        # evidence that does not exist.
        out.append(("Design & Creation (Design Science)", "default (no facet fired)"))
    return out


def primary_design(facets: dict[str, bool]) -> str:
    return derive_designs(facets)[0][0]


def facet_summary() -> str:
    lines = ["Methodological facets (observable properties):"]
    for f in FACETS:
        lines.append(f"  {f.key:15s} {f.question}")
    return "\n".join(lines)


__all__ = [
    "BY_KEY",
    "DESIGN_RULES",
    "FACETS",
    "FACET_KEYS",
    "VOCABULARY_VERSION",
    "Facet",
    "derive_designs",
    "extended_evidence",
    "facet_summary",
    "primary_design",
]
