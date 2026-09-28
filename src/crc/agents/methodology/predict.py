"""Agent 3 inference — methodological facets, and the research design they imply.

Agent 3 predicts eight **observable facets** of a paper and derives the research
design from them, rather than forcing one of nine designs directly. Three measured
reasons (`docs/AGENT3_FACETS_RESULTS.md`):

* Two *independent* labellers of the single design label agreed only **43%** of the
  time, so the design's apparent accuracy was agreement with one noisy source.
* A large share of computing papers genuinely **build something and evaluate it**,
  so a single label had to guess — which is where ~52% of the old errors lived.
* Facets are checkable without methodological training. "Does this paper prove
  theorems?" needs a reader; "is this design science or a case study?" needs a
  methodologist. That difference is what let the model be validated **without a
  hand-labelled gold set**.

Two consequences worth knowing at the call site:

**All nine designs are now reachable.** The previous model was trained on four and
could never return the other five. The derivation rules cover all nine, so a case
study or a literature review can be named — though the rare ones rest on
low-prevalence facets and their confidence should be read accordingly.

**The design is derived, not predicted.** `derived_rule` states exactly which
condition fired, so any answer can be audited back to the facets that produced it.
When more than one design is compatible, all of them are reported rather than
hidden behind an arbitrary tie-break.

Input is the abstract, which earlier measurement established as the best single
source for methodology (0.692 against 0.652 for the methods section, at 100%
coverage).

Standing caveat: facet labels come from a label model over cue-based labelling
functions, so the numbers establish *reliability*, not the validity of the
taxonomy. No gold-free method can establish the latter.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path

import numpy as np

from crc.agents.methodology.retrieval import STORE_FILE, ReferenceStore
from crc.ingest import Document, ingest
from crc.ingest.schema import DocType, Section
from crc.taxonomy.facets import (
    DESIGN_RULES,
    FACET_KEYS,
    derive_designs,
)
from crc.taxonomy.methodology import method_of, worldview_of

PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"
#: The facet model the system runs, and the label file it was trained on.
#: v2 (2026-09-26): vocabulary v2 labels; beat the August model on all six arXiv
#: comment / journal-ref signals and left 14.8% of papers without a facet instead
#: of 17.6% (`results/facet_models_comparison.json`).
DEPLOYED_MODEL = "methodology-facets-v2"
DEPLOYED_LABELS = Path(r"C:\Users\ziada\gp_data\corpus\facets_pool_v2.parquet")

#: Sections to read, best first. The abstract is what the model was trained on;
#: the rest exist so a document without a detectable abstract still gets an answer.
READ_ORDER = (Section.ABSTRACT.value, Section.METHODS.value,
              Section.INTRODUCTION.value, Section.CONCLUSION.value,
              Section.BODY.value)

#: A facet is asserted above this probability, unless the model directory carries
#: per-facet thresholds fitted on validation (`THRESHOLDS_FILE`).
FACET_THRESHOLD = 0.5
THRESHOLDS_FILE = "thresholds.json"
#: Reference papers shown with every answer when a store is present.
N_SIMILAR = 3
#: Flag the answer when the derived design's own probability is weak, or when the
#: facets leave several designs equally compatible.
LOW_CONFIDENCE = 0.45


@dataclass
class MethodologyPrediction:
    """Agent 3's answer: the facets observed, and the design they imply."""

    design: str
    confidence: float
    runner_up: str
    gap: float
    #: Probability of each design's rule condition, from the facet probabilities.
    probs: dict[str, float]
    #: Probability of each observable facet — the actual model output.
    facets: dict[str, float]
    facets_present: list[str]
    #: The exact rule that produced ``design``, so the answer is auditable.
    derived_rule: str
    #: Every design compatible with these facets, when more than one is.
    compatible_designs: list[str]
    read_from: str
    n_blocks: int
    borderline: bool
    borderline_reason: str | None = None
    #: Derived from the design's taxonomy prior — a lookup, not a prediction.
    worldview: str = ""
    method: str = ""
    notes: list[str] = dc_field(default_factory=list)
    #: Where the design came from: "facets" (the rules); when no facet fired,
    #: "top_facet" (the most probable facet, below threshold), "similar_papers"
    #: (the most similar reference papers' designs) or "default".
    design_source: str = "facets"
    #: The most similar reference papers and the design each was given -- evidence
    #: a reader can check, not an input to the answer (see `retrieval.py`).
    similar_papers: list[dict] = dc_field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    def summary(self) -> str:
        s = f"{self.design} ({self.confidence:.0%})"
        if self.facets_present:
            s += "  [" + ", ".join(self.facets_present) + "]"
        return s


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


FALLBACKS = ("top_facet", "similar_papers")


def normalise_fallback(value: str | bool | None,
                       store: ReferenceStore | None) -> str | None:
    """``True`` is the original meaning (similar papers); that mode needs a store."""
    if value is True:
        value = "similar_papers"
    if value not in FALLBACKS:
        return None
    if value == "similar_papers" and store is None:
        return None
    return value


def rule_probability(rule: str, p: dict[str, float]) -> float:
    """Probability that a derivation rule holds, from independent facet odds.

    Facets are modelled as independent, which is an approximation — building and
    evaluating clearly correlate. It is stated rather than hidden, and it is only
    used to rank and to report a confidence, never to choose the design (that is
    the deterministic rule order, so the answer stays auditable).
    """
    if rule.startswith(("default", "no facet")):
        return 0.0
    prob = 1.0
    for term in rule.split(" and "):
        term = term.strip()
        if term.startswith("not "):
            prob *= 1.0 - p.get(term[4:].strip(), 0.0)
        else:
            prob *= p.get(term, 0.0)
    return float(prob)


class MethodologyClassifier:
    """Multi-label facet model, with the research design derived from its output."""

    #: Class-level defaults, so a classifier built without ``__init__`` (the test
    #: stubs) behaves as the plain model: 0.5 everywhere, no retrieval.
    thresholds: dict[str, float] | None = None
    store: ReferenceStore | None = None
    #: What to answer when no facet fires: None (the taxonomy default, flagged),
    #: "top_facet" or "similar_papers" -- chosen on validation by
    #: `evaluate_facet_decisions.py` and recorded in thresholds.json.
    no_evidence_fallback: str | None = None

    def __init__(self, model, tokenizer, max_length: int = 256,
                 device: str | None = None, name: str = "methodology",
                 thresholds: dict[str, float] | None = None,
                 store: ReferenceStore | None = None,
                 no_evidence_fallback: str | bool | None = None):
        import torch

        self.model = model
        self.tok = tokenizer
        self.max_length = max_length
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.thresholds = thresholds
        self.store = store
        self.no_evidence_fallback = normalise_fallback(no_evidence_fallback, store)

    @classmethod
    def load(cls, model_dir: str | Path | None = None,
             **kwargs) -> "MethodologyClassifier":
        """Load the model, plus its fitted thresholds and reference store if present.

        ``thresholds.json`` also records whether the no-evidence fallback passed its
        validation gate; it is only switched on when it did.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        d = Path(model_dir) if model_dir else (MODELS / DEPLOYED_MODEL)
        if not d.is_absolute() and not d.exists():
            d = MODELS / d.name
        model = AutoModelForSequenceClassification.from_pretrained(str(d))
        tok = AutoTokenizer.from_pretrained(str(d))
        kwargs.setdefault("name", d.name)
        if (d / THRESHOLDS_FILE).exists():
            cfg = json.loads((d / THRESHOLDS_FILE).read_text())
            kwargs.setdefault("thresholds", cfg.get("thresholds"))
            kwargs.setdefault("no_evidence_fallback", cfg.get("no_evidence_fallback"))
        if (d / STORE_FILE).exists():
            kwargs.setdefault("store", ReferenceStore.load(d / STORE_FILE))
        return cls(model, tok, **kwargs)

    def threshold(self, facet: str) -> float:
        return float((self.thresholds or {}).get(facet, FACET_THRESHOLD))

    def _select_blocks(self, doc: Document) -> tuple[list, str]:
        chunks = doc.classifiable_chunks()
        by_section: dict[str, list] = {}
        for c in chunks:
            by_section.setdefault(c.section.value, []).append(c)
        for sec in READ_ORDER:
            if by_section.get(sec):
                return by_section[sec], sec
        if chunks:
            return sorted(chunks, key=lambda c: -c.n_words)[:2], "fallback"
        return [], "none"

    def encode(self, texts: list[str],
               batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Facet probabilities and the unit-normalised [CLS] embedding, one pass."""
        import torch

        if not texts:
            return np.zeros((0, len(FACET_KEYS))), np.zeros((0, 0), np.float32)
        logits, embs = [], []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], truncation=True,
                               max_length=self.max_length, padding=True,
                               return_tensors="pt").to(self.device)
                out = self.model(**enc, output_hidden_states=True)
                logits.append(out.logits.float().cpu().numpy())
                cls_vec = out.hidden_states[-1][:, 0, :].float()
                embs.append(torch.nn.functional.normalize(cls_vec, dim=-1).cpu().numpy())
        return (_sigmoid(np.concatenate(logits, axis=0)),
                np.concatenate(embs, axis=0).astype(np.float32))

    def facet_probs(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self.encode(texts, batch_size)[0]

    def predict(self, doc: Document) -> MethodologyPrediction:
        blocks, read_from = self._select_blocks(doc)
        notes: list[str] = []

        if not blocks:
            flat = {f: 0.0 for f in FACET_KEYS}
            return MethodologyPrediction(
                design="Design & Creation (Design Science)", confidence=0.0,
                runner_up="", gap=0.0, probs={}, facets=flat, facets_present=[],
                derived_rule="default (no readable text)", compatible_designs=[],
                read_from="none", n_blocks=0, borderline=True,
                borderline_reason="no readable text",
                worldview="", method="",
                notes=["No classifiable text was found; this is not a prediction."])

        pasted = doc.doc_type in (DocType.PARAGRAPH, DocType.ARTICLE)
        if read_from == Section.BODY.value and pasted:
            notes.append("Read from pasted text with no section structure; "
                         "treated as an abstract, which is what the model expects.")
        elif read_from != Section.ABSTRACT.value:
            notes.append(f"No abstract was detected in a structured document, so "
                         f"the facets were read from '{read_from}'. The model was "
                         f"validated on abstracts; treat this with more caution.")

        # One row per block; take the most confident reading per facet, matching
        # the pooling rule the earlier experiment selected on validation.
        texts = [c.text for c in blocks]
        query = None
        if self.store is not None:
            probs, emb = self.encode(texts)
            query = emb.mean(axis=0)
        else:
            probs = self.facet_probs(texts)
        fp = probs.max(axis=0)
        facets = {f: round(float(v), 4) for f, v in zip(FACET_KEYS, fp)}
        hard = {f: v >= self.threshold(f) for f, v in facets.items()}
        present = [f for f in FACET_KEYS if hard[f]]

        compatible = derive_designs(hard)
        design, rule = compatible[0]
        source = "facets" if present else "default"
        conf = rule_probability(rule, facets)

        # No facet fired: the rules would return Design & Creation by default, a
        # guess that agreed with the paper's own methods/results text on only
        # 20-30% of such papers. Both replacements below read the paper and agree
        # on 33-38%; which one is used was decided on validation. Either way the
        # answer is flagged and says where it came from.
        if not present and self.no_evidence_fallback == "top_facet":
            top = max(facets, key=facets.get)
            design, base = derive_designs({f: f == top for f in FACET_KEYS})[0]
            rule = (f"no facet cleared its threshold; the most probable one "
                    f"({top}, {facets[top]:.2f}) was taken as the evidence -> {base}")
            compatible = [(design, rule)]
            source = "top_facet"
            conf = facets[top]
        elif (not present and self.no_evidence_fallback == "similar_papers"
              and query is not None):
            design, share, vote = self.store.design_vote(query)
            rule = (f"no facet in the text; the design most common among the most "
                    f"similar reference papers ({share:.0%} of their weighted vote)")
            compatible = [(design, rule)]
            source = "similar_papers"
            conf = share

        # Rank every design by how probable its condition is, for the audit trail.
        ranked = sorted(
            {d: rule_probability(r, facets) for d, r in DESIGN_RULES}.items(),
            key=lambda kv: -kv[1])
        runner_up, gap = "", 0.0
        others = [(d, p) for d, p in ranked if d != design]
        if others:
            runner_up, gap = others[0][0], round(conf - others[0][1], 4)

        borderline, reason = False, None
        if source != "facets":
            borderline, reason = True, (
                "NO_EVIDENCE: no methodological facet was detected in the text"
                + {"similar_papers": "; the design is taken from the most similar "
                                     "reference papers",
                   "top_facet": "; the design follows the single most probable "
                                "facet, which is below its threshold",
                   "default": "; the design is the taxonomy's default, not an "
                              "observation"}[source])
        elif conf < LOW_CONFIDENCE:
            borderline, reason = True, (
                f"LOW_CONFIDENCE: the derived design's own condition holds with "
                f"probability {conf:.2f} < {LOW_CONFIDENCE}")
        elif len(compatible) > 1:
            borderline, reason = True, (
                f"UNDERDETERMINED: {len(compatible)} designs are compatible with "
                f"these facets")

        if len(compatible) > 1:
            notes.append(
                "The facets are compatible with more than one design ("
                + ", ".join(d for d, _ in compatible)
                + "); the first is reported, by the precedence in the taxonomy.")
        if not present:
            notes.append("No facet was detected above threshold, so the design "
                         "rests on no positive evidence.")
        if source in ("similar_papers", "top_facet"):
            notes.append(f"No-evidence fallback: {rule}. Worldview and method are "
                         f"looked up from the design, not predicted.")
        else:
            notes.append(f"Design derived from facets by the rule: {rule}. "
                         f"Worldview and method are looked up from the design, not "
                         f"predicted.")
        similar = ([n.to_dict() for n in self.store.neighbours(query, N_SIMILAR)]
                   if query is not None else [])

        return MethodologyPrediction(
            design=design,
            confidence=round(conf, 4),
            runner_up=runner_up,
            gap=gap,
            probs={d: round(p, 4) for d, p in ranked},
            facets=facets,
            facets_present=present,
            derived_rule=rule,
            compatible_designs=[d for d, _ in compatible],
            read_from=read_from,
            n_blocks=len(blocks),
            borderline=borderline,
            borderline_reason=reason,
            worldview=worldview_of(design),
            method=method_of(design),
            notes=notes,
            design_source=source,
            similar_papers=similar,
        )

    def classify(self, source, **kwargs) -> MethodologyPrediction:
        doc = source if isinstance(source, Document) else ingest(source, **kwargs)
        return self.predict(doc)


__all__ = [
    "DEPLOYED_MODEL",
    "FACET_THRESHOLD",
    "LOW_CONFIDENCE",
    "READ_ORDER",
    "MethodologyClassifier",
    "MethodologyPrediction",
    "rule_probability",
]
