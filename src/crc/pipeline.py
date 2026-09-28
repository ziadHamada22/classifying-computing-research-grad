"""The coordinated pipeline: ingest once, then run the agents in order.

Agent 2's label space is chosen by Agent 1's answer, so Agent 1's errors do not
merely add to the error rate — they *cascade*: a paper sent to the wrong
discipline cannot possibly get the right field, because the right field is not
on the ballot. Two things follow, and both are implemented here.

**Ingest once.** Parsing and chunking is the same work for every agent, and on a
long PDF it costs more than inference does. The document is ingested a single
time and the parsed object is passed along.

**Propagate uncertainty rather than hiding it.** When the discipline is contested
the pipeline does not silently commit to one ballot. It evaluates the field under
every discipline the calibrated prediction set leaves open and returns each
reading, so a reader sees the fork instead of a confident-looking answer resting
on a coin flip.

Two properties of that mechanism are worth stating, because they shape the code:

*The shortlist is nearly free.* Conditioning is a mask over Agent 2's 38 logits,
so one encoder pass serves every candidate discipline. Widening the shortlist
costs presentation, not computation — which is why the pipeline can afford a more
generous alpha here (``DEFAULT_SHORTLIST_ALPHA``) than Agent 1 uses for deciding
whether to *call* its own answer contested.

*Nothing waits for a second opinion.* Both agents always commit to a top-1 label,
so the pipeline produces a complete answer on its own. The set and the extra
readings are context attached to that answer. A local LLM second opinion on
contested documents was built and measured (`agents/discipline/llm_review.py`);
on held-out contested papers it changed nothing, so it is not in this path.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path

from crc.agents.discipline import DEPLOYED_MODEL
from crc.agents.discipline.predict import (
    ClassificationResult,
    DisciplineClassifier,
    EnsembleClassifier,
)
from crc.agents.discipline.conformal import (
    ConformalCalibration,
    load_bank,
    pick_alpha,
)
from crc.agents.field.predict import FieldClassifier, FieldPrediction
from crc.agents.methodology.predict import (
    DEPLOYED_MODEL as METHODOLOGY_MODEL,
    MethodologyClassifier,
    MethodologyPrediction,
)
from crc.ingest import Document, ingest
from crc.taxonomy import DISCIPLINES

PROJECT = Path(__file__).resolve().parents[2]
MODELS = PROJECT / "models"

#: Below this many words the input is flagged: the classifiers were trained on
#: abstracts of 20 words or more (Agent 3 filters its corpus at exactly that).
MIN_WORDS = 20
#: English function words. Real English abstracts run at 25-40% of tokens; French,
#: Spanish or German text sits near zero. Below 8% on 20+ words, the text is
#: flagged as probably not English (the models only saw English).
_EN_FUNCTION = frozenset(
    "the of and to in a is we for on with that this are by as an be from our "
    "which it at or these can their its".split())


def input_warnings(doc: Document) -> list[str]:
    """Plain-language warnings about inputs the models were not built for.

    They never change an answer; they are shown before it, so a one-word or
    non-English input is not mistaken for a reliable classification.
    """
    words = doc.text.split()
    out = []
    if len(words) < MIN_WORDS:
        out.append(f"Very short input ({len(words)} word{'' if len(words) == 1 else 's'}). The models were trained "
                   f"on abstracts of {MIN_WORDS}+ words, so treat these answers as guesses.")
    else:
        toks = [w.strip(".,;:()[]\"'").lower() for w in words]
        share = sum(t in _EN_FUNCTION for t in toks) / len(toks)
        if share < 0.08:
            out.append("The text does not look like English. The models were trained "
                       "on English papers only, so treat these answers with caution.")
    return out


#: Miscoverage for the *shortlist* of disciplines handed to Agent 2. Tighter than
#: Agent 1's own reporting alpha because extra candidates cost no GPU time here,
#: and a wider shortlist measurably recovers cascade loss.
DEFAULT_SHORTLIST_ALPHA = 0.10


@dataclass
class PipelineResult:
    """Everything the system concluded about one document."""

    discipline: str
    discipline_confidence: float
    field: str | None
    field_confidence: float | None
    doc_type: str
    n_chunks: int
    #: The document's title when the parser found one (the proposal's output has it).
    title: str | None = None
    #: Input problems the models were not built for (very short, not English),
    #: shown before the answers. They never change an answer.
    warnings: list[str] = dc_field(default_factory=list)
    #: Agent 3. The design is derived from observable facets by an auditable rule
    #: (see `agents/methodology/predict.py`); worldview/method are looked up from
    #: the design's taxonomy prior rather than predicted.
    design: str | None = None
    design_confidence: float | None = None
    design_read_from: str | None = None
    design_borderline: bool = False
    worldview: str | None = None
    research_method: str | None = None
    design_probs: dict = dc_field(default_factory=dict)
    #: The observable facets Agent 3 actually predicted; the design above is
    #: derived from them by an auditable rule, reported in `derived_rule`.
    facets: dict = dc_field(default_factory=dict)
    facets_present: list = dc_field(default_factory=list)
    derived_rule: str | None = None
    #: "facets", or "similar_papers" when no facet fired and the design was taken
    #: from the most similar reference papers.
    design_source: str | None = None
    #: The reference papers most like this one, with the design each was given.
    similar_papers: list = dc_field(default_factory=list)
    #: Present when Agent 1 was borderline: the field reading under the
    #: runner-up discipline, so the fork is visible rather than hidden.
    alternative: dict | None = None
    #: One entry per additional live discipline (empty when uncontested). The
    #: singular ``alternative`` above mirrors the first of these.
    alternatives: list[dict] = dc_field(default_factory=list)
    #: The disciplines Agent 1 could not rule out, and the alpha behind them.
    discipline_set: list[str] = dc_field(default_factory=list)
    shortlist_alpha: float | None = None
    discipline_certified: bool = False
    discipline_borderline: bool = False
    field_borderline: bool = False
    #: Fields of the committed discipline that cannot be ruled out at the
    #: calibrated level (Agent 2's conformal set; empty without a bank).
    field_set: list[str] = dc_field(default_factory=list)
    field_certified: bool = False
    notes: list[str] = dc_field(default_factory=list)
    timing_ms: dict = dc_field(default_factory=dict)
    discipline_probs: dict = dc_field(default_factory=dict)
    field_probs: dict = dc_field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    def summary(self) -> str:
        s = f"{self.discipline} ({self.discipline_confidence:.0%})"
        if self.field:
            s += f" → {self.field} ({self.field_confidence:.0%})"
        for a in (self.alternatives or
                  ([self.alternative] if self.alternative else [])):
            s += f"   [or {a['discipline']} → {a['field']}]"
        if self.design:
            s += (chr(10) + f"  design: {self.design} "
                  f"({self.design_confidence:.0%})")
        return s


class Pipeline:
    """Agent 1 -> Agent 2 (field), plus Agent 3 (research design)."""

    def __init__(self, discipline_clf, field_clf: FieldClassifier | None = None,
                 conditional_on_borderline: bool = True,
                 shortlist: ConformalCalibration | None = None,
                 methodology_clf: MethodologyClassifier | None = None):
        self.discipline_clf = discipline_clf
        self.field_clf = field_clf
        self.methodology_clf = methodology_clf
        self.conditional_on_borderline = conditional_on_borderline
        self.shortlist = shortlist

    def _candidate_disciplines(self, disc: ClassificationResult) -> list[str]:
        """Which disciplines get a field reading, best first.

        Preference order: the calibrated shortlist, else Agent 1's own prediction
        set, else the old top-2 fork when it flagged the answer, else just the
        committed label. The committed label always leads the list, so the primary
        reading never changes because of how the shortlist happened to be built.
        """
        names: list[str] = []
        if self.shortlist is not None and disc.probs:
            import numpy as np

            vec = np.array([disc.probs.get(d, 0.0) for d in DISCIPLINES])
            names = self.shortlist.set_labels(vec[None, :])[0]
        elif disc.prediction_set:
            names = list(disc.prediction_set)
        elif disc.borderline and self.conditional_on_borderline:
            names = [disc.label, disc.runner_up]

        if not self.conditional_on_borderline:
            return [disc.label]
        # order by the model's own confidence, committed label first
        rest = sorted((n for n in names if n != disc.label),
                      key=lambda n: -disc.probs.get(n, 0.0))
        return [disc.label] + rest

    @classmethod
    def load(cls, discipline_dir: str | Path | None = None,
             field_dir: str | Path | None = None,
             methodology_dir: str | Path | None = None,
             ensemble: bool = False,
             shortlist_alpha: float | None = DEFAULT_SHORTLIST_ALPHA,
             **kwargs) -> "Pipeline":
        """Load the deployed agents.

        ``ensemble=True`` swaps Agent 1 for the calibrated SciBERT + DeBERTa +
        TF-IDF ensemble (``models/ensemble``), whose conformal bank was fitted on
        the blend's own outputs -- so its prediction sets and the field shortlist
        stay calibrated. Otherwise Agent 1 is the single deployed model
        (``DEPLOYED_MODEL``).
        """
        if ensemble:
            edir = Path(discipline_dir) if discipline_dir else MODELS / "ensemble"
            if not (edir / "ensemble.json").exists():
                raise FileNotFoundError(
                    f"no ensemble manifest in {edir}; run scripts/build_ensemble.py")
            disc = EnsembleClassifier.load(edir)
            calib_dir = edir
        else:
            calib_dir = Path(discipline_dir or MODELS / DEPLOYED_MODEL)
            disc = DisciplineClassifier.load(calib_dir)

        fld = None
        fdir = Path(field_dir) if field_dir else MODELS / "field-scibert-v2"
        if (fdir / "config.json").exists():
            fld = FieldClassifier.load(fdir)

        # Agent 3 is independent of the other two: the research design is not
        # conditioned on discipline or field, so it needs no ballot and cannot
        # inherit their errors.
        meth = None
        mdir = Path(methodology_dir) if methodology_dir else (
            MODELS / METHODOLOGY_MODEL)
        if (mdir / "config.json").exists():
            meth = MethodologyClassifier.load(mdir)

        # The shortlist bank must come from the same scorer Agent 1 is using.
        shortlist = None
        cpath = calib_dir / "conformal.json"
        if shortlist_alpha is not None and cpath.exists():
            shortlist = pick_alpha(load_bank(cpath), shortlist_alpha)
        return cls(disc, fld, shortlist=shortlist,
                   methodology_clf=meth, **kwargs)

    def stages(self, doc: Document) -> dict:
        """Run each agent once on an ingested document; return the raw results.

        Both ``run`` (CLI, API) and the web UI build their output from this, so
        the two cannot drift apart: the same discipline decision, the same
        shortlist, the same field readings.
        """
        out: dict = {"discipline": None, "candidates": [], "fields": [],
                     "methodology": None, "timing_ms": {},
                     "title": doc.title, "warnings": input_warnings(doc)}
        t1 = time.perf_counter()
        disc: ClassificationResult = self.discipline_clf.classify_document(doc)
        out["discipline"] = disc
        out["timing_ms"]["discipline"] = round((time.perf_counter() - t1) * 1000, 1)

        if self.field_clf is not None:
            t2 = time.perf_counter()
            # The cascade guard. Agent 2 can only ever name a field belonging to
            # the discipline it was pointed at, so a wrong discipline makes the
            # right field unreachable. Reading under every live candidate is what
            # keeps that from silently costing accuracy -- and it is one encoder
            # pass regardless of how many candidates there are.
            out["candidates"] = self._candidate_disciplines(disc)
            out["fields"] = self.field_clf.predict_many(doc, out["candidates"])
            out["timing_ms"]["field"] = round((time.perf_counter() - t2) * 1000, 1)

        # Agent 3: research design. Independent of Agents 1 and 2 -- the design of
        # a study does not depend on its discipline -- so it runs on the same
        # ingested document without a ballot and cannot inherit the cascade.
        if self.methodology_clf is not None:
            t3 = time.perf_counter()
            out["methodology"] = self.methodology_clf.predict(doc)
            out["timing_ms"]["methodology"] = round((time.perf_counter() - t3) * 1000, 1)
        return out

    def run(self, source) -> PipelineResult:
        t0 = time.perf_counter()
        doc = source if isinstance(source, Document) else ingest(source)
        t_ingest = (time.perf_counter() - t0) * 1000
        st = self.stages(doc)
        disc: ClassificationResult = st["discipline"]

        notes: list[str] = list(st.get("warnings", []))
        result = PipelineResult(
            discipline=disc.label,
            discipline_confidence=disc.confidence,
            field=None, field_confidence=None,
            doc_type=disc.doc_type, n_chunks=disc.n_chunks,
            title=st.get("title"),
            warnings=list(st.get("warnings", [])),
            discipline_borderline=disc.borderline,
            discipline_probs=disc.probs,
            discipline_set=list(disc.prediction_set),
            discipline_certified=disc.certified,
            shortlist_alpha=(self.shortlist.alpha
                             if self.shortlist is not None else None),
        )
        if disc.borderline:
            notes.append(f"Agent 1 contested — {disc.borderline_reason}")

        if st["fields"]:
            candidates, readings = st["candidates"], st["fields"]
            primary = readings[0]
            result.field = primary.label
            result.field_confidence = primary.confidence
            result.field_borderline = primary.borderline
            result.field_probs = primary.probs
            result.field_set = list(primary.prediction_set)
            result.field_certified = primary.certified
            if primary.borderline:
                notes.append(f"Agent 2 borderline — {primary.borderline_reason}")
            for name, alt in zip(candidates[1:], readings[1:]):
                result.alternatives.append({
                    "discipline": name,
                    "discipline_confidence": round(disc.probs.get(name, 0.0), 4),
                    "field": alt.label,
                    "field_confidence": alt.confidence,
                })
            if result.alternatives:
                result.alternative = result.alternatives[0]
                others = ", ".join(a["discipline"] for a in result.alternatives)
                notes.append(
                    f"The discipline is contested, so the field is also reported "
                    f"under {others}; treat all readings as candidates.")
        elif self.field_clf is None:
            notes.append("Agent 2 not loaded — field prediction skipped.")

        m: MethodologyPrediction | None = st["methodology"]
        if m is not None:
            result.design = m.design
            result.design_confidence = m.confidence
            result.design_read_from = m.read_from
            result.design_borderline = m.borderline
            result.worldview = m.worldview
            result.research_method = m.method
            result.design_probs = m.probs
            result.facets = m.facets
            result.facets_present = list(m.facets_present)
            result.derived_rule = m.derived_rule
            result.design_source = m.design_source
            result.similar_papers = list(m.similar_papers)
            notes.extend(m.notes)
            if m.borderline:
                notes.append(f"Agent 3 borderline -- {m.borderline_reason}")
        else:
            notes.append("Agent 3 not loaded -- research design skipped.")

        result.notes = notes
        tm = st["timing_ms"]
        result.timing_ms = {
            "ingest": round(t_ingest, 1),
            "discipline": tm.get("discipline", 0.0),
            "field": tm.get("field", 0.0),
            "methodology": tm.get("methodology", 0.0),
        }
        result.timing_ms["total"] = round(sum(result.timing_ms.values()), 1)
        return result


__all__ = ["Pipeline", "PipelineResult"]
