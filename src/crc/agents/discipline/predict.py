"""Agent 1 inference: document in, calibrated discipline prediction out.

Ties the three layers together — ingest/chunk, per-chunk classification,
document-level aggregation — behind one call:

    clf = DisciplineClassifier.load("models/deberta-v3-base")
    result = clf.classify("paper.pdf")

An ensemble of backbones is supported through the same interface, because the
chunk-probability step is the only part that differs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from crc.agents.discipline.aggregate import (
    DEFAULT_SECTION_WEIGHTS,
    DocumentPrediction,
    predict_document,
)
from crc.agents.discipline.conformal import (
    ConformalCalibration,
    load_bank,
    pick_alpha,
)
from crc.agents.discipline.features import format_chunk
from crc.ingest import Document, DocType, ingest
from crc.taxonomy import DISCIPLINES

#: A prediction is borderline when the top class is weakly held, or when the top
#: two are nearly tied. Both are re-tuned against the v2 ensemble rather than
#: inherited from the prototype's single-model constants, which were mis-scaled
#: for averaged distributions.
#:
#: These are only the fallback for a scorer with no conformal calibration (the
#: pipeline's ensemble mode, for instance); a calibrated model always prefers the
#: conformal route (see `conformal.py`). The values are SciBERT's, swept on val at
#: a 25% review budget by `tune_thresholds.py`, which also writes them to
#: `<model>/thresholds.json` -- read per model by `DisciplineClassifier.load`.
#: On the v2 test split they catch 64.8% of errors at a 27.8% fire rate.
LOW_CONFIDENCE = 0.80
TIGHT_GAP = 0.04
#: The prototype-era pair the shipped code actually ran with until 2026-09, when
#: the tuned values above were finally written back. They catch only 20.6% of
#: test errors. Kept so evaluations that reported them stay reproducible.
LEGACY_LOW_CONFIDENCE = 0.55
LEGACY_TIGHT_GAP = 0.12

#: Default miscoverage for the shipped prediction set. At 0.20 roughly a quarter
#: of out-of-period documents are reported as contested, and a document reported
#: as a single discipline is right about 80% of the time. Override per call site.
DEFAULT_CONFORMAL_ALPHA = 0.20


@dataclass
class ChunkPrediction:
    index: int
    section: str
    n_words: int
    label: str
    confidence: float
    weight: float
    text_preview: str


@dataclass
class ClassificationResult:
    """Everything a curator needs to audit one decision."""

    label: str
    confidence: float
    runner_up: str
    gap: float
    probs: dict[str, float]
    doc_type: str
    n_chunks: int
    strategy: str
    borderline: bool
    borderline_reason: str | None = None
    #: Disciplines that cannot be ruled out at the calibrated confidence level.
    #: Empty when no conformal calibration is loaded. ``label`` above is always
    #: committed regardless, so the system answers with or without this.
    prediction_set: list[str] = field(default_factory=list)
    set_size: int = 0
    #: True when the committed label is the *only* one in the set — i.e. it
    #: carries the 1 - alpha certificate on its own.
    certified: bool = False
    conformal_alpha: float | None = None
    section_mass: dict[str, float] = field(default_factory=dict)
    chunks: list[ChunkPrediction] = field(default_factory=list)
    parser: str = ""
    source: str = ""

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class DisciplineClassifier:
    """A single fine-tuned backbone, optionally temperature-calibrated."""

    def __init__(self, model, tokenizer, temperature: float = 1.0,
                 max_length: int = 256, device: str | None = None,
                 section_weights: dict[str, float] | None = None,
                 strategy: str = "weighted_mean", name: str = "model",
                 conformal: ConformalCalibration | None = None,
                 low_confidence: float = LOW_CONFIDENCE,
                 tight_gap: float = TIGHT_GAP):
        import torch

        #: Calibration is tied to the scorer it was fitted on, so it must not be
        #: shared with an ensemble or a different backbone.
        self.conformal = conformal
        #: Fallback borderline thresholds, used only when ``conformal`` is None.
        self.low_confidence = float(low_confidence)
        self.tight_gap = float(tight_gap)
        self.model = model
        self.tok = tokenizer
        self.temperature = float(temperature) if temperature else 1.0
        self.max_length = max_length
        self.strategy = strategy
        self.section_weights = section_weights or dict(DEFAULT_SECTION_WEIGHTS)
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(cls, model_dir: str | Path,
             conformal_alpha: float | None = DEFAULT_CONFORMAL_ALPHA,
             **kwargs) -> "DisciplineClassifier":
        """Load a backbone plus whatever calibration sits beside it.

        ``conformal_alpha=None`` disables prediction sets entirely, which is the
        right choice for an ensemble or any scorer the calibration was not fitted
        on.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        tok = AutoTokenizer.from_pretrained(str(model_dir))

        temperature = 1.0
        tpath = model_dir / "temperature.json"
        if tpath.exists():
            temperature = float(json.loads(tpath.read_text()).get("temperature", 1.0))

        weights = None
        wpath = model_dir / "section_weights.json"
        if wpath.exists():
            weights = json.loads(wpath.read_text())

        conformal = None
        cpath = model_dir / "conformal.json"
        if conformal_alpha is not None and cpath.exists():
            conformal = pick_alpha(load_bank(cpath), conformal_alpha)

        # Tuned fallback thresholds for this model, when they have been fitted.
        thpath = model_dir / "thresholds.json"
        if thpath.exists():
            th = json.loads(thpath.read_text()).get("thresholds", {})
            kwargs.setdefault("low_confidence", th.get("low_confidence", LOW_CONFIDENCE))
            kwargs.setdefault("tight_gap", th.get("tight_gap", TIGHT_GAP))

        kwargs.setdefault("name", model_dir.name)
        return cls(model, tok, temperature=temperature,
                   section_weights=weights, conformal=conformal, **kwargs)

    def chunk_probs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Calibrated per-chunk probability matrix, shape (len(texts), 6)."""
        import torch

        if not texts:
            return np.zeros((0, len(DISCIPLINES)))
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tok(batch, truncation=True, max_length=self.max_length,
                               padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.float().cpu().numpy()
                out.append(logits)
        logits = np.concatenate(out, axis=0) / self.temperature
        return _softmax(logits, axis=-1)

    def classify_document(self, doc: Document,
                          strategy: str | None = None) -> ClassificationResult:
        chunks = doc.classifiable_chunks()
        texts = [format_chunk(c.text, c.section, doc.title) for c in chunks]
        probs = self.chunk_probs(texts)

        sections = [c.section.value for c in chunks]
        n_words = np.array([c.n_words for c in chunks], dtype=np.float64)
        pred: DocumentPrediction = predict_document(
            probs, sections, n_words,
            strategy=strategy or self.strategy,
            section_weights=self.section_weights,
        )
        return self._to_result(doc, chunks, probs, pred)

    def classify(self, source, **kwargs) -> ClassificationResult:
        """Ingest then classify. Accepts a path, a raw string, or a Document."""
        doc = source if isinstance(source, Document) else ingest(source, **kwargs)
        return self.classify_document(doc)

    def _to_result(self, doc: Document, chunks, probs: np.ndarray,
                   pred: DocumentPrediction,
                   conformal: ConformalCalibration | None = None
                   ) -> ClassificationResult:
        # The prediction set decides how the answer is *presented*, never whether
        # there is one: `pred.label` is committed either way, so the agent works
        # standalone. With no calibration we fall back to the old thresholds.
        cal = conformal if conformal is not None else self.conformal
        pset: list[str] = []
        if cal is not None:
            pset = cal.set_labels(np.asarray(pred.probs)[None, :])[0]
            certified = pset == [pred.label]
            borderline = len(pset) != 1
            if borderline:
                reason = (f"CONFORMAL: {len(pset)} disciplines cannot be ruled "
                          f"out at alpha={cal.alpha:g}"
                          + (f" ({', '.join(pset)})" if pset else
                             " (not even the top label clears the bar)"))
            else:
                reason = None
        else:
            borderline, reason = borderline_check(pred.confidence, pred.gap,
                                                  self.low_confidence, self.tight_gap)
            certified = False
        chunk_preds = []
        for c, p, w in zip(chunks, probs, pred.chunk_weights):
            j = int(np.argmax(p))
            chunk_preds.append(ChunkPrediction(
                index=c.index, section=c.section.value, n_words=c.n_words,
                label=DISCIPLINES[j], confidence=round(float(p[j]), 4),
                weight=round(float(w), 4),
                text_preview=c.text[:120],
            ))
        return ClassificationResult(
            label=pred.label,
            confidence=round(pred.confidence, 4),
            runner_up=pred.runner_up,
            gap=round(pred.gap, 4),
            probs={d: round(float(v), 4) for d, v in zip(DISCIPLINES, pred.probs)},
            doc_type=doc.doc_type.value,
            n_chunks=pred.n_chunks,
            strategy=pred.strategy,
            borderline=borderline,
            borderline_reason=reason,
            prediction_set=pset,
            set_size=len(pset),
            certified=certified,
            conformal_alpha=(cal.alpha if cal is not None else None),
            section_mass=pred.per_section,
            chunks=chunk_preds,
            parser=doc.parser,
            source=doc.source,
        )


class TfidfMember:
    """The TF-IDF baseline as an ensemble member at inference time (CPU, sklearn).

    The fitted ensemble's best-calibrated member set includes TF-IDF, but only
    transformer backbones could be loaded at inference, so the documented
    ensemble was never the one that ran. This wraps the saved sklearn bundle in
    the one method an ensemble member needs.
    """

    def __init__(self, bundle: dict, name: str = "tfidf", temperature: float = 1.0):
        self.vec = bundle["vectorizer"]
        self.selector = bundle.get("selector")
        self.clf = bundle["classifier"]
        self.name = name
        #: Applied to log-probabilities, exactly as the ensemble evaluation did
        #: when it fitted the member weights (``ensemble.load_member``).
        self.temperature = float(temperature) if temperature else 1.0

    @classmethod
    def load(cls, model_dir: str | Path) -> "TfidfMember":
        import joblib

        model_dir = Path(model_dir)
        t = 1.0
        tpath = model_dir / "temperature.json"
        if tpath.exists():
            t = float(json.loads(tpath.read_text()).get("temperature", 1.0))
        return cls(joblib.load(model_dir / "pipeline.joblib"), name=model_dir.name,
                   temperature=t)

    def chunk_probs(self, texts: list[str], batch_size: int = 0) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(DISCIPLINES)))
        X = self.vec.transform(texts)
        if self.selector is not None:
            X = self.selector.transform(X)
        proba = self.clf.predict_proba(X)
        col = {int(c): i for i, c in enumerate(self.clf.classes_)}
        proba = proba[:, [col[j] for j in range(len(DISCIPLINES))]]
        if self.temperature != 1.0:
            proba = _softmax(np.log(np.clip(proba, 1e-9, 1.0)) / self.temperature)
        return proba


class EnsembleClassifier:
    """Probability-averaged ensemble over several backbones."""

    #: Where the deployable ensemble lives: a manifest of members and weights,
    #: plus a conformal bank fitted on the *ensemble's own* outputs.
    DEFAULT_DIR = Path(__file__).resolve().parents[4] / "models" / "ensemble"

    def __init__(self, members: list,
                 weights: list[float] | None = None,
                 strategy: str = "weighted_mean",
                 section_weights: dict[str, float] | None = None,
                 conformal: ConformalCalibration | None = None):
        if not members:
            raise ValueError("ensemble needs at least one member")
        #: Averaged probabilities are a different scorer from any single member,
        #: so a member's calibration is invalid here. Defaults to None; pass a
        #: calibration fitted on this ensemble's own outputs to enable sets.
        self.conformal = conformal
        self.members = members
        w = np.array(weights if weights else [1.0] * len(members), dtype=np.float64)
        self.weights = w / w.sum()
        self.strategy = strategy
        self.section_weights = section_weights or dict(DEFAULT_SECTION_WEIGHTS)

    @classmethod
    def load(cls, ensemble_dir: str | Path | None = None,
             conformal_alpha: float | None = DEFAULT_CONFORMAL_ALPHA
             ) -> "EnsembleClassifier":
        """Load the ensemble described by ``ensemble.json`` and its own bank.

        Members are named model directories under ``models/``; a directory
        holding ``pipeline.joblib`` is loaded as a TF-IDF member. The conformal
        bank beside the manifest was fitted on the blended probabilities, so it
        is valid for this ensemble and no other scorer.
        """
        d = Path(ensemble_dir) if ensemble_dir else cls.DEFAULT_DIR
        manifest = json.loads((d / "ensemble.json").read_text())
        models = d.parent
        members, weights = [], []
        for tag, w in manifest["members"]:
            md = models / tag
            if (md / "pipeline.joblib").exists() and not (md / "config.json").exists():
                members.append(TfidfMember.load(md))
            else:
                # a member's own calibration is not valid for the blend
                members.append(DisciplineClassifier.load(md, conformal_alpha=None))
            weights.append(float(w))
        conformal = None
        cpath = d / "conformal.json"
        if conformal_alpha is not None and cpath.exists():
            conformal = pick_alpha(load_bank(cpath), conformal_alpha)
        ens = cls(members, weights, strategy=manifest.get("strategy", "weighted_mean"),
                  section_weights=members[0].section_weights, conformal=conformal)
        ens.name = " + ".join(tag for tag, _ in manifest["members"])
        return ens

    def chunk_probs(self, texts: list[str]) -> np.ndarray:
        acc = None
        for m, w in zip(self.members, self.weights):
            p = m.chunk_probs(texts) * w
            acc = p if acc is None else acc + p
        return acc

    def classify_document(self, doc: Document) -> ClassificationResult:
        chunks = doc.classifiable_chunks()
        texts = [format_chunk(c.text, c.section, doc.title) for c in chunks]
        probs = self.chunk_probs(texts)
        sections = [c.section.value for c in chunks]
        n_words = np.array([c.n_words for c in chunks], dtype=np.float64)
        pred = predict_document(probs, sections, n_words,
                                strategy=self.strategy,
                                section_weights=self.section_weights)
        if not hasattr(self.members[0], "_to_result"):
            # the first member formats the audit trail, so it must be a
            # transformer backbone; sklearn members may follow it
            raise TypeError("the first ensemble member must be a DisciplineClassifier")
        return self.members[0]._to_result(doc, chunks, probs, pred,
                                          conformal=self.conformal)

    def classify(self, source, **kwargs) -> ClassificationResult:
        doc = source if isinstance(source, Document) else ingest(source, **kwargs)
        return self.classify_document(doc)


def borderline_check(confidence: float, gap: float,
                     low_confidence: float = LOW_CONFIDENCE,
                     tight_gap: float = TIGHT_GAP) -> tuple[bool, str | None]:
    """Should this prediction get a second opinion?"""
    if confidence < low_confidence:
        return True, (f"LOW_CONFIDENCE: top class at {confidence:.2f} "
                      f"< {low_confidence}")
    if gap < tight_gap:
        return True, f"TIGHT_GAP: top two separated by {gap:.2f} < {tight_gap}"
    return False, None


__all__ = [
    "DisciplineClassifier",
    "EnsembleClassifier",
    "TfidfMember",
    "ClassificationResult",
    "ChunkPrediction",
    "borderline_check",
    "LOW_CONFIDENCE",
    "TIGHT_GAP",
    "LEGACY_LOW_CONFIDENCE",
    "LEGACY_TIGHT_GAP",
]
