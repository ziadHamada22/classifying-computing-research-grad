"""Agent 2 inference — field prediction, conditioned on a discipline.

The model has one flat 38-way head; conditioning happens here, by masking the
logits to the fields of the given discipline before the softmax. That means the
returned probabilities are a proper distribution *over that discipline's fields*
rather than a renormalised slice of a global one, so a confidence of 0.8 means
"80% among the seven CS fields", which is what a reader expects.

Because Agent 1 can be wrong, the caller may pass more than one candidate
discipline (see `predict_conditional`), which is how the pipeline surfaces both
readings when the discipline itself is borderline.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path

import numpy as np

from crc.agents.discipline.conformal import (
    ConformalCalibration,
    load_bank,
    pick_alpha,
)
from crc.agents.discipline.features import format_chunk
from crc.ingest import Document, ingest
from crc.ingest.schema import Section
from crc.taxonomy.fields import (
    DISCIPLINE_FIELD_IDS,
    GLOBAL_ID2LABEL,
    FIELDS_BY_DISCIPLINE,
)

PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"

#: What Agent 2 reads from a document. A paper's field is stated in its abstract,
#: and the rest of the text dilutes it: on 2,123 leak-free full papers, reading
#: the abstract chunk alone gives 0.835 field accuracy (true discipline given),
#: against 0.821 for the plain mean over every chunk that ran until 2026-09
#: (McNemar p = 0.035) and 0.830 for section-weighted pooling fitted on val
#: (``crc.eval.evaluate_field_documents``). So a document with an abstract is
#: read through its abstract -- one encoder pass instead of one per chunk -- and
#: anything without one (pasted text, unstructured articles) through every chunk.
FIELD_POOLING = "abstract"

#: A field answer is *contested* when the calibrated field set holds more than
#: one field (``conformal.json`` beside the model: split conformal, LAC, over
#: the discipline's ballot, calibrated on the 2025+ field slice). At alpha 0.10,
#: on unseen 2025+ papers, the set covers the true field 89.7% of the time, flags
#: 28% of papers, catches 66% of field errors, and a single-field answer is
#: right 90% of the time (``crc.eval.evaluate_field_uncertainty``).
DEFAULT_FIELD_ALPHA = 0.10
#: Fallback flag for a model with no conformal bank: swept on val at a 25% review
#: budget (catches 62% of test errors).
LOW_CONFIDENCE = 0.80
TIGHT_GAP = 0.02
#: The prototype-era pair copied from Agent 1, which ran until 2026-09: it
#: catches only 26% of field errors. Kept so older results stay reproducible.
LEGACY_LOW_CONFIDENCE = 0.55
LEGACY_TIGHT_GAP = 0.12


@dataclass
class FieldPrediction:
    """Agent 2's answer for one document, within one discipline."""

    discipline: str
    label: str
    confidence: float
    runner_up: str
    gap: float
    probs: dict[str, float]
    n_chunks: int
    borderline: bool
    borderline_reason: str | None = None
    #: Set when this is the *second* reading of a borderline discipline.
    conditional_on: str | None = None
    #: Fields of this discipline that cannot be ruled out at the calibrated level
    #: (empty when no conformal bank is loaded). ``label`` is committed either way.
    prediction_set: list[str] = dc_field(default_factory=list)
    set_size: int = 0
    certified: bool = False
    conformal_alpha: float | None = None

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class FieldClassifier:
    """Shared-encoder field model, masked to a discipline at inference."""

    #: Class-level defaults, so an instance built without ``__init__`` (as the
    #: tests do with stub encoders) behaves as "no conformal bank, deployed pooling".
    conformal: ConformalCalibration | None = None
    pooling: str = FIELD_POOLING

    def __init__(self, model, tokenizer, max_length: int = 256,
                 device: str | None = None, name: str = "field",
                 conformal: ConformalCalibration | None = None,
                 pooling: str = FIELD_POOLING):
        import torch

        if pooling not in ("abstract", "mean"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling

        #: Calibrated over ballot-masked probabilities with the discipline given,
        #: so its guarantee reads "if the discipline is right, the true field is
        #: in the set with probability >= 1 - alpha".
        self.conformal = conformal
        self.model = model
        self.tok = tokenizer
        self.max_length = max_length
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(cls, model_dir: str | Path = None,
             conformal_alpha: float | None = DEFAULT_FIELD_ALPHA,
             **kwargs) -> "FieldClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        d = Path(model_dir) if model_dir else (MODELS / "field-scibert-v2")
        if not d.is_absolute() and not d.exists():
            d = MODELS / d.name
        model = AutoModelForSequenceClassification.from_pretrained(str(d))
        tok = AutoTokenizer.from_pretrained(str(d))
        kwargs.setdefault("name", d.name)
        cpath = d / "conformal.json"
        if conformal_alpha is not None and cpath.exists():
            kwargs.setdefault("conformal", pick_alpha(load_bank(cpath), conformal_alpha))
        return cls(model, tok, **kwargs)

    def chunk_logits(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, len(GLOBAL_ID2LABEL)))
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], truncation=True,
                               max_length=self.max_length, padding=True,
                               return_tensors="pt").to(self.device)
                out.append(self.model(**enc).logits.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def evidence_chunks(self, doc: Document) -> list:
        """The chunks Agent 2 reads: the abstract when there is one (``pooling
        == "abstract"``), otherwise every classifiable chunk."""
        chunks = doc.classifiable_chunks()
        if self.pooling == "abstract":
            abstract = [c for c in chunks if c.section == Section.ABSTRACT]
            if abstract:
                return abstract
        return chunks

    def document_logits(self, doc: Document) -> np.ndarray:
        """Encode the evidence once: ``(n_chunks, 38)`` unmasked field logits.

        Conditioning is only a mask over these columns, so one encoder pass
        serves every candidate discipline, and with abstract pooling a whole
        paper costs a single short pass.
        """
        chunks = self.evidence_chunks(doc)
        texts = [format_chunk(c.text, c.section, doc.title) for c in chunks]
        return self.chunk_logits(texts)

    def predict(self, doc: Document, discipline: str,
                conditional_on: str | None = None,
                logits: np.ndarray | None = None) -> FieldPrediction:
        """Predict the field of ``doc``, restricted to ``discipline``'s fields.

        ``logits`` may be supplied by a caller that already encoded the document
        (see :meth:`predict_many`), in which case no GPU work happens here.
        """
        if logits is None:
            logits = self.document_logits(doc)

        allowed = DISCIPLINE_FIELD_IDS[discipline]
        names = [f.name for f in FIELDS_BY_DISCIPLINE[discipline]]

        if logits.size == 0:
            uniform = 1.0 / len(allowed)
            return FieldPrediction(
                discipline=discipline, label=names[0], confidence=uniform,
                runner_up=names[min(1, len(names) - 1)], gap=0.0,
                probs={n: uniform for n in names}, n_chunks=0,
                borderline=True, borderline_reason="no classifiable text",
                conditional_on=conditional_on)

        # Mask first, then softmax — so the distribution is over this
        # discipline's fields rather than a slice of the global one.
        masked = logits[:, allowed]
        probs = _softmax(masked, axis=-1)
        # Mean over the evidence chunks: the abstract alone for a paper (see
        # FIELD_POOLING), every chunk for text without an abstract.
        doc_probs = probs.mean(axis=0)
        doc_probs = doc_probs / doc_probs.sum()

        order = np.argsort(-doc_probs)
        top, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        conf = float(doc_probs[top])
        gap = float(doc_probs[top] - doc_probs[second])

        borderline, reason = False, None
        pset: list[str] = []
        if self.conformal is not None:
            vec = np.zeros(len(GLOBAL_ID2LABEL))
            vec[allowed] = doc_probs
            mask = self.conformal.predict_set(vec[None, :])[0]
            pset = [GLOBAL_ID2LABEL[i] for i in np.where(mask)[0]]
            if len(pset) != 1:
                borderline = True
                reason = (f"CONFORMAL: {len(pset)} fields cannot be ruled out at "
                          f"alpha={self.conformal.alpha:g}"
                          + (f" ({', '.join(pset)})" if pset else
                             " (not even the top field clears the bar)"))
        elif conf < LOW_CONFIDENCE:
            borderline, reason = True, (
                f"LOW_CONFIDENCE: top field at {conf:.2f} < {LOW_CONFIDENCE}")
        elif gap < TIGHT_GAP:
            borderline, reason = True, (
                f"TIGHT_GAP: top two separated by {gap:.2f} < {TIGHT_GAP}")

        return FieldPrediction(
            discipline=discipline,
            label=names[top],
            confidence=round(conf, 4),
            runner_up=names[second],
            gap=round(gap, 4),
            probs={n: round(float(p), 4) for n, p in zip(names, doc_probs)},
            n_chunks=int(logits.shape[0]),
            borderline=borderline,
            borderline_reason=reason,
            conditional_on=conditional_on,
            prediction_set=pset,
            set_size=len(pset),
            certified=pset == [names[top]],
            conformal_alpha=(self.conformal.alpha if self.conformal is not None else None),
        )

    def predict_many(self, doc: Document,
                     disciplines: list[str]) -> list[FieldPrediction]:
        """Predict under several candidate disciplines for one encoder pass.

        Used when Agent 1 reports a contested discipline set: rather than
        committing to a discipline that may be wrong, the pipeline reports the
        field reading under each live candidate. Because conditioning is just a
        mask, the second and later readings are effectively free — so widening
        the candidate set costs presentation, not computation.
        """
        if not disciplines:
            return []
        logits = self.document_logits(doc)
        return [self.predict(doc, d, conditional_on=None if i == 0 else d,
                             logits=logits)
                for i, d in enumerate(disciplines)]

    #: Kept as the previous name for this behaviour.
    predict_conditional = predict_many

    def classify(self, source, discipline: str, **kwargs) -> FieldPrediction:
        doc = source if isinstance(source, Document) else ingest(source, **kwargs)
        return self.predict(doc, discipline)


__all__ = ["FieldClassifier", "FieldPrediction", "DEFAULT_FIELD_ALPHA", "FIELD_POOLING",
           "LOW_CONFIDENCE", "TIGHT_GAP", "LEGACY_LOW_CONFIDENCE", "LEGACY_TIGHT_GAP"]
