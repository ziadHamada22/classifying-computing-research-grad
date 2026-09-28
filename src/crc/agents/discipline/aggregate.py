"""Combine per-chunk probabilities into one document-level prediction.

A 30-page paper yields dozens of chunks of very unequal value. The abstract
states the contribution; a methods paragraph reciting optimiser settings could
belong to any discipline; a related-work chunk often describes *other* fields
entirely and actively misleads. Plain averaging lets the uninformative majority
outvote the informative minority, so the aggregators here weight each chunk by
three independent signals:

  section     how much this kind of section says about discipline
  length      longer chunks carry more evidence (sub-linear, via sqrt)
  certainty   a chunk whose distribution is near-uniform says little, so its
              weight decays with normalised entropy

The whole thing must also degrade to the identity when there is exactly one
chunk, because a pasted paragraph is the most common interactive input.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from crc.ingest.schema import SECTION_PRIOR, Section
from crc.taxonomy import DISCIPLINES

EPS = 1e-9
_N_CLASSES = len(DISCIPLINES)
_MAX_ENTROPY = float(np.log(_N_CLASSES))

#: Default section weights, seeded from the hand-set priors in the ingest
#: schema. `fit_section_weights` replaces these with values fitted on val.
DEFAULT_SECTION_WEIGHTS: dict[str, float] = {
    s.value: SECTION_PRIOR.get(s, 1.0) for s in Section
}


@dataclass
class DocumentPrediction:
    """Aggregated result for one document."""

    probs: np.ndarray                      # (6,) document-level distribution
    label: str
    confidence: float
    runner_up: str
    gap: float
    n_chunks: int
    strategy: str
    chunk_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    per_section: dict[str, float] = field(default_factory=dict)

    @property
    def ranked(self) -> list[tuple[str, float]]:
        order = np.argsort(-self.probs)
        return [(DISCIPLINES[i], float(self.probs[i])) for i in order]


def normalised_entropy(probs: np.ndarray) -> np.ndarray:
    """Entropy of each row, scaled to [0, 1]."""
    p = np.clip(probs, EPS, 1.0)
    ent = -(p * np.log(p)).sum(axis=-1)
    return ent / _MAX_ENTROPY


def chunk_weights(sections: list[str] | None, n_words: np.ndarray | None,
                  probs: np.ndarray,
                  section_weights: dict[str, float] | None = None,
                  use_length: bool = True,
                  use_certainty: bool = True,
                  length_ref: float = 200.0) -> np.ndarray:
    """Per-chunk weight from section, length and certainty."""
    n = probs.shape[0]
    w = np.ones(n, dtype=np.float64)

    if sections is not None:
        table = section_weights or DEFAULT_SECTION_WEIGHTS
        w *= np.array([max(0.0, table.get(s, 1.0)) for s in sections])

    if use_length and n_words is not None:
        # Sub-linear: a 400-word chunk is better evidence than a 100-word one,
        # but not four times better.
        w *= np.sqrt(np.clip(np.asarray(n_words, dtype=np.float64), 1.0, None)
                     / length_ref)

    if use_certainty:
        # Near-uniform chunks contribute little. Floor at 0.05 so a document of
        # uniformly uncertain chunks still produces a prediction.
        w *= np.clip(1.0 - normalised_entropy(probs), 0.05, 1.0)

    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(n, dtype=np.float64)
    return w


def aggregate(probs: np.ndarray,
              sections: list[str] | None = None,
              n_words: np.ndarray | None = None,
              strategy: str = "weighted_mean",
              section_weights: dict[str, float] | None = None,
              **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Reduce (n_chunks, 6) chunk probabilities to a (6,) document distribution.

    Returns ``(document_probs, chunk_weights)``.
    """
    probs = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    n = probs.shape[0]
    if n == 1:
        return probs[0].copy(), np.ones(1)

    if strategy == "mean":
        w = np.ones(n)
        doc = probs.mean(axis=0)
    elif strategy == "max":
        w = np.zeros(n)
        w[int(np.max(probs, axis=1).argmax())] = 1.0
        doc = probs[int(np.max(probs, axis=1).argmax())].copy()
    elif strategy in ("weighted_mean", "weighted_geometric"):
        w = chunk_weights(sections, n_words, probs, section_weights,
                          use_length=kwargs.get("use_length", True),
                          use_certainty=kwargs.get("use_certainty", True))
        if strategy == "weighted_mean":
            doc = (probs * w[:, None]).sum(axis=0) / w.sum()
        else:
            # Geometric mean in log space: sharper than arithmetic, and a class
            # that any confident chunk strongly rejects gets suppressed.
            logp = np.log(np.clip(probs, EPS, 1.0))
            doc = np.exp((logp * w[:, None]).sum(axis=0) / w.sum())
            doc = doc / doc.sum()
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    doc = np.clip(doc, EPS, None)
    return doc / doc.sum(), w


def predict_document(probs: np.ndarray,
                     sections: list[str] | None = None,
                     n_words: np.ndarray | None = None,
                     strategy: str = "weighted_mean",
                     section_weights: dict[str, float] | None = None,
                     **kwargs) -> DocumentPrediction:
    doc, w = aggregate(probs, sections, n_words, strategy, section_weights, **kwargs)
    order = np.argsort(-doc)
    top, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    per_section: dict[str, float] = {}
    if sections is not None:
        for s, wi in zip(sections, w):
            per_section[s] = per_section.get(s, 0.0) + float(wi)
        total = sum(per_section.values()) or 1.0
        per_section = {k: round(v / total, 4) for k, v in per_section.items()}
    return DocumentPrediction(
        probs=doc,
        label=DISCIPLINES[top],
        confidence=float(doc[top]),
        runner_up=DISCIPLINES[second],
        gap=float(doc[top] - doc[second]),
        n_chunks=int(np.atleast_2d(probs).shape[0]),
        strategy=strategy,
        chunk_weights=w,
        per_section=per_section,
    )


def fit_section_weights(doc_chunk_probs: list[np.ndarray],
                        doc_sections: list[list[str]],
                        doc_n_words: list[np.ndarray],
                        labels: np.ndarray,
                        strategy: str = "weighted_mean",
                        max_iter: int = 200,
                        verbose: bool = True,
                        prior: dict[str, float] | None = None) -> dict[str, float]:
    """Fit one non-negative weight per section by minimising document NLL on val.

    Weights are parameterised as ``exp(theta)`` so they stay positive without a
    constrained optimiser. Falls back to the hand-set priors if SciPy is absent
    or the optimisation fails to beat them.

    ``prior`` supplies the starting point and the fallback. It defaults to the
    discipline priors, where the abstract dominates -- which is the right start
    for Agent 1 but a poor one for any task whose evidence lives elsewhere (the
    research design is declared in the methods section, not the abstract).
    Passing a neutral or task-specific prior keeps Nelder-Mead from starting in
    the wrong corner of an eight-dimensional space.
    """
    base_table = dict(prior) if prior else dict(DEFAULT_SECTION_WEIGHTS)
    sections_seen = sorted({s for ss in doc_sections for s in ss})
    if not sections_seen:
        return dict(base_table)
    idx = {s: i for i, s in enumerate(sections_seen)}

    def nll(theta: np.ndarray) -> float:
        table = {s: float(np.exp(theta[idx[s]])) for s in sections_seen}
        total = 0.0
        for p, secs, nw, y in zip(doc_chunk_probs, doc_sections, doc_n_words, labels):
            doc, _ = aggregate(p, secs, nw, strategy, table)
            total -= float(np.log(max(doc[int(y)], EPS)))
        return total / max(1, len(labels))

    theta0 = np.array([np.log(max(base_table.get(s, 1.0), 0.05))
                       for s in sections_seen])
    base = nll(theta0)

    try:
        from scipy.optimize import minimize

        res = minimize(nll, theta0, method="Nelder-Mead",
                       options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-4})
        if res.fun < base:
            fitted = {s: float(np.exp(res.x[idx[s]])) for s in sections_seen}
            if verbose:
                print(f"section weights fitted: NLL {base:.4f} -> {res.fun:.4f}")
            # Normalise so the median weight is 1 — keeps numbers interpretable.
            med = float(np.median(list(fitted.values()))) or 1.0
            fitted = {k: round(v / med, 4) for k, v in fitted.items()}
            out = dict(base_table)
            out.update(fitted)
            return out
        if verbose:
            print(f"section weights: optimiser did not improve on prior "
                  f"({res.fun:.4f} vs {base:.4f}); keeping prior")
    except ImportError:
        if verbose:
            print("scipy unavailable; keeping prior section weights")
    return dict(base_table)


__all__ = [
    "DocumentPrediction",
    "aggregate",
    "predict_document",
    "chunk_weights",
    "normalised_entropy",
    "fit_section_weights",
    "DEFAULT_SECTION_WEIGHTS",
]
