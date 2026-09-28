"""Hierarchy-aware helpers shared by Agent 1, the pipeline and the evaluation.

Agent 2's head is a flat distribution over all 38 fields, and every field
belongs to exactly one discipline. Summing its field probabilities within each
discipline therefore gives a *second discipline distribution*, learned from a
different corpus (111,917 papers) with a different objective. Nothing in the
original design used it: the pipeline let Agent 1 choose the ballot and only then
asked Agent 2 which field on that ballot fits.

This module holds the arithmetic for using that signal and for scoring the
hierarchy as a whole:

``field_to_discipline_marginal``
    ``(n, 38)`` field probabilities -> ``(n, 6)`` discipline probabilities.
``joint_discipline_probs``
    A weighted geometric pool of Agent 1's distribution and that marginal
    (a product of experts). ``weight = 0`` is Agent 1 unchanged.
``hierarchical_scores``
    Hierarchical precision / recall / F1 in the sense of Kiritchenko et al.:
    each prediction is expanded to its ancestor set, so naming the right
    discipline with the wrong field earns half credit and a wrong discipline
    earns none.
"""
from __future__ import annotations

import numpy as np

from crc.taxonomy import DISCIPLINES
from crc.taxonomy.fields import DISCIPLINE_FIELD_IDS

_EPS = 1e-9


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def field_to_discipline_marginal(field_probs: np.ndarray) -> np.ndarray:
    """Sum each discipline's field probabilities: ``(n, 38) -> (n, 6)``.

    Columns follow ``DISCIPLINES`` order, like every other discipline
    distribution in the system. Rows sum to one when the input rows do.
    """
    fp = np.atleast_2d(np.asarray(field_probs, dtype=np.float64))
    out = np.zeros((fp.shape[0], len(DISCIPLINES)))
    for j, d in enumerate(DISCIPLINES):
        out[:, j] = fp[:, DISCIPLINE_FIELD_IDS[d]].sum(axis=1)
    return out


def joint_discipline_probs(p_disc: np.ndarray, p_marginal: np.ndarray,
                           weight: float) -> np.ndarray:
    """Weighted geometric pool: ``p ~ p_disc^(1-w) * p_marginal^w``.

    Geometric rather than arithmetic because the two experts were trained on
    different corpora with different label noise: a product lets either expert
    veto a discipline it finds implausible, where an average would let a
    confident-but-wrong expert drag the answer. ``weight`` is fitted on
    validation data, never on the data it is reported on.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    a = np.atleast_2d(np.asarray(p_disc, dtype=np.float64))
    b = np.atleast_2d(np.asarray(p_marginal, dtype=np.float64))
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if weight == 0.0:
        return a / a.sum(axis=1, keepdims=True)
    logp = (1.0 - weight) * np.log(a + _EPS) + weight * np.log(b + _EPS)
    return softmax(logp, axis=1)


def conditioned_field(field_logits: np.ndarray, disciplines) -> np.ndarray:
    """Global field id of the best field inside each row's given discipline."""
    fl = np.atleast_2d(np.asarray(field_logits))
    out = np.empty(len(fl), dtype=int)
    for i, d in enumerate(disciplines):
        allowed = DISCIPLINE_FIELD_IDS[d]
        out[i] = allowed[int(np.argmax(fl[i, allowed]))]
    return out


def hierarchical_scores(true_disc, pred_disc, true_field, pred_field) -> dict:
    """Hierarchical P / R / F1 over the discipline -> field tree.

    Every paper has exactly one true path and one predicted path of depth two,
    so the expanded label sets always have size two and hierarchical precision
    equals hierarchical recall. The F1 is then the mean of the two levels'
    accuracies -- reported explicitly because the proposal promised it, and
    because it credits a right-discipline/wrong-field answer that flat field
    accuracy scores as a plain miss.
    """
    td, pd_ = np.asarray(true_disc), np.asarray(pred_disc)
    tf, pf = np.asarray(true_field), np.asarray(pred_field)
    disc_hit = td == pd_
    # A field belongs to one discipline, so a field hit implies a discipline hit.
    field_hit = (tf == pf) & disc_hit
    overlap = disc_hit.astype(float) + field_hit.astype(float)
    hp = float(overlap.sum() / (2 * len(td))) if len(td) else 0.0
    return {
        "hierarchical_precision": round(hp, 4),
        "hierarchical_recall": round(hp, 4),
        "hierarchical_f1": round(hp, 4),
        "level_accuracy": {"discipline": round(float(disc_hit.mean()), 4),
                           "field": round(float(field_hit.mean()), 4)},
        "partial_credit_share": round(float((disc_hit & ~field_hit).mean()), 4),
    }


__all__ = ["softmax", "field_to_discipline_marginal", "joint_discipline_probs",
           "conditioned_field", "hierarchical_scores"]
