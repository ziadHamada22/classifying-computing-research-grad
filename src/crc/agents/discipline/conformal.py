"""Conformal prediction — calibrated uncertainty to replace hand-tuned thresholds.

The classifier has always had to answer a second question alongside "which
discipline?": *how much should anyone trust this answer?* Until now that was
decided by two hand-swept constants (``LOW_CONFIDENCE``, ``TIGHT_GAP``), which
have three measured problems:

1. **It carries no guarantee.** "Top class below 0.55" is a heuristic; nothing
   ties it to an error rate you can state in advance.
2. **It does not transfer across time.** A pair fitted on the iid val split fires
   on 31.2% of 2025 papers against a 25% budget while catching 7 points *fewer*
   errors, because ECE nearly triples out of period (0.042 -> 0.100).
3. **It answers the wrong question.** It returns a yes/no flag, so the pipeline
   falls back to "report the top-2 disciplines". Two is arbitrary: some papers are
   a genuine three-way tie, most confident ones need one.

Split conformal prediction fixes all three. Given a calibration set and a target
miscoverage ``alpha``, it returns a **prediction set** whose probability of
containing the true label is at least ``1 - alpha`` — a distribution-free,
finite-sample guarantee that assumes only exchangeability between calibration and
test data. The set is read as:

    |set| == 1   ->  the single label is certified at 1 - alpha
    |set| >= 2   ->  genuinely contested, and the set names *which* disciplines
                     are live (an adaptive replacement for a fixed "top-2")
    |set| == 0   ->  not even the top label clears the bar at this alpha

**The set never withholds an answer.** Each agent always commits to its top-1
label and works standalone; the set is calibrated *context* attached to that
answer, so nothing downstream depends on a reviewer or a language model being
present. A local LLM may later consult the set on contested documents only, but it
is strictly optional and no part of the classification path.

Crucially, the exchangeability assumption is exactly what temporal drift violates.
So instead of the drift silently degrading a threshold, it shows up as *measured
under-coverage* — the failure becomes diagnosable, and the fix (calibrate on a
temporal slice) is principled rather than a re-tune.

**Two score functions** are implemented, and the choice is deliberate:

``lac`` (a.k.a. THR / "least ambiguous set-valued classifier")
    ``s(x, y) = 1 - p(y|x)``. Provably the smallest average set size at a given
    coverage. This is the default.
``aps`` (adaptive prediction sets, Romano et al. 2020)
    Cumulative mass down to the true label, with the standard uniform
    randomisation so coverage is exact rather than conservative. Better
    *conditional* coverage, at the cost of larger sets.

**Class-conditional (Mondrian) calibration** is offered for both: a separate
quantile per class, which upgrades the marginal guarantee to a per-class one.
That matters here because marginal coverage can be met while the weakest class
quietly under-covers — and Computer Science is measurably Agent 1's weakest class
(recall 0.742), the one whose errors cost the most downstream.

RAPS is deliberately *not* implemented: its regularisation exists to stop set
sizes exploding over hundreds or thousands of labels, and with six disciplines
LAC's sets are already near-minimal.

Everything works from saved probabilities, so it never re-runs a model.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from crc.taxonomy import DISCIPLINES

#: Score functions available to :func:`calibrate`.
METHODS = ("lac", "aps")


def lac_score_matrix(probs: np.ndarray) -> np.ndarray:
    """``s[i, k] = 1 - p(k | x_i)`` — the LAC/THR nonconformity score."""
    return 1.0 - np.asarray(probs, dtype=np.float64)


def aps_score_matrix(probs: np.ndarray,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """APS score: cumulative probability mass down to each candidate label.

    ``s[i, k]`` is the mass of all labels ranked above ``k`` plus ``k``'s own
    share. With ``rng`` supplied, ``k``'s own mass is scaled by U(0,1) — the
    randomisation from Romano et al. (2020) that makes coverage exact instead of
    conservative. Without it the score is the deterministic upper variant.
    """
    p = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-p, axis=1)
    p_sorted = np.take_along_axis(p, order, axis=1)
    csum = np.cumsum(p_sorted, axis=1)
    if rng is not None:
        # exclusive cumulative mass + U * own mass
        s_sorted = csum - p_sorted * rng.random(size=p.shape)
    else:
        s_sorted = csum
    s = np.empty_like(s_sorted)
    np.put_along_axis(s, order, s_sorted, axis=1)
    return s


def score_matrix(probs: np.ndarray, method: str = "lac",
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Nonconformity score of every (sample, label) pair.

    Unifying both methods as a score *matrix* is what keeps set construction and
    calibration provably consistent: calibration reads the true label's column,
    and the prediction set is exactly ``{k : s[i, k] <= qhat}``. Any other set
    rule would break the guarantee the quantile was chosen to provide.
    """
    if method == "lac":
        return lac_score_matrix(probs)
    if method == "aps":
        return aps_score_matrix(probs, rng)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected ``1 - alpha`` quantile of calibration scores.

    Uses the ``ceil((n + 1)(1 - alpha)) / n`` level rather than the plain
    ``1 - alpha``; that correction is what turns an empirical quantile into a
    valid coverage guarantee. When the calibration set is too small to support
    the requested alpha the honest answer is an infinite threshold — every label
    is included — rather than a silently invalid one.
    """
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(s, level, method="higher"))


@dataclass
class ConformalCalibration:
    """A fitted conformal predictor: the score rule plus its threshold(s)."""

    method: str
    alpha: float
    class_conditional: bool
    #: Marginal threshold (used when ``class_conditional`` is False).
    qhat: float | None = None
    #: Per-class thresholds, indexed like :data:`crc.taxonomy.DISCIPLINES`.
    qhat_per_class: list[float] | None = None
    n_calib: int = 0
    n_calib_per_class: list[int] = field(default_factory=list)
    calib_source: str = ""
    seed: int = 0

    # -- thresholds as an array, whichever mode we are in -------------------
    def thresholds(self, n_classes: int) -> np.ndarray:
        if self.class_conditional:
            if self.qhat_per_class is None:
                raise ValueError("class-conditional calibration has no per-class qhat")
            return np.asarray(self.qhat_per_class, dtype=np.float64)
        if self.qhat is None:
            raise ValueError("marginal calibration has no qhat")
        return np.full(n_classes, float(self.qhat), dtype=np.float64)

    def predict_set(self, probs: np.ndarray,
                    rng: np.random.Generator | None = None) -> np.ndarray:
        """Boolean mask ``(n, K)``: is label ``k`` in the prediction set?"""
        probs = np.atleast_2d(np.asarray(probs, dtype=np.float64))
        if self.method == "aps" and rng is None:
            rng = np.random.default_rng(self.seed)
        s = score_matrix(probs, self.method, rng)
        return s <= self.thresholds(probs.shape[1])[None, :]

    def route(self, probs: np.ndarray,
              rng: np.random.Generator | None = None) -> np.ndarray:
        """True where the label is *not* certified alone (``|set| != 1``).

        A flag for how to present or review the answer — never a reason to
        suppress it; the top-1 label stands either way.
        """
        return self.predict_set(probs, rng).sum(axis=1) != 1

    def set_labels(self, probs: np.ndarray,
                   labels: list[str] | None = None,
                   rng: np.random.Generator | None = None) -> list[list[str]]:
        """Prediction sets as label names, for a human-readable audit trail."""
        names = list(labels or DISCIPLINES)
        mask = self.predict_set(probs, rng)
        return [[names[k] for k in np.where(row)[0]] for row in mask]

    # -- persistence --------------------------------------------------------
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ConformalCalibration":
        return cls(**json.loads(Path(path).read_text()))


def calibrate(probs: np.ndarray, labels: np.ndarray, alpha: float = 0.1,
              method: str = "lac", class_conditional: bool = False,
              calib_source: str = "", seed: int = 0) -> ConformalCalibration:
    """Fit a conformal predictor on held-out calibration data.

    ``probs`` must come from data the model did not train on, and must be
    exchangeable with the data it will be applied to — that second condition is
    the one temporal drift breaks, which is why the calibration source is
    recorded on the returned object.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    probs = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    labels = np.asarray(labels).astype(int)
    n, k = probs.shape
    rng = np.random.default_rng(seed) if method == "aps" else None
    s = score_matrix(probs, method, rng)
    true_scores = s[np.arange(n), labels]

    cal = ConformalCalibration(
        method=method, alpha=float(alpha),
        class_conditional=bool(class_conditional),
        n_calib=int(n), calib_source=calib_source, seed=int(seed),
    )
    if class_conditional:
        qs, counts = [], []
        for c in range(k):
            m = labels == c
            counts.append(int(m.sum()))
            qs.append(conformal_quantile(true_scores[m], alpha))
        cal.qhat_per_class = qs
        cal.n_calib_per_class = counts
    else:
        cal.qhat = conformal_quantile(true_scores, alpha)
        cal.n_calib_per_class = [int((labels == c).sum()) for c in range(k)]
    return cal


def evaluate_sets(cal: ConformalCalibration, probs: np.ndarray,
                  labels: np.ndarray,
                  rng: np.random.Generator | None = None,
                  class_names: list[str] | None = None) -> dict:
    """Coverage, set size and routing behaviour of a fitted predictor.

    ``coverage`` is the quantity the guarantee is about: it should land at or just
    above ``1 - alpha`` on exchangeable data. Falling below is the signature of
    an exchangeability violation (drift), not of a coding error.
    """
    probs = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    labels = np.asarray(labels).astype(int)
    n, k = probs.shape
    mask = cal.predict_set(probs, rng)
    sizes = mask.sum(axis=1)
    covered = mask[np.arange(n), labels]

    top1 = probs.argmax(axis=1)
    wrong = top1 != labels
    routed = sizes != 1
    singleton = sizes == 1

    per_class = {}
    for c in range(k):
        m = labels == c
        if not m.any():
            continue
        per_class[(class_names or DISCIPLINES)[c]] = {
            "n": int(m.sum()),
            "coverage": round(float(covered[m].mean()), 4),
            "avg_set_size": round(float(sizes[m].mean()), 4),
            "routed_share": round(float(routed[m].mean()), 4),
        }

    return {
        "n": int(n),
        "alpha": cal.alpha,
        "target_coverage": round(1.0 - cal.alpha, 4),
        "coverage": round(float(covered.mean()), 4),
        "coverage_gap": round(float(covered.mean() - (1.0 - cal.alpha)), 4),
        "avg_set_size": round(float(sizes.mean()), 4),
        "median_set_size": int(np.median(sizes)),
        "size_distribution": {int(v): int((sizes == v).sum())
                              for v in range(k + 1) if (sizes == v).any()},
        "routed_share": round(float(routed.mean()), 4),
        "singleton_share": round(float(singleton.mean()), 4),
        "empty_share": round(float((sizes == 0).mean()), 4),
        # Accuracy among the papers the predictor certifies with one label —
        # the number that justifies committing without a second opinion.
        "singleton_accuracy": (round(float((top1 == labels)[singleton].mean()), 4)
                               if singleton.any() else None),
        # Routing viewed as an error detector, comparable to the hand-tuned
        # thresholds: of the top-1 errors, how many did we send for review?
        "error_recall": (round(float(routed[wrong].mean()), 4)
                         if wrong.any() else None),
        "error_precision": (round(float(wrong[routed].mean()), 4)
                            if routed.any() else None),
        "top1_accuracy": round(float((top1 == labels).mean()), 4),
        "per_class": per_class,
    }



# ---------------------------------------------------------------- alpha banks
# alpha is an operating choice, not a property of the model: it trades how often
# the system hedges against how often the hedge contains the truth. Storing one
# quantile per alpha means that choice can be revisited at any time without
# re-running calibration, and the deployed default can move with a config edit
# rather than a retraining job.


def save_bank(path: str | Path, cals: dict[float, ConformalCalibration]) -> Path:
    """Write several fitted predictors (one per alpha) to a single file."""
    if not cals:
        raise ValueError("nothing to save")
    first = next(iter(cals.values()))
    doc = {
        "format": "conformal-bank-1",
        "method": first.method,
        "class_conditional": first.class_conditional,
        "calib_source": first.calib_source,
        "n_calib": first.n_calib,
        "alphas": {f"{a:g}": asdict(c) for a, c in sorted(cals.items())},
    }
    pth = Path(path)
    pth.parent.mkdir(parents=True, exist_ok=True)
    pth.write_text(json.dumps(doc, indent=2))
    return pth


def load_bank(path: str | Path) -> dict[float, ConformalCalibration]:
    """Read a bank, or a single-calibration file, into ``{alpha: predictor}``."""
    raw = json.loads(Path(path).read_text())
    if raw.get("format") == "conformal-bank-1":
        return {float(a): ConformalCalibration(**c)
                for a, c in raw["alphas"].items()}
    cal = ConformalCalibration(**raw)          # legacy single-alpha file
    return {cal.alpha: cal}


def pick_alpha(bank: dict[float, ConformalCalibration],
               alpha: float) -> ConformalCalibration:
    """The predictor for ``alpha``, or the closest one available.

    Falling back to the nearest alpha keeps inference working when a config asks
    for an alpha nobody calibrated, instead of failing shut on a live document.
    """
    if not bank:
        raise ValueError("empty conformal bank")
    if alpha in bank:
        return bank[alpha]
    return bank[min(bank, key=lambda a: abs(a - alpha))]


__all__ = [
    "METHODS",
    "ConformalCalibration",
    "aps_score_matrix",
    "calibrate",
    "conformal_quantile",
    "evaluate_sets",
    "lac_score_matrix",
    "load_bank",
    "pick_alpha",
    "save_bank",
    "score_matrix",
]
