"""Temperature scaling for Agent 1 (Guo et al., 2017).

A confidence score is only useful if it is honest: a paper predicted at 0.90
should be right about 90% of the time. Fine-tuned transformer classifiers are
systematically over-confident, and this project's whole cost-aware design leans
on confidence — the borderline detector decides when to spend a local-LLM call,
and a curator decides which papers to audit. Both are meaningless if the
confidences are inflated.

A single scalar T is fitted on the *validation* split by minimising NLL on
logits / T. Because argmax is invariant under positive rescaling, accuracy and
F1 are unchanged; only the spread of the distribution is corrected.

Run:
    python -m crc.agents.discipline.calibrate --model-dir models/scibert
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> float:
    """Standard binned ECE over the top-class confidence."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    max_iter: int = 200, verbose: bool = True) -> float:
    """Fit T by minimising NLL of softmax(logits / T) with LBFGS."""
    import torch

    lg = torch.tensor(logits, dtype=torch.float32)
    ll = torch.tensor(labels, dtype=torch.long)
    # Optimise log T so T stays strictly positive.
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(lg / torch.exp(log_t), ll)
        loss.backward()
        return loss

    before = float(nll(lg, ll))
    opt.step(closure)
    t = float(torch.exp(log_t).item())
    after = float(nll(lg / t, ll))
    if verbose:
        print(f"  NLL {before:.4f} -> {after:.4f}   T = {t:.4f}")
    if after > before or not np.isfinite(t) or t <= 0:
        if verbose:
            print("  optimiser did not improve NLL; falling back to T = 1.0")
        return 1.0
    return t


def reliability_bins(probs: np.ndarray, labels: np.ndarray,
                     n_bins: int = 15) -> list[dict]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append({
            "lo": round(float(lo), 3), "hi": round(float(hi), 3),
            "n": int(m.sum()),
            "confidence": round(float(conf[m].mean()), 4) if m.any() else None,
            "accuracy": round(float(correct[m].mean()), 4) if m.any() else None,
        })
    return out


def adopt_temperature(val_raw: float, val_fitted: float,
                      guard_raw: float | None = None,
                      guard_fitted: float | None = None) -> bool:
    """Adopt a fitted temperature only if it lowers ECE where it will be used.

    It must improve validation ECE and -- when a composition-matched slice
    (the 2025+ calibration half) is available -- that slice's ECE too.
    """
    if not val_fitted < val_raw:
        return False
    if guard_raw is not None and guard_fitted is not None:
        return guard_fitted < guard_raw
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--bins", type=int, default=15)
    args = ap.parse_args()

    d = Path(args.model_dir)
    if not d.is_absolute():
        d = MODELS / d.name if (MODELS / d.name).exists() else d

    val = np.load(d / "val_predictions.npz")
    test = np.load(d / "test_predictions.npz")

    print(f"fitting temperature on {len(val['labels']):,} val examples")
    t_fit = fit_temperature(val["logits"], val["labels"])

    # Guo et al. fit T by minimising NLL, but the NLL optimum is not always the
    # ECE optimum -- and a model that is already calibrated can be made worse by
    # rescaling it. Only adopt T if it actually improves calibration on val;
    # otherwise report the fitted value and ship T = 1.0.
    val_probs_raw = softmax(val["logits"])
    val_probs_cal = softmax(val["logits"] / t_fit)
    ece_raw_val = expected_calibration_error(val_probs_raw, val["labels"], args.bins)
    ece_cal_val = expected_calibration_error(val_probs_cal, val["labels"], args.bins)

    # Second condition (added 2026-09): validation holds no ambiguous papers,
    # real input does. A T that helps on val can hurt where it matters -- the
    # soft-label model's val-fitted T = 0.82 roughly doubled its 2025+ ECE. So T
    # must also help on the composition-matched slice the conformal bank is
    # calibrated on (the seed-42 half of the 2025+ holdout), when it exists.
    guard = None
    tpath = d / "temporal_predictions.npz"
    if tpath.exists():
        tz = np.load(tpath)
        idx = np.random.default_rng(42).permutation(len(tz["labels"]))
        ci = idx[: len(idx) // 2]
        g_raw = expected_calibration_error(softmax(tz["logits"][ci]),
                                           tz["labels"][ci], args.bins)
        g_cal = expected_calibration_error(softmax(tz["logits"][ci] / t_fit),
                                           tz["labels"][ci], args.bins)
        guard = {"slice": "2025+ calibration half (seed 42)",
                 "ece_raw": round(g_raw, 4), "ece_fitted": round(g_cal, 4)}

    applied = adopt_temperature(ece_raw_val, ece_cal_val,
                                None if guard is None else guard["ece_raw"],
                                None if guard is None else guard["ece_fitted"])
    t = t_fit if applied else 1.0
    if applied:
        print(f"  adopting T = {t:.4f} (val ECE {ece_raw_val:.4f} -> {ece_cal_val:.4f})")
    elif ece_cal_val < ece_raw_val:
        print(f"  REJECTING T = {t_fit:.4f}: it improves val ECE "
              f"({ece_raw_val:.4f} -> {ece_cal_val:.4f}) but worsens the 2025+ "
              f"calibration half ({guard['ece_raw']:.4f} -> {guard['ece_fitted']:.4f}),"
              f" whose composition matches real input. Shipping T = 1.0.")
    else:
        print(f"  REJECTING T = {t_fit:.4f}: it worsens val ECE "
              f"({ece_raw_val:.4f} -> {ece_cal_val:.4f}). Shipping T = 1.0 — "
              f"this model is already calibrated.")

    report = {
        "temperature": t,
        "temperature_fitted": t_fit,
        "temperature_applied": bool(applied),
        "val_ece_raw": round(ece_raw_val, 4),
        "val_ece_fitted": round(ece_cal_val, 4),
        "composition_guard": guard,
        "model_dir": str(d),
    }
    for name, npz in (("val", val), ("test", test)):
        logits, labels = npz["logits"], npz["labels"]
        raw = softmax(logits)
        cal = softmax(logits / t)
        acc = float((raw.argmax(1) == labels).mean())
        ece_raw = expected_calibration_error(raw, labels, args.bins)
        ece_cal = expected_calibration_error(cal, labels, args.bins)
        report[name] = {
            "n": int(len(labels)),
            "accuracy": round(acc, 4),
            "ece_raw": round(ece_raw, 4),
            "ece_calibrated": round(ece_cal, 4),
            "ece_reduction": round(1 - ece_cal / ece_raw, 4) if ece_raw else 0.0,
            "mean_confidence_raw": round(float(raw.max(1).mean()), 4),
            "mean_confidence_calibrated": round(float(cal.max(1).mean()), 4),
            "reliability": reliability_bins(cal, labels, args.bins),
        }
        print(f"  {name:5s} n={len(labels):>6,} acc={acc:.4f}  "
              f"ECE {ece_raw:.4f} -> {ece_cal:.4f}  "
              f"(mean conf {raw.max(1).mean():.3f} -> {cal.max(1).mean():.3f})")

    (d / "temperature.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {d / 'temperature.json'}")


if __name__ == "__main__":
    main()
