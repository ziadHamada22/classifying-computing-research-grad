"""Build and evaluate a probability-averaged ensemble of trained backbones.

Two independent averaging axes are available and both are cheap once the models
are trained, because each backbone's val/test logits are already saved to
`.npz` by `train.py`:

  seed ensemble       several seeds of one architecture, averaged. Reduces the
                      variance of a single fine-tuning run.
  architecture        different backbones (SciBERT + DeBERTa + ...), averaged.
  ensemble            reduces correlated error because the backbones have
                      different inductive biases and pre-training corpora.

Ensemble weights are optional and, when requested, fitted on val by a small grid
search over the simplex, then applied unchanged to test. Working from saved
logits means this never re-runs a model, so the whole search is seconds.

Run:
    python -m crc.agents.discipline.ensemble --members scibert deberta-v3-base
    python -m crc.agents.discipline.ensemble --members scibert deberta-v3-base --fit-weights
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

from crc.agents.discipline.calibrate import expected_calibration_error, softmax
from crc.taxonomy import DISCIPLINES

PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def load_member(tag: str) -> dict:
    d = MODELS / tag
    t = 1.0
    tpath = d / "temperature.json"
    if tpath.exists():
        t = float(json.loads(tpath.read_text()).get("temperature", 1.0))
    out = {"tag": tag, "temperature": t}
    for split in ("val", "test"):
        npz = np.load(d / f"{split}_predictions.npz")
        out[split] = {"probs": softmax(npz["logits"] / t), "labels": npz["labels"]}
    return out


def simplex_grid(n: int, step: int = 10):
    """All weight vectors on the n-simplex with resolution 1/step."""
    for combo in itertools.combinations_with_replacement(range(n), step):
        w = np.zeros(n)
        for c in combo:
            w[c] += 1
        yield w / w.sum()


def blend(members: list[dict], split: str, weights: np.ndarray) -> np.ndarray:
    acc = None
    for m, w in zip(members, weights):
        p = m[split]["probs"] * w
        acc = p if acc is None else acc + p
    return acc


def macro_f1(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, probs.argmax(1), average="macro"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--fit-weights", action="store_true")
    ap.add_argument("--grid-step", type=int, default=10)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    members = [load_member(t) for t in args.members]
    labels_val = members[0]["val"]["labels"]
    labels_test = members[0]["test"]["labels"]
    for m in members[1:]:
        assert np.array_equal(m["val"]["labels"], labels_val), \
            "members must share the val split ordering"
        assert np.array_equal(m["test"]["labels"], labels_test), \
            "members must share the test split ordering"

    n = len(members)
    print("individual members (val / test macro-F1):")
    for m in members:
        print(f"  {m['tag']:20s} "
              f"{macro_f1(m['val']['probs'], labels_val):.4f} / "
              f"{macro_f1(m['test']['probs'], labels_test):.4f}")

    equal = np.ones(n) / n
    if args.fit_weights and n > 1:
        best_w, best_f1 = equal, -1.0
        for w in simplex_grid(n, args.grid_step):
            f = macro_f1(blend(members, "val", w), labels_val)
            if f > best_f1:
                best_f1, best_w = f, w
        weights = best_w
        print(f"\nfitted weights (val macro-F1 {best_f1:.4f}): "
              + ", ".join(f"{m['tag']}={w:.2f}" for m, w in zip(members, weights)))
    else:
        weights = equal
        print(f"\nequal weights: {1/n:.3f} each")

    val_probs = blend(members, "val", weights)
    test_probs = blend(members, "test", weights)
    tag = args.tag or "ensemble_" + "_".join(m["tag"] for m in members)

    report = {
        "tag": tag,
        "members": [m["tag"] for m in members],
        "weights": {m["tag"]: round(float(w), 4) for m, w in zip(members, weights)},
        "val": {
            "macro_f1": round(macro_f1(val_probs, labels_val), 4),
            "accuracy": round(float((val_probs.argmax(1) == labels_val).mean()), 4),
            "ece": round(expected_calibration_error(val_probs, labels_val), 4),
        },
        "test": {
            "macro_f1": round(macro_f1(test_probs, labels_test), 4),
            "accuracy": round(float((test_probs.argmax(1) == labels_test).mean()), 4),
            "ece": round(expected_calibration_error(test_probs, labels_test), 4),
        },
    }
    best_single = max(macro_f1(m["test"]["probs"], labels_test) for m in members)
    report["test"]["lift_over_best_single"] = round(
        report["test"]["macro_f1"] - best_single, 4)

    print(f"\n=== {tag} ===")
    print(f"  val  macro-F1 {report['val']['macro_f1']:.4f}  "
          f"ECE {report['val']['ece']:.4f}")
    print(f"  test macro-F1 {report['test']['macro_f1']:.4f}  "
          f"ECE {report['test']['ece']:.4f}  "
          f"(lift over best single: {report['test']['lift_over_best_single']:+.4f})")
    print("\n" + classification_report(
        [DISCIPLINES[i] for i in labels_test],
        [DISCIPLINES[i] for i in test_probs.argmax(1)],
        labels=DISCIPLINES, digits=4, zero_division=0))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
