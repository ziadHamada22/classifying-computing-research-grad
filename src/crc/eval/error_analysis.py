"""Decompose Agent 1's residual error into defensible vs genuine mistakes.

The strict metric caps at ~0.82 on corpus v2 because a large share of the test
set is genuinely multi-disciplinary. This script answers the question that
number cannot: *when the system is "wrong", how wrong is it?* It splits every
error into three kinds, from most to least excusable:

  co-listed      the predicted discipline appears among those the paper's own
                 arXiv categories point to -- the authors filed it there too
  adjacent       predicted a discipline that is the declared `secondary` of one
                 of the paper's categories -- a defensible neighbour
  genuine        neither -- a real confusion worth investigating

It runs from saved logits (any single model or a fitted ensemble), so no GPU and
no re-inference. The output is the material for the evaluation chapter's
"honest error" discussion.

Run:
    python -m crc.eval.error_analysis --members scibert deberta-v3-base tfidf
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from crc.agents.discipline.calibrate import softmax
from crc.taxonomy import BY_CATEGORY, DISCIPLINES, LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS = WORK / "corpus" / "corpus_v2.parquet"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def load_probs(tag: str, split: str):
    d = MODELS / tag
    t = 1.0
    tp = d / "temperature.json"
    if tp.exists():
        t = float(json.loads(tp.read_text()).get("temperature", 1.0))
    npz = np.load(d / f"{split}_predictions.npz")
    return softmax(npz["logits"] / t), npz["labels"]


def primary_and_secondary(cats) -> tuple[set[str], set[str]]:
    prim, sec = set(), set()
    if cats is None:
        return prim, sec
    for c in list(cats):
        m = BY_CATEGORY.get(c)
        if m:
            prim.add(m.discipline)
            if m.secondary:
                sec.add(m.secondary)
    return prim, sec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    probs, labels = None, None
    w = args.weights or [1.0] * len(args.members)
    w = np.array(w) / sum(w)
    for tag, wi in zip(args.members, w):
        p, l = load_probs(tag, args.split)
        labels = l if labels is None else labels
        probs = p * wi if probs is None else probs + p * wi

    df = pd.read_parquet(CORPUS)
    df = df[df["split"] == args.split].reset_index(drop=True)
    assert len(df) == len(labels), (len(df), len(labels))
    assert np.array_equal(df["discipline"].map(LABEL2ID).to_numpy(), labels)

    pred = probs.argmax(1)
    pred_name = [DISCIPLINES[i] for i in pred]
    gold_name = [DISCIPLINES[i] for i in labels]
    wrong = pred != labels

    kinds = Counter()
    genuine_pairs = Counter()
    for i in range(len(df)):
        if not wrong[i]:
            kinds["correct"] += 1
            continue
        prim, sec = primary_and_secondary(df.iloc[i]["mapped_categories"])
        if pred_name[i] in prim:
            kinds["co_listed"] += 1
        elif pred_name[i] in sec:
            kinds["adjacent"] += 1
        else:
            kinds["genuine"] += 1
            genuine_pairs[(gold_name[i], pred_name[i])] += 1

    n = len(df)
    n_wrong = int(wrong.sum())
    print(f"split={args.split}  n={n:,}  members={args.members}")
    print(f"strict accuracy: {1 - n_wrong / n:.4f}   ({n_wrong:,} errors)\n")

    print("error decomposition (share of ALL papers / share of ERRORS):")
    for k in ("co_listed", "adjacent", "genuine"):
        c = kinds[k]
        print(f"  {k:10s} {c:>5,}   {c/n:6.1%} of all   {c/max(1,n_wrong):6.1%} of errors")

    defensible = kinds["co_listed"] + kinds["adjacent"]
    print(f"\n  defensible (co-listed + adjacent): {defensible:,} "
          f"= {defensible/max(1,n_wrong):.1%} of all errors")
    print(f"  genuine confusion:                 {kinds['genuine']:,} "
          f"= {kinds['genuine']/max(1,n_wrong):.1%} of all errors")
    print(f"\n  strict accuracy      {1-n_wrong/n:.4f}")
    print(f"  crediting defensible {1-kinds['genuine']/n:.4f}  "
          f"(+{defensible/n:.4f})")

    print("\ntop genuine-confusion pairs (gold -> pred):")
    for (g, p), c in genuine_pairs.most_common(10):
        print(f"  {g:24s} -> {p:24s} {c:>4d}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = "_".join(args.members)
    out = RESULTS / f"error_analysis_{tag}_{args.split}.json"
    out.write_text(json.dumps({
        "members": args.members, "split": args.split, "n": n,
        "n_errors": n_wrong,
        "decomposition": dict(kinds),
        "strict_accuracy": round(1 - n_wrong / n, 4),
        "defensible_credited_accuracy": round(1 - kinds["genuine"] / n, 4),
        "top_genuine_pairs": [
            {"gold": g, "pred": p, "count": c}
            for (g, p), c in genuine_pairs.most_common(15)],
    }, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
