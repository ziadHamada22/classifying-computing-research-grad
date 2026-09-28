"""Evaluate Agent 1 honestly, on accuracy and on time.

Three accuracy numbers are reported side by side rather than one headline,
because a single figure would hide what the v1 diagnosis exposed:

  strict          exact match against the single v2 consensus label
  co-listed       credits a prediction that appears among the disciplines the
                  paper's own arXiv categories point to. Under v1, 24% of
                  "errors" were of this kind — the model picked a discipline the
                  authors themselves had filed the paper under. Reported
                  separately and never as the headline, because it is a
                  strictly easier metric.
  temporal        the 2025+ holdout, which no training paper predates. This is
                  the number that says whether the system works on research
                  published after it was built.

Plus a breakdown over the ambiguous test papers, which train/val never saw.

Run:
    python -m crc.eval.evaluate --model-dir models/scibert
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from crc.agents.discipline.calibrate import expected_calibration_error, softmax
from crc.agents.discipline.features import build_text_column
from crc.taxonomy import BY_CATEGORY, DISCIPLINES, LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def allowed_disciplines(mapped_categories) -> set[str]:
    """Disciplines the paper's own arXiv categories point to."""
    out = set()
    # Parquet list columns arrive as numpy arrays, whose truth value is
    # ambiguous -- so test for None explicitly rather than with `or []`.
    if mapped_categories is None:
        return out
    for c in list(mapped_categories):
        m = BY_CATEGORY.get(c)
        if m:
            out.add(m.discipline)
            if m.secondary:
                out.add(m.secondary)
    return out


def batched_logits(model, tok, texts: list[str], device: str,
                   max_length: int = 256, batch_size: int = 64
                   ) -> tuple[np.ndarray, float]:
    import torch

    out = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True,
                      max_length=max_length, padding=True,
                      return_tensors="pt").to(device)
            out.append(model(**enc).logits.float().cpu().numpy())
    elapsed = time.perf_counter() - t0
    return np.concatenate(out, axis=0), elapsed


def evaluate_split(df: pd.DataFrame, probs: np.ndarray, name: str) -> dict:
    y_true = df["discipline"].map(LABEL2ID).to_numpy()
    y_pred = probs.argmax(axis=1)
    pred_names = [DISCIPLINES[i] for i in y_pred]

    strict_macro = f1_score(y_true, y_pred, average="macro")
    strict_acc = float((y_pred == y_true).mean())

    allowed = df["mapped_categories"].map(allowed_disciplines)
    co_listed_hit = np.array([
        p in a or p == t for p, a, t in zip(pred_names, allowed, df["discipline"])
    ])

    res = {
        "split": name,
        "n": int(len(df)),
        "strict_accuracy": round(strict_acc, 4),
        "strict_macro_f1": round(float(strict_macro), 4),
        "strict_weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "co_listed_accuracy": round(float(co_listed_hit.mean()), 4),
        "ece": round(expected_calibration_error(probs, y_true), 4),
        "mean_confidence": round(float(probs.max(axis=1).mean()), 4),
        "report": classification_report(
            [DISCIPLINES[i] for i in y_true], pred_names,
            labels=DISCIPLINES, digits=4, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred,
                                      labels=list(range(len(DISCIPLINES)))).tolist(),
    }

    if "ambiguous" in df.columns and df["ambiguous"].any():
        for flag, tag in ((False, "unambiguous"), (True, "ambiguous")):
            m = (df["ambiguous"] == flag).to_numpy()
            if m.sum() == 0:
                continue
            res[tag] = {
                "n": int(m.sum()),
                "accuracy": round(float((y_pred[m] == y_true[m]).mean()), 4),
                "macro_f1": round(float(f1_score(y_true[m], y_pred[m],
                                                 average="macro")), 4),
                "co_listed_accuracy": round(float(co_listed_hit[m].mean()), 4),
            }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-temperature", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    d = Path(args.model_dir)
    if not d.is_absolute() and not d.exists():
        d = MODELS / d.name
    tag = d.name

    temperature = 1.0
    tpath = d / "temperature.json"
    if tpath.exists() and not args.no_temperature:
        temperature = float(json.loads(tpath.read_text()).get("temperature", 1.0))
    print(f"model {tag}  |  T = {temperature:.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(str(d)).to(device).eval()
    tok = AutoTokenizer.from_pretrained(str(d))

    corpus = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet")
    temporal = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet")

    splits = {
        "test": corpus[corpus["split"] == "test"].reset_index(drop=True),
        "temporal_test": temporal.reset_index(drop=True),
    }

    report: dict = {"model_dir": str(d), "tag": tag, "temperature": temperature,
                    "device": device, "splits": {}}

    for name, df in splits.items():
        texts = build_text_column(df).tolist()
        logits, elapsed = batched_logits(model, tok, texts, device,
                                         args.max_length, args.batch_size)
        probs = softmax(logits / temperature)
        res = evaluate_split(df, probs, name)
        res["latency"] = {
            "total_seconds": round(elapsed, 2),
            "docs_per_second": round(len(df) / elapsed, 1),
            "ms_per_doc": round(1000 * elapsed / len(df), 2),
            "batch_size": args.batch_size,
        }
        report["splits"][name] = res

        print(f"\n===== {name}  (n={len(df):,}) =====")
        print(f"  strict macro-F1   {res['strict_macro_f1']:.4f}")
        print(f"  strict accuracy   {res['strict_accuracy']:.4f}")
        print(f"  co-listed acc     {res['co_listed_accuracy']:.4f}")
        print(f"  ECE               {res['ece']:.4f}")
        print(f"  throughput        {res['latency']['docs_per_second']:.1f} docs/s "
              f"({res['latency']['ms_per_doc']:.2f} ms/doc)")
        for tagname in ("unambiguous", "ambiguous"):
            if tagname in res:
                r = res[tagname]
                print(f"  {tagname:12s} n={r['n']:>5,} acc={r['accuracy']:.4f} "
                      f"co-listed={r['co_listed_accuracy']:.4f}")
        print(res["report"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"eval_{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
