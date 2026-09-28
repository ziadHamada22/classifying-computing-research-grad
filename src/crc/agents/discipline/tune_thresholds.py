"""Tune the borderline thresholds against the actual v2 document distribution.

The borderline detector decides when to spend a local-LLM second opinion. Its
two constants — LOW_CONFIDENCE and TIGHT_GAP — were inherited from the prototype,
where they were set for a single model's confidence range. Two things changed:

  * document-level aggregation produces a *sharper* distribution than any single
    chunk, so a gap that was "tight" for one chunk is routine after pooling;
  * calibration drifts over time (ECE 0.042 iid -> 0.100 temporal), so a
    threshold fitted on the iid val split misfires on new papers.

These thresholds are now only the *fallback* for a scorer with no conformal
calibration (`conformal.py` is the primary route), but the fallback still runs --
for example in the pipeline's ensemble mode -- so it must use tuned values. The
result is written to ``<model>/thresholds.json``, which
``DisciplineClassifier.load`` reads.

This script sweeps both thresholds on validation and picks the pair that
captures the most *misclassified* papers (the ones worth a second opinion) while
firing on no more than a target fraction of all papers (the cost budget). It
reports the operating point on both the iid and temporal test splits so the
time-transfer cost is visible rather than hidden.

Run:
    python -m crc.agents.discipline.tune_thresholds --model-dir models/scibert
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crc.agents.discipline.calibrate import softmax
from crc.agents.discipline.features import build_text_column
from crc.taxonomy import LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"


def predict_probs(model, tok, texts, device, temperature, max_length=256,
                  batch_size=64):
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True,
                      max_length=max_length, padding=True,
                      return_tensors="pt").to(device)
            out.append(model(**enc).logits.float().cpu().numpy())
    return softmax(np.concatenate(out, 0) / temperature)


def borderline_mask(probs, low_conf, tight_gap):
    srt = np.sort(probs, axis=1)
    top, second = srt[:, -1], srt[:, -2]
    return (top < low_conf) | ((top - second) < tight_gap)


def operating_stats(probs, labels, low_conf, tight_gap):
    pred = probs.argmax(1)
    wrong = pred != labels
    flagged = borderline_mask(probs, low_conf, tight_gap)
    n = len(labels)
    fire_rate = flagged.mean()
    # Of the papers that are actually wrong, how many did we flag? (recall)
    caught = (flagged & wrong).sum() / max(1, wrong.sum())
    # Of the papers we flagged, how many were actually wrong? (precision)
    precision = (flagged & wrong).sum() / max(1, flagged.sum())
    return {
        "low_confidence": round(float(low_conf), 3),
        "tight_gap": round(float(tight_gap), 3),
        "fire_rate": round(float(fire_rate), 4),
        "error_recall": round(float(caught), 4),
        "error_precision": round(float(precision), 4),
        "n_flagged": int(flagged.sum()),
        "n_wrong": int(wrong.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-fire-rate", type=float, default=0.25,
                    help="Cost budget: fire on at most this fraction of papers.")
    args = ap.parse_args()

    d = Path(args.model_dir)
    if not d.is_absolute() and not d.exists():
        d = MODELS / d.name
    temperature = 1.0
    tpath = d / "temperature.json"
    if tpath.exists():
        temperature = float(json.loads(tpath.read_text()).get("temperature", 1.0))

    _model = {}

    def model_and_tok():
        if not _model:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _model["m"] = (AutoModelForSequenceClassification.from_pretrained(str(d))
                           .to(dev).eval(), AutoTokenizer.from_pretrained(str(d)), dev)
        return _model["m"]

    corpus = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet")
    temporal = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet")
    val = corpus[corpus["split"] == "val"].reset_index(drop=True)
    test = corpus[corpus["split"] == "test"].reset_index(drop=True)

    def probs_for(df, split):
        # Saved logits from training/evaluation, when they match this split in
        # order, are the same numbers the model would produce -- no GPU needed.
        npz = d / f"{split}_predictions.npz"
        if npz.exists():
            z = np.load(npz)
            if np.array_equal(z["labels"], labels_for(df)):
                return softmax(z["logits"] / temperature)
        model, tok, device = model_and_tok()
        return predict_probs(model, tok, build_text_column(df).tolist(),
                             device, temperature)

    def labels_for(df):
        return df["discipline"].map(LABEL2ID).to_numpy()

    pv, lv = probs_for(val, "val"), labels_for(val)

    # Sweep the two thresholds; keep the pair with the best error recall inside
    # the cost budget, tie-broken by error precision.
    best = None
    for lc in np.round(np.arange(0.40, 0.86, 0.02), 3):
        for tg in np.round(np.arange(0.04, 0.31, 0.02), 3):
            s = operating_stats(pv, lv, lc, tg)
            if s["fire_rate"] > args.max_fire_rate:
                continue
            key = (s["error_recall"], s["error_precision"])
            if best is None or key > best["_key"]:
                best = {**s, "_key": key}

    if best is None:
        print("no threshold pair fit the budget; loosen --max-fire-rate")
        return
    lc, tg = best["low_confidence"], best["tight_gap"]
    print(f"budget: fire on <= {args.max_fire_rate:.0%} of papers\n")
    print(f"chosen thresholds: LOW_CONFIDENCE={lc}  TIGHT_GAP={tg}")
    print(f"  val: fires {best['fire_rate']:.1%}, catches "
          f"{best['error_recall']:.1%} of errors, "
          f"{best['error_precision']:.1%} of flags are real errors")

    print("\noperating point on each split:")
    print(f"  {'split':16s} {'fire%':>7s} {'err_recall':>11s} {'err_prec':>9s}")
    temporal = temporal.reset_index(drop=True)
    rows = {"val": (pv, lv), "test": (probs_for(test, "test"), labels_for(test)),
            "temporal": (probs_for(temporal, "temporal"), labels_for(temporal))}
    report = {"thresholds": {"low_confidence": lc, "tight_gap": tg},
              "max_fire_rate": args.max_fire_rate, "splits": {}}
    for name, (p, l) in rows.items():
        s = operating_stats(p, l, lc, tg)
        report["splits"][name] = s
        print(f"  {name:16s} {s['fire_rate']*100:>6.1f}% "
              f"{s['error_recall']*100:>10.1f}% {s['error_precision']*100:>8.1f}%")

    (d / "thresholds.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {d / 'thresholds.json'}")


if __name__ == "__main__":
    main()
