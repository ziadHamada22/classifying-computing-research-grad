"""Workstream A — Agent 1 discipline classifier, external validation on WoS.

This is the honesty test the arXiv corpus cannot provide: 96% of our Software
Engineering papers are `cs.SE`, so a good SE score on arXiv might just mean the
model learned `cs.SE` house style. WoS papers come from a different source and
different venues, so if SE holds up here the confound is disproven — and if it
drops, we have found a real weakness. Either way it is a reportable result.

Because WoS labels are at discipline granularity we can only score Agent 1
(discipline), not Agent 2 (field). The two CS judgement-call areas are excluded
from the headline ("clean") figures and reported separately.

Treat every number as **out-of-distribution**: WoS text is older, from other
venues, and — unlike our training rows — has no title. It is a robustness probe,
not an iid held-out split.

Run:
    python -m crc.eval.evaluate_wos                # SciBERT (deployed default)
    python -m crc.eval.evaluate_wos --ensemble     # SciBERT + DeBERTa
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from crc.agents.discipline import DEPLOYED_MODEL
from crc.agents.discipline.features import build_text_column
from crc.agents.discipline.predict import DisciplineClassifier, EnsembleClassifier
from crc.taxonomy import DISCIPLINES
from crc.taxonomy.disciplines import DISCIPLINE_ABBR

WORK = Path(r"C:\Users\ziada\gp_data")
POOL = WORK / "corpus" / "wos_pool.parquet"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def build_classifier(ensemble: bool, model_dir: str | None = None):
    if ensemble:
        members, weights = [], []
        for tag, w in (("scibert", 0.5), ("deberta-v3-base", 0.2)):
            d = MODELS / tag
            if (d / "config.json").exists():
                members.append(DisciplineClassifier.load(d))
                weights.append(w)
        if len(members) > 1:
            return EnsembleClassifier(members, weights), "scibert+deberta"
        return members[0], members[0].name
    d = Path(model_dir) if model_dir else MODELS / DEPLOYED_MODEL
    return DisciplineClassifier.load(d), d.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--model-dir", default=None,
                    help="Single discipline model (default: the deployed one).")
    args = ap.parse_args()

    df = pd.read_parquet(args.pool)
    print(f"WoS pool: {len(df):,} papers "
          f"({int(df['ambiguous'].sum()):,} ambiguous-area)")

    clf, tag = build_classifier(args.ensemble, args.model_dir)
    print(f"model: {tag}")

    texts = build_text_column(df).tolist()
    t0 = time.perf_counter()
    probs = clf.chunk_probs(texts)            # one abstract == one chunk
    elapsed = time.perf_counter() - t0
    pred_idx = probs.argmax(axis=1)
    pred = np.array([DISCIPLINES[i] for i in pred_idx])
    true = df["discipline"].to_numpy()
    clean = ~df["ambiguous"].to_numpy()

    acc_all = float((pred == true).mean())
    acc_clean = float((pred[clean] == true[clean]).mean())
    print(f"\noverall accuracy   all {acc_all:.4f}   "
          f"clean {acc_clean:.4f}   ({int(clean.sum()):,} clean papers)")
    print(f"throughput         {len(df)/elapsed:.1f} docs/s")

    # Per-discipline recall on the clean subset (does Agent 1 recognise these
    # non-arXiv papers as their WoS-assigned discipline?).
    print(f"\n{'discipline':24s} {'n':>6s} {'recall':>8s}")
    per_discipline = {}
    for d in DISCIPLINES:
        m = clean & (true == d)
        if not m.any():
            continue
        rec = float((pred[m] == d).mean())
        per_discipline[d] = {"n": int(m.sum()), "recall": round(rec, 4)}
        star = "   <== SE" if d == "Software Engineering" else ""
        print(f"  {d:22s} {int(m.sum()):>6,} {rec:>8.4f}{star}")

    # Per WoS area: where does Agent 1 actually send each area?
    print(f"\n{'WoS area':26s} {'->disc(mapped)':22s} {'n':>5s} {'hit':>6s}  top-preds")
    per_area = {}
    for area, g in df.groupby("wos_area"):
        idx = g.index.to_numpy()
        gp = pred[idx]
        mapped = g["discipline"].iloc[0]
        hit = float((gp == mapped).mean())
        dist = Counter(gp).most_common(3)
        per_area[area] = {
            "mapped_discipline": mapped,
            "n": int(len(g)),
            "hit_rate": round(hit, 4),
            "ambiguous": bool(g["ambiguous"].iloc[0]),
            "pred_distribution": {k: int(v) for k, v in Counter(gp).items()},
        }
        top = ", ".join(f"{DISCIPLINE_ABBR[k]}:{v}" for k, v in dist)
        amb = " [A]" if g["ambiguous"].iloc[0] else ""
        print(f"  {area:24s}{amb:4s} {mapped[:20]:22s} {len(g):>5,} "
              f"{hit:>6.3f}  {top}")

    report_clean = classification_report(
        true[clean], pred[clean], labels=DISCIPLINES, digits=4, zero_division=0)
    print("\n" + report_clean)

    out = {
        "model": tag,
        "n": int(len(df)),
        "n_clean": int(clean.sum()),
        "accuracy_all": round(acc_all, 4),
        "accuracy_clean": round(acc_clean, 4),
        "macro_f1_clean": round(float(
            f1_score(true[clean], pred[clean], average="macro")), 4),
        "docs_per_second": round(len(df) / elapsed, 1),
        "per_discipline_recall": per_discipline,
        "per_area": per_area,
        "confusion_labels": DISCIPLINES,
        "confusion_clean": confusion_matrix(
            true[clean], pred[clean], labels=DISCIPLINES).tolist(),
        "note": "OUT-OF-DISTRIBUTION probe: non-arXiv, title-free, older text.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"wos_discipline_eval_{tag.replace('+', '_')}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
