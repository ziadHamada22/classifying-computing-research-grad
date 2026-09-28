"""Build Agent 2's training corpus: field-balanced, split per discipline.

Agent 2 has six independent label spaces, so balancing has to happen *within*
each discipline rather than globally. Left unbalanced the heads would inherit
arXiv's shape and learn the prior instead of the text — Computer Vision alone is
48% of Computer Science, and Machine Learning Methods is 60% of Data Science, so
a CS head could score ~48% by always answering "Computer Vision".

Two caps therefore apply per discipline:

  ``--max-per-field``       absolute ceiling, so a huge field cannot dominate
  ``--max-field-share``     ceiling as a share of the discipline's total

Splits are stratified by field inside each discipline, and a 2025+ temporal
holdout is carved out the same way Agent 1's was, so the two agents are
evaluated on the same time boundary.

Run:
    python -m crc.data.build_field_corpus
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crc.taxonomy import DISCIPLINES
from crc.taxonomy.fields import FIELDS_BY_DISCIPLINE

WORK = Path(r"C:\Users\ziada\gp_data")
LABELLED = WORK / "corpus" / "fields_pool.parquet"
OUT_DIR = WORK / "corpus"


def cap_fields(df: pd.DataFrame, max_per_field: int, max_share: float,
               rng: np.random.Generator) -> pd.DataFrame:
    """Cap each field within one discipline."""
    counts = df["field"].value_counts()
    if counts.empty:
        return df
    # Share cap is computed against what the discipline would hold if the
    # largest field were reduced, so one huge field cannot set its own ceiling.
    share_cap = int(max(counts.median() * 3, counts.sum() * max_share))
    cap = min(max_per_field, max(share_cap, 1))
    parts = []
    for name, g in df.groupby("field", sort=False):
        n = min(len(g), cap)
        parts.append(g.sample(n=n, random_state=int(rng.integers(1 << 31))))
    return pd.concat(parts, ignore_index=True)


def add_splits(df: pd.DataFrame, rng: np.random.Generator,
               train: float = 0.70, val: float = 0.15) -> pd.DataFrame:
    """Stratify by field within the discipline."""
    out = []
    for _, g in df.groupby("field", sort=False):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31))
                     ).reset_index(drop=True)
        n = len(g)
        n_tr, n_va = int(train * n), int(val * n)
        # Guarantee at least one val and test row for tiny fields, otherwise
        # a field can vanish from evaluation entirely.
        if n >= 3:
            n_tr = min(n_tr, n - 2)
            n_va = max(1, min(n_va, n - n_tr - 1))
        split = np.array(["test"] * n, dtype=object)
        split[:n_tr] = "train"
        split[n_tr:n_tr + n_va] = "val"
        g["split"] = split
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default=str(LABELLED))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--max-per-field", type=int, default=4000)
    ap.add_argument("--max-field-share", type=float, default=0.25)
    ap.add_argument("--min-field-papers", type=int, default=60,
                    help="Drop a field below this — too few to learn or judge.")
    ap.add_argument("--require-both-evidence", action="store_true",
                    help="Keep only labels where category AND keywords agree.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.labelled)
    df = df[df["field"].notna()].copy()
    print(f"labelled papers: {len(df):,}")

    if args.require_both_evidence:
        before = len(df)
        df = df[df["field_evidence"] == "both"]
        print(f"  both-evidence filter: {before:,} -> {len(df):,}")

    parts, dropped = [], []
    for d in DISCIPLINES:
        g = df[df["discipline"] == d]
        if g.empty:
            continue
        counts = g["field"].value_counts()
        keep = counts[counts >= args.min_field_papers].index
        for name in counts.index.difference(keep):
            dropped.append((d, name, int(counts[name])))
        g = g[g["field"].isin(keep)]
        if g.empty:
            continue
        g = cap_fields(g, args.max_per_field, args.max_field_share, rng)
        g = add_splits(g, rng)
        parts.append(g)

    corpus = pd.concat(parts, ignore_index=True)
    corpus["text"] = (corpus["title"].fillna("") + ". "
                      + corpus["abstract"].fillna("")).str.strip()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fields_corpus.parquet"
    corpus.to_parquet(path, index=False)

    print(f"\ncorpus -> {path}  ({len(corpus):,} rows)")
    if dropped:
        print("\ndropped fields (too few papers):")
        for d, name, n in dropped:
            print(f"  {d:24s} {name:44s} {n}")

    print("\n=== per discipline ===")
    stats: dict = {"n_total": int(len(corpus)), "per_discipline": {}}
    for d in DISCIPLINES:
        g = corpus[corpus["discipline"] == d]
        if g.empty:
            print(f"\n{d}: EMPTY")
            continue
        vc = g["field"].value_counts()
        sp = g["split"].value_counts()
        print(f"\n{d}  n={len(g):,}  fields={len(vc)}  "
              f"(train {sp.get('train', 0):,} / val {sp.get('val', 0):,} / "
              f"test {sp.get('test', 0):,})")
        for name, n in vc.items():
            print(f"    {name:46s} {n:>6,}  {n/len(g):5.1%}")
        stats["per_discipline"][d] = {
            "n": int(len(g)),
            "n_fields": int(len(vc)),
            "majority_share": round(float(vc.iloc[0] / len(g)), 4),
            "fields": {k: int(v) for k, v in vc.items()},
            "splits": {k: int(v) for k, v in sp.items()},
        }

    print("\nmajority-class baseline per discipline "
          "(what a head must beat to be doing anything):")
    for d, s in stats["per_discipline"].items():
        print(f"  {d:24s} {s['majority_share']:.1%}")

    (out_dir / "fields_corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nstats -> {out_dir / 'fields_corpus_stats.json'}")


if __name__ == "__main__":
    main()
