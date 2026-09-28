"""Build Agent 3's training corpus: the four trainable research designs, balanced.

The weak-labelled pool is dominated by Design & Creation (5,878) while Simulation
has 698, so left unbalanced a 4-way head would learn the prior. A per-design cap
levels the big classes; Simulation stays at its natural size (still trainable).
Splits are stratified by design so every split sees every class. The five
arXiv-starved human-centric designs are excluded here (see
`taxonomy/methodology.TRAINABLE_DESIGNS`); they are the local-LLM reviewer's job.

Run:
    python -m crc.data.build_methodology_corpus
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crc.taxonomy.methodology import TRAINABLE_DESIGNS

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS = WORK / "corpus"
POOL = CORPUS / "methodology_pool.parquet"
OUT = CORPUS / "methodology_corpus.parquet"


def add_splits(df: pd.DataFrame, rng: np.random.Generator,
               train: float = 0.70, val: float = 0.15) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("design", sort=False):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31))
                     ).reset_index(drop=True)
        n = len(g)
        n_tr, n_va = int(train * n), int(val * n)
        split = np.array(["test"] * n, dtype=object)
        split[:n_tr] = "train"
        split[n_tr:n_tr + n_va] = "val"
        g = g.drop(columns=["split"], errors="ignore")
        g["split"] = split
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-per-design", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.pool)
    df = df[df["design"].isin(TRAINABLE_DESIGNS)].copy()
    print(f"trainable-design papers: {len(df):,}")

    parts = []
    for name, g in df.groupby("design", sort=False):
        n = min(len(g), args.max_per_design)
        parts.append(g.sample(n=n, random_state=int(rng.integers(1 << 31))))
    capped = pd.concat(parts, ignore_index=True)
    capped = add_splits(capped, rng)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    capped.to_parquet(args.out, index=False)
    print(f"\ncorpus -> {args.out}  ({len(capped):,} rows)")

    print("\n=== design x split ===")
    print(capped.groupby(["design", "split"]).size()
          .unstack(fill_value=0).to_string())
    maj = capped["design"].value_counts().iloc[0] / len(capped)
    print(f"\nmajority-class baseline: {maj:.1%} "
          f"(what the 4-way head must beat)")

    stats = {
        "n_total": int(len(capped)),
        "max_per_design": args.max_per_design,
        "majority_baseline": round(float(maj), 4),
        "designs": {k: int(v) for k, v in capped["design"].value_counts().items()},
        "splits": {k: int(v) for k, v in capped["split"].value_counts().items()},
    }
    (CORPUS / "methodology_corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"stats -> {CORPUS / 'methodology_corpus_stats.json'}")


if __name__ == "__main__":
    main()
