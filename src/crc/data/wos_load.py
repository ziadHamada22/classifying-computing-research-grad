"""Load WoS-46985 into a clean, discipline-mapped, arXiv-deduped parquet.

Web of Science abstracts are the only *non-arXiv, human-labelled* computing
papers we have, which is what makes them a real external test for Agent 1 (see
`docs/WOS_AUGMENTATION_SCOPE.md`). This step:

  1. reads the river-martin CSV mirror already cached under gp_data (the HDLTex
     loader is broken; this mirror is plain CSVs),
  2. keeps only the two in-scope domains (CS, ECE) and maps each area onto one of
     our six disciplines, dropping the pure-electrical ECE areas,
  3. **dedupes against the arXiv discipline corpus** so a paper the model trained
     on cannot inflate the external number,

and writes `gp_data/corpus/wos_pool.parquet`. WoS carries no title, so the text
is the abstract alone — an honestly harder, title-free condition that is part of
the out-of-distribution nature of this test.

Run:
    python -m crc.data.wos_load
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from crc.taxonomy.wos_map import IN_SCOPE_DOMAINS, is_ambiguous, map_area

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS = WORK / "corpus"
HF_HUB = WORK / "hf" / "hub"
WOS_REPO = "datasets--river-martin--web-of-science-with-label-texts"
OUT = CORPUS / "wos_pool.parquet"

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Normalised abstract prefix for exact-overlap detection."""
    return _WS.sub(" ", (s or "").strip().lower())[:160]


def _find_csvs() -> list[Path]:
    hits = sorted((HF_HUB / WOS_REPO).glob("snapshots/*/train.csv"))
    if not hits:
        raise FileNotFoundError(
            f"WoS CSVs not found under {HF_HUB / WOS_REPO}. Expected the "
            "river-martin snapshot with train/validate/test.csv.")
    return sorted(hits[0].parent.glob("*.csv"))


def _arxiv_abstract_keys() -> set[str]:
    """Normalised abstracts the discipline model was exposed to."""
    keys: set[str] = set()
    for name in ("corpus_v2.parquet", "corpus_v2_temporal.parquet"):
        p = CORPUS / name
        if p.exists():
            col = pd.read_parquet(p, columns=["abstract"])["abstract"]
            keys.update(_norm(a) for a in col)
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args()

    frames = [pd.read_csv(p) for p in _find_csvs()]
    df = pd.concat(frames, ignore_index=True)
    print(f"raw WoS rows: {len(df):,}   columns: {list(df.columns)}")

    df = df[df["domain"].isin(IN_SCOPE_DOMAINS)].copy()
    print(f"in-scope domains (CS, ECE): {len(df):,}")

    df["discipline"] = df["area"].map(map_area)
    df["ambiguous"] = df["area"].map(is_ambiguous)
    dropped = df["discipline"].isna()
    print(f"dropped out-of-scope areas (pure EE etc.): {int(dropped.sum()):,}")
    df = df[df["discipline"].notna()].copy()

    df = df.rename(columns={"area": "wos_area"})
    df["abstract"] = df["abstract"].fillna("").astype(str)
    before = len(df)
    df = df[df["abstract"].str.len() > 0]
    df = df.drop_duplicates(subset=["abstract"])
    print(f"after empty/dup-abstract cleanup: {before:,} -> {len(df):,}")

    if not args.no_dedup:
        keys = _arxiv_abstract_keys()
        if keys:
            mask = df["abstract"].map(_norm).isin(keys)
            print(f"arXiv-overlap removed: {int(mask.sum()):,} "
                  f"(of {len(keys):,} arXiv keys)")
            df = df[~mask].copy()

    df = df[["abstract", "domain", "wos_area", "discipline", "ambiguous"]]
    df = df.reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df):,} rows)")

    print("\n=== per discipline (mapped) ===")
    for d, g in df.groupby("discipline"):
        amb = int(g["ambiguous"].sum())
        print(f"  {d:24s} {len(g):>6,}   ({amb} ambiguous-area)")
    print("\n=== per WoS area ===")
    vc = df.groupby(["discipline", "wos_area"]).size().sort_values(ascending=False)
    for (d, a), n in vc.items():
        flag = "  [AMBIG]" if is_ambiguous(a) else ""
        print(f"  {d:22s} {a:26s} {n:>5,}{flag}")

    stats = {
        "n_total": int(len(df)),
        "per_discipline": {d: int(len(g)) for d, g in df.groupby("discipline")},
        "n_ambiguous": int(df["ambiguous"].sum()),
    }
    (CORPUS / "wos_pool_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
