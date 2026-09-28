"""Pass 2 — sample a balanced corpus from the computing pool.

Fixes the three composition defects measured in the v1 corpus:

  * **Temporal bias.** v1 was built by greedily taking the first N papers in
    stream order, so it covered 2007-2013 only, with a median year of 2007 for
    Computer Science. v2 samples evenly across a configurable modern window and
    holds out later years entirely as a temporal test set.
  * **Category domination.** v1's Computer Engineering class was 62% cs.SY and
    its Data Science class was 66% classical statistics, so each "discipline"
    was really a proxy for one arXiv category. v2 water-fills a quota across the
    categories of each discipline so no category may exceed ``--max-cat-share``.
  * **Contradictory supervision.** Papers whose category list does not yield a
    clear winner under the v2 vote are excluded from train/val, but retained and
    flagged in the test set so evaluation stays honest about the hard cases.

Run:
    python -m crc.data.sample_corpus --per-discipline 10000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crc.taxonomy import BY_CATEGORY, DISCIPLINES

WORK = Path(r"C:\Users\ziada\gp_data")
POOL = WORK / "corpus" / "computing_pool.parquet"
OUT_DIR = WORK / "corpus"


def _fill_at_cap(available: dict[str, int], target: int,
                 max_share: float) -> dict[str, int]:
    """One water-filling pass at a fixed per-category cap."""
    cap = max(1, int(target * max_share))
    quota = {c: 0 for c in available}
    remaining = target
    active = {c for c, n in available.items() if n > 0}

    while remaining > 0 and active:
        even = max(1, remaining // len(active))
        progressed = False
        for c in sorted(active):
            room = min(available[c] - quota[c], cap - quota[c], remaining, even)
            if room <= 0:
                active.discard(c)
                continue
            quota[c] += room
            remaining -= room
            progressed = True
            if quota[c] >= min(available[c], cap):
                active.discard(c)
            if remaining <= 0:
                break
        if not progressed:
            break
    return {c: q for c, q in quota.items() if q > 0}


def water_fill(available: dict[str, int], target: int,
               max_share: float) -> dict[str, int]:
    """Spread ``target`` samples across categories as evenly as supply allows.

    Categories with plenty of papers converge on an equal share; categories with
    few contribute everything they have and their shortfall is redistributed.

    The per-category cap is a *preference*, not a hard constraint, because some
    disciplines cannot satisfy it: arXiv has exactly one Software Engineering
    category, so cs.SE is 96% of SE's supply and a hard 30% cap would starve the
    class to a third of its target. The cap is therefore relaxed in steps until
    the target is reachable, which balances categories wherever supply allows
    and quietly gives up only where arXiv itself offers no alternative.
    """
    total_supply = sum(available.values())
    target = min(target, total_supply)
    if target <= 0:
        return {}

    quota: dict[str, int] = {}
    for share in (max_share, max_share * 1.5, max_share * 2.0, 0.5, 0.75, 1.0):
        quota = _fill_at_cap(available, target, min(1.0, share))
        if sum(quota.values()) >= target:
            return quota
    return quota


def sample_discipline(df: pd.DataFrame, target: int, max_cat_share: float,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Sample one discipline, balanced over category and then over year."""
    avail = df["primary_category"].value_counts().to_dict()
    quota = water_fill(avail, target, max_cat_share)

    picks = []
    for cat, q in quota.items():
        sub = df[df["primary_category"] == cat]
        # Spread the category's quota evenly across the years it spans.
        years = sorted(sub["year"].dropna().unique())
        if not years:
            picks.append(sub.sample(n=min(q, len(sub)), random_state=int(rng.integers(1 << 31))))
            continue
        per_year = water_fill(
            {int(y): int((sub["year"] == y).sum()) for y in years}, q, 1.0)
        for y, qy in per_year.items():
            s = sub[sub["year"] == y]
            picks.append(s.sample(n=min(qy, len(s)),
                                  random_state=int(rng.integers(1 << 31))))
    if not picks:
        return df.head(0)
    return pd.concat(picks, ignore_index=True)


def add_splits(df: pd.DataFrame, rng: np.random.Generator,
               train=0.70, val=0.15) -> pd.DataFrame:
    """Stratified train/val/test assignment within each discipline."""
    out = []
    for _, grp in df.groupby("discipline", sort=False):
        g = grp.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
        n = len(g)
        n_tr, n_va = int(train * n), int(val * n)
        split = np.array(["test"] * n, dtype=object)
        split[:n_tr] = "train"
        split[n_tr:n_tr + n_va] = "val"
        g["split"] = split
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--per-discipline", type=int, default=10_000)
    ap.add_argument("--year-min", type=int, default=2015)
    ap.add_argument("--year-max", type=int, default=2024)
    ap.add_argument("--temporal-min", type=int, default=2025,
                    help="Papers from this year on are held out entirely.")
    ap.add_argument("--temporal-per-discipline", type=int, default=1_500)
    ap.add_argument("--max-cat-share", type=float, default=0.30)
    ap.add_argument("--keep-cross-listed", action="store_true",
                    help="Keep papers whose primary category is non-computing.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pool = pd.read_parquet(args.pool)
    print(f"pool: {len(pool):,} rows")

    # The pool keeps any paper with a computing category *anywhere* in its list,
    # which is right for downstream reuse but wrong for training Agent 1: 11.4%
    # of it is math.NA / math.OC / quant-ph papers that merely cross-list into
    # computing. Requiring the author-chosen primary category to be a computing
    # category is the cleanest "this is a computing paper" filter.
    if not args.keep_cross_listed:
        before = len(pool)
        pool = pool[pool["primary_category"].isin(BY_CATEGORY)]
        print(f"primary-category filter: {before:,} -> {len(pool):,} "
              f"(dropped {before - len(pool):,} cross-list-only papers)")

    pool = pool[pool["year"].notna()]
    main_win = pool[(pool["year"] >= args.year_min) & (pool["year"] <= args.year_max)]
    temp_win = pool[pool["year"] >= args.temporal_min]
    print(f"main window {args.year_min}-{args.year_max}: {len(main_win):,}")
    print(f"temporal window {args.temporal_min}+: {len(temp_win):,}")

    # ---- main corpus -------------------------------------------------------
    # train/val need clean supervision; the test split keeps ambiguous papers so
    # evaluation reflects the real input distribution.
    clean = main_win[~main_win["ambiguous"]]
    n_test_target = int(args.per_discipline * 0.15)

    parts = []
    for d in DISCIPLINES:
        dc = clean[clean["discipline"] == d]
        take = sample_discipline(dc, args.per_discipline, args.max_cat_share, rng)
        parts.append(take)
        print(f"  {d:24s} clean_avail={len(dc):>7,}  sampled={len(take):>6,}")
    main_df = pd.concat(parts, ignore_index=True)
    main_df = add_splits(main_df, rng)

    # Inject ambiguous papers into the test split only, proportionally.
    amb = main_win[main_win["ambiguous"]]
    amb_parts = []
    for d in DISCIPLINES:
        da = amb[amb["discipline"] == d]
        n = min(len(da), int(n_test_target * 0.20))
        if n:
            amb_parts.append(da.sample(n=n, random_state=int(rng.integers(1 << 31))))
    if amb_parts:
        amb_df = pd.concat(amb_parts, ignore_index=True)
        amb_df["split"] = "test"
        main_df = pd.concat([main_df, amb_df], ignore_index=True)
        print(f"  + {len(amb_df):,} ambiguous papers added to test only")

    main_df = main_df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # ---- temporal holdout --------------------------------------------------
    temp_parts = []
    for d in DISCIPLINES:
        dt = temp_win[temp_win["discipline"] == d]
        take = sample_discipline(dt, args.temporal_per_discipline,
                                 args.max_cat_share, rng)
        temp_parts.append(take)
    temporal = pd.concat(temp_parts, ignore_index=True)
    temporal = temporal[~temporal["id"].isin(set(main_df["id"]))]
    temporal["split"] = "temporal_test"
    temporal = temporal.drop_duplicates(subset=["id"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_path = out_dir / "corpus_v2.parquet"
    temp_path = out_dir / "corpus_v2_temporal.parquet"
    main_df.to_parquet(main_path, index=False)
    temporal.to_parquet(temp_path, index=False)

    print(f"\nmain corpus  -> {main_path}  ({len(main_df):,} rows)")
    print(main_df.groupby(["discipline", "split"]).size().unstack(fill_value=0).to_string())
    print(f"\ntemporal test -> {temp_path}  ({len(temporal):,} rows)")
    print(temporal["discipline"].value_counts().to_string())

    print("\nper-discipline category mix (top 5) — compare against v1's 62%/100% domination:")
    for d, g in main_df.groupby("discipline"):
        top = g["primary_category"].value_counts().head(5)
        s = ", ".join(f"{c}={n/len(g):.0%}" for c, n in top.items())
        print(f"  {d:24s} {s}")

    print("\nyear spread:")
    print(main_df.groupby(["discipline"])["year"].agg(["min", "median", "max"]).to_string())

    stats = {
        "main_rows": int(len(main_df)),
        "temporal_rows": int(len(temporal)),
        "year_min": args.year_min,
        "year_max": args.year_max,
        "temporal_min": args.temporal_min,
        "max_cat_share": args.max_cat_share,
        "per_discipline": args.per_discipline,
        "splits": {k: int(v) for k, v in main_df["split"].value_counts().items()},
        "ambiguous_in_test": int(main_df[main_df["split"] == "test"]["ambiguous"].sum()),
        "category_mix": {
            d: {c: int(n) for c, n in g["primary_category"].value_counts().items()}
            for d, g in main_df.groupby("discipline")
        },
    }
    (out_dir / "corpus_v2_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nstats -> {out_dir / 'corpus_v2_stats.json'}")


if __name__ == "__main__":
    main()
