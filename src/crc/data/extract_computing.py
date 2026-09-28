"""Pass 1 — scan the arXiv metadata snapshot and extract every computing paper.

Reads the `librarian-bots/arxiv-metadata-snapshot` parquet shards (~3.1M rows),
applies the v2 taxonomy vote to each paper's full category list, and writes one
row per computing paper with its discipline, ambiguity flag and submission year.

The output is the *pool* the balanced corpus is sampled from (pass 2). It is
deliberately unsampled and unsplit so that Agent 2 (field classification) and
any re-sampling can reuse it without re-scanning 5 GB.

Run:
    python -m crc.data.extract_computing
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from crc.taxonomy import DISCIPLINES, label_paper

WORK = Path(r"C:\Users\ziada\gp_data")
SNAPSHOT_GLOB = "**/librarian-bots--arxiv-metadata-snapshot/**/*.parquet"
OUT = WORK / "corpus" / "computing_pool.parquet"

# Columns we actually need; keeps peak memory well under control.
COLUMNS = ["id", "title", "abstract", "categories", "versions", "update_date"]

_NEW_ID = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}")
_OLD_ID = re.compile(r"^[a-zA-Z\-\.]+/(\d{2})(\d{2})\d+")


def _year_from_id(aid: str) -> int | None:
    """arXiv identifiers encode YYMM. New style 2305.00379, old style cs/0701001."""
    aid = str(aid)
    m = _NEW_ID.match(aid)
    if m:
        return 2000 + int(m.group(1))
    m = _OLD_ID.match(aid)
    if m:
        yy = int(m.group(1))
        return 1900 + yy if yy > 50 else 2000 + yy
    return None


def _year_from_versions(versions) -> int | None:
    """First version's `created` date is the true submission date."""
    if versions is None:
        return None
    try:
        seq = list(versions)
    except TypeError:
        return None
    if not seq:
        return None
    first = seq[0]
    created = None
    if isinstance(first, dict):
        created = first.get("created")
    else:
        created = getattr(first, "created", None)
    if not created:
        return None
    # e.g. "Mon, 2 Apr 2007 19:18:42 GMT"
    m = re.search(r"\b(19|20)\d{2}\b", str(created))
    return int(m.group(0)) if m else None


def _split_categories(raw) -> list[str]:
    """The snapshot stores categories as one space-separated string."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return raw.split()
    try:
        seq = [str(x) for x in raw]
    except TypeError:
        return []
    if len(seq) == 1 and " " in seq[0]:
        return seq[0].split()
    return seq


def _clean(s) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def find_shards(work: Path) -> list[Path]:
    shards = sorted(work.glob(SNAPSHOT_GLOB))
    if not shards:
        shards = sorted(work.glob("**/*.parquet"))
        shards = [s for s in shards if "arxiv-metadata-snapshot" in str(s)]
    return shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--batch-size", type=int, default=100_000)
    ap.add_argument("--min-abstract-words", type=int, default=20,
                    help="Drop stubs; withdrawn papers often have 1-line abstracts.")
    args = ap.parse_args()

    work = Path(args.work)
    shards = find_shards(work)
    if not shards:
        raise SystemExit(f"No snapshot parquet shards found under {work}")
    print(f"Found {len(shards)} shard(s):")
    for s in shards:
        print(f"  {s.name}  {s.stat().st_size/1e9:.2f} GB")

    t0 = time.time()
    kept: list[pd.DataFrame] = []
    n_seen = 0
    n_unmapped = 0

    for si, shard in enumerate(shards, 1):
        pf = pq.ParquetFile(shard)
        avail = set(pf.schema_arrow.names)
        cols = [c for c in COLUMNS if c in avail]
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
            df = batch.to_pandas()
            n_seen += len(df)

            cats = df["categories"].map(_split_categories)
            labels = cats.map(label_paper)

            disc = labels.map(lambda l: l.discipline)
            mask = disc.notna()
            n_unmapped += int((~mask).sum())
            if not mask.any():
                continue

            sub = df.loc[mask].copy()
            lab = labels[mask]
            sub["categories"] = cats[mask]
            sub["primary_category"] = sub["categories"].map(
                lambda c: c[0] if c else None)
            sub["discipline"] = [l.discipline for l in lab]
            sub["margin"] = [round(l.margin, 4) for l in lab]
            sub["ambiguous"] = [l.ambiguous for l in lab]
            sub["co_listed"] = [l.co_listed for l in lab]
            sub["mapped_categories"] = [l.mapped_categories for l in lab]

            if "versions" in sub.columns:
                yr = sub["versions"].map(_year_from_versions)
            else:
                yr = pd.Series([None] * len(sub), index=sub.index)
            yr = yr.fillna(sub["id"].map(_year_from_id))
            sub["year"] = pd.to_numeric(yr, errors="coerce").astype("Int32")

            sub["title"] = sub["title"].map(_clean)
            sub["abstract"] = sub["abstract"].map(_clean)
            sub = sub[sub["abstract"].str.count(r"\s+") >= args.min_abstract_words]

            keep_cols = ["id", "title", "abstract", "primary_category", "categories",
                         "mapped_categories", "co_listed", "discipline", "margin",
                         "ambiguous", "year"]
            kept.append(sub[keep_cols].reset_index(drop=True))

        print(f"  [{si}/{len(shards)}] {shard.name}: seen={n_seen:,} "
              f"kept={sum(len(k) for k in kept):,}  ({time.time()-t0:.0f}s)",
              flush=True)

    pool = pd.concat(kept, ignore_index=True)
    pool = pool.drop_duplicates(subset=["id"]).reset_index(drop=True)
    pool["text"] = (pool["title"] + ". " + pool["abstract"]).str.strip()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(out, index=False)

    print(f"\nScanned {n_seen:,} arXiv papers in {time.time()-t0:.0f}s")
    print(f"Non-computing / unmapped: {n_unmapped:,}")
    print(f"Computing pool: {len(pool):,} rows -> {out}")

    print("\nPer-discipline counts (all / unambiguous):")
    for d in DISCIPLINES:
        g = pool[pool["discipline"] == d]
        print(f"  {d:24s} {len(g):>8,}  /  {int((~g['ambiguous']).sum()):>8,}")
    print(f"\nAmbiguous overall: {pool['ambiguous'].mean():.1%}")

    print("\nPapers per year (2010+):")
    yc = pool[pool["year"] >= 2010]["year"].value_counts().sort_index()
    for y, n in yc.items():
        print(f"  {y}  {n:>8,}")

    stats = {
        "scanned": int(n_seen),
        "pool_rows": int(len(pool)),
        "ambiguous_rate": float(pool["ambiguous"].mean()),
        "per_discipline": {
            d: {
                "total": int((pool["discipline"] == d).sum()),
                "unambiguous": int(((pool["discipline"] == d) & ~pool["ambiguous"]).sum()),
            }
            for d in DISCIPLINES
        },
        "per_year": {str(k): int(v) for k, v in yc.items()},
    }
    (out.parent / "computing_pool_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nStats -> {out.parent / 'computing_pool_stats.json'}")


if __name__ == "__main__":
    main()
