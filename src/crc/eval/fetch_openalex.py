"""Fetch OpenAlex primary-topic labels for a corpus split, for the external
Agent-1 baseline (`evaluate_baselines.py`).

Resolution is by arXiv DOI (``10.48550/arXiv.<id>``) against the per-work
endpoint, which resolves alias DOIs the bulk `filter=doi:` endpoint misses.
Results are cached to disk keyed by arXiv id, so the (network-bound) fetch runs
once and every re-run is free; interrupt and resume freely.

This is an OFFLINE EVALUATION baseline, not part of the deployed pipeline — the
"no online LLM / no network at inference" constraint governs the shipped system,
not a one-off research comparison built from a public index.

Run:
    python -m crc.eval.fetch_openalex                 # test split
    python -m crc.eval.fetch_openalex --split temporal_test
    python -m crc.eval.fetch_openalex --limit 200     # smoke test
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

MAILTO = "ziadaiman103@gmail.com"          # OpenAlex "polite pool"
WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
CACHE = CORPUS_DIR / "openalex_test_cache.json"

_lock = threading.Lock()


def _fetch_one(sess: requests.Session, aid: str) -> dict:
    """Resolve one arXiv id to its OpenAlex primary-topic axes (or a miss)."""
    url = f"https://api.openalex.org/works/doi:10.48550/arXiv.{aid}"
    for attempt in range(4):
        try:
            r = sess.get(url, params={"mailto": MAILTO,
                                      "select": "doi,primary_topic"}, timeout=25)
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            pt = r.json().get("primary_topic") or {}
            return {
                "topic": pt.get("display_name"),
                "subfield": (pt.get("subfield") or {}).get("display_name"),
                "field": (pt.get("field") or {}).get("display_name"),
                "domain": (pt.get("domain") or {}).get("display_name"),
            }
        if r.status_code == 404:
            return {"topic": None, "subfield": None, "field": None,
                    "domain": None, "http": 404}
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2.0 * (attempt + 1))
            continue
        return {"topic": None, "subfield": None, "field": None,
                "domain": None, "http": r.status_code}
    return {"topic": None, "subfield": None, "field": None, "domain": None,
            "http": "retry_exhausted"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["test", "temporal_test"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", default="", help="optional existing cache json to merge in")
    args = ap.parse_args()

    if args.split == "temporal_test":
        df = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet")
    else:
        corpus = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet")
        df = corpus[corpus["split"] == "test"]
    ids = [str(x) for x in df["id"].tolist()]
    if args.limit:
        ids = ids[: args.limit]
    print(f"split={args.split}  papers={len(ids):,}")

    cache: dict[str, dict] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
    if args.seed and Path(args.seed).exists():
        seed = json.loads(Path(args.seed).read_text())
        merged = sum(1 for k, v in seed.items() if k not in cache and v.get("field"))
        cache.update({k: v for k, v in seed.items() if k not in cache})
        print(f"seeded {merged:,} entries from {args.seed}")

    todo = [a for a in ids if a not in cache]
    print(f"cached={len(ids) - len(todo):,}  to_fetch={len(todo):,}")

    sess = requests.Session()
    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_fetch_one, sess, a): a for a in todo}
        for fut in as_completed(futs):
            aid = futs[fut]
            with _lock:
                cache[aid] = fut.result()
                done += 1
            if done % 250 == 0:
                CACHE.write_text(json.dumps(cache))
                rate = done / (time.perf_counter() - t0)
                print(f"  {done:,}/{len(todo):,}  {rate:.1f} req/s")

    CACHE.write_text(json.dumps(cache))
    present = [cache[a] for a in ids if a in cache]
    resolved = sum(1 for c in present if c.get("field"))
    print(f"\nsaved -> {CACHE}")
    print(f"resolved {resolved:,}/{len(ids):,} ({resolved/len(ids):.1%}) to a topic")


if __name__ == "__main__":
    main()
