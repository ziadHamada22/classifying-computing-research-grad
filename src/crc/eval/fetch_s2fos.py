"""Fetch Semantic Scholar S2FOS (field-of-study) predictions for a corpus split.

S2FOS (Semantic Scholar's open field-of-study classifier) is a multilabel linear
SVM over character n-grams; its label space is the ~23 top-level science fields
(Computer Science, Mathematics, Engineering, ...). We pull it straight from the
Semantic Scholar Graph API's ``s2FieldsOfStudy`` (source ``s2-fos-model``) via the
batch endpoint (up to 500 ids/request) — no install, no model download.

Its top-level "Computer Science" is our *entire* domain, so S2FOS provides no
signal to separate the six CC2020 disciplines: the baseline's role is to show
exactly that. Offline evaluation only, not part of the deployed pipeline.

Run:
    python -m crc.eval.fetch_s2fos                # test split
    python -m crc.eval.fetch_s2fos --split temporal_test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
CACHE = CORPUS_DIR / "s2fos_test_cache.json"
BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"


def _post_batch(ids: list[str]) -> list:
    for attempt in range(5):
        try:
            r = requests.post(BATCH, params={"fields": "s2FieldsOfStudy"},
                              json={"ids": ids}, timeout=60)
        except requests.RequestException:
            time.sleep(2.0 * (attempt + 1)); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(3.0 * (attempt + 1)); continue
        # Bad id in the batch can 400 the whole request; caller splits on []
        return []
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "temporal_test"])
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()

    if args.split == "temporal_test":
        df = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet")
    else:
        corpus = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet")
        df = corpus[corpus["split"] == "test"]
    ids = [str(x) for x in df["id"].tolist()]
    print(f"split={args.split}  papers={len(ids):,}")

    cache: dict[str, dict] = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [a for a in ids if a not in cache]
    print(f"cached={len(ids) - len(todo):,}  to_fetch={len(todo):,}")

    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        res = _post_batch([f"arXiv:{a}" for a in chunk])
        if not res:                      # whole-batch failure: mark as misses
            for a in chunk:
                cache[a] = {"model": [], "http": "batch_fail"}
        else:
            for a, w in zip(chunk, res):
                if w is None:
                    cache[a] = {"model": None}       # not in S2
                    continue
                s2 = w.get("s2FieldsOfStudy") or []
                cache[a] = {"model": sorted({f["category"] for f in s2
                                             if f.get("source") == "s2-fos-model"})}
        CACHE.write_text(json.dumps(cache))
        print(f"  {min(i + args.batch_size, len(todo)):,}/{len(todo):,}")
        time.sleep(1.1)                  # be polite to the shared pool

    present = [cache[a] for a in ids if a in cache]
    resolved = sum(1 for c in present if c.get("model"))
    says_cs = sum(1 for c in present if c.get("model") and "Computer Science" in c["model"])
    print(f"\nsaved -> {CACHE}")
    print(f"resolved {resolved:,}/{len(ids):,}; of resolved, "
          f"{says_cs:,} ({says_cs / max(resolved,1):.1%}) tagged 'Computer Science'")


if __name__ == "__main__":
    main()
