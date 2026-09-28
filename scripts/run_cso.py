
import argparse
import json
import time
from pathlib import Path

from cso_classifier import CSOClassifier

CORPUS = Path(r"C:\Users\ziada\gp_data\corpus")
INPUT = CORPUS / "cso_input.json"
CACHE = CORPUS / "cso_test_cache.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    rows = json.loads(INPUT.read_text())
    if args.limit:
        rows = rows[: args.limit]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [r for r in rows if r["id"] not in cache]
    print(f"papers={len(rows)}  to_run={len(todo)}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    # Single-process loop over cc.run() rather than batch_run(): CSO's batch
    # multiprocessing is fragile on Windows (silent child death) and only writes
    # at the very end. Constructing once keeps the model loaded; caching every 50
    # makes the (~1.7 s/paper) run resumable and gives a progress signal.
    cc = CSOClassifier(modules="both", enhancement="all", explanation=False,
                       get_weights=False)
    t0 = time.perf_counter()
    for i, r in enumerate(todo, 1):
        res = cc.run({"title": r["title"], "abstract": r["abstract"], "keywords": ""})
        topics = set()
        for key in ("syntactic", "semantic", "union", "enhanced"):
            for t in (res.get(key) or []):
                topics.add(str(t).lower().strip())
        cache[r["id"]] = {"topics": sorted(topics)}
        if i % 50 == 0 or i == len(todo):
            CACHE.write_text(json.dumps(cache))
            rate = i / (time.perf_counter() - t0)
            print(f"  {i}/{len(todo)}  {rate:.2f} paper/s", flush=True)
    print(f"saved {len(cache)} -> {CACHE}", flush=True)


if __name__ == "__main__":
    main()
