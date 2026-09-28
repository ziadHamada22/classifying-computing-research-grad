"""Train every candidate backbone in sequence, then print a comparison table.

Runs unattended for several hours. Per-model batch sizes are set so each fits
the 6 GB RTX 3060 Laptop; the effective batch is held at 32 across all of them
via gradient accumulation, so the comparison isolates the backbone rather than
confounding it with optimisation-schedule differences.

    python scripts/bake_off.py                    # all backbones
    python scripts/bake_off.py --only deberta-v3-base modernbert-base
    python scripts/bake_off.py --skip scibert     # scibert already trained
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SYSTEM = Path(__file__).resolve().parents[1]
SRC = SYSTEM / "src"
PY = SYSTEM / ".venv" / "Scripts" / "python.exe"
RESULTS = SYSTEM / "results"

# (tag, hf id, per-device batch, grad accum, learning rate)
# effective batch = batch * accum = 32 for every entry
BACKBONES: list[tuple[str, str, int, int, float]] = [
    ("scibert",           "allenai/scibert_scivocab_uncased", 16, 2, 2e-5),
    ("deberta-v3-base",   "microsoft/deberta-v3-base",         8, 4, 2e-5),
    ("modernbert-base",   "answerdotai/ModernBERT-base",       8, 4, 3e-5),
    ("specter2",          "allenai/specter2_base",            16, 2, 2e-5),
    # Vanilla BERT: the proposal's "no domain adaptation" baseline. Same size and
    # recipe as SciBERT, so the comparison isolates scientific pre-training.
    ("bert-base-uncased", "bert-base-uncased",                16, 2, 2e-5),
]


def run(tag: str, model: str, batch: int, accum: int, lr: float,
        epochs: float, corpus: str) -> bool:
    cmd = [
        str(PY), "-m", "crc.agents.discipline.train",
        "--model", model, "--tag", tag, "--corpus", corpus,
        "--epochs", str(epochs), "--batch-size", str(batch),
        "--grad-accum", str(accum), "--lr", str(lr),
    ]
    print(f"\n{'='*72}\n  {tag}  ({model})\n"
          f"  batch={batch} x accum={accum} (effective {batch*accum})  lr={lr}\n"
          f"{'='*72}", flush=True)
    env = {"PYTHONPATH": str(SRC)}
    import os

    full_env = {**os.environ, **env}
    t0 = time.time()
    proc = subprocess.run(cmd, env=full_env, cwd=str(SYSTEM))
    ok = proc.returncode == 0
    print(f"  -> {'OK' if ok else 'FAILED'} in {(time.time()-t0)/60:.1f} min",
          flush=True)
    return ok


def comparison_table() -> None:
    rows = []
    for f in sorted(RESULTS.glob("metrics_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        rows.append({
            "tag": d.get("tag", f.stem),
            "model": d.get("model", ""),
            "corpus": d.get("corpus", ""),
            "val_macro_f1": d.get("val", {}).get("macro_f1"),
            "test_macro_f1": d.get("test", {}).get("macro_f1"),
            "test_accuracy": d.get("test", {}).get("accuracy"),
            "train_min": round(d.get("train_seconds", 0) / 60, 1),
        })
    if not rows:
        print("\nno metrics files found")
        return
    rows.sort(key=lambda r: (r["test_macro_f1"] or 0), reverse=True)
    print(f"\n{'='*88}\n  BACKBONE COMPARISON (sorted by test macro-F1)\n{'='*88}")
    print(f"  {'tag':20s} {'corpus':9s} {'val F1':>8s} {'test F1':>8s} "
          f"{'test acc':>9s} {'train min':>10s}")
    for r in rows:
        v = f"{r['val_macro_f1']:.4f}" if r["val_macro_f1"] else "  -   "
        t = f"{r['test_macro_f1']:.4f}" if r["test_macro_f1"] else "  -   "
        a = f"{r['test_accuracy']:.4f}" if r["test_accuracy"] else "  -   "
        print(f"  {r['tag']:20s} {r['corpus']:9s} {v:>8s} {t:>8s} {a:>9s} "
              f"{r['train_min']:>10.1f}")
    (RESULTS / "bake_off_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\n  saved -> {RESULTS / 'bake_off_summary.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--corpus", default="abstract",
                    choices=["abstract", "chunk", "both"])
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()

    if args.table_only:
        comparison_table()
        return 0

    todo = [b for b in BACKBONES
            if (args.only is None or b[0] in args.only) and b[0] not in args.skip]
    print(f"training {len(todo)} backbone(s): {[t[0] for t in todo]}")

    failed = []
    for tag, model, batch, accum, lr in todo:
        if not run(tag, model, batch, accum, lr, args.epochs, args.corpus):
            failed.append(tag)

    comparison_table()
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
