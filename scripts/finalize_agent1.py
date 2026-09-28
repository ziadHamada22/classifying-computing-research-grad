"""Run the whole post-bake-off pipeline unattended, once the GPU is free.

After the backbone bake-off finishes there is a fixed sequence of GPU and CPU
steps to turn raw trained models into the final, reported Agent 1. Chaining them
here means the moment the GPU frees, everything runs without a human issuing ten
commands in order:

  per backbone:  calibrate (ECE-guarded)  ->  three-way evaluate
  best backbone: document-level eval (full-text vs abstract)  ->  tune thresholds
  across all:    best ensemble  ->  error decomposition  ->  refresh SUMMARY.md

The "best backbone" is chosen by validation macro-F1 from the metrics files, so
this adapts to whichever model won without hard-coding it.

    python scripts/finalize_agent1.py
    python scripts/finalize_agent1.py --backbones scibert deberta-v3-base modernbert-base specter2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SYSTEM = Path(__file__).resolve().parents[1]
PY = SYSTEM / ".venv" / "Scripts" / "python.exe"
MODELS = SYSTEM / "models"
RESULTS = SYSTEM / "results"


def run(args: list[str], label: str) -> bool:
    print(f"\n{'='*72}\n▶ {label}\n{'='*72}", flush=True)
    import os

    env = {**os.environ, "PYTHONPATH": str(SYSTEM / "src")}
    proc = subprocess.run([str(PY), *args], cwd=str(SYSTEM), env=env)
    ok = proc.returncode == 0
    print(f"{'✓' if ok else '✗'} {label} (exit {proc.returncode})", flush=True)
    return ok


def val_f1(tag: str) -> float:
    f = RESULTS / f"metrics_{tag}.json"
    if not f.exists():
        return -1.0
    return json.loads(f.read_text()).get("val", {}).get("macro_f1", -1.0) or -1.0


def present_backbones(requested: list[str] | None) -> list[str]:
    have = [d.name for d in MODELS.iterdir()
            if d.is_dir() and (d / "val_predictions.npz").exists()]
    if requested:
        return [t for t in requested if t in have]
    # Transformer backbones only for the doc-level / threshold steps; tfidf has
    # no chunk-level model, but it still joins the ensemble.
    return sorted(have)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=None)
    ap.add_argument("--transformer-only-for-docs", action="store_true", default=True)
    args = ap.parse_args()

    backbones = present_backbones(args.backbones)
    if not backbones:
        print("no trained backbones found; nothing to finalize")
        return 1
    print(f"finalizing over: {backbones}")

    transformers = [b for b in backbones if b != "tfidf"]

    failures = []

    # 1. Calibrate + evaluate every backbone (idempotent; re-runs are cheap).
    for tag in backbones:
        if not run(["-m", "crc.agents.discipline.calibrate", "--model-dir",
                    f"models/{tag}"], f"calibrate {tag}"):
            failures.append(f"calibrate {tag}")
    for tag in transformers:
        if not run(["-m", "crc.eval.evaluate", "--model-dir", f"models/{tag}"],
                   f"evaluate {tag}"):
            failures.append(f"evaluate {tag}")

    # 2. Best transformer by val macro-F1 gets the document-level treatment.
    best = max(transformers, key=val_f1) if transformers else None
    if best:
        print(f"\nbest transformer by val macro-F1: {best} ({val_f1(best):.4f})")
        if not run(["-m", "crc.eval.evaluate_documents", "--model-dir",
                    f"models/{best}"], f"document-level eval ({best})"):
            failures.append("document eval")
        if not run(["-m", "crc.agents.discipline.tune_thresholds", "--model-dir",
                    f"models/{best}"], f"tune thresholds ({best})"):
            failures.append("tune thresholds")

    # 3. Full ensemble over everything, weights fitted on val.
    if len(backbones) > 1:
        if not run(["-m", "crc.agents.discipline.ensemble", "--members",
                    *backbones, "--fit-weights"], "fit full ensemble"):
            failures.append("ensemble")
        if not run(["-m", "crc.eval.error_analysis", "--members", *backbones],
                   "error decomposition (equal-weight ensemble)"):
            failures.append("error analysis")

    # 4. Refresh the consolidated summary.
    run(["scripts/summarize_results.py"], "refresh SUMMARY.md")

    print(f"\n{'='*72}")
    if failures:
        print(f"finished with {len(failures)} failure(s): {failures}")
        return 1
    print("finalize complete — all steps succeeded")
    print(f"see {RESULTS / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
