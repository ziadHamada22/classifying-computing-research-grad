"""Evaluate the facet model without any gold labels — and say what that cannot show.

There is a tempting mistake to avoid here. The label model (Dawid-Skene) estimates
how accurate each labelling function is from agreement alone, so it is natural to
drop the trained model in as one more voter and read off its accuracy. **That would
be invalid.** The model was trained on the label model's own posteriors, so it is a
distillation of those labelling functions, not an independent observer of the
paper. Dawid-Skene assumes conditional independence; a distillation violates it and
would score itself highly by construction.

What *is* valid rests on a structural fact: **the model reads only the abstract.**
The methods and results sections are text it never saw. So agreement between the
model's prediction and a labelling function that reads those unseen regions is a
real test — it asks whether the model learned what a facet *means*, or merely
which words appear in abstracts.

That gives the one comparison that matters:

    does the MODEL, reading the abstract, agree with the unseen methods/results
    text BETTER than the abstract's own surface cues do?

If yes, the model generalises beyond cue matching. If it merely ties, the model is
an expensive regex. Neither answer needs a single hand-labelled paper.

Three things are reported and one is explicitly disclaimed:

1. estimated accuracy of each labelling function (valid, gold-free);
2. the model-vs-unseen-region comparison above (valid, gold-free);
3. the derived-design distribution and how it differs from the old single-label
   model (descriptive);
4. **not** the model's absolute accuracy against truth — no gold-free method
   establishes that, and this module refuses to imply otherwise.

Run:
    python -m crc.eval.evaluate_facets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crc.data.label_facets import (
    ABSENT,
    PRESENT,
    REGION_NAMES,
    collect_votes,
    dawid_skene,
)
from crc.agents.methodology.predict import DEPLOYED_MODEL
from crc.taxonomy.facets import FACET_KEYS, derive_designs

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS = WORK / "corpus"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"

#: Regions the model never reads, so their votes are genuinely unseen evidence.
UNSEEN_REGIONS = ("methods", "results_conclusion")


def agreement(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    """Share of items where two vote vectors agree, over items both voted on."""
    m = mask & (a != 0) & (b != 0)
    if not m.any():
        return float("nan"), 0
    return float((a[m] == b[m]).mean()), int(m.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODELS / DEPLOYED_MODEL))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--legacy-cues", action="store_true",
                    help="judge against the v1 cue vocabulary (the August yardstick)")
    ap.add_argument("--threshold-file", action="store_true",
                    help="assert facets at the model's fitted per-facet thresholds")
    args = ap.parse_args()

    d = Path(args.model_dir)
    z = np.load(d / f"{args.split}_facet_predictions.npz", allow_pickle=True)
    probs = z["probs"]
    pids = z["paper_ids"].astype(str)
    facets = [str(f) for f in z["facets"]]
    assert facets == FACET_KEYS, "facet order changed since training"
    print(f"{args.split}: {len(pids):,} papers  | yardstick: "
          f"{'v1 cues' if args.legacy_cues else 'v2 cues'}")
    thr = np.full(len(FACET_KEYS), 0.5)
    if args.threshold_file and (d / "thresholds.json").exists():
        t = json.loads((d / "thresholds.json").read_text())["thresholds"]
        thr = np.array([t.get(f, 0.5) for f in FACET_KEYS])
    if args.out is None:
        tag = "" if d.name == "methodology-facets" else f"_{d.name}"
        args.out = str(RESULTS / f"facets_eval{tag}{'_v1cues' if args.legacy_cues else ''}.json")

    # ---- rebuild the labelling-function votes for exactly these papers
    n = len(pids)
    votes = collect_votes(set(pids), order=list(pids), extended=not args.legacy_cues)

    report: dict = {"split": args.split, "n": int(n),
                    "model_dir": d.name, "regions": REGION_NAMES,
                    "yardstick": "v1 cues" if args.legacy_cues else "v2 cues",
                    "thresholds": dict(zip(FACET_KEYS, thr.tolist()))}

    # ---- (1) labelling-function accuracy, estimated with no ground truth
    print("\n" + "=" * 78)
    print("1 - estimated accuracy of each labelling function (no gold labels used)")
    print("=" * 78)
    hdr = f"{'facet':16s} " + " ".join(f"{r[:9]:>11s}" for r in REGION_NAMES)
    print("\n" + hdr)
    lf_acc = {}
    for f in FACET_KEYS:
        m = dawid_skene(votes[f])
        lf_acc[f] = dict(zip(REGION_NAMES,
                             [round(a, 4) for a in m["estimated_accuracy"]]))
        print(f"{f:16s} " + " ".join(
            f"{a:>11.3f}" for a in m["estimated_accuracy"]))
    report["labelling_function_accuracy"] = lf_acc

    # ---- (2) the valid model test: agreement with text the model never read
    print("\n" + "=" * 78)
    print("2 - does the model beat the abstract's surface cues, judged by text")
    print("    it never saw (methods / results)?")
    print("=" * 78)
    ja = REGION_NAMES.index("abstract")
    model_votes = {f: np.where(probs[:, k] >= thr[k], PRESENT, ABSENT).astype(np.int8)
                   for k, f in enumerate(FACET_KEYS)}

    print(f"\n{'facet':16s} {'unseen region':18s} {'abstract cues':>14s} "
          f"{'model':>8s} {'delta':>8s} {'n':>7s}")
    comp = {}
    wins = 0
    total = 0
    for f in FACET_KEYS:
        comp[f] = {}
        for reg in UNSEEN_REGIONS:
            jr = REGION_NAMES.index(reg)
            ref = votes[f][:, jr]
            mask = ref != 0
            a_cue, n_cue = agreement(votes[f][:, ja], ref, mask)
            a_mod, n_mod = agreement(model_votes[f], ref, mask)
            delta = a_mod - a_cue
            comp[f][reg] = {"abstract_cues": round(a_cue, 4),
                            "model": round(a_mod, 4),
                            "delta": round(delta, 4), "n": n_mod}
            if not np.isnan(delta):
                wins += int(delta > 0)
                total += 1
            print(f"{f:16s} {reg:18s} {a_cue:>14.3f} {a_mod:>8.3f} "
                  f"{delta:>+8.3f} {n_mod:>7,}")
    report["model_vs_cues_on_unseen_regions"] = comp
    report["model_wins"] = {"wins": wins, "comparisons": total}
    print(f"\nmodel agrees better than the abstract's own cues in "
          f"{wins}/{total} comparisons")

    # ---- (3) derived designs, and how they differ from the single-label model
    print("\n" + "=" * 78)
    print("3 - derived designs (descriptive)")
    print("=" * 78)
    hard = {f: probs[:, k] >= thr[k] for k, f in enumerate(FACET_KEYS)}
    derived = [derive_designs({f: bool(hard[f][i]) for f in FACET_KEYS})
               for i in range(n)]
    top = [x[0][0] for x in derived]
    multi = int(np.mean([len(x) > 1 for x in derived]) * n)
    both = int((hard["builds"] & hard["evaluates"]).sum())
    print(f"\npapers the facets leave underdetermined (>1 compatible design): "
          f"{multi:,} ({multi/n:.1%})")
    print(f"papers that BOTH build and evaluate: {both:,} ({both/n:.1%}) "
          f"- a single label had to guess on these")
    vc = pd.Series(top).value_counts()
    print(f"\n{'derived design':50s} {'n':>7s} {'share':>7s}")
    for k, v in vc.items():
        print(f"{k:50s} {v:>7,} {v/n:>7.1%}")
    report["derived_design_counts"] = {k: int(v) for k, v in vc.items()}
    report["both_builds_and_evaluates"] = both
    report["underdetermined"] = multi

    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {args.out}")
    print("\nNOT ESTABLISHED: the model's absolute accuracy against ground truth. "
          "No gold-free method can show that; these numbers establish reliability "
          "and generalisation beyond surface cues, not validity of the taxonomy.")


if __name__ == "__main__":
    main()
