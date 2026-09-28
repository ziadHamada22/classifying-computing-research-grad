"""Score Agent 3 against the human gold labels — the honest evaluation.

Run this after labelling with `gold/annotate_methodology.html` and exporting
`gold_labels.json` into `system/gold/`. It compares:

  * **model vs gold** on the four trainable designs — the true accuracy of the
    encoder on the designs it is meant to handle (the 0.657 weak-label number is
    NOT this).
  * **weak-labeller vs gold** — how good the cue-based weak supervision actually
    is, which bounds what any model trained on it can reach.
  * **rare designs** — for papers a human called a rare human-centric design,
    what the encoder predicted instead (it can only output the four): the
    measured size of the local-LLM reviewer's job.

Optionally give a second annotator's file to get Cohen's kappa — methodology is
subjective, so the human-human ceiling matters.

Run:
    python -m crc.eval.score_gold
    python -m crc.eval.score_gold --labels2 gold/gold_labels_ziad2.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, f1_score

from crc.taxonomy.methodology import TRAINABLE_DESIGNS

PROJECT = Path(__file__).resolve().parents[3]
GOLD = PROJECT / "gold"
NON_LABELS = {"Unsure / borderline", "Not empirical research / N/A", None, ""}


def load_labels(path: Path) -> dict[str, str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["paper_id"]: r.get("design") for r in rows
            if r.get("design") not in NON_LABELS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(GOLD / "gold_labels.json"))
    ap.add_argument("--key", default=str(GOLD / "gold_key.json"))
    ap.add_argument("--labels2", default=None)
    args = ap.parse_args()

    gold = load_labels(Path(args.labels))
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    all_rows = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    n_unsure = sum(1 for r in all_rows
                   if r.get("design") in {"Unsure / borderline"})
    n_na = sum(1 for r in all_rows
               if r.get("design") == "Not empirical research / N/A")
    print(f"gold: {len(gold)} usable labels "
          f"({n_unsure} unsure, {n_na} N/A, {len(all_rows)} shown)")

    # Restrict to the four trainable designs for the headline model comparison.
    tr = [pid for pid, g in gold.items()
          if g in TRAINABLE_DESIGNS and pid in key]
    if tr:
        y = [gold[p] for p in tr]
        m = [key[p]["model_pred"] for p in tr]
        acc = sum(a == b for a, b in zip(y, m)) / len(tr)
        mf1 = f1_score(y, m, labels=TRAINABLE_DESIGNS, average="macro",
                       zero_division=0)
        print(f"\n=== MODEL vs GOLD  (4 trainable designs, n={len(tr)}) ===")
        print(f"  accuracy {acc:.3f}   macro-F1 {mf1:.3f}")
        print("  per-design (gold n | model recall):")
        for d in TRAINABLE_DESIGNS:
            idx = [i for i, g in enumerate(y) if g == d]
            if idx:
                r = sum(m[i] == d for i in idx) / len(idx)
                print(f"    {d:36s} n={len(idx):>3d}  recall={r:.3f}")

        # Weak labeller vs gold, same papers that carry a weak label.
        wl = [(gold[p], key[p]["weak_design"]) for p in tr
              if key[p]["weak_design"] in TRAINABLE_DESIGNS]
        if wl:
            wacc = sum(a == b for a, b in wl) / len(wl)
            print(f"\n=== WEAK-LABELLER vs GOLD (n={len(wl)}) ===")
            print(f"  accuracy {wacc:.3f}   "
                  f"(upper bound on what training on weak labels can reach)")

    # Rare designs: what did the encoder do with them?
    rare = {pid: g for pid, g in gold.items() if g not in TRAINABLE_DESIGNS}
    if rare:
        print(f"\n=== RARE designs a human identified (n={len(rare)}) ===")
        by = Counter(g for g in rare.values())
        for d, c in by.most_common():
            preds = Counter(key[p]["model_pred"] for p in rare
                            if gold[p] == d and p in key)
            top = ", ".join(f"{k.split(' /')[0][:16]}:{v}" for k, v in preds.most_common(2))
            print(f"  {d:40s} n={c:<3d} model->  {top}")
        print("  (these are the local-LLM reviewer's territory.)")

    if args.labels2:
        g2 = load_labels(Path(args.labels2))
        shared = [p for p in gold if p in g2]
        if shared:
            k = cohen_kappa_score([gold[p] for p in shared],
                                  [g2[p] for p in shared])
            print(f"\n=== INTER-ANNOTATOR (n={len(shared)}) ===")
            print(f"  Cohen's kappa {k:.3f}  "
                  f"(the human-human ceiling; the model cannot beat this)")


if __name__ == "__main__":
    main()
