"""Aggregate Agent 3's block predictions into document predictions, and compare.

Three questions:

**Q1 - does reading the whole paper beat reading a truncated methods+abstract?**
The paper-level Agent 3 scored 0.657 accuracy / 0.631 macro-F1 on 1,410 test
papers, reading only the first 320 tokens of a methods+abstract concatenation.
This scores the block model on **the same papers** so the difference is the
architecture and nothing else.

**Q2 - which sections actually reveal the research design?** Two independent
reads: single-section ablations (classify each paper from one section only), and
the fitted section weights. Agent 1's equivalent fit independently drove the
``other`` section to exactly zero, which is what makes this worth trusting as
evidence rather than decoration. Note that ``related_work`` describes *other*
people's methods, so it is a live candidate for a near-zero or harmful weight.

**Q3 - does it close the Design-&-Creation vs Experiment bleed?** That confusion
was about half of all the paper-level model's errors: the fuzzy line between
building a system and measuring one. If whole-document reading helps anywhere, it
should help there, because the distinction is usually stated outside the first
320 tokens.

Everything runs from the saved block logits, so the model is never re-run.

**Standing caveat, unchanged:** the labels are weak (cue-derived), so every number
here measures agreement with the cue labeller, not true methodological accuracy.
An improvement means the encoder generalises the weak labels better. Real accuracy
still needs the hand-labelled gold set.

Run:
    python -m crc.eval.evaluate_methodology_blocks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from crc.agents.discipline.aggregate import aggregate, fit_section_weights
from crc.taxonomy.methodology import TRAINABLE_DESIGNS

PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"

#: Neutral starting point for the weight fit: every section equally plausible.
#: Deliberately not Agent 1's abstract-dominant prior -- the research design is
#: declared in the methods section, so that prior starts in the wrong corner.
NEUTRAL_PRIOR = {"methods": 1.0, "abstract": 1.0, "introduction": 1.0,
                 "background": 1.0, "related_work": 1.0, "results": 1.0,
                 "discussion": 1.0, "conclusion": 1.0}

STRATEGIES = ("mean", "max", "weighted_mean", "weighted_geometric")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def load_blocks(model_dir: Path, split: str):
    """Blocks regrouped per paper, preserving document structure."""
    d = np.load(model_dir / f"{split}_block_predictions.npz", allow_pickle=True)
    probs = _softmax(d["logits"].astype(np.float64))
    pids = d["paper_ids"].astype(str)
    secs = d["sections"].astype(str)
    nw = d["n_words"].astype(np.float64)
    labels = d["labels"].astype(int)

    order = np.argsort(pids, kind="stable")
    pids, probs, secs, nw, labels = (pids[order], probs[order], secs[order],
                                     nw[order], labels[order])
    bounds = np.flatnonzero(np.r_[True, pids[1:] != pids[:-1]])
    groups = np.split(np.arange(len(pids)), bounds[1:])

    docs = []
    for g in groups:
        docs.append({
            "paper_id": pids[g[0]],
            "probs": probs[g],
            "sections": [str(s) for s in secs[g]],
            "n_words": nw[g],
            # every block of a paper carries the paper's label
            "label": int(labels[g[0]]),
        })
    return docs


def score(docs, strategy: str, weights: dict | None = None) -> dict:
    y = np.array([d["label"] for d in docs])
    pred = np.empty(len(docs), dtype=int)
    for i, d in enumerate(docs):
        doc, _ = aggregate(d["probs"], d["sections"], d["n_words"],
                           strategy=strategy, section_weights=weights)
        pred[i] = int(doc.argmax())
    return {
        "n_papers": int(len(docs)),
        "accuracy": round(float((pred == y).mean()), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "_pred": pred, "_y": y,
    }


def score_single_section(docs, section: str) -> dict | None:
    """Classify each paper from one section only (papers lacking it are skipped)."""
    keep = [d for d in docs if section in d["sections"]]
    if not keep:
        return None
    y, pred = [], []
    for d in keep:
        m = np.array([s == section for s in d["sections"]])
        doc, _ = aggregate(d["probs"][m], None, d["n_words"][m],
                           strategy="mean")
        pred.append(int(doc.argmax()))
        y.append(d["label"])
    y, pred = np.array(y), np.array(pred)
    return {
        "n_papers": int(len(keep)),
        "coverage": round(len(keep) / len(docs), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODELS / "methodology-blocks"))
    ap.add_argument("--baseline",
                    default=str(RESULTS / "metrics_methodology-scibert.json"))
    ap.add_argument("--out", default=str(RESULTS / "methodology_blocks_eval.json"))
    args = ap.parse_args()

    d = Path(args.model_dir)
    val, test = load_blocks(d, "val"), load_blocks(d, "test")
    print(f"val  {len(val):,} papers, "
          f"{sum(len(x['probs']) for x in val):,} blocks")
    print(f"test {len(test):,} papers, "
          f"{sum(len(x['probs']) for x in test):,} blocks")

    report: dict = {"model_dir": d.name, "labels": TRAINABLE_DESIGNS}

    # ---- baseline for comparison
    base = None
    bp = Path(args.baseline)
    if bp.exists():
        b = json.loads(bp.read_text())
        base = {"accuracy": round(b["test"]["accuracy"], 4),
                "macro_f1": round(b["test"]["macro_f1"], 4),
                "n_papers": b["test"]["n"], "tag": b.get("tag")}
        report["paper_level_baseline"] = base

    # ---- Q2a: fit section weights on val (neutral prior)
    print("\nfitting section weights on val ...")
    fitted = fit_section_weights(
        [x["probs"] for x in val], [x["sections"] for x in val],
        [x["n_words"] for x in val], np.array([x["label"] for x in val]),
        strategy="weighted_mean", prior=NEUTRAL_PRIOR)
    seen = sorted({s for x in val for s in x["sections"]})
    fitted_seen = {k: round(v, 4) for k, v in fitted.items() if k in seen}
    report["fitted_section_weights"] = fitted_seen

    # ---- Q1: strategy comparison
    print("\n" + "=" * 74)
    print("Q1 - document-level accuracy by aggregation strategy")
    print("=" * 74)
    print(f"\n{'strategy':34s} {'val acc':>8s} {'val F1':>8s} "
          f"{'test acc':>9s} {'test F1':>8s}")
    rows: dict = {}
    combos = [(s, None, s) for s in STRATEGIES] + [
        ("weighted_mean", fitted, "weighted_mean + fitted weights"),
        ("weighted_geometric", fitted, "weighted_geometric + fitted weights"),
    ]
    best = None
    for strategy, w, name in combos:
        rv, rt = score(val, strategy, w), score(test, strategy, w)
        rows[name] = {"val": {k: v for k, v in rv.items() if not k.startswith("_")},
                      "test": {k: v for k, v in rt.items() if not k.startswith("_")}}
        print(f"{name:34s} {rv['accuracy']:>8.4f} {rv['macro_f1']:>8.4f} "
              f"{rt['accuracy']:>9.4f} {rt['macro_f1']:>8.4f}")
        # selection on VAL only, so test stays honest
        if best is None or rv["macro_f1"] > best[0]:
            best = (rv["macro_f1"], name, rt)
    report["strategies"] = rows

    if base:
        print(f"\n{'paper-level baseline (truncated)':34s} {'':>8s} {'':>8s} "
              f"{base['accuracy']:>9.4f} {base['macro_f1']:>8.4f}")

    _, best_name, best_test = best
    print(f"\nbest on val: {best_name}")
    print(f"  test accuracy {best_test['accuracy']:.4f}  "
          f"macro-F1 {best_test['macro_f1']:.4f}")
    if base:
        print(f"  vs paper-level baseline: "
              f"{best_test['accuracy'] - base['accuracy']:+.4f} accuracy, "
              f"{best_test['macro_f1'] - base['macro_f1']:+.4f} macro-F1")
        report["delta_vs_baseline"] = {
            "accuracy": round(best_test["accuracy"] - base["accuracy"], 4),
            "macro_f1": round(best_test["macro_f1"] - base["macro_f1"], 4),
            "selected_strategy": best_name,
        }

    # ---- Q2b: single-section ablation
    print("\n" + "=" * 74)
    print("Q2 - which sections reveal the research design?")
    print("=" * 74)
    print(f"\nsingle-section only (test):")
    print(f"  {'section':16s} {'papers':>7s} {'cover':>7s} {'acc':>7s} {'F1':>7s} "
          f"{'fitted w':>9s}")
    abl = {}
    for sec in sorted(NEUTRAL_PRIOR):
        r = score_single_section(test, sec)
        if r is None:
            continue
        abl[sec] = r
        print(f"  {sec:16s} {r['n_papers']:>7,} {r['coverage']:>7.1%} "
              f"{r['accuracy']:>7.4f} {r['macro_f1']:>7.4f} "
              f"{fitted_seen.get(sec, float('nan')):>9.3f}")
    report["single_section_ablation_test"] = abl

    # ---- Q3: the Design & Creation vs Experiment bleed
    print("\n" + "=" * 74)
    print("Q3 - the Design & Creation / Experiment confusion")
    print("=" * 74)
    y, pred = best_test["_y"], best_test["_pred"]
    names = TRAINABLE_DESIGNS
    cm = confusion_matrix(y, pred, labels=list(range(len(names))))
    report["confusion_matrix_test"] = {"labels": names, "matrix": cm.tolist()}
    print("\nrows = true, cols = predicted")
    print(f"  {'':34s}" + "".join(f"{n[:10]:>12s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n[:32]:34s}" + "".join(f"{v:>12,}" for v in cm[i]))

    try:
        dc = names.index("Design & Creation (Design Science)")
        ex = names.index("Experiment / Empirical Evaluation")
        bleed = int(cm[dc, ex] + cm[ex, dc])
        errors = int(cm.sum() - np.trace(cm))
        report["dc_experiment_bleed"] = {
            "dc_to_exp": int(cm[dc, ex]), "exp_to_dc": int(cm[ex, dc]),
            "total": bleed, "all_errors": errors,
            "share_of_errors": round(bleed / max(1, errors), 4),
        }
        print(f"\n  D&C -> Experiment {cm[dc, ex]:,} + "
              f"Experiment -> D&C {cm[ex, dc]:,} = {bleed:,} "
              f"of {errors:,} errors ({bleed / max(1, errors):.1%})")
        print("  (paper-level baseline: 252 of ~484 errors, about half)")
    except ValueError:
        pass

    print("\n" + classification_report(
        [names[i] for i in y], [names[i] for i in pred],
        labels=names, digits=3, zero_division=0))

    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved -> {args.out}")
    print("\nCAVEAT: measured against WEAK (cue-derived) labels, so this is "
          "'does the encoder generalise the weak labels better', not true "
          "methodological accuracy. The gold set is still required for that.")


if __name__ == "__main__":
    main()
