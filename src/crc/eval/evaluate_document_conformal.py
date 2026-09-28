"""Does the conformal guarantee hold when the input is a whole paper?

The deployed conformal bank was calibrated on abstract-level scores (one chunk
per paper, 2025+). A PDF is read chunk by chunk and pooled into one document
distribution by the section-weighted mean, and pooled distributions are shaped
differently: for the soft-label model they come out *under*-confident (test ECE
0.08 against 0.04 for the abstract alone). Exchangeability between calibration
scores and test scores is therefore not given for full-document input, and the
guarantee was never measured there -- for either model.

This measures it on the full-text evaluation papers (2015-2024, never trained
on), from the chunk probabilities ``evaluate_documents`` cached, so no GPU is
needed:

* coverage and set size of the deployed bank at alpha = 0.10 / 0.20, for the
  abstract read alone vs the pooled whole document, split clean / ambiguous;
* one candidate remedy, a document-level temperature fitted on the validation
  papers' pooled distributions (NLL) and applied to test.

Run (after ``evaluate_documents --chunks .../chunks_eval.parquet``):
    python -m crc.eval.evaluate_document_conformal
"""
from __future__ import annotations

from crc.agents.discipline import DEPLOYED_MODEL

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crc.agents.discipline.calibrate import expected_calibration_error
from crc.agents.discipline.conformal import evaluate_sets, load_bank, pick_alpha
from crc.agents.discipline.hierarchy import softmax
from crc.eval.evaluate_documents import CACHE, apply_strategy, group_papers

WORK = Path(r"C:\Users\ziada\gp_data")
CHUNKS = WORK / "corpus" / "chunks_eval.parquet"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def doc_probs(tag: str, split: str, chunks: pd.DataFrame, strategy: str,
              weights: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = chunks[chunks["split"] == split].reset_index(drop=True)
    cache = CACHE / f"{tag}_{CHUNKS.stem}_{split}.npz"
    if not cache.exists():
        raise SystemExit(f"{cache} missing: run evaluate_documents for {tag} first")
    probs = np.load(cache)["probs"]
    assert len(probs) == len(sub), "cached chunk probabilities out of date"
    amb = sub.groupby("paper_id")["ambiguous"].first()
    docs = list(group_papers(sub, probs))
    P = np.stack([apply_strategy(strategy, x[2], x[3], x[4], weights) for x in docs])
    y = np.array([x[1] for x in docs])
    a = np.array([bool(amb[x[0]]) for x in docs])
    return P, y, a


def fit_doc_temperature(P: np.ndarray, y: np.ndarray) -> float:
    logp = np.log(np.clip(P, 1e-12, 1.0))
    best, best_nll = 1.0, np.inf
    for T in np.geomspace(0.2, 5.0, 120):
        q = softmax(logp / T)
        nll = -np.mean(np.log(q[np.arange(len(y)), y] + 1e-12))
        if nll < best_nll:
            best, best_nll = float(T), nll
    return best


def rescale(P: np.ndarray, T: float) -> np.ndarray:
    return softmax(np.log(np.clip(P, 1e-12, 1.0)) / T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODELS / DEPLOYED_MODEL))
    ap.add_argument("--strategy", default="weighted_mean",
                    help="The deployed pooling rule (DisciplineClassifier default).")
    args = ap.parse_args()

    d = Path(args.model_dir)
    tag = d.name
    weights = json.loads((d / "section_weights.json").read_text())
    bank = load_bank(d / "conformal.json")
    chunks = pd.read_parquet(CHUNKS, columns=["paper_id", "discipline", "split",
                                              "ambiguous", "section", "n_words"])

    V = {s: doc_probs(tag, "val", chunks, s, weights)
         for s in ("abstract_only", args.strategy)}
    Te = {s: doc_probs(tag, "test", chunks, s, weights)
          for s in ("abstract_only", args.strategy)}
    T_doc = fit_doc_temperature(V[args.strategy][0], V[args.strategy][1])

    variants = {
        "abstract only": Te["abstract_only"],
        f"whole document ({args.strategy})": Te[args.strategy],
        f"whole document, doc temperature {T_doc:.3f}": (
            rescale(Te[args.strategy][0], T_doc),) + Te[args.strategy][1:],
    }
    report: dict = {"model": tag, "strategy": args.strategy,
                    "bank": str(d / "conformal.json"),
                    "doc_temperature_fitted_on_val": round(T_doc, 4),
                    "test_papers": int(len(Te["abstract_only"][1])),
                    "test_ambiguous_share": round(float(Te["abstract_only"][2].mean()), 4),
                    "variants": {}}
    print(f"{tag}: {report['test_papers']:,} full-text test papers "
          f"({report['test_ambiguous_share']:.1%} ambiguous); "
          f"doc-level temperature fitted on val = {T_doc:.3f}\n")
    print(f"{'input':44s} {'acc':>6s} {'ECE':>6s} | {'cov@.10':>8s} {'size':>5s} "
          f"| {'cov@.20':>8s} {'size':>5s} | clean/ambig cov@.10")
    for name, (P, y, a) in variants.items():
        r = {"accuracy": round(float((P.argmax(1) == y).mean()), 4),
             "ece": round(expected_calibration_error(P, y), 4), "alphas": {}}
        for alpha in (0.10, 0.20):
            cal = pick_alpha(bank, alpha)
            e = evaluate_sets(cal, P, y)
            r["alphas"][f"{alpha:g}"] = {
                "coverage": e["coverage"], "avg_set_size": e["avg_set_size"],
                "contested_share": e["routed_share"],
                "coverage_clean": evaluate_sets(cal, P[~a], y[~a])["coverage"],
                "coverage_ambiguous": (evaluate_sets(cal, P[a], y[a])["coverage"]
                                       if a.any() else None),
            }
        report["variants"][name] = r
        a1, a2 = r["alphas"]["0.1"], r["alphas"]["0.2"]
        print(f"{name:44s} {r['accuracy']:>6.4f} {r['ece']:>6.4f} | "
              f"{a1['coverage']:>8.4f} {a1['avg_set_size']:>5.2f} | "
              f"{a2['coverage']:>8.4f} {a2['avg_set_size']:>5.2f} | "
              f"{a1['coverage_clean']:.4f} / {a1['coverage_ambiguous']}")

    out = RESULTS / f"document_conformal_{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
