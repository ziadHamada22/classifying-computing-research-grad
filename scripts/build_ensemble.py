"""Build the deployable Agent 1 ensemble: manifest + its own conformal bank.

Two gaps closed here. The best-calibrated ensemble in the results (SciBERT +
DeBERTa + TF-IDF, fitted weights 0.5 / 0.2 / 0.3, test ECE 0.0099) could never
run at inference, because only transformer members were loadable; and the
ensemble the web UI actually served had no conformal calibration at all, so it
fell back to hand-set thresholds. Now TF-IDF loads as a member
(``predict.TfidfMember``) and this script fits a conformal bank on the
*ensemble's own* blended probabilities -- a member's calibration is not valid
for a blend.

Protocol identical to the deployed SciBERT bank (``crc.eval.evaluate_conformal``):
class-conditional LAC, calibrated on the seed-42 half of the 2025+ holdout,
alphas {0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3}, coverage reported on the other
half. Missing member predictions (DeBERTa never scored the 2025+ papers) are
computed once and saved beside the member.

Run:
    python scripts/build_ensemble.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM / "src"))

from crc.agents.discipline.calibrate import expected_calibration_error  # noqa: E402
from crc.agents.discipline.conformal import (  # noqa: E402
    calibrate,
    evaluate_sets,
    save_bank,
)
from crc.agents.discipline.features import build_text_column  # noqa: E402
from crc.agents.discipline.hierarchy import softmax  # noqa: E402
from crc.taxonomy import LABEL2ID  # noqa: E402

MODELS = SYSTEM / "models"
RESULTS = SYSTEM / "results"
CORPUS_DIR = Path(r"C:\Users\ziada\gp_data\corpus")

MEMBERS = [("scibert", 0.5), ("deberta-v3-base", 0.2), ("tfidf", 0.3)]
ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
SEED = 42


def frames() -> dict[str, pd.DataFrame]:
    c = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet")
    out = {s: c[c["split"] == s].reset_index(drop=True) for s in ("val", "test")}
    out["temporal"] = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet").reset_index(drop=True)
    return out


def member_probs(tag: str, split: str, df: pd.DataFrame) -> np.ndarray:
    d = MODELS / tag
    y = df["discipline"].map(LABEL2ID).to_numpy()
    npz = d / f"{split}_predictions.npz"
    if not npz.exists():
        from crc.agents.discipline.predict import DisciplineClassifier
        print(f"  scoring {tag} on {split} ({len(df):,} papers) ...", flush=True)
        clf = DisciplineClassifier.load(d, conformal_alpha=None)
        texts = build_text_column(df).tolist()
        import torch
        logits = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = clf.tok(texts[i:i + 64], truncation=True, max_length=256,
                              padding=True, return_tensors="pt").to(clf.device)
                logits.append(clf.model(**enc).logits.float().cpu().numpy())
        np.savez(npz, logits=np.concatenate(logits), labels=y)
        del clf
    z = np.load(npz)
    assert np.array_equal(z["labels"], y), f"{tag}/{split} out of order"
    t = 1.0
    tp = d / "temperature.json"
    if tp.exists():   # the same temperature the member applies at inference
        t = float(json.loads(tp.read_text()).get("temperature", 1.0))
    return softmax(z["logits"] / t)


def main() -> None:
    F = frames()
    Y = {s: f["discipline"].map(LABEL2ID).to_numpy() for s, f in F.items()}
    w = np.array([m[1] for m in MEMBERS], dtype=np.float64)
    w = w / w.sum()
    P = {}
    single = {}
    for s, df in F.items():
        parts = [member_probs(tag, s, df) for tag, _ in MEMBERS]
        P[s] = sum(wi * p for wi, p in zip(w, parts))
        single[s] = parts[0]

    from sklearn.metrics import f1_score
    report = {"members": MEMBERS, "normalised_weights": w.round(4).tolist(), "splits": {}}
    for s in ("val", "test", "temporal"):
        row = {}
        for name, p in (("ensemble", P[s]), ("scibert", single[s])):
            pred = p.argmax(1)
            row[name] = {"macro_f1": round(float(f1_score(Y[s], pred, average="macro")), 4),
                         "accuracy": round(float((pred == Y[s]).mean()), 4),
                         "ece": round(expected_calibration_error(p, Y[s]), 4)}
        report["splits"][s] = row
        print(f"{s:9s} ensemble F1 {row['ensemble']['macro_f1']:.4f} ECE "
              f"{row['ensemble']['ece']:.4f}  |  SciBERT F1 {row['scibert']['macro_f1']:.4f} "
              f"ECE {row['scibert']['ece']:.4f}")

    idx = np.random.default_rng(SEED).permutation(len(Y["temporal"]))
    ci, ei = idx[: len(idx) // 2], idx[len(idx) // 2:]
    source = "temporal-half (2025+), 28.9% ambiguous -- ensemble outputs"
    bank = {a: calibrate(P["temporal"][ci], Y["temporal"][ci], alpha=a, method="lac",
                         class_conditional=True, calib_source=source, seed=SEED)
            for a in ALPHAS}
    out_dir = MODELS / "ensemble"
    save_bank(out_dir / "conformal.json", bank)
    # Blended predictions in the shared .npz shape (log-probs), so every
    # evaluator that reads <model>/<split>_predictions.npz can score the ensemble.
    for s in ("val", "test", "temporal"):
        np.savez(out_dir / f"{s}_predictions.npz",
                 logits=np.log(np.clip(P[s], 1e-12, 1.0)), labels=Y[s])
    report["conformal_eval_half"] = {f"{a:g}": evaluate_sets(bank[a], P["temporal"][ei],
                                                            Y["temporal"][ei])
                                     for a in ALPHAS}
    for a in (0.1, 0.2):
        r = report["conformal_eval_half"][f"{a:g}"]
        print(f"alpha={a}: coverage {r['coverage']:.4f} (target {1-a:.2f}), "
              f"avg set {r['avg_set_size']:.3f}, contested {r['routed_share']:.1%}")

    (out_dir / "ensemble.json").write_text(json.dumps({
        "members": MEMBERS,
        "strategy": "weighted_mean",
        "weights_source": "simplex search on val macro-F1 (crc.agents.discipline.ensemble)",
        "conformal": "conformal.json -- fitted on these blended probabilities",
    }, indent=2))
    (RESULTS / "deployable_ensemble.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out_dir / 'ensemble.json'} and {out_dir / 'conformal.json'}")


if __name__ == "__main__":
    main()
