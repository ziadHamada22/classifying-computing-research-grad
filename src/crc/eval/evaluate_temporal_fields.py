"""The cascade on neutral, genuinely drifted ground: a 2025+ field slice.

Every end-to-end number so far was measured on the field corpus's test papers,
and two facts make that a biased place to measure the cascade:

* The field corpus holds no 2025+ papers, so the deployed conformal shortlist --
  calibrated on 2025+ papers, because that is what real input looks like --
  over-covers there (0.964 against a 0.90 target). Its benefit *under genuine
  drift* was never measured. (Open caveat from the 2026-08-12 session.)
* Those papers are Agent 2's home distribution: on them Agent 2's own discipline
  marginal (0.860) beats Agent 1 (0.818), which is why pooling the two looked
  like a win there and nowhere else (``evaluate_joint``).

This builds the missing slice. The 9,000-paper 2025+ holdout is labelled with a
field by the *same* weak labeller that built the field corpus
(``crc.data.label_fields.label_frame``), unchanged, and only confidently labelled
papers are kept -- exactly the rule the field corpus used. Ambiguous-discipline
papers are kept too, because real 2025+ input contains them (28.9%).

The deployed conformal bank was calibrated on one half of the holdout (seed-42
permutation, see ``evaluate_conformal``), so every shortlist number here is
scored on the **other** half only. Neither Agent 1 nor Agent 2 trained on any
2025+ paper.

Reported: oracle vs end-to-end field accuracy, hierarchical F1, the routing
policies (top-1, fixed top-2, conformal shortlists), and pooled decoding.

Run:
    python -m crc.eval.evaluate_temporal_fields
"""
from __future__ import annotations

from crc.agents.discipline import DEPLOYED_MODEL

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crc.agents.discipline.conformal import load_bank, pick_alpha
from crc.agents.discipline.hierarchy import (
    conditioned_field,
    field_to_discipline_marginal,
    hierarchical_scores,
    joint_discipline_probs,
    softmax,
)
from crc.data.label_fields import label_frame
from crc.eval.evaluate_joint import mcnemar
from crc.eval.evaluate_pipeline import conformal_routing_metrics, field_pred_under_every_discipline
from crc.taxonomy import DISCIPLINES, LABEL2ID
from crc.taxonomy.fields import GLOBAL_LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
CACHE = WORK / "cache" / "joint"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def field_logits_for(df: pd.DataFrame, field_dir: Path, cache_name: str) -> np.ndarray:
    path = CACHE / f"{cache_name}.npy"
    if path.exists():
        z = np.load(path)
        if len(z) == len(df):
            return z
    from crc.agents.field.predict import FieldClassifier
    z = FieldClassifier.load(field_dir).chunk_logits(df["text"].tolist(), batch_size=64)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, z)
    return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discipline-dir", default=str(MODELS / DEPLOYED_MODEL))
    ap.add_argument("--field-dir", default=str(MODELS / "field-scibert-v2"))
    ap.add_argument("--seed", type=int, default=42,
                    help="Must match the seed evaluate_conformal split the holdout with.")
    ap.add_argument("--out", default=str(RESULTS / "temporal_field_eval.json"))
    args = ap.parse_args()

    ddir, fdir = Path(args.discipline_dir), Path(args.field_dir)
    df = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet").reset_index(drop=True)

    # --- same calibration/evaluation halves as the deployed conformal bank ----
    idx = np.random.default_rng(args.seed).permutation(len(df))
    eval_half = np.zeros(len(df), bool)
    eval_half[idx[len(df) // 2:]] = True

    # --- field labels, by the unchanged corpus labeller ----------------------
    lab = label_frame(df)
    df["field"] = lab["field"].to_numpy()
    df["field_evidence"] = lab["field_evidence"].to_numpy()
    labelled = df["field"].notna().to_numpy() & df["field"].map(
        lambda f: f in GLOBAL_LABEL2ID).to_numpy()
    print(f"2025+ papers: {len(df):,}   field-labelled: {labelled.sum():,} "
          f"({labelled.mean():.1%})   of which in the evaluation half: "
          f"{(labelled & eval_half).sum():,}")

    # --- model outputs --------------------------------------------------------
    z1 = np.load(ddir / "temporal_predictions.npz")
    y_disc = df["discipline"].map(LABEL2ID).to_numpy()
    assert np.array_equal(z1["labels"], y_disc), "temporal predictions out of order"
    p1 = softmax(z1["logits"])
    z2 = field_logits_for(df, fdir, f"a2_{fdir.name}_temporal")
    p2m = field_to_discipline_marginal(softmax(z2))

    report: dict = {"discipline_model": ddir.name, "field_model": fdir.name,
                    "n_holdout": int(len(df)), "n_field_labelled": int(labelled.sum()),
                    "labelled_share": round(float(labelled.mean()), 4),
                    "field_evidence": df.loc[labelled, "field_evidence"]
                    .value_counts().to_dict(),
                    "subsets": {}}

    true_field = np.full(len(df), -1)
    true_field[labelled] = df.loc[labelled, "field"].map(GLOBAL_LABEL2ID).astype(int)
    pred_all = field_pred_under_every_discipline(z2)
    true_names = [DISCIPLINES[j] for j in y_disc]

    jw_path = RESULTS / "joint_decoding_eval.json"
    joint_w = (json.loads(jw_path.read_text())["selected_weight"]
               if jw_path.exists() else None)

    def score(mask: np.ndarray) -> dict:
        i = np.where(mask)[0]
        tf = true_field[i]
        out = {"n": int(len(i)),
               "ambiguous_share": round(float(df["ambiguous"].to_numpy()[i].mean()), 4)}
        oracle = conditioned_field(z2[i], [true_names[k] for k in i])
        out["oracle_field_accuracy"] = round(float((oracle == tf).mean()), 4)
        variants = {"agent1": p1[i]}
        hits = {}
        if joint_w is not None:
            variants[f"pooled_w{joint_w}"] = joint_discipline_probs(p1[i], p2m[i], joint_w)
        for name, pr in variants.items():
            pd_names = [DISCIPLINES[j] for j in pr.argmax(1)]
            pf = conditioned_field(z2[i], pd_names)
            hits[name] = (pf == tf, pr.argmax(1) == y_disc[i])
            out[name] = {
                "discipline_accuracy": round(float((pr.argmax(1) == y_disc[i]).mean()), 4),
                "end_to_end_field_accuracy": round(float((pf == tf).mean()), 4),
                "field_acc_given_discipline_correct": round(float(
                    (pf == tf)[pr.argmax(1) == y_disc[i]].mean()), 4),
                "hierarchical": hierarchical_scores(
                    [true_names[k] for k in i], pd_names, tf, pf),
            }
        for name in hits:
            if name != "agent1":
                out[name]["mcnemar_e2e_vs_agent1"] = mcnemar(hits["agent1"][0], hits[name][0])
                out[name]["mcnemar_discipline_vs_agent1"] = mcnemar(hits["agent1"][1],
                                                                     hits[name][1])
        out["cascade_loss"] = round(out["oracle_field_accuracy"]
                                    - out["agent1"]["end_to_end_field_accuracy"], 4)
        return out

    for name, mask in (("all_labelled", labelled),
                       ("eval_half", labelled & eval_half),
                       ("eval_half_clean", labelled & eval_half & ~df["ambiguous"].to_numpy()),
                       ("eval_half_ambiguous", labelled & eval_half & df["ambiguous"].to_numpy())):
        report["subsets"][name] = score(mask)

    # --- routing policies on the evaluation half (the bank never saw it) -----
    m = labelled & eval_half
    order = np.argsort(-p1, axis=1)
    conf = p1[np.arange(len(p1)), order[:, 0]]
    gap = conf - p1[np.arange(len(p1)), order[:, 1]]
    policies = {}
    top1 = np.zeros_like(p1, dtype=bool)
    top1[np.arange(len(p1)), order[:, 0]] = True
    policies["top-1 only"] = top1
    from crc.agents.discipline.predict import (
        LEGACY_LOW_CONFIDENCE, LEGACY_TIGHT_GAP, LOW_CONFIDENCE, TIGHT_GAP)
    for lc, tg, what in ((LEGACY_LOW_CONFIDENCE, LEGACY_TIGHT_GAP, "legacy"),
                         (LOW_CONFIDENCE, TIGHT_GAP, "tuned")):
        fork = top1.copy()
        bl = (conf < lc) | (gap < tg)
        fork[np.where(bl)[0], order[bl, 1]] = True
        policies[f"fixed top-2 when borderline, {what} ({lc}/{tg})"] = fork
    if (ddir / "conformal.json").exists():       # a candidate model may have none yet
        bank = load_bank(ddir / "conformal.json")
        for a in (0.05, 0.10, 0.20):
            cal = pick_alpha(bank, a)
            s = cal.predict_set(p1)
            s[np.arange(len(p1)), order[:, 0]] = True   # top-1 is always read
            policies[f"deployed conformal alpha={a}"] = s
    report["routing_eval_half"] = {
        name: conformal_routing_metrics(m, disc_set=s, pred_all=pred_all,
                                        true_field=true_field, true_disc_idx=y_disc)
        for name, s in policies.items()}

    Path(args.out).write_text(json.dumps(report, indent=2))

    e = report["subsets"]["eval_half"]
    print(f"\nEVALUATION HALF (n={e['n']:,}, {e['ambiguous_share']:.1%} ambiguous)")
    print(f"  oracle field acc        {e['oracle_field_accuracy']:.4f}")
    for k, v in e.items():
        if isinstance(v, dict) and "end_to_end_field_accuracy" in v:
            print(f"  {k:18s} disc {v['discipline_accuracy']:.4f}  e2e field "
                  f"{v['end_to_end_field_accuracy']:.4f}  hF1 "
                  f"{v['hierarchical']['hierarchical_f1']:.4f}  "
                  f"field|disc-right {v['field_acc_given_discipline_correct']:.4f}")
    print("\n  routing (right field anywhere in the shortlist / avg candidates):")
    for k, v in report["routing_eval_half"].items():
        print(f"  {k:44s} {v['field_in_candidate_set']:.4f}   "
              f"disc coverage {v['discipline_set_coverage']:.4f}   "
              f"avg {v['avg_candidates']:.2f}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
