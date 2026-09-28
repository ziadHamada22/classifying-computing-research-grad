"""Agent 2's confidence and errors: calibration, the borderline flag, conformal
field sets, what the errors are, and how much it drifts on 2025+ papers.

Agent 1's uncertainty was measured, corrected and replaced by conformal sets;
Agent 2's never was. Its "borderline" flag still uses the prototype-era Agent 1
constants (top field below 0.55, or a gap below 0.12), copied without tuning, and
its calibration (ECE) has never been reported. Everything here runs from saved
logits (no GPU), with the discipline fixed to the truth, because the question is
Agent 2's own confidence *given* the right ballot:

1. **Calibration** -- ECE of the ballot-masked probabilities on val, test and the
   2025+ field slice.
2. **The borderline flag** -- how many errors the inherited 0.55 / 0.12 flag
   catches, and the pair a validation sweep would choose.
3. **Conformal field sets** -- split conformal (LAC) over each discipline's
   ballot, calibrated on validation or on the 2025+ calibration half, scored on
   test and on the 2025+ evaluation half.
4. **Error anatomy** -- how many errors name the weak labeller's own runner-up
   field (a defensible disagreement), accuracy by the evidence that produced the
   label, and the most-confused field pairs.
5. **Drift** -- per-discipline oracle accuracy, 2015-2024 test vs 2025+.

Run:
    python -m crc.eval.evaluate_field_uncertainty
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from crc.agents.discipline.calibrate import expected_calibration_error
from crc.agents.discipline.conformal import calibrate, evaluate_sets, save_bank
from crc.agents.discipline.hierarchy import softmax
from crc.data.label_fields import label_frame
from crc.taxonomy import DISCIPLINES
from crc.taxonomy.fields import DISCIPLINE_FIELD_IDS, GLOBAL_ID2LABEL, GLOBAL_LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
CACHE = WORK / "cache" / "joint"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"

N_FIELDS = len(GLOBAL_ID2LABEL)
FIELD_NAMES = [GLOBAL_ID2LABEL[i] for i in range(N_FIELDS)]
ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
LEGACY = (0.55, 0.12)


def ballot_matrix(logits: np.ndarray, disciplines) -> np.ndarray:
    """(n, 38) probabilities: masked softmax over each row's own ballot, zero
    elsewhere -- the distribution ``FieldClassifier.predict`` reports."""
    out = np.zeros((len(logits), N_FIELDS))
    for d in DISCIPLINES:
        m = np.asarray(disciplines) == d
        if m.any():
            allowed = DISCIPLINE_FIELD_IDS[d]
            out[np.ix_(m, allowed)] = softmax(logits[m][:, allowed], axis=1)
    return out


def flag_stats(P: np.ndarray, y: np.ndarray, low: float, gap: float) -> dict:
    srt = np.sort(P, axis=1)
    top, second = srt[:, -1], srt[:, -2]
    flagged = (top < low) | ((top - second) < gap)
    wrong = P.argmax(1) != y
    return {"low_confidence": low, "tight_gap": gap,
            "fire_rate": round(float(flagged.mean()), 4),
            "error_recall": round(float((flagged & wrong).sum() / max(1, wrong.sum())), 4),
            "error_precision": round(float((flagged & wrong).sum() / max(1, flagged.sum())), 4)}


def sweep(P: np.ndarray, y: np.ndarray, budget: float = 0.25) -> tuple[float, float]:
    best = None
    for low in np.round(np.arange(0.40, 0.96, 0.02), 3):
        for gap in np.round(np.arange(0.02, 0.41, 0.02), 3):
            s = flag_stats(P, y, low, gap)
            if s["fire_rate"] > budget:
                continue
            key = (s["error_recall"], s["error_precision"])
            if best is None or key > best[0]:
                best = (key, low, gap)
    return float(best[1]), float(best[2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-dir", default=str(MODELS / "field-scibert-v2"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deploy", action="store_true",
                    help="Write the 2025+-calibrated field bank to <field-dir>/conformal.json.")
    args = ap.parse_args()
    fdir = Path(args.field_dir)

    fc = pd.read_parquet(CORPUS_DIR / "fields_corpus.parquet")
    S = {}
    for split in ("val", "test"):
        z = np.load(fdir / f"{split}_predictions.npz", allow_pickle=True)
        df = fc[fc["split"] == split].reset_index(drop=True)
        assert np.array_equal(df["field"].map(GLOBAL_LABEL2ID).to_numpy(), z["labels"])
        S[split] = {"P": ballot_matrix(z["logits"], z["disciplines"]), "y": z["labels"],
                    "disc": np.asarray(z["disciplines"]), "df": df}

    # 2025+ field slice: same labeller, same halves as the conformal bank
    t = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet").reset_index(drop=True)
    lab = label_frame(t)
    ok = lab["field"].map(lambda f: f in GLOBAL_LABEL2ID).to_numpy() & lab["field"].notna().to_numpy()
    z2 = np.load(CACHE / f"a2_{fdir.name}_temporal.npy")
    idx = np.random.default_rng(args.seed).permutation(len(t))
    calib = np.zeros(len(t), bool)
    calib[idx[: len(t) // 2]] = True
    Pt = ballot_matrix(z2, t["discipline"].to_numpy())
    yt = np.full(len(t), -1)
    yt[ok] = lab.loc[ok, "field"].map(GLOBAL_LABEL2ID).astype(int)
    for name, m in (("2025+ calib half", ok & calib), ("2025+ eval half", ok & ~calib)):
        S[name] = {"P": Pt[m], "y": yt[m], "disc": t["discipline"].to_numpy()[m]}

    report: dict = {"field_model": fdir.name, "conditioning": "oracle (true discipline)"}

    # 1 · calibration ------------------------------------------------------------
    report["calibration"] = {}
    for name, s in S.items():
        acc = float((s["P"].argmax(1) == s["y"]).mean())
        report["calibration"][name] = {
            "n": int(len(s["y"])), "accuracy": round(acc, 4),
            "ece": round(expected_calibration_error(s["P"], s["y"]), 4),
            "mean_confidence": round(float(s["P"].max(1).mean()), 4),
            "per_discipline_accuracy": {
                d: round(float((s["P"].argmax(1) == s["y"])[s["disc"] == d].mean()), 4)
                for d in DISCIPLINES if (s["disc"] == d).any()}}

    # 2 · the borderline flag ----------------------------------------------------
    tuned = sweep(S["val"]["P"], S["val"]["y"])
    report["borderline_flag"] = {
        "legacy": {k: flag_stats(s["P"], s["y"], *LEGACY) for k, s in S.items()},
        "tuned_on_val": {k: flag_stats(s["P"], s["y"], *tuned) for k, s in S.items()},
        "tuned_thresholds": {"low_confidence": tuned[0], "tight_gap": tuned[1],
                             "budget": 0.25}}

    # 3 · conformal field sets ---------------------------------------------------
    report["conformal"] = {}
    banks = {}
    for src in ("val", "2025+ calib half"):
        banks[src] = {a: calibrate(S[src]["P"], S[src]["y"], alpha=a, method="lac",
                                   class_conditional=False,
                                   calib_source=f"{src}, oracle ballot", seed=args.seed)
                      for a in ALPHAS}
        report["conformal"][f"calibrated on {src}"] = {
            ev: {f"{a:g}": {k: v for k, v in evaluate_sets(
                     banks[src][a], S[ev]["P"], S[ev]["y"], class_names=FIELD_NAMES).items()
                     if k != "per_class"}
                 for a in (0.05, 0.1, 0.2)}
            for ev in ("test", "2025+ eval half")}
    if args.deploy:
        path = save_bank(fdir / "conformal.json", banks["2025+ calib half"])
        report["deployed_bank"] = str(path)

    # 4 · error anatomy (test) ---------------------------------------------------
    s, df = S["test"], S["test"]["df"]
    pred = s["P"].argmax(1)
    wrong = pred != s["y"]
    runner = df["field_runner_up"].map(lambda f: GLOBAL_LABEL2ID.get(f, -1)).to_numpy()
    report["errors"] = {
        "n_errors": int(wrong.sum()),
        "share_naming_labeller_runner_up": round(float((pred == runner)[wrong].mean()), 4),
        "accuracy_by_label_evidence": {
            ev: round(float((~wrong)[df["field_evidence"].to_numpy() == ev].mean()), 4)
            for ev in ("category", "both", "keyword")},
        "accuracy_by_label_margin": {
            f"{lo:.1f}-{hi:.1f}": round(float((~wrong)[(df["field_margin"] >= lo).to_numpy()
                                                       & (df["field_margin"] < hi).to_numpy()].mean()), 4)
            for lo, hi in ((0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))},
        "top_confusions": [
            {"true": GLOBAL_ID2LABEL[a], "pred": GLOBAL_ID2LABEL[b], "count": c}
            for (a, b), c in Counter(zip(s["y"][wrong], pred[wrong])).most_common(10)],
        "weakest_fields": sorted(
            [{"field": GLOBAL_ID2LABEL[f], "n": int((s["y"] == f).sum()),
              "recall": round(float((pred == s["y"])[s["y"] == f].mean()), 4)}
             for f in np.unique(s["y"])], key=lambda r: r["recall"])[:8],
    }

    out = RESULTS / "field_uncertainty_eval.json"
    out.write_text(json.dumps(report, indent=2))

    # ---- print --------------------------------------------------------------
    print("1 · calibration (oracle ballot)")
    for k, v in report["calibration"].items():
        print(f"   {k:18s} n={v['n']:6,} acc {v['accuracy']:.4f}  ECE {v['ece']:.4f}  "
              f"mean conf {v['mean_confidence']:.4f}")
    print("\n2 · borderline flag   (fire rate / error recall / precision)")
    for kind in ("legacy", "tuned_on_val"):
        th = LEGACY if kind == "legacy" else tuned
        for k, v in report["borderline_flag"][kind].items():
            print(f"   {kind:12s} {th[0]:.2f}/{th[1]:.2f} {k:18s} {v['fire_rate']:.3f} / "
                  f"{v['error_recall']:.3f} / {v['error_precision']:.3f}")
    print("\n3 · conformal field sets (coverage / avg set size / contested share)")
    for src, r in report["conformal"].items():
        for ev, rr in r.items():
            print(f"   {src:32s} -> {ev:16s} " + "  ".join(
                f"a={a}: {x['coverage']:.3f}/{x['avg_set_size']:.2f}/{x['routed_share']:.2f}"
                for a, x in rr.items()))
    e = report["errors"]
    print(f"\n4 · errors: {e['n_errors']:,}; naming the labeller's runner-up field: "
          f"{e['share_naming_labeller_runner_up']:.1%}")
    print(f"   accuracy by label evidence: {e['accuracy_by_label_evidence']}")
    print(f"   accuracy by label margin:   {e['accuracy_by_label_margin']}")
    print(f"   weakest fields: " + "; ".join(f"{w['field']} {w['recall']:.2f} (n={w['n']})"
                                             for w in e["weakest_fields"][:5]))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
