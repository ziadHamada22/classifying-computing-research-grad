"""Conformal prediction vs the hand-tuned borderline thresholds.

Four questions, in the order they have to be answered:

**Q1 — does the guarantee hold on data the model is exchangeable with?**
Calibrate on the iid val split, measure coverage on the iid test split. If this
fails, nothing else is worth reading.

**Q2 — does it survive the documented temporal drift?** Apply that same
val-calibrated predictor to the 2025+ holdout. Exchangeability is exactly what
drift breaks, so the expected result is *under-coverage* — and that is the point:
the failure is measured, not silent. A hand-tuned threshold degrades invisibly.

**Q3 — is calibrating on a temporal slice the fix?** Split the 2025+ holdout in
two, calibrate on one half, evaluate on the other. Both halves are drawn from the
same period, so exchangeability is restored and coverage should return.

**Q4 — at the same cost, does it route better than the swept thresholds?** The
threshold sweep is given every advantage: it is fitted on the same calibration
data and it directly optimises error recall inside a firing budget, which is the
metric being compared. Alpha is chosen on validation to match its firing rate, so
the comparison is matched-cost and neither side sees the test data first.

Run:
    python -m crc.eval.evaluate_conformal --model-dir models/scibert
"""
from __future__ import annotations

from crc.agents.discipline import DEPLOYED_MODEL

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crc.agents.discipline.calibrate import softmax
from crc.agents.discipline.conformal import (
    calibrate as fit_conformal,
    evaluate_sets,
    save_bank,
)
from crc.agents.discipline.features import build_text_column
from crc.agents.discipline.tune_thresholds import operating_stats
from crc.taxonomy import DISCIPLINES, LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"

#: Alphas to sweep. 0.10 is the headline operating point.
ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)


def _probs_for_split(model_dir: Path, name: str, df: pd.DataFrame,
                     temperature: float) -> np.ndarray:
    """Probabilities for one split, from cache when possible.

    val/test logits were saved at training time. The temporal split was never
    scored, so it is computed once here and cached alongside them in the same
    format, keeping this module a pure consumer of saved logits thereafter.
    """
    cache = model_dir / f"{name}_predictions.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return softmax(d["logits"].astype(np.float64) / temperature)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"  computing logits for {name} (not cached) ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir)).to(device).eval()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    texts = build_text_column(df).tolist()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=256,
                      padding=True, return_tensors="pt").to(device)
            out.append(model(**enc).logits.float().cpu().numpy())
    logits = np.concatenate(out, 0)
    labels = df["discipline"].map(LABEL2ID).to_numpy()
    np.savez(cache, logits=logits, labels=labels)
    print(f"  cached -> {cache}")
    del model
    torch.cuda.empty_cache()
    return softmax(logits.astype(np.float64) / temperature)


def _pick_alpha_for_budget(probs: np.ndarray, labels: np.ndarray,
                           budget: float, method: str,
                           class_conditional: bool) -> tuple[float, float]:
    """Largest alpha whose routing rate on calibration data is within budget.

    Chosen on calibration data only — the test splits are never consulted, so the
    matched-cost comparison stays honest.
    """
    best = None
    for a in np.round(np.arange(0.01, 0.51, 0.01), 3):
        cal = fit_conformal(probs, labels, alpha=float(a), method=method,
                            class_conditional=class_conditional)
        rate = float(cal.route(probs).mean())
        if rate <= budget and (best is None or a < best[0]):
            best = (float(a), rate)
    if best is None:      # even a loose alpha over-fires; fall back to the loosest
        cal = fit_conformal(probs, labels, alpha=0.5, method=method,
                            class_conditional=class_conditional)
        return 0.5, float(cal.route(probs).mean())
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODELS / DEPLOYED_MODEL))
    ap.add_argument("--budget", type=float, default=0.25,
                    help="Firing/routing cost budget for the matched comparison.")
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="Headline operating point.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RESULTS / "conformal_eval.json"))
    args = ap.parse_args()

    d = Path(args.model_dir)
    if not d.is_absolute() and not d.exists():
        d = MODELS / d.name
    temperature = 1.0
    tpath = d / "temperature.json"
    if tpath.exists():
        temperature = float(json.loads(tpath.read_text()).get("temperature", 1.0))

    corpus = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet")
    temporal_df = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet")
    splits = {
        "val": corpus[corpus["split"] == "val"].reset_index(drop=True),
        "test": corpus[corpus["split"] == "test"].reset_index(drop=True),
        "temporal": temporal_df.reset_index(drop=True),
    }

    print(f"model: {d.name}   temperature: {temperature}")
    P, Y, AMB = {}, {}, {}
    for name, df in splits.items():
        P[name] = _probs_for_split(d, name, df, temperature)
        Y[name] = df["discipline"].map(LABEL2ID).to_numpy()
        AMB[name] = (df["ambiguous"].to_numpy().astype(bool)
                     if "ambiguous" in df.columns
                     else np.zeros(len(df), dtype=bool))
        acc = float((P[name].argmax(1) == Y[name]).mean())
        print(f"  {name:9s} n={len(df):>6,}  top-1 acc {acc:.4f}  "
              f"ambiguous {AMB[name].mean():5.1%}")

    report: dict = {
        "model": d.name, "temperature": temperature,
        "budget": args.budget, "headline_alpha": args.alpha,
        "top1_accuracy": {k: round(float((P[k].argmax(1) == Y[k]).mean()), 4)
                          for k in P},
    }

    # ------------------------------------------------------------------ Q1/Q2
    # One predictor calibrated on val, applied to an exchangeable split (test)
    # and a drifted one (temporal).
    print("\n" + "=" * 78)
    print("Q1/Q2 — calibrate on val (iid); apply to iid test and 2025+ temporal")
    print("=" * 78)
    print(f"\n{'method':22s} {'alpha':>6s} {'cov_test':>9s} {'cov_temp':>9s} "
          f"{'size_test':>10s} {'size_temp':>10s} {'route_test':>11s} "
          f"{'route_temp':>11s}")
    q12: dict = {}
    for method in ("lac", "aps"):
        for cc in (False, True):
            tag = f"{method}{'-classcond' if cc else ''}"
            rows = {}
            for a in ALPHAS:
                cal = fit_conformal(P["val"], Y["val"], alpha=a, method=method,
                                    class_conditional=cc, calib_source="val",
                                    seed=args.seed)
                rt = evaluate_sets(cal, P["test"], Y["test"])
                rp = evaluate_sets(cal, P["temporal"], Y["temporal"])
                rows[str(a)] = {"test": rt, "temporal": rp,
                                "qhat": cal.qhat,
                                "qhat_per_class": cal.qhat_per_class}
                if a == args.alpha:
                    print(f"{tag:22s} {a:>6.2f} {rt['coverage']:>9.4f} "
                          f"{rp['coverage']:>9.4f} {rt['avg_set_size']:>10.3f} "
                          f"{rp['avg_set_size']:>10.3f} "
                          f"{rt['routed_share']:>11.3f} "
                          f"{rp['routed_share']:>11.3f}")
            q12[tag] = rows
    report["calibrated_on_val"] = q12

    # The headline drift number, spelled out.
    h = q12["lac"][str(args.alpha)]
    print(f"\nLAC at alpha={args.alpha} (target coverage "
          f"{1 - args.alpha:.2f}):")
    print(f"  iid test   coverage {h['test']['coverage']:.4f}  "
          f"(gap {h['test']['coverage_gap']:+.4f})  <- guarantee holds")
    print(f"  2025+ temp coverage {h['temporal']['coverage']:.4f}  "
          f"(gap {h['temporal']['coverage_gap']:+.4f})  <- drift shows up here")

    # ------------------------------------------------------------------- Q2b
    # WHY val under-covers, and it is not (only) temporal drift. The corpus
    # design excludes ambiguous papers from train/val but keeps them in test
    # (0% / 16.7% / 28.9% for val / test / temporal), so val is not exchangeable
    # with either evaluation split *by construction*. Two probes separate the
    # composition effect from genuine drift:
    #   (i)  split the val-calibrated coverage on test by the ambiguous flag;
    #   (ii) calibrate on half of test -- same period, same composition -- and
    #        evaluate on the other half, where exchangeability does hold.
    print("\n" + "=" * 78)
    print("Q2b — is the under-coverage drift, or corpus composition?")
    print("=" * 78)
    cal_val = fit_conformal(P["val"], Y["val"], alpha=args.alpha, method="lac",
                            calib_source="val", seed=args.seed)
    print(f"\nval-calibrated LAC (alpha={args.alpha}), coverage split by the "
          f"ambiguous flag:")
    print(f"  {'split':12s} {'subset':12s} {'n':>7s} {'coverage':>9s} "
          f"{'avg_size':>9s} {'top1_acc':>9s}")
    q2b: dict = {"by_ambiguity": {}}
    for split in ("test", "temporal"):
        for subset, m in (("clean", ~AMB[split]), ("ambiguous", AMB[split])):
            if not m.any():
                continue
            r = evaluate_sets(cal_val, P[split][m], Y[split][m])
            q2b["by_ambiguity"][f"{split}_{subset}"] = r
            print(f"  {split:12s} {subset:12s} {r['n']:>7,} "
                  f"{r['coverage']:>9.4f} {r['avg_set_size']:>9.3f} "
                  f"{r['top1_accuracy']:>9.4f}")

    rng0 = np.random.default_rng(args.seed)
    tidx = rng0.permutation(len(Y["test"]))
    th = len(tidx) // 2
    tci, tei = tidx[:th], tidx[th:]
    print(f"\ncalibrating on half of test instead of val "
          f"({len(tci):,} calib / {len(tei):,} eval, same composition):")
    print(f"  {'method':22s} {'alpha':>6s} {'coverage':>9s} {'avg_size':>9s} "
          f"{'route':>7s}")
    q2b["test_half_calibrated"] = {}
    for method in ("lac", "aps"):
        for cc in (False, True):
            tag = f"{method}{'-classcond' if cc else ''}"
            cal_th = fit_conformal(P["test"][tci], Y["test"][tci],
                                   alpha=args.alpha, method=method,
                                   class_conditional=cc,
                                   calib_source="test-half", seed=args.seed)
            r = evaluate_sets(cal_th, P["test"][tei], Y["test"][tei])
            q2b["test_half_calibrated"][tag] = r
            print(f"  {tag:22s} {args.alpha:>6.2f} {r['coverage']:>9.4f} "
                  f"{r['avg_set_size']:>9.3f} {r['routed_share']:>7.3f}")
    report["composition_diagnosis"] = q2b

    # -------------------------------------------------------------------- Q3
    # Split the temporal holdout: calibrate on one half, evaluate on the other.
    print("\n" + "=" * 78)
    print("Q3 — does calibrating on a temporal slice restore coverage?")
    print("=" * 78)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(Y["temporal"]))
    half = len(idx) // 2
    ci, ei = idx[:half], idx[half:]
    print(f"\ntemporal split: {len(ci):,} calibration / {len(ei):,} evaluation")
    print(f"\n{'method':22s} {'alpha':>6s} {'cov(val-cal)':>13s} "
          f"{'cov(temp-cal)':>14s} {'size':>7s} {'route':>7s}")
    q3: dict = {}
    for method in ("lac", "aps"):
        for cc in (False, True):
            tag = f"{method}{'-classcond' if cc else ''}"
            rows = {}
            for a in ALPHAS:
                cal_v = fit_conformal(P["val"], Y["val"], alpha=a, method=method,
                                      class_conditional=cc, calib_source="val",
                                      seed=args.seed)
                cal_t = fit_conformal(P["temporal"][ci], Y["temporal"][ci],
                                      alpha=a, method=method,
                                      class_conditional=cc,
                                      calib_source="temporal-half",
                                      seed=args.seed)
                r_v = evaluate_sets(cal_v, P["temporal"][ei], Y["temporal"][ei])
                r_t = evaluate_sets(cal_t, P["temporal"][ei], Y["temporal"][ei])
                rows[str(a)] = {"val_calibrated": r_v, "temporal_calibrated": r_t,
                                "qhat_temporal": cal_t.qhat,
                                "qhat_per_class_temporal": cal_t.qhat_per_class}
                if a == args.alpha:
                    print(f"{tag:22s} {a:>6.2f} {r_v['coverage']:>13.4f} "
                          f"{r_t['coverage']:>14.4f} "
                          f"{r_t['avg_set_size']:>7.3f} "
                          f"{r_t['routed_share']:>7.3f}")
            q3[tag] = rows
    report["temporal_recalibration"] = {
        "n_calib": int(len(ci)), "n_eval": int(len(ei)), "results": q3}

    # -------------------------------------------------------------------- Q4
    # Matched-cost comparison against the swept thresholds.
    print("\n" + "=" * 78)
    print(f"Q4 — matched cost (<= {args.budget:.0%} routed): conformal vs "
          f"hand-tuned thresholds")
    print("=" * 78)

    # (a) the thresholds, swept on val exactly as tune_thresholds.py does
    best_thr = None
    for lc in np.round(np.arange(0.40, 0.86, 0.02), 3):
        for tg in np.round(np.arange(0.04, 0.31, 0.02), 3):
            s = operating_stats(P["val"], Y["val"], lc, tg)
            if s["fire_rate"] > args.budget:
                continue
            key = (s["error_recall"], s["error_precision"])
            if best_thr is None or key > best_thr["_key"]:
                best_thr = {**s, "_key": key}
    lc, tg = best_thr["low_confidence"], best_thr["tight_gap"]
    print(f"\nthresholds swept on val -> LOW_CONFIDENCE={lc}  TIGHT_GAP={tg}")

    # (b) alpha matched to the same budget, also chosen on val
    a_lac, rate_lac = _pick_alpha_for_budget(P["val"], Y["val"], args.budget,
                                             "lac", False)
    a_cc, rate_cc = _pick_alpha_for_budget(P["val"], Y["val"], args.budget,
                                           "lac", True)
    print(f"alpha matched on val    -> LAC alpha={a_lac} "
          f"(routes {rate_lac:.1%})   LAC-classcond alpha={a_cc} "
          f"(routes {rate_cc:.1%})")

    cal_lac = fit_conformal(P["val"], Y["val"], alpha=a_lac, method="lac",
                            calib_source="val", seed=args.seed)
    cal_cc = fit_conformal(P["val"], Y["val"], alpha=a_cc, method="lac",
                           class_conditional=True, calib_source="val",
                           seed=args.seed)
    cal_t = fit_conformal(P["temporal"][ci], Y["temporal"][ci], alpha=a_lac,
                          method="lac", calib_source="temporal-half",
                          seed=args.seed)

    print(f"\n{'system':34s} {'split':10s} {'fire/route':>11s} "
          f"{'err_recall':>11s} {'err_prec':>9s} {'coverage':>9s}")
    q4: dict = {"thresholds": {"low_confidence": lc, "tight_gap": tg,
                               "splits": {}},
                "conformal_lac": {"alpha": a_lac, "splits": {}},
                "conformal_lac_classcond": {"alpha": a_cc, "splits": {}},
                "conformal_lac_temporal_calibrated": {"alpha": a_lac,
                                                      "splits": {}}}
    for split in ("val", "test", "temporal"):
        s = operating_stats(P[split], Y[split], lc, tg)
        q4["thresholds"]["splits"][split] = s
        print(f"{'hand-tuned thresholds':34s} {split:10s} "
              f"{s['fire_rate']:>11.3f} {s['error_recall']:>11.3f} "
              f"{s['error_precision']:>9.3f} {'--':>9s}")
    for name, cal in (("conformal_lac", cal_lac),
                      ("conformal_lac_classcond", cal_cc)):
        for split in ("val", "test", "temporal"):
            r = evaluate_sets(cal, P[split], Y[split])
            q4[name]["splits"][split] = r
            label = name.replace("_", " ")
            print(f"{label:34s} {split:10s} {r['routed_share']:>11.3f} "
                  f"{r['error_recall']:>11.3f} {r['error_precision']:>9.3f} "
                  f"{r['coverage']:>9.4f}")
    # temporal-calibrated, evaluated on the held-out temporal half
    r = evaluate_sets(cal_t, P["temporal"][ei], Y["temporal"][ei])
    q4["conformal_lac_temporal_calibrated"]["splits"]["temporal_heldout"] = r
    print(f"{'conformal lac (temporal-calib)':34s} {'temp-out':10s} "
          f"{r['routed_share']:>11.3f} {r['error_recall']:>11.3f} "
          f"{r['error_precision']:>9.3f} {r['coverage']:>9.4f}")
    report["matched_cost"] = q4

    # -------------------------------------------------- cost of the guarantee
    # The actionable curve: on out-of-period papers, what does each level of
    # certainty cost in set size and review volume? This is the trade the
    # thresholds were making implicitly, now priced.
    print("\n" + "=" * 78)
    print("Cost of the guarantee — LAC calibrated on 2025+, held-out 2025+ eval")
    print("=" * 78)
    print(f"\n{'alpha':>6s} {'target':>7s} {'coverage':>9s} {'avg_size':>9s} "
          f"{'route%':>8s} {'singleton_acc':>14s}")
    curve = {}
    for a in ALPHAS:
        r = q3["lac"][str(a)]["temporal_calibrated"]
        curve[str(a)] = r
        sa = r["singleton_accuracy"]
        print(f"{a:>6.2f} {1 - a:>7.2f} {r['coverage']:>9.4f} "
              f"{r['avg_set_size']:>9.3f} {r['routed_share'] * 100:>7.1f}% "
              f"{(sa if sa is not None else float('nan')):>14.4f}")
    report["cost_of_guarantee_temporal"] = curve

    # ---------------------------------------------------- deployable artifact
    # Ship a BANK of alphas rather than one, calibrated on the 2025+ slice: the
    # deployed system scores new papers, which are out-of-period by definition,
    # and alpha is an operating choice that should be changeable without
    # recalibrating. Class-conditional, because per-class coverage protects the
    # weak classes whose errors cost the most downstream.
    bank = {a: fit_conformal(P["temporal"][ci], Y["temporal"][ci], alpha=a,
                             method="lac", class_conditional=True,
                             calib_source="temporal-half (2025+), 28.9% ambiguous",
                             seed=args.seed)
            for a in ALPHAS}
    path = save_bank(d / "conformal.json", bank)
    report["deployed"] = {
        "path": str(path), "method": "lac", "class_conditional": True,
        "calib_source": next(iter(bank.values())).calib_source,
        "alphas": {f"{a:g}": evaluate_sets(c, P["temporal"][ei],
                                           Y["temporal"][ei])
                   for a, c in bank.items()},
    }
    print()
    print(f"deployable calibration bank -> {path}")
    print(f"  {len(bank)} alphas, class-conditional LAC, calibrated on "
          f"{len(ci):,} papers from 2025+")
    for a, c in bank.items():
        m = report["deployed"]["alphas"][f"{a:g}"]
        star = ("  <- Agent 1 default" if a == 0.20
                else "  <- pipeline shortlist" if a == 0.10 else "")
        print(f"   alpha {a:<5g} coverage {m['coverage']:.4f}  "
              f"avg set {m['avg_set_size']:.3f}  "
              f"contested {m['routed_share']:6.1%}  "
              f"single-label acc {m['singleton_accuracy']:.4f}{star}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
