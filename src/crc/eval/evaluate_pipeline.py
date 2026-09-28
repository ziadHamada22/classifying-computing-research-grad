"""End-to-end pipeline evaluation: Agent 1 -> Agent 2 on *predicted* disciplines.

Every field number reported so far (conditioned acc ~0.840) uses an **oracle**
discipline: Agent 2 is masked to each paper's *true* discipline. That measures
Agent 2 in isolation, but it is not how the system runs. In deployment Agent 1
predicts the discipline, and that prediction chooses Agent 2's ballot -- so a
paper Agent 1 sends to the wrong discipline *cannot* get the right field, because
the right field is not on the ballot at all. Errors do not add; they cascade.

This module measures the honest compounded number. It runs both real models on
the same field-test papers, feeding each agent the text format it was trained on
(Agent 1: an ``[abstract]``-tagged chunk; Agent 2: raw ``title. abstract``), so
each reproduces its published behaviour and the compounding is faithful.

**Leakage guard.** The field corpus and the discipline corpus were sampled
independently, so ~20% of the field-test papers happen to sit in Agent 1's
*training* split. Scoring Agent 1 on those would flatter it. The headline is
therefore reported on the papers Agent 1 never trained on; the full-set number is
shown alongside so the size of the leakage effect is visible.

Because each field belongs to exactly one discipline, strict field-correct
implies discipline-correct, which gives a clean identity:

    end_to_end_field_acc = discipline_acc x field_acc_given_discipline_correct

so the cascade loss is entirely attributable to Agent 1, and
``field_acc_given_discipline_correct`` should reproduce the oracle number --
confirming Agent 2 itself is not degraded, only starved of correct ballots.

Run:
    python -m crc.eval.evaluate_pipeline
"""
from __future__ import annotations

from crc.agents.discipline import DEPLOYED_MODEL

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from crc.agents.discipline.calibrate import softmax
from crc.agents.discipline.conformal import (
    calibrate as fit_conformal,
    load_bank,
    pick_alpha,
)
from crc.agents.discipline.features import format_abstract_row
from crc.agents.discipline.hierarchy import hierarchical_scores
# The fork baseline below is the one this evaluation has always reported, which
# ran on the prototype-era thresholds; pin them so its numbers stay reproducible.
from crc.agents.discipline.predict import (
    LEGACY_LOW_CONFIDENCE as LOW_CONFIDENCE,
    LEGACY_TIGHT_GAP as TIGHT_GAP,
    DisciplineClassifier,
)
from crc.agents.field.predict import FieldClassifier
from crc.taxonomy import DISCIPLINES
from crc.taxonomy.fields import (
    DISCIPLINE_FIELD_IDS,
    GLOBAL_ID2LABEL,
    GLOBAL_LABEL2ID,
)

WORK = Path(r"C:\Users\ziada\gp_data")
FIELDS_CORPUS = WORK / "corpus" / "fields_corpus.parquet"
DISC_CORPUS = WORK / "corpus" / "corpus_v2.parquet"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def conditioned_field_pred(field_logits: np.ndarray,
                           disciplines: list[str]) -> np.ndarray:
    """Argmax field restricted to each paper's given discipline's ballot."""
    out = np.empty(len(field_logits), dtype=int)
    for i, d in enumerate(disciplines):
        allowed = DISCIPLINE_FIELD_IDS[d]
        out[i] = allowed[int(np.argmax(field_logits[i, allowed]))]
    return out


def _metrics_for_subset(mask: np.ndarray, *, true_field: np.ndarray,
                        true_disc: np.ndarray, pred_disc: np.ndarray,
                        top2_disc: np.ndarray, borderline: np.ndarray,
                        co_listed: list, oracle_field: np.ndarray,
                        e2e_field: np.ndarray, e2e_fork_field: np.ndarray) -> dict:
    """All end-to-end numbers for one subset of papers (given by ``mask``)."""
    idx = np.where(mask)[0]
    n = len(idx)
    tf, td, pd_ = true_field[idx], true_disc[idx], pred_disc[idx]
    of, ef, eff = oracle_field[idx], e2e_field[idx], e2e_fork_field[idx]

    disc_correct = pd_ == td
    disc_acc = float(disc_correct.mean())
    co_correct = np.array([pd_[k] in co_listed[idx[k]] for k in range(n)])
    disc_colisted_acc = float(co_correct.mean())

    oracle_acc = float((of == tf).mean())
    e2e_acc = float((ef == tf).mean())
    fork_acc = float((eff == tf).mean())

    # field accuracy on the papers where the discipline was right -- should
    # reproduce the oracle, proving the loss is Agent 1's, not Agent 2's.
    fac = float((ef[disc_correct] == tf[disc_correct]).mean()) if disc_correct.any() else 0.0

    # of Agent 1's discipline errors, how many named a discipline the paper is
    # itself co-listed under (a defensible mistake, not a random one)?
    err = ~disc_correct
    defensible_share = float(co_correct[err].mean()) if err.any() else 0.0

    return {
        "n": int(n),
        "discipline_accuracy": round(disc_acc, 4),
        "discipline_colisted_accuracy": round(disc_colisted_acc, 4),
        "discipline_macro_f1": round(
            f1_score(td, pd_, average="macro", labels=DISCIPLINES), 4),
        "oracle_field_accuracy": round(oracle_acc, 4),
        "oracle_field_macro_f1": round(f1_score(tf, of, average="macro"), 4),
        "end_to_end_field_accuracy": round(e2e_acc, 4),
        "end_to_end_field_macro_f1": round(f1_score(tf, ef, average="macro"), 4),
        "end_to_end_with_borderline_fork": round(fork_acc, 4),
        "cascade_loss": round(oracle_acc - e2e_acc, 4),
        "field_acc_given_discipline_correct": round(fac, 4),
        "defensible_discipline_error_share": round(defensible_share, 4),
        "borderline_share": round(float(borderline[idx].mean()), 4),
        # Hierarchical P/R/F1 over discipline -> field (the proposal's metric).
        "hierarchical": hierarchical_scores(td, pd_, tf, ef),
    }


def field_pred_under_every_discipline(field_logits: np.ndarray) -> np.ndarray:
    """``(n, 6)``: Agent 2's best field under each discipline's ballot.

    Precomputing all six lets any routing policy — top-1, top-2, or a conformal
    set — be scored by indexing rather than re-running Agent 2.
    """
    out = np.empty((len(field_logits), len(DISCIPLINES)), dtype=int)
    for j, d in enumerate(DISCIPLINES):
        allowed = np.asarray(DISCIPLINE_FIELD_IDS[d])
        out[:, j] = allowed[np.argmax(field_logits[:, allowed], axis=1)]
    return out


def conformal_routing_metrics(mask: np.ndarray, *, disc_set: np.ndarray,
                              pred_all: np.ndarray, true_field: np.ndarray,
                              true_disc_idx: np.ndarray) -> dict:
    """Score a set-valued routing policy on one subset of papers.

    The question a prediction set answers is not "is the top guess right?" but
    "does the shortlist contain the right answer?" — which is what a reviewer or
    a local-LLM second opinion actually consumes. Cost is the shortlist length.
    """
    idx = np.where(mask)[0]
    s = disc_set[idx]                       # (m, 6) bool
    sizes = s.sum(axis=1)
    hits = ((pred_all[idx] == true_field[idx][:, None]) & s).any(axis=1)
    disc_covered = s[np.arange(len(idx)), true_disc_idx[idx]]
    singleton = sizes == 1

    return {
        "n": int(len(idx)),
        "discipline_set_coverage": round(float(disc_covered.mean()), 4),
        "field_in_candidate_set": round(float(hits.mean()), 4),
        "avg_candidates": round(float(sizes.mean()), 4),
        "routed_share": round(float((sizes != 1).mean()), 4),
        "empty_share": round(float((sizes == 0).mean()), 4),
        # the auto-commit path: papers the predictor certifies with one discipline
        "singleton_share": round(float(singleton.mean()), 4),
        "singleton_field_accuracy": (
            round(float(hits[singleton].mean()), 4) if singleton.any() else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discipline-dir", default=str(MODELS / DEPLOYED_MODEL))
    ap.add_argument("--field-dir", default=str(MODELS / "field-scibert-v2"))
    ap.add_argument("--fields-corpus", default=str(FIELDS_CORPUS))
    ap.add_argument("--disc-corpus", default=str(DISC_CORPUS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--conformal-alphas", default="0.02,0.05,0.10,0.20,0.30",
                    help="Alphas to sweep for the set-valued routing comparison.")
    ap.add_argument("--no-conformal", action="store_true")
    ap.add_argument("--use-deployed-bank", action="store_true",
                    help="Score the SHIPPED conformal.json instead of fitting a "
                         "fresh predictor on val. Measures what actually runs.")
    ap.add_argument("--out", default=str(RESULTS / "pipeline_eval.json"))
    args = ap.parse_args()

    df = pd.read_parquet(args.fields_corpus)
    df = df[df["split"] == args.split].reset_index(drop=True)
    df["label"] = df["field"].map(GLOBAL_LABEL2ID)
    df = df[df["label"].notna()].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    print(f"field-{args.split}: {len(df):,} papers")

    # --- leakage sets: which of these did Agent 1 see? -----------------------
    disc = pd.read_parquet(args.disc_corpus, columns=["id", "split"])
    a1_train = set(disc[disc["split"] == "train"]["id"])
    a1_all = set(disc["id"])
    ids = df["id"].tolist()
    in_a1_train = np.array([i in a1_train for i in ids])
    in_a1_any = np.array([i in a1_all for i in ids])
    print(f"  in Agent-1 train: {in_a1_train.sum():,} "
          f"({in_a1_train.mean()*100:.1f}%)   "
          f"unseen by Agent 1: {(~in_a1_any).sum():,} "
          f"({(~in_a1_any).mean()*100:.1f}%)")

    # --- run both models (batched; abstracts are single-chunk) ---------------
    print("loading models ...")
    a1 = DisciplineClassifier.load(args.discipline_dir)
    a2 = FieldClassifier.load(args.field_dir)
    # sanity: the discipline head's columns must be in canonical order.
    id2label = a1.model.config.id2label
    got = [id2label.get(i, id2label.get(str(i))) for i in range(len(DISCIPLINES))]
    assert got == list(DISCIPLINES), f"discipline order mismatch: {got}"

    a1_texts = [format_abstract_row(t, ab)
                for t, ab in zip(df["title"], df["abstract"])]
    a2_texts = df["text"].tolist()

    print("Agent 1 (discipline) ...")
    disc_probs = a1.chunk_probs(a1_texts, batch_size=args.batch_size)  # (N,6)
    print("Agent 2 (field) ...")
    field_logits = a2.chunk_logits(a2_texts, batch_size=args.batch_size)  # (N,38)

    # --- derive predictions --------------------------------------------------
    order = np.argsort(-disc_probs, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    pred_disc = np.array([DISCIPLINES[j] for j in top1])
    top2_disc = np.array([DISCIPLINES[j] for j in top2])
    conf = disc_probs[np.arange(len(df)), top1]
    gap = conf - disc_probs[np.arange(len(df)), top2]
    borderline = (conf < LOW_CONFIDENCE) | (gap < TIGHT_GAP)

    true_disc = df["discipline"].to_numpy()
    true_field = df["label"].to_numpy()
    co_listed = [set(x) for x in df["co_listed"]]

    oracle_field = conditioned_field_pred(field_logits, list(true_disc))
    e2e_field = conditioned_field_pred(field_logits, list(pred_disc))
    # The pipeline's uncertainty fork: on a borderline discipline it also reports
    # the field under the runner-up discipline. Credit a hit under either.
    e2e_fork_disc = [top2_disc[i] if borderline[i] else pred_disc[i]
                     for i in range(len(df))]
    e2e_fork_field = conditioned_field_pred(field_logits, list(e2e_fork_disc))
    # fork "correct" = correct under top-1, OR (borderline) correct under top-2
    fork_hit = (e2e_field == true_field) | (
        borderline & (e2e_fork_field == true_field))
    # encode fork as a pseudo-prediction for the shared metric helper
    e2e_fork_effective = np.where(fork_hit, true_field, e2e_field)

    common = dict(
        true_field=true_field, true_disc=true_disc, pred_disc=pred_disc,
        top2_disc=top2_disc, borderline=borderline, co_listed=co_listed,
        oracle_field=oracle_field, e2e_field=e2e_field,
        e2e_fork_field=e2e_fork_effective,
    )

    subsets = {
        "no_agent1_train_leak": ~in_a1_train,     # headline: Agent 1 never trained on these
        "unseen_by_agent1": ~in_a1_any,           # strictest
        "full_field_test": np.ones(len(df), bool),  # includes leaked papers
    }
    report = {
        "discipline_model": Path(args.discipline_dir).name,
        "field_model": Path(args.field_dir).name,
        "split": args.split,
        "subsets": {name: _metrics_for_subset(mask, **common)
                    for name, mask in subsets.items()},
    }

    # per-true-discipline breakdown on the headline subset
    head_mask = subsets["no_agent1_train_leak"]
    per_disc = {}
    for d in DISCIPLINES:
        m = head_mask & (true_disc == d)
        if not m.any():
            continue
        idx = np.where(m)[0]
        per_disc[d] = {
            "n": int(m.sum()),
            "discipline_recall": round(float((pred_disc[idx] == d).mean()), 4),
            "oracle_field_acc": round(float(
                (oracle_field[idx] == true_field[idx]).mean()), 4),
            "end_to_end_field_acc": round(float(
                (e2e_field[idx] == true_field[idx]).mean()), 4),
        }
    report["per_true_discipline_headline"] = per_disc

    # ------------------------------------------------- conformal set routing
    # The pipeline's fork reports the field under the top-2 disciplines when the
    # discipline is borderline. "Two" is arbitrary. A conformal set adapts: one
    # label where the model is sure, three where it genuinely is not, with a
    # stated coverage guarantee. This measures whether adapting beats top-2 at
    # comparable cost.
    #
    # Calibration source: the discipline model's own val split. That is a valid
    # choice *here* precisely because both it and the field-test set are 0%
    # ambiguous and drawn from 2015-2024 — the exchangeability that fails for the
    # temporal holdout holds for this pair (verified by the coverage figure).
    if not args.no_conformal:
        vpath = Path(args.discipline_dir)
        if not vpath.is_absolute() and not vpath.exists():
            vpath = MODELS / vpath.name
        npz = vpath / "val_predictions.npz"
        if not npz.exists():
            print(f"\n[conformal skipped: {npz} not found]")
        else:
            temperature = 1.0
            tj = vpath / "temperature.json"
            if tj.exists():
                temperature = float(
                    json.loads(tj.read_text()).get("temperature", 1.0))
            vd = np.load(npz, allow_pickle=True)
            vprobs = softmax(vd["logits"].astype(np.float64) / temperature)
            vlabels = vd["labels"].astype(int)

            pred_all = field_pred_under_every_discipline(field_logits)
            true_disc_idx = np.array([DISCIPLINES.index(x) for x in true_disc])
            alphas = [float(a) for a in args.conformal_alphas.split(",")]

            print("\n" + "=" * 72)
            print("CONFORMAL SET ROUTING vs the fixed top-2 fork")
            print("=" * 72)
            print("\n(headline subset: no Agent-1 train leakage, "
                  f"n={int(head_mask.sum()):,})")
            print(f"\n{'policy':26s} {'disc_cov':>9s} {'field_in_set':>13s} "
                  f"{'avg_cand':>9s} {'routed':>8s}")

            # baselines, expressed as set policies so the costs are comparable
            n = len(df)
            base_top1 = np.zeros((n, len(DISCIPLINES)), dtype=bool)
            base_top1[np.arange(n), top1] = True
            base_fork = base_top1.copy()
            base_fork[np.arange(n), top2] |= borderline

            conf: dict = {"calibration_source": "discipline val split (0% ambiguous, 2015-2024)",
                          "policies": {}}
            for name, m in (("top-1 only", base_top1),
                            ("fixed top-2 when borderline", base_fork)):
                r = conformal_routing_metrics(
                    head_mask, disc_set=m, pred_all=pred_all,
                    true_field=true_field, true_disc_idx=true_disc_idx)
                conf["policies"][name] = r
                print(f"{name:26s} {r['discipline_set_coverage']:>9.4f} "
                      f"{r['field_in_candidate_set']:>13.4f} "
                      f"{r['avg_candidates']:>9.3f} {r['routed_share']:>8.3f}")

            # Raw conformal sets. At loose alpha LAC can return an EMPTY set
            # (no discipline clears 1 - qhat), which as a routing policy is
            # strictly worse than guessing: it discards the top-1 answer. Both
            # variants are reported so the effect is visible rather than hidden,
            # and the second is the deployable one.
            bank = (load_bank(vpath / "conformal.json")
                    if args.use_deployed_bank else None)

            def _cal(a):
                """The predictor for alpha: the shipped one, or fitted on val."""
                if bank is not None:
                    return pick_alpha(bank, a)
                return fit_conformal(vprobs, vlabels, alpha=a, method="lac",
                                     calib_source="val")

            if bank is not None:
                src = next(iter(bank.values())).calib_source
                conf["calibration_source"] = f"DEPLOYED bank: {src}"
                print(f"  [scoring the shipped calibration: {src}]")

            for a in alphas:
                cal = _cal(a)
                m = cal.predict_set(disc_probs)
                r = conformal_routing_metrics(
                    head_mask, disc_set=m, pred_all=pred_all,
                    true_field=true_field, true_disc_idx=true_disc_idx)
                conf["policies"][f"conformal LAC alpha={a}"] = r
                print(f"{'conformal LAC a=' + format(a, '.2f'):26s} "
                      f"{r['discipline_set_coverage']:>9.4f} "
                      f"{r['field_in_candidate_set']:>13.4f} "
                      f"{r['avg_candidates']:>9.3f} {r['routed_share']:>8.3f}")

            print("\n  ... same, with an empty set falling back to top-1:")
            for a in alphas:
                cal = _cal(a)
                m = cal.predict_set(disc_probs)
                empty = m.sum(axis=1) == 0
                m = m.copy()
                m[empty, top1[empty]] = True
                r = conformal_routing_metrics(
                    head_mask, disc_set=m, pred_all=pred_all,
                    true_field=true_field, true_disc_idx=true_disc_idx)
                conf["policies"][f"conformal LAC alpha={a} +top1-fallback"] = r
                print(f"{'  LAC a=' + format(a, '.2f') + ' +fallback':26s} "
                      f"{r['discipline_set_coverage']:>9.4f} "
                      f"{r['field_in_candidate_set']:>13.4f} "
                      f"{r['avg_candidates']:>9.3f} {r['routed_share']:>8.3f}")
            report["conformal_routing"] = conf

    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    # --- print ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("END-TO-END PIPELINE  (Agent 1 predicts the discipline Agent 2 uses)")
    print("=" * 72)
    hdr = ("subset", "n", "disc_acc", "oracle_fld", "e2e_fld", "cascade", "+fork")
    print(f"\n{hdr[0]:22s} {hdr[1]:>7s} {hdr[2]:>9s} {hdr[3]:>11s} "
          f"{hdr[4]:>9s} {hdr[5]:>9s} {hdr[6]:>7s}")
    for name, s in report["subsets"].items():
        print(f"{name:22s} {s['n']:>7,} {s['discipline_accuracy']:>9.4f} "
              f"{s['oracle_field_accuracy']:>11.4f} "
              f"{s['end_to_end_field_accuracy']:>9.4f} "
              f"{s['cascade_loss']:>9.4f} "
              f"{s['end_to_end_with_borderline_fork']:>7.4f}")

    h = report["subsets"]["no_agent1_train_leak"]
    print(f"\nHEADLINE (no Agent-1 train leakage, n={h['n']:,}):")
    print(f"  Agent 1 discipline   : {h['discipline_accuracy']:.4f} strict "
          f"/ {h['discipline_colisted_accuracy']:.4f} co-listed")
    print(f"  Oracle field (true disc) : {h['oracle_field_accuracy']:.4f}")
    print(f"  END-TO-END field (pred disc): {h['end_to_end_field_accuracy']:.4f}"
          f"   <-- the honest number")
    print(f"  cascade loss              : {h['cascade_loss']:.4f} "
          f"(all attributable to Agent 1)")
    print(f"  field acc | disc correct  : {h['field_acc_given_discipline_correct']:.4f}"
          f"   (~oracle => Agent 2 not degraded)")
    print(f"  with borderline fork      : {h['end_to_end_with_borderline_fork']:.4f}"
          f"   (uncertainty propagation recovers some loss)")
    print(f"  of disc errors, defensible: {h['defensible_discipline_error_share']:.1%}"
          f" were co-listed on the paper")

    print(f"\nper true discipline (headline subset):")
    print(f"  {'discipline':24s} {'n':>6s} {'disc_rec':>9s} "
          f"{'oracle':>8s} {'e2e':>8s}")
    for d, s in per_disc.items():
        print(f"  {d:24s} {s['n']:>6,} {s['discipline_recall']:>9.4f} "
              f"{s['oracle_field_acc']:>8.4f} {s['end_to_end_field_acc']:>8.4f}")

    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
