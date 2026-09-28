"""Does Agent 2 know the discipline too? Hierarchy-consistent discipline decoding.

The cascade costs the pipeline 14.7 points of field accuracy, and all of it is
Agent 1's: a paper sent to the wrong discipline cannot get the right field. Agent
2, meanwhile, already computes a full 38-way field distribution for every paper,
and summing it within each discipline yields a discipline distribution learned
from a different corpus. This script measures whether pooling the two -- a
product of experts, ``p ~ p_A1^(1-w) * p_A2marg^w`` -- beats Agent 1 alone.

**Leakage guard, both directions.** The discipline and field corpora were sampled
independently, so they overlap:

* 26% of Agent 1's *test* papers sit in Agent 2's *training* split, so scoring the
  pooled model on them would flatter Agent 2's vote. They are excluded.
* ~20% of the field-test papers sit in Agent 1's training split (the guard
  `evaluate_pipeline` already applies). They are excluded too.

The 2025+ temporal holdout overlaps neither corpus, which makes it the cleanest
test here. The pooling weight is chosen on Agent 1's validation papers (minus
Agent 2's training papers) and then frozen; no test or temporal paper influences
it.

Reported per evaluation set: discipline accuracy / macro-F1 / per-class recall /
ECE, the ambiguous vs unambiguous split, and -- on the field-test papers -- the
end-to-end field accuracy and hierarchical F1. Discipline changes are tested with
an exact McNemar test on the same papers.

Run:
    python -m crc.eval.evaluate_joint
"""
from __future__ import annotations

from crc.agents.discipline import DEPLOYED_MODEL

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from crc.agents.discipline.calibrate import expected_calibration_error
from crc.agents.discipline.features import format_abstract_row
from crc.agents.discipline.hierarchy import (
    conditioned_field,
    field_to_discipline_marginal,
    hierarchical_scores,
    joint_discipline_probs,
    softmax,
)
from crc.taxonomy import DISCIPLINES, LABEL2ID
from crc.taxonomy.fields import GLOBAL_LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
CACHE = WORK / "cache" / "joint"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"

WEIGHT_GRID = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]


# ------------------------------------------------------------------ inference
def _cached(name: str, fn) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npy"
    if path.exists():
        return np.load(path)
    arr = fn()
    np.save(path, arr)
    return arr


def agent1_probs(tag: str, split: str, df: pd.DataFrame, a1_loader) -> np.ndarray:
    """Agent 1 probabilities, from its saved logits when they match ``df``."""
    npz = MODELS / tag / f"{split}_predictions.npz"
    if npz.exists():
        z = np.load(npz)
        labels = df["discipline"].map(LABEL2ID).to_numpy()
        if len(z["labels"]) == len(df) and np.array_equal(z["labels"], labels):
            return softmax(z["logits"])
    texts = [format_abstract_row(t, a) for t, a in zip(df["title"], df["abstract"])]
    return _cached(f"a1_{tag}_{split}", lambda: _probs(a1_loader(), texts))


def agent2_logits(tag: str, name: str, df: pd.DataFrame, a2_loader) -> np.ndarray:
    # Agent 2 was trained on raw "title. abstract" -- feed it exactly that.
    texts = df["text"].tolist()
    return _cached(f"a2_{tag}_{name}",
                   lambda: a2_loader().chunk_logits(texts, batch_size=64))


def _probs(clf, texts: list[str]) -> np.ndarray:
    """Chunk probabilities from a single classifier or an ensemble."""
    try:
        return clf.chunk_probs(texts, batch_size=64)
    except TypeError:                      # EnsembleClassifier takes no batch_size
        return clf.chunk_probs(texts)


# ------------------------------------------------------------------ scoring
def mcnemar(base_hit: np.ndarray, new_hit: np.ndarray) -> dict:
    fixed = int((~base_hit & new_hit).sum())
    broken = int((base_hit & ~new_hit).sum())
    p = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else 1.0
    return {"fixed": fixed, "broken": broken, "p_value": round(float(p), 5)}


def disc_metrics(probs: np.ndarray, y: np.ndarray,
                 ambiguous: np.ndarray | None = None) -> dict:
    pred = probs.argmax(1)
    hit = pred == y
    out = {
        "n": int(len(y)),
        "accuracy": round(float(hit.mean()), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro",
                                         labels=list(range(len(DISCIPLINES))))), 4),
        "ece": round(expected_calibration_error(probs, y), 4),
        "recall": {d: round(float(hit[y == j].mean()), 4)
                   for j, d in enumerate(DISCIPLINES) if (y == j).any()},
    }
    if ambiguous is not None and ambiguous.any():
        out["ambiguous_accuracy"] = round(float(hit[ambiguous].mean()), 4)
        out["unambiguous_accuracy"] = round(float(hit[~ambiguous].mean()), 4)
    return out


def select_weight(p1: np.ndarray, p2: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    curve = {}
    for w in WEIGHT_GRID:
        pred = joint_discipline_probs(p1, p2, w).argmax(1)
        curve[w] = round(float(f1_score(y, pred, average="macro")), 4)
    best = max(curve.values())
    # smallest weight reaching the best score: prefer the less-changed system
    w_star = min(w for w, s in curve.items() if s == best)
    return w_star, curve


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discipline-dir", default=str(MODELS / DEPLOYED_MODEL))
    ap.add_argument("--field-dir", default=str(MODELS / "field-scibert-v2"))
    ap.add_argument("--weight", type=float, default=None,
                    help="Skip selection and use this pooling weight.")
    ap.add_argument("--out", default=str(RESULTS / "joint_decoding_eval.json"))
    args = ap.parse_args()

    a1_tag, a2_tag = Path(args.discipline_dir).name, Path(args.field_dir).name
    _a1, _a2 = {}, {}

    def a1_loader():
        if "m" not in _a1:
            from crc.agents.discipline.predict import (
                DisciplineClassifier,
                EnsembleClassifier,
            )
            d = Path(args.discipline_dir)
            if (d / "ensemble.json").exists():
                _a1["m"] = EnsembleClassifier.load(d, conformal_alpha=None)
            else:
                _a1["m"] = DisciplineClassifier.load(d, conformal_alpha=None)
        return _a1["m"]

    def a2_loader():
        if "m" not in _a2:
            from crc.agents.field.predict import FieldClassifier
            _a2["m"] = FieldClassifier.load(args.field_dir)
        return _a2["m"]

    disc = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet")
    temporal = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet").reset_index(drop=True)
    fields = pd.read_parquet(CORPUS_DIR / "fields_corpus.parquet")
    fields = fields[fields["field"].map(GLOBAL_LABEL2ID).notna()]

    a1_train_ids = set(disc.loc[disc["split"] == "train", "id"])
    a2_train_ids = set(fields.loc[fields["split"] == "train", "id"])

    sets: dict[str, dict] = {}
    for split in ("val", "test"):
        df = disc[disc["split"] == split].reset_index(drop=True)
        clean = ~df["id"].isin(a2_train_ids).to_numpy()
        sets[f"a1_{split}"] = {"df": df, "clean": clean, "a1_split": split}
    sets["temporal"] = {"df": temporal, "clean": np.ones(len(temporal), bool),
                        "a1_split": "temporal"}
    for split in ("val", "test"):
        df = fields[fields["split"] == split].reset_index(drop=True)
        clean = ~df["id"].isin(a1_train_ids).to_numpy()
        sets[f"field_{split}"] = {"df": df, "clean": clean, "a1_split": None}

    for name, s in sets.items():
        df = s["df"]
        print(f"{name:10s} n={len(df):6,}  leak-free={s['clean'].sum():6,}")
        s["p1"] = (agent1_probs(a1_tag, s["a1_split"], df, a1_loader)
                   if s["a1_split"] else
                   _cached(f"a1_{a1_tag}_{name}", lambda df=df: _probs(
                       a1_loader(), [format_abstract_row(t, a) for t, a in
                                     zip(df["title"], df["abstract"])])))
        s["z2"] = agent2_logits(a2_tag, name, df, a2_loader)
        s["p2"] = field_to_discipline_marginal(softmax(s["z2"]))
        s["y"] = df["discipline"].map(LABEL2ID).to_numpy()
        s["amb"] = df["ambiguous"].to_numpy().astype(bool)

    # ---- select the pooling weight on validation (leak-free rows only) ------
    v = sets["a1_val"]
    m = v["clean"]
    if args.weight is None:
        w_star, curve = select_weight(v["p1"][m], v["p2"][m], v["y"][m])
    else:
        w_star, curve = args.weight, {}
    print(f"\nselected weight w = {w_star}  (val macro-F1 curve: {curve})")

    report: dict = {
        "discipline_model": a1_tag, "field_model": a2_tag,
        "pooling": "geometric (product of experts)",
        "selected_weight": w_star,
        "selection": "Agent-1 val papers not in Agent-2 train, macro-F1",
        "val_curve": {str(k): v for k, v in curve.items()},
        "sets": {},
    }

    for name in ("a1_val", "a1_test", "temporal", "field_test"):
        s = sets[name]
        m = s["clean"]
        y, p1, p2 = s["y"][m], s["p1"][m], s["p2"][m]
        pj = joint_discipline_probs(p1, p2, w_star)
        amb = s["amb"][m]
        entry = {
            "n": int(m.sum()),
            "agent1": disc_metrics(p1, y, amb),
            "agent2_marginal": disc_metrics(p2, y, amb),
            "joint": disc_metrics(pj, y, amb),
            "mcnemar_joint_vs_agent1": mcnemar(p1.argmax(1) == y, pj.argmax(1) == y),
        }
        if name.startswith("field"):
            df = s["df"][m].reset_index(drop=True)
            tf = df["field"].map(GLOBAL_LABEL2ID).to_numpy().astype(int)
            z2 = s["z2"][m]
            true_d = [DISCIPLINES[j] for j in y]
            for key, probs in (("agent1", p1), ("joint", pj)):
                pd_names = [DISCIPLINES[j] for j in probs.argmax(1)]
                pf = conditioned_field(z2, pd_names)
                entry[key]["end_to_end_field_accuracy"] = round(float((pf == tf).mean()), 4)
                entry[key]["hierarchical"] = hierarchical_scores(true_d, pd_names, tf, pf)
            of = conditioned_field(z2, true_d)
            entry["oracle_field_accuracy"] = round(float((of == tf).mean()), 4)
            e1 = conditioned_field(z2, [DISCIPLINES[j] for j in p1.argmax(1)]) == tf
            ej = conditioned_field(z2, [DISCIPLINES[j] for j in pj.argmax(1)]) == tf
            entry["mcnemar_e2e_field"] = mcnemar(e1, ej)
            per = {}
            for j, d in enumerate(DISCIPLINES):
                k = y == j
                if k.any():
                    per[d] = {"n": int(k.sum()),
                              "agent1_e2e": round(float(e1[k].mean()), 4),
                              "joint_e2e": round(float(ej[k].mean()), 4)}
            entry["per_discipline_e2e"] = per
        report["sets"][name] = entry

        a, j = entry["agent1"], entry["joint"]
        print(f"\n== {name} (leak-free n={entry['n']:,})")
        print(f"   discipline acc   A1 {a['accuracy']:.4f} -> joint {j['accuracy']:.4f}   "
              f"(A2-marginal alone {entry['agent2_marginal']['accuracy']:.4f})")
        print(f"   macro-F1         A1 {a['macro_f1']:.4f} -> joint {j['macro_f1']:.4f}")
        print(f"   CS recall        A1 {a['recall'].get('Computer Science', 0):.4f} -> "
              f"joint {j['recall'].get('Computer Science', 0):.4f}")
        print(f"   ECE              A1 {a['ece']:.4f} -> joint {j['ece']:.4f}")
        if "ambiguous_accuracy" in a:
            print(f"   ambiguous acc    A1 {a['ambiguous_accuracy']:.4f} -> "
                  f"joint {j['ambiguous_accuracy']:.4f}")
        print(f"   McNemar          {entry['mcnemar_joint_vs_agent1']}")
        if "end_to_end_field_accuracy" in a:
            print(f"   e2e field acc    A1 {a['end_to_end_field_accuracy']:.4f} -> "
                  f"joint {j['end_to_end_field_accuracy']:.4f}  "
                  f"(oracle {entry['oracle_field_accuracy']:.4f})  "
                  f"McNemar {entry['mcnemar_e2e_field']}")
            print(f"   hierarchical F1  A1 {a['hierarchical']['hierarchical_f1']:.4f} -> "
                  f"joint {j['hierarchical']['hierarchical_f1']:.4f}")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
