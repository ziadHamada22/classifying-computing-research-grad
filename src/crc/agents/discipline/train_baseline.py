"""Classical TF-IDF baselines for Agent 1: logistic regression, SVM, XGBoost,
random forest, naive Bayes and kNN -- the proposal's classical lineup.

Two reasons classical models earn their place in a transformer-era system:

  1. **A comparison floor.** The report needs to show what strong bag-of-words
     models achieve, so the transformer's gain is quantified rather than assumed.
     The proposal named TF-IDF + SVM and XGBoost explicitly; logistic regression
     is the member the ensemble actually uses.
  2. **A decorrelated ensemble member.** TF-IDF sees surface lexical evidence;
     the transformer sees contextual semantics. Their errors are less correlated
     than two transformers', so averaging them lifts the ensemble more than
     adding a third transformer would -- and their *disagreement* is itself an
     uncertainty signal.

All three share one TF-IDF configuration, so the comparison isolates the
classifier. Each saves probabilities in the same ``.npz`` shape as the
transformers (val / test / temporal), plus a ``metrics_<tag>.json`` and an
``eval_<tag>.json`` in the schemas ``train.py`` and ``crc.eval.evaluate`` write,
so the ensemble, calibration and summary tooling treat them identically.

Hyperparameters are chosen on validation only:

  logreg    C fixed at 4.0 (the original baseline; unchanged so its numbers
            reproduce exactly)
  svm       LinearSVC, C picked from a small grid on val macro-F1, then wrapped
            in sigmoid calibration so it emits probabilities (needed for ECE and
            ensembling -- a raw margin is not a distribution)
  xgboost   gradient-boosted trees on the 20k most class-informative n-grams
            (chi-squared, fitted on train), early-stopped on val log-loss
  rf        400 trees on the same 20k n-grams; min_samples_leaf 1 or 3 on val
  nb        Complement naive Bayes; alpha from a small grid on val
  knn       cosine k-nearest neighbours; k in {5, 15, 45, 135} on val

Run:
    python -m crc.agents.discipline.train_baseline                 # logreg -> tfidf
    python -m crc.agents.discipline.train_baseline --clf svm       # -> tfidf-svm
    python -m crc.agents.discipline.train_baseline --clf xgboost   # -> tfidf-xgboost
    python -m crc.agents.discipline.train_baseline --clf rf|nb|knn # -> tfidf-rf / -nb / -knn
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score

from crc.agents.discipline.features import build_text_column
from crc.taxonomy import DISCIPLINES, ID2LABEL, LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS = WORK / "corpus" / "corpus_v2.parquet"
TEMPORAL = WORK / "corpus" / "corpus_v2_temporal.parquet"
PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"

DEFAULT_TAGS = {"logreg": "tfidf", "svm": "tfidf-svm", "xgboost": "tfidf-xgboost",
                "rf": "tfidf-rf", "nb": "tfidf-nb", "knn": "tfidf-knn"}
MODEL_NAMES = {"logreg": "tfidf+logreg", "svm": "tfidf+svm",
               "xgboost": "tfidf+xgboost", "rf": "tfidf+random-forest",
               "nb": "tfidf+complement-nb", "knn": "tfidf+knn"}


def make_vectorizer(max_features: int, min_df: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=min_df, max_df=0.9,
        sublinear_tf=True, strip_accents="unicode",
        stop_words="english", max_features=max_features)


def to_canonical(proba: np.ndarray, classes) -> np.ndarray:
    """Reorder predict_proba columns to DISCIPLINES order.

    Every saved distribution across the system shares one column convention, so
    a classifier whose ``classes_`` come out in another order must be permuted.
    """
    col = {int(c): i for i, c in enumerate(classes)}
    out = np.zeros((len(proba), len(DISCIPLINES)))
    for j in range(len(DISCIPLINES)):
        out[:, j] = proba[:, col[j]]
    return out


def fit_logreg(Xtr, ytr, Xva, yva, args):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced",
                             n_jobs=-1)
    clf.fit(Xtr, ytr)
    return clf, None, {"C": args.C}


def fit_svm(Xtr, ytr, Xva, yva, args):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    # Select C on val with the raw margin (argmax is all macro-F1 needs), then
    # refit once with probability calibration for the chosen C.
    grid = [float(c) for c in args.svm_grid.split(",")]
    scores = {}
    for c in grid:
        m = LinearSVC(C=c, class_weight="balanced", max_iter=5000)
        m.fit(Xtr, ytr)
        scores[c] = float(f1_score(yva, m.predict(Xva), average="macro"))
        print(f"  svm C={c:<6g} val macro-F1 {scores[c]:.4f}")
    best_c = max(scores, key=scores.get)
    clf = CalibratedClassifierCV(
        LinearSVC(C=best_c, class_weight="balanced", max_iter=5000),
        method="sigmoid", cv=3)
    clf.fit(Xtr, ytr)
    return clf, None, {"C": best_c, "val_grid": scores, "calibration": "sigmoid, cv=3"}


def fit_xgboost(Xtr, ytr, Xva, yva, args):
    from sklearn.feature_selection import SelectKBest, chi2
    from xgboost import XGBClassifier

    # Trees over 100k sparse n-grams mostly split on noise; keep the n-grams that
    # are most class-informative on TRAIN (never val/test).
    sel = SelectKBest(chi2, k=min(args.xgb_features, Xtr.shape[1]))
    Xtr_s = sel.fit_transform(Xtr, ytr)
    Xva_s = sel.transform(Xva)
    clf = XGBClassifier(
        objective="multi:softprob", num_class=len(DISCIPLINES),
        n_estimators=args.xgb_rounds, learning_rate=args.xgb_lr,
        max_depth=args.xgb_depth, subsample=0.8, colsample_bytree=0.5,
        tree_method="hist", eval_metric="mlogloss",
        early_stopping_rounds=50, n_jobs=-1, random_state=42,
    )
    clf.fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)], verbose=100)
    params = {"k_features": int(Xtr_s.shape[1]), "max_depth": args.xgb_depth,
              "learning_rate": args.xgb_lr,
              "best_iteration": int(clf.best_iteration),
              "early_stopping": "50 rounds on val mlogloss"}
    return clf, sel, params


def fit_rf(Xtr, ytr, Xva, yva, args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, chi2

    # Same informative-n-gram selection as XGBoost (fitted on train only).
    sel = SelectKBest(chi2, k=min(args.xgb_features, Xtr.shape[1]))
    Xtr_s = sel.fit_transform(Xtr, ytr)
    Xva_s = sel.transform(Xva)
    scores, models = {}, {}
    for leaf in (1, 3):
        m = RandomForestClassifier(n_estimators=args.rf_trees, max_features="sqrt",
                                   min_samples_leaf=leaf, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=42)
        m.fit(Xtr_s, ytr)
        scores[leaf] = float(f1_score(yva, m.predict(Xva_s), average="macro"))
        models[leaf] = m
        print(f"  rf min_samples_leaf={leaf} val macro-F1 {scores[leaf]:.4f}")
    best = max(scores, key=scores.get)
    return models[best], sel, {"n_estimators": args.rf_trees, "k_features": int(Xtr_s.shape[1]),
                               "min_samples_leaf": best, "val_grid": scores}


def fit_nb(Xtr, ytr, Xva, yva, args):
    from sklearn.naive_bayes import ComplementNB

    # Complement NB: the text-classification variant that copes with imbalance.
    scores, models = {}, {}
    for a in (0.03, 0.1, 0.3, 1.0):
        m = ComplementNB(alpha=a).fit(Xtr, ytr)
        scores[a] = float(f1_score(yva, m.predict(Xva), average="macro"))
        models[a] = m
        print(f"  nb alpha={a:<5g} val macro-F1 {scores[a]:.4f}")
    best = max(scores, key=scores.get)
    return models[best], None, {"alpha": best, "val_grid": scores}


def fit_knn(Xtr, ytr, Xva, yva, args):
    from sklearn.neighbors import KNeighborsClassifier

    # Cosine kNN over the TF-IDF vectors (the lexical twin of the SPECTER2 kNN).
    scores, models = {}, {}
    for k in (5, 15, 45, 135):
        m = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute",
                                 weights="distance", n_jobs=-1).fit(Xtr, ytr)
        scores[k] = float(f1_score(yva, m.predict(Xva), average="macro"))
        models[k] = m
        print(f"  knn k={k:<3d} val macro-F1 {scores[k]:.4f}")
    best = max(scores, key=scores.get)
    return models[best], None, {"k": best, "val_grid": scores}


FITTERS = {"logreg": fit_logreg, "svm": fit_svm, "xgboost": fit_xgboost,
           "rf": fit_rf, "nb": fit_nb, "knn": fit_knn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf", default="logreg", choices=sorted(FITTERS))
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--temporal", default=str(TEMPORAL))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--C", type=float, default=4.0, help="logreg only")
    ap.add_argument("--svm-grid", default="0.03,0.1,0.3,1.0")
    ap.add_argument("--xgb-features", type=int, default=20_000)
    ap.add_argument("--xgb-rounds", type=int, default=2000)
    ap.add_argument("--xgb-lr", type=float, default=0.1)
    ap.add_argument("--xgb-depth", type=int, default=6)
    ap.add_argument("--rf-trees", type=int, default=400)
    ap.add_argument("--max-features", type=int, default=100_000)
    ap.add_argument("--min-df", type=int, default=3)
    args = ap.parse_args()

    import joblib

    from crc.eval.evaluate import evaluate_split

    tag = args.tag or DEFAULT_TAGS[args.clf]
    out_dir = MODELS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.corpus)
    df["text_in"] = build_text_column(df)
    df["label"] = df["discipline"].map(LABEL2ID).astype(int)
    splits = {s: df[df["split"] == s].reset_index(drop=True)
              for s in ("train", "val", "test")}
    temporal = pd.read_parquet(args.temporal).reset_index(drop=True)
    temporal["text_in"] = build_text_column(temporal)
    temporal["label"] = temporal["discipline"].map(LABEL2ID).astype(int)
    print(f"{MODEL_NAMES[args.clf]}: train {len(splits['train']):,} · "
          f"val {len(splits['val']):,} · test {len(splits['test']):,} · "
          f"temporal {len(temporal):,}")

    t0 = time.time()
    vec = make_vectorizer(args.max_features, args.min_df)
    Xtr = vec.fit_transform(splits["train"]["text_in"])
    ytr = splits["train"]["label"].to_numpy()
    Xva = vec.transform(splits["val"]["text_in"])
    yva = splits["val"]["label"].to_numpy()
    clf, selector, params = FITTERS[args.clf](Xtr, ytr, Xva, yva, args)
    fit_secs = time.time() - t0
    print(f"fit in {fit_secs:.1f}s  {params}")

    def predict(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
        t = time.perf_counter()
        X = vec.transform(frame["text_in"])
        if selector is not None:
            X = selector.transform(X)
        p = to_canonical(clf.predict_proba(X), clf.classes_)
        return p, time.perf_counter() - t

    results = {"model": MODEL_NAMES[args.clf], "tag": tag,
               "train_seconds": round(fit_secs, 1), "params": params}
    evaluation = {"model_dir": str(out_dir), "tag": tag, "temperature": 1.0,
                  "device": "cpu", "splits": {}}
    for name, frame in (("val", splits["val"]), ("test", splits["test"]),
                        ("temporal", temporal)):
        probs, secs = predict(frame)
        labels = frame["label"].to_numpy()
        preds = probs.argmax(1)
        rep = classification_report(
            [ID2LABEL[i] for i in labels], [ID2LABEL[i] for i in preds],
            labels=DISCIPLINES, digits=4, zero_division=0)
        results[name] = {
            "macro_f1": float(f1_score(labels, preds, average="macro")),
            "weighted_f1": float(f1_score(labels, preds, average="weighted")),
            "accuracy": float((preds == labels).mean()),
            "report": rep,
        }
        # Saved as log-probs so the tooling's softmax recovers the distribution.
        np.savez(out_dir / f"{name}_predictions.npz",
                 logits=np.log(np.clip(probs, 1e-9, 1.0)), labels=labels)
        if name in ("test", "temporal"):
            key = "test" if name == "test" else "temporal_test"
            r = evaluate_split(frame, probs, key)
            r["latency"] = {"total_seconds": round(secs, 2),
                            "docs_per_second": round(len(frame) / secs, 1),
                            "ms_per_doc": round(1000 * secs / len(frame), 2)}
            evaluation["splits"][key] = r
        print(f"\n===== {name} (macro-F1 {results[name]['macro_f1']:.4f}) =====")
        print(rep)

    joblib.dump({"vectorizer": vec, "selector": selector, "classifier": clf},
                out_dir / "pipeline.joblib")
    (RESULTS / f"metrics_{tag}.json").write_text(json.dumps(results, indent=2))
    (RESULTS / f"eval_{tag}.json").write_text(json.dumps(evaluation, indent=2))
    print(f"model   -> {out_dir / 'pipeline.joblib'}")
    print(f"metrics -> {RESULTS / f'metrics_{tag}.json'}")


if __name__ == "__main__":
    main()
