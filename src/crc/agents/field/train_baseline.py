"""Classical TF-IDF baselines for Agent 2 (field), for the comparison floor.

Same idea as Agent 1's classical baselines: quantify what a bag-of-words model
achieves on the field task, so the fine-tuned encoder's gain is measured rather
than assumed. Trained on exactly the data the deployed ``field-scibert-v2`` saw
(the self-training-augmented corpus, train split), one flat 38-way classifier,
and scored the way Agent 2 runs: **conditioned** on the true discipline (argmax
within that discipline's ballot), plus the flat 38-way number for reference.

Evaluated on the field-test split, the 2025+ field slice (the unchanged corpus
labeller applied to the 2025+ holdout), and -- the fair referee -- the human
labelled Web of Science fields (``evaluate_wos_fields``' papers and mapping):
the weak labels are keyword- and category-derived, so a bag-of-words model is
naturally good at reproducing them, and only labels the labeller never touched
can say whether the encoder learned more than the labeller's vocabulary.

Run:
    python -m crc.agents.field.train_baseline            # logreg and svm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from crc.data.label_fields import label_frame
from crc.taxonomy.wos_map import WOS_AREA_TO_FIELD
from crc.taxonomy.fields import DISCIPLINE_FIELD_IDS, GLOBAL_LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def conditioned(scores: np.ndarray, classes: np.ndarray, disciplines) -> np.ndarray:
    """Argmax within each row's discipline ballot (global field ids)."""
    col = {int(c): i for i, c in enumerate(classes)}
    out = np.empty(len(scores), dtype=int)
    for i, d in enumerate(disciplines):
        allowed = [f for f in DISCIPLINE_FIELD_IDS[d] if f in col]
        out[i] = allowed[int(np.argmax([scores[i, col[f]] for f in allowed]))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clfs", default="logreg,svm")
    args = ap.parse_args()

    st = pd.read_parquet(CORPUS_DIR / "fields_corpus_st.parquet")
    st = st[st["field"].map(lambda f: f in GLOBAL_LABEL2ID)]
    train = st[st["split"] == "train"]
    test = st[st["split"] == "test"].reset_index(drop=True)
    t = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet").reset_index(drop=True)
    lab = label_frame(t)
    keep = lab["field"].map(lambda f: f in GLOBAL_LABEL2ID).to_numpy() & lab["field"].notna().to_numpy()
    temporal = lab[keep].reset_index(drop=True)
    wos = pd.read_parquet(WORK / "corpus" / "wos_pool.parquet")
    wos = wos[wos["wos_area"].isin(WOS_AREA_TO_FIELD)].reset_index(drop=True)
    wos["field"] = wos["wos_area"].map(WOS_AREA_TO_FIELD)
    wos["text"] = wos["abstract"].fillna("").astype(str)     # WoS has no titles
    print(f"train {len(train):,} · test {len(test):,} · 2025+ labelled {len(temporal):,} · "
          f"WoS {len(wos):,}")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True,
                          strip_accents="unicode", stop_words="english",
                          max_features=150_000)
    t0 = time.time()
    Xtr = vec.fit_transform(train["text"])
    ytr = train["field"].map(GLOBAL_LABEL2ID).to_numpy()
    sets = {"test": test, "2025+": temporal, "wos": wos}
    X = {k: vec.transform(v["text"]) for k, v in sets.items()}
    Y = {k: v["field"].map(GLOBAL_LABEL2ID).to_numpy() for k, v in sets.items()}

    report: dict = {"train_rows": int(len(train)), "features": int(Xtr.shape[1]),
                    "conditioning": "oracle (true discipline)", "models": {}}
    for name in args.clfs.split(","):
        t1 = time.time()
        if name == "logreg":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=4.0, max_iter=2000)
        else:
            from sklearn.svm import LinearSVC
            clf = LinearSVC(C=0.3, max_iter=5000)
        clf.fit(Xtr, ytr)
        secs = time.time() - t1
        r = {"train_seconds": round(secs + (t1 - t0), 1)}
        for k in sets:
            S = (clf.decision_function(X[k]) if name == "svm" else clf.predict_log_proba(X[k]))
            pc = conditioned(S, clf.classes_, sets[k]["discipline"].to_numpy())
            flat = clf.classes_[S.argmax(1)]
            r[k] = {"conditioned_accuracy": round(float((pc == Y[k]).mean()), 4),
                    "conditioned_macro_f1": round(float(f1_score(Y[k], pc, average="macro")), 4),
                    "flat_accuracy": round(float((flat == Y[k]).mean()), 4)}
        report["models"][f"tfidf-{name}"] = r
        print(f"tfidf+{name}: test cond. acc {r['test']['conditioned_accuracy']:.4f} "
              f"(macro-F1 {r['test']['conditioned_macro_f1']:.4f}, flat {r['test']['flat_accuracy']:.4f}) · "
              f"2025+ cond. acc {r['2025+']['conditioned_accuracy']:.4f} · "
              f"WoS {r['wos']['conditioned_accuracy']:.4f}  [{r['train_seconds']:.0f}s]")
    (RESULTS / "field_baselines_eval.json").write_text(json.dumps(report, indent=2))
    print(f"saved -> {RESULTS / 'field_baselines_eval.json'}")


if __name__ == "__main__":
    main()
