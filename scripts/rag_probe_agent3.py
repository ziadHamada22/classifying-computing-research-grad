"""Read-only probe: would retrieval (RAG / kNN over labelled papers) help Agent 3?

Nothing in Agent 3 is changed. For each test paper, retrieve its k nearest
training papers by an embedding, predict its facets as the similarity-weighted
mean of the neighbours' soft facet labels, and compare with the deployed facet
model -- alone, and blended 50/50 -- on:
  (a) AUC against the label model's facets (the weak-label measure Agent 3 reports)
  (b) the gold-free test: agreement with the methods / results text the model
      never read (the region labelling functions), where the facet model beat the
      abstract's own cues 14/16.
Three embeddings: SPECTER2 (citation/topic), all-MiniLM-L6-v2 (general sentence
semantics), and the facet model's own encoder (task-aligned).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crc.agents.methodology.train_facets import build_corpus  # noqa: E402
from crc.data.label_facets import ABSENT, PRESENT, REGION_NAMES, load_regions, vote_region  # noqa: E402
from crc.taxonomy.facets import FACET_KEYS  # noqa: E402

CORPUS = Path(r"C:\Users\ziada\gp_data\corpus")
MODELS = ROOT / "models"
CACHE = Path(r"C:\Users\ziada\gp_data\cache\rag_probe")
CACHE.mkdir(parents=True, exist_ok=True)
DEV = "cuda"


def embed(name: str, split: str, texts: list[str]) -> np.ndarray:
    path = CACHE / f"{name}_{split}.npy"
    if path.exists() and len(np.load(path)) == len(texts):
        return np.load(path)
    from transformers import AutoModel, AutoTokenizer
    src = {"specter2": "allenai/specter2_base",
           "minilm": "sentence-transformers/all-MiniLM-L6-v2",
           "facet_encoder": str(MODELS / "methodology-facets")}[name]
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModel.from_pretrained(src).to(DEV).eval().half()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            if name == "specter2":
                batch = [tok.sep_token + t for t in batch]     # no titles here
            enc = tok(batch, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(DEV)
            h = model(**enc).last_hidden_state.float()
            if name == "minilm":        # its documented pooling: attention-masked mean
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
            else:                        # [CLS]
                v = h[:, 0, :]
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    e = np.concatenate(out).astype(np.float32)
    np.save(path, e)
    return e


def knn(q: np.ndarray, store: np.ndarray, soft: np.ndarray, k: int = 20, tau: float = 0.05):
    out = np.zeros((len(q), soft.shape[1]))
    for s in range(0, len(q), 1024):
        S = q[s:s + 1024] @ store.T
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sim = np.take_along_axis(S, idx, axis=1)
        w = np.exp((sim - sim.max(1, keepdims=True)) / tau)
        out[s:s + 1024] = (w[:, :, None] * soft[idx]).sum(1) / w.sum(1, keepdims=True)
    return out


def main():
    df = build_corpus(CORPUS / "facets_pool.parquet")
    z = np.load(MODELS / "methodology-facets" / "test_facet_predictions.npz", allow_pickle=True)
    pids = z["paper_ids"].astype(str)
    model_p = z["probs"].astype(float)
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df.set_index("paper_id").loc[pids].reset_index()
    soft_cols = [f"p_{f}" for f in FACET_KEYS]
    y_bin = test[FACET_KEYS].to_numpy().astype(int)
    print(f"store (train) {len(train):,} papers · test {len(test):,}")

    # unseen-region votes (methods, results_conclusion) for the test papers
    regions = load_regions(set(pids))
    idx = {p: i for i, p in enumerate(pids)}
    votes = {f: np.zeros((len(pids), len(REGION_NAMES)), np.int8) for f in FACET_KEYS}
    for r in regions.itertuples(index=False):
        if r.region == "title" or r.paper_id not in idx:
            continue
        j = REGION_NAMES.index(r.region)
        for f, v in vote_region(r.text, r.region, int(r.n_words)).items():
            votes[f][idx[r.paper_id], j] = v
    unseen = [REGION_NAMES.index("methods"), REGION_NAMES.index("results_conclusion")]

    def unseen_agreement(P):
        acc, n = [], 0
        for k, f in enumerate(FACET_KEYS):
            pv = np.where(P[:, k] >= 0.5, PRESENT, ABSENT)
            for j in unseen:
                rv = votes[f][:, j]
                m = rv != 0
                acc.append(float((pv[m] == rv[m]).mean()))
        return float(np.mean(acc))      # mean over 16 facet x region comparisons

    def aucs(P):
        return {f: roc_auc_score(y_bin[:, k], P[:, k]) for k, f in enumerate(FACET_KEYS)
                if 0 < y_bin[:, k].sum() < len(y_bin)}

    rows = {"facet model (deployed)": model_p}
    for name in ("specter2", "minilm", "facet_encoder"):
        E_tr = embed(name, "train", train["text"].tolist())
        E_te = embed(name, "test", test["text"].tolist())
        kp = knn(E_te, E_tr, train[soft_cols].to_numpy())
        rows[f"kNN over {name}"] = kp
        rows[f"blend: model + kNN({name})"] = 0.5 * model_p + 0.5 * kp

    report = {}
    print(f"\n{'predictor':34s} {'macro AUC':>9s} {'rare-facet AUC':>14s} {'unseen-text agreement':>22s}")
    rare = ["human_data", "field_context", "secondary", "intervenes"]
    for name, P in rows.items():
        a = aucs(P)
        r = {"macro_auc": round(float(np.mean(list(a.values()))), 4),
             "rare_facet_auc": round(float(np.mean([a[f] for f in rare])), 4),
             "unseen_region_agreement": round(unseen_agreement(P), 4),
             "per_facet_auc": {f: round(v, 4) for f, v in a.items()}}
        report[name] = r
        print(f"{name:34s} {r['macro_auc']:>9.4f} {r['rare_facet_auc']:>14.4f} "
              f"{r['unseen_region_agreement']:>22.4f}")
    # neighbour topicality: how often the nearest neighbour shares the discipline
    for name in ("specter2", "minilm", "facet_encoder"):
        E_tr, E_te = np.load(CACHE / f"{name}_train.npy"), np.load(CACHE / f"{name}_test.npy")
        nn = (E_te @ E_tr.T).argmax(1)
        same_disc = float((train["discipline"].to_numpy()[nn] == test["discipline"].to_numpy()).mean())
        same_design = float((train["derived_design"].to_numpy()[nn] == test["derived_design"].to_numpy()).mean())
        report[f"nearest-neighbour {name}"] = {"same_discipline": round(same_disc, 4),
                                               "same_derived_design": round(same_design, 4)}
        print(f"nearest neighbour ({name:13s}): same discipline {same_disc:.3f} | same derived design {same_design:.3f}")
    out = ROOT / "results" / "agent3_rag_probe.json"
    out.write_text(json.dumps(report, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
