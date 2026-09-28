"""Retrieval-based discipline classifiers over SPECTER2 embeddings.

The proposal planned a retrieval store alongside the fine-tuned models: a kNN
index over embedded training papers, and -- after Karval & Singh (2025) -- a
principal-vector classifier that represents each class by the principal
directions of its embeddings. Both are implemented here, plus the neighbour
lookup the local LLM uses for retrieval-augmented few-shot prompts.

**Encoder.** ``allenai/specter2_base``, frozen, with its documented input format
(``title [SEP] abstract``, [CLS] pooling). It is trained on citation structure,
not on our labels, so the store is an *independent* view of each paper -- which
is the point of a second opinion.

**Store.** Agent 1's training split only (50,400 papers). Validation selects
every hyperparameter; test and the 2025+ holdout are touched once.

``knn``        cosine similarity to the store; the top ``k`` neighbours vote with
               weight ``exp(sim / tau)``.
``principal``  each class keeps the top ``r`` right singular vectors of its
               (unit-normalised) embedding matrix; a paper's class score is the
               share of its length lying in that class's subspace (the CLAFIC
               subspace method). ``r = 1`` is exactly "cosine to the class's
               principal vector".

Probabilities are produced for both (a temperature fitted on validation log-
loss), so they join the ensemble, calibration and summary tooling like any other
member. Outputs: ``models/specter2-knn``, ``models/specter2-principal``,
``results/metrics_*.json`` and ``results/eval_*.json`` in the shared schemas.

Run:
    python -m crc.agents.discipline.retrieval
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from crc.taxonomy import DISCIPLINES, ID2LABEL, LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
CACHE = WORK / "cache" / "retrieval"
PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"

ENCODER = "allenai/specter2_base"
N_CLASSES = len(DISCIPLINES)


# ------------------------------------------------------------------ encoder
class Specter2Encoder:
    """Frozen SPECTER2 base: title [SEP] abstract -> unit-norm [CLS] vector."""

    def __init__(self, device: str | None = None, max_length: int = 320):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(ENCODER)
        self.model = AutoModel.from_pretrained(ENCODER).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.max_length = max_length

    def texts(self, titles, abstracts) -> list[str]:
        sep = self.tok.sep_token
        return [f"{t or ''}{sep}{a or ''}" for t, a in zip(titles, abstracts)]

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt",
                               return_token_type_ids=False).to(self.device)
                cls = self.model(**enc).last_hidden_state[:, 0, :].float()
                out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


def cached_embeddings(name: str, df: pd.DataFrame, encoder_factory) -> np.ndarray:
    """Embeddings for ``df`` (title, abstract), cached by name + id list."""
    CACHE.mkdir(parents=True, exist_ok=True)
    emb_p, ids_p = CACHE / f"specter2_{name}.npy", CACHE / f"specter2_{name}_ids.json"
    ids = df["id"].tolist()
    if emb_p.exists() and ids_p.exists() and json.loads(ids_p.read_text()) == ids:
        return np.load(emb_p)
    enc = encoder_factory()
    t0 = time.time()
    emb = enc.encode(enc.texts(df["title"], df["abstract"]))
    print(f"  embedded {name}: {len(df):,} papers in {time.time()-t0:.0f}s")
    np.save(emb_p, emb)
    ids_p.write_text(json.dumps(ids))
    return emb


# ------------------------------------------------------------------ classifiers
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def topk_neighbours(queries: np.ndarray, store: np.ndarray, k: int,
                    chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """(indices, similarities) of the ``k`` most cosine-similar store rows."""
    idx = np.empty((len(queries), k), dtype=np.int64)
    sim = np.empty((len(queries), k), dtype=np.float32)
    for s in range(0, len(queries), chunk):
        S = queries[s:s + chunk] @ store.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-ps, axis=1)
        idx[s:s + chunk] = np.take_along_axis(part, order, axis=1)
        sim[s:s + chunk] = np.take_along_axis(ps, order, axis=1)
    return idx, sim


def knn_probs(nbr_idx: np.ndarray, nbr_sim: np.ndarray, store_labels: np.ndarray,
              k: int, tau: float) -> np.ndarray:
    w = np.exp((nbr_sim[:, :k] - nbr_sim[:, :1]) / tau)   # stable: max weight 1
    lab = store_labels[nbr_idx[:, :k]]
    out = np.zeros((len(nbr_idx), N_CLASSES))
    for c in range(N_CLASSES):
        out[:, c] = (w * (lab == c)).sum(axis=1)
    return out / out.sum(axis=1, keepdims=True)


class PrincipalSubspaceClassifier:
    """Each class = the span of its top-``r`` principal directions (CLAFIC)."""

    def __init__(self, r: int):
        self.r = r
        self.bases: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrincipalSubspaceClassifier":
        self.bases = []
        for c in range(N_CLASSES):
            # uncentred SVD: the directions carrying most of the class's energy
            _, _, vt = np.linalg.svd(X[y == c].astype(np.float64), full_matrices=False)
            self.bases.append(vt[: self.r].astype(np.float32))
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Share of each (unit) vector's squared length inside each subspace."""
        return np.stack([((X @ B.T) ** 2).sum(axis=1) for B in self.bases], axis=1)


def fit_temperature(scores: np.ndarray, y: np.ndarray,
                    grid=np.geomspace(1e-3, 10, 60)) -> float:
    """Temperature minimising validation log-loss of softmax(scores / T)."""
    best, best_nll = 1.0, np.inf
    for T in grid:
        p = _softmax(scores / T)
        nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        if nll < best_nll:
            best, best_nll = float(T), nll
    return best


# ------------------------------------------------------------------ store for the LLM
class RetrievalStore:
    """Nearest training papers for a query, as (text, label) few-shot examples."""

    def __init__(self, embeddings: np.ndarray, frame: pd.DataFrame):
        self.emb = embeddings
        self.frame = frame.reset_index(drop=True)

    @classmethod
    def load_train(cls) -> "RetrievalStore":
        df = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet",
                             columns=["id", "title", "abstract", "discipline", "split"])
        df = df[df["split"] == "train"].reset_index(drop=True)
        emb = cached_embeddings("train", df, Specter2Encoder)
        return cls(emb, df)

    def examples(self, query_emb: np.ndarray, k: int = 4,
                 exclude_ids: set | None = None) -> list[tuple[str, str]]:
        idx, _ = topk_neighbours(query_emb[None, :].astype(np.float32), self.emb,
                                 k + (len(exclude_ids) if exclude_ids else 0))
        out = []
        for i in idx[0]:
            row = self.frame.iloc[int(i)]
            if exclude_ids and row["id"] in exclude_ids:
                continue
            out.append((f"{row['title']}. {row['abstract']}", row["discipline"]))
            if len(out) == k:
                break
        # most similar last, so it sits closest to the query in the prompt
        return out[::-1]


# ------------------------------------------------------------------ build + evaluate
def _save(tag: str, name: str, probs: np.ndarray, frame: pd.DataFrame,
          results: dict, evaluation: dict, secs: float,
          encode_docs_per_second: float | None = None) -> None:
    from crc.eval.evaluate import evaluate_split

    out_dir = MODELS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    y = frame["discipline"].map(LABEL2ID).to_numpy()
    pred = probs.argmax(1)
    results[name] = {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "accuracy": float((pred == y).mean()),
        "report": classification_report(
            [ID2LABEL[i] for i in y], [ID2LABEL[i] for i in pred],
            labels=DISCIPLINES, digits=4, zero_division=0),
    }
    np.savez(out_dir / f"{name}_predictions.npz",
             logits=np.log(np.clip(probs, 1e-9, 1.0)), labels=y)
    if name in ("test", "temporal"):
        key = "test" if name == "test" else "temporal_test"
        r = evaluate_split(frame, probs, key)
        # A real document must be encoded first, and that dominates the cost:
        # report end-to-end throughput, with the classification step alongside.
        per_doc = secs / len(frame)
        if encode_docs_per_second:
            per_doc += 1.0 / encode_docs_per_second
        r["latency"] = {"docs_per_second": round(1.0 / per_doc, 1),
                        "ms_per_doc": round(1000 * per_doc, 2),
                        "classify_only_seconds": round(secs, 3),
                        "encode_docs_per_second": encode_docs_per_second,
                        "note": "end to end: SPECTER2 encode (timed on a 1,000-paper "
                                "sample, GPU otherwise idle) + classification"}
        evaluation["splits"][key] = r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-grid", default="5,10,25,50,100,200,400")
    ap.add_argument("--tau-grid", default="0.0025,0.005,0.01,0.02,0.05,0.1")
    ap.add_argument("--r-grid", default="1,2,5,10,15,25,50,100")
    args = ap.parse_args()

    corpus = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet")
    temporal = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet")
    frames = {s: corpus[corpus["split"] == s].reset_index(drop=True)
              for s in ("train", "val", "test")}
    frames["temporal"] = temporal.reset_index(drop=True)

    _enc = {}

    def factory():
        if "e" not in _enc:
            _enc["e"] = Specter2Encoder()
        return _enc["e"]

    E = {s: cached_embeddings(s, f, factory) for s, f in frames.items()}
    # Time the encoder on a fixed sample so throughput is honest even when the
    # embeddings above came from the cache.
    sample = frames["test"].head(1000)
    enc = factory()
    texts = enc.texts(sample["title"], sample["abstract"])
    enc.encode(texts[:64])                      # warm-up
    t0 = time.perf_counter()
    enc.encode(texts)
    encode_rate = round(len(texts) / (time.perf_counter() - t0), 1)
    print(f"SPECTER2 encode throughput: {encode_rate} docs/s")
    Y = {s: f["discipline"].map(LABEL2ID).to_numpy() for s, f in frames.items()}

    # ---------------- kNN
    ks = [int(k) for k in args.k_grid.split(",")]
    taus = [float(t) for t in args.tau_grid.split(",")]
    kmax = max(ks)
    nbrs = {s: topk_neighbours(E[s], E["train"], kmax) for s in ("val", "test", "temporal")}
    grid = {}
    for k in ks:
        for tau in taus:
            p = knn_probs(*nbrs["val"], Y["train"], k, tau)
            grid[(k, tau)] = float(f1_score(Y["val"], p.argmax(1), average="macro"))
    (k_star, tau_star) = max(grid, key=grid.get)
    print(f"kNN: k={k_star}, tau={tau_star} (val macro-F1 {grid[(k_star, tau_star)]:.4f})")
    res = {"model": "specter2 kNN", "tag": "specter2-knn", "train_seconds": 0.0,
           "params": {"k": k_star, "tau": tau_star, "encoder": ENCODER,
                      "store": "Agent 1 train split",
                      "val_grid": {f"k={k},tau={t}": round(v, 4) for (k, t), v in grid.items()}}}
    ev = {"model_dir": str(MODELS / "specter2-knn"), "tag": "specter2-knn",
          "temperature": 1.0, "device": "cpu", "splits": {}}
    for s in ("val", "test", "temporal"):
        t0 = time.perf_counter()
        p = knn_probs(*nbrs[s], Y["train"], k_star, tau_star)
        _save("specter2-knn", s, p, frames[s], res, ev, time.perf_counter() - t0,
              encode_rate)
        print(f"  {s:9s} macro-F1 {res[s]['macro_f1']:.4f}")
    (RESULTS / "metrics_specter2-knn.json").write_text(json.dumps(res, indent=2))
    (RESULTS / "eval_specter2-knn.json").write_text(json.dumps(ev, indent=2))

    # ---------------- principal subspaces
    rs = [int(r) for r in args.r_grid.split(",")]
    t0 = time.time()
    fitted = {r: PrincipalSubspaceClassifier(r).fit(E["train"], Y["train"]) for r in rs}
    fit_secs = time.time() - t0
    rgrid = {r: float(f1_score(Y["val"], m.scores(E["val"]).argmax(1), average="macro"))
             for r, m in fitted.items()}
    r_star = max(rgrid, key=rgrid.get)
    clf = fitted[r_star]
    T = fit_temperature(clf.scores(E["val"]), Y["val"])
    print(f"principal: r={r_star}, T={T:.4f} (val macro-F1 {rgrid[r_star]:.4f})")
    res = {"model": "specter2 principal subspace", "tag": "specter2-principal",
           "train_seconds": round(fit_secs, 1),
           "params": {"r": r_star, "temperature": T, "encoder": ENCODER,
                      "val_grid": {str(r): round(v, 4) for r, v in rgrid.items()}}}
    ev = {"model_dir": str(MODELS / "specter2-principal"), "tag": "specter2-principal",
          "temperature": T, "device": "cpu", "splits": {}}
    for s in ("val", "test", "temporal"):
        t0 = time.perf_counter()
        p = _softmax(clf.scores(E[s]) / T)
        _save("specter2-principal", s, p, frames[s], res, ev, time.perf_counter() - t0,
              encode_rate)
        print(f"  {s:9s} macro-F1 {res[s]['macro_f1']:.4f}")
    np.savez(MODELS / "specter2-principal" / "bases.npz",
             **{DISCIPLINES[c]: b for c, b in enumerate(clf.bases)})
    (RESULTS / "metrics_specter2-principal.json").write_text(json.dumps(res, indent=2))
    (RESULTS / "eval_specter2-principal.json").write_text(json.dumps(ev, indent=2))


if __name__ == "__main__":
    main()
