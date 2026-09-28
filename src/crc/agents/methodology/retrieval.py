"""Retrieval for Agent 3: the reference papers most like this one, and how they were read.

What retrieval is for here, and what it is not for, were both measured before this
was built (`scripts/rag_probe_agent3.py` -> `results/agent3_rag_probe.json`).

* **Not for editing the facet probabilities.** Blending the neighbours' facet
  labels into the model's moved macro AUC by -0.004 to +0.003 and pulled the rare
  facets under threshold (``human_data`` predicted on 2% of papers -> ~0%), which
  would make Survey, Qualitative and Case Study unreachable. The neighbours' labels
  come from the same label model the classifier was trained on, so averaging them
  in adds smoothing, not information.
* **For explanation.** Every answer can show the reader the most similar reference
  papers and the design each was given. That is evidence a reader can check, and it
  needs no accuracy claim.
* **For the case where the model has no evidence.** When no facet fires, the
  taxonomy used to fall back to Design & Creation by default -- a guess dressed as
  an answer. The neighbours' designs are an honest, data-driven replacement,
  reported as such (`predict.py`, `evaluate_facet_decisions.py`).

The embedding is the facet model's own [CLS] vector, taken from the forward pass
Agent 3 already runs, so retrieval costs no extra encoder. It is also the only one
of three candidates that retrieves papers of the same *design* rather than the same
*topic*: the nearest neighbour shares the derived design 59% of the time, against
39% for SPECTER2 and 40% for MiniLM, which retrieve by subject.

The store holds only the training split, so nothing evaluated on validation or
test can retrieve itself.

Build:
    python -m crc.agents.methodology.retrieval
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crc.taxonomy.facets import FACET_KEYS

STORE_FILE = "reference_store.npz"
#: Neighbours consulted for the no-evidence fallback; similarity temperature.
DEFAULT_K = 20
TAU = 0.05
DEFAULT_DESIGN = "Design & Creation (Design Science)"


@dataclass
class Neighbour:
    paper_id: str
    title: str
    design: str
    facets: list[str]
    similarity: float

    def to_dict(self) -> dict:
        return {"paper_id": self.paper_id, "title": self.title,
                "design": self.design, "facets": self.facets,
                "similarity": round(float(self.similarity), 4)}


class ReferenceStore:
    """Unit-normalised embeddings of labelled reference papers, searched by cosine."""

    def __init__(self, emb: np.ndarray, paper_ids, titles, designs,
                 facets: np.ndarray, has_evidence: np.ndarray | None = None):
        self.emb = np.asarray(emb, dtype=np.float32)
        self.paper_ids = np.asarray(paper_ids).astype(str)
        self.titles = np.asarray(titles).astype(str)
        self.designs = np.asarray(designs).astype(str)
        self.facets = np.asarray(facets, dtype=bool)
        self.has_evidence = (self.facets.any(1) if has_evidence is None
                             else np.asarray(has_evidence, dtype=bool))

    def __len__(self) -> int:
        return len(self.paper_ids)

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceStore":
        z = np.load(path, allow_pickle=True)
        return cls(z["emb"], z["paper_ids"], z["titles"], z["designs"], z["facets"])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, emb=self.emb.astype(np.float16),
                            paper_ids=self.paper_ids, titles=self.titles,
                            designs=self.designs, facets=self.facets)

    def search(self, q: np.ndarray, k: int = DEFAULT_K) -> tuple[np.ndarray, np.ndarray]:
        """Top-k indices and cosine similarities for one query vector."""
        q = np.asarray(q, dtype=np.float32).ravel()
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self.emb @ q
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]

    def neighbours(self, q: np.ndarray, k: int = 3) -> list[Neighbour]:
        idx, sims = self.search(q, k)
        return [Neighbour(self.paper_ids[i], self.titles[i], self.designs[i],
                          [f for f, on in zip(FACET_KEYS, self.facets[i]) if on],
                          float(s)) for i, s in zip(idx, sims)]

    def design_vote(self, q: np.ndarray, k: int = DEFAULT_K,
                    tau: float = TAU) -> tuple[str, float, dict[str, float]]:
        """Similarity-weighted design vote among neighbours that have evidence.

        Neighbours that themselves had no facet are excluded: their design is the
        old default, and letting them vote would launder the default back in.
        Returns (design, its share of the vote, full vote).
        """
        idx, sims = self.search(q, k * 3)
        keep = self.has_evidence[idx]
        idx, sims = idx[keep][:k], sims[keep][:k]
        if len(idx) == 0:
            return DEFAULT_DESIGN, 0.0, {}
        w = np.exp((sims - sims.max()) / tau)
        vote: dict[str, float] = {}
        for i, wi in zip(idx, w):
            vote[self.designs[i]] = vote.get(self.designs[i], 0.0) + float(wi)
        total = sum(vote.values())
        vote = {d: v / total for d, v in sorted(vote.items(), key=lambda kv: -kv[1])}
        best = next(iter(vote))
        return best, vote[best], vote


def build(model_dir: Path, corpus: Path) -> Path:
    """Embed the training split with the facet model and save it beside the model."""
    import pandas as pd

    from crc.agents.methodology.predict import MethodologyClassifier
    from crc.agents.methodology.train_facets import build_corpus
    from crc.taxonomy.facets import derive_designs

    df = build_corpus(corpus)
    train = df[df["split"] == "train"].reset_index(drop=True)
    titles = pd.read_parquet(corpus.parent / "computing_pool.parquet",
                             columns=["id", "title"])
    tmap = dict(zip(titles["id"].astype(str), titles["title"].fillna("")))

    clf = MethodologyClassifier.load(model_dir)
    _, emb = clf.encode(train["text"].tolist(), batch_size=64)
    hard = train[FACET_KEYS].to_numpy().astype(bool)
    designs = [derive_designs(dict(zip(FACET_KEYS, row)))[0][0] for row in hard]
    store = ReferenceStore(
        emb, train["paper_id"].astype(str),
        [" ".join(tmap.get(p, "").split()) for p in train["paper_id"].astype(str)],
        designs, hard)
    out = model_dir / STORE_FILE
    store.save(out)
    print(f"reference store: {len(store):,} training papers, "
          f"{int(store.has_evidence.sum()):,} with at least one facet -> {out}")
    return out


def main() -> None:
    from crc.agents.methodology.predict import DEPLOYED_LABELS, DEPLOYED_MODEL, MODELS

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODELS / DEPLOYED_MODEL))
    ap.add_argument("--facets", default=str(DEPLOYED_LABELS))
    args = ap.parse_args()
    out = build(Path(args.model_dir), Path(args.facets))
    print(json.dumps({"store": str(out)}))


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_K", "STORE_FILE", "Neighbour", "ReferenceStore", "build"]
