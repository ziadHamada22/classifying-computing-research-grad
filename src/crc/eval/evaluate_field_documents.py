"""Agent 2 on the input it actually receives: formats and whole documents.

Agent 2 was trained on raw ``title. abstract`` strings and has only ever been
evaluated on them. Inside the pipeline it receives something else:

* a PDF's chunks are formatted as Agent 1's are -- ``[abstract] Title. text``,
  ``[methods] Title. text`` -- so every chunk carries a section tag the field
  model never saw;
* a whole paper arrives as many chunks, pooled by a plain mean that the code
  itself describes as "the honest default until it is fitted" -- never fitted,
  never measured.

This measures both, with the discipline fixed to the truth (oracle
conditioning), because the question is Agent 2's own behaviour, not the cascade:

Part A -- abstract formats, on the 16,804 field-test papers:
    ``raw``      ``title. abstract``          (training format)
    ``tagged``   ``[abstract] Title. abstract``  (a PDF's abstract chunk)
    ``bare``     the abstract alone          (pasted text: no tag, no title)

Part B -- whole documents, on the full-text evaluation papers that carry a field
label and are not in Agent 2's training split: abstract only, title + abstract,
mean (deployed), max, and the section-weighted mean / geometric pool with
weights fitted on validation. Chunks are scored in the deployed format and, as a
check, in the untagged training-style format.

Chunk logits are cached under ``gp_data/cache/documents``.

Run:
    python -m crc.eval.evaluate_field_documents
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from crc.agents.discipline.aggregate import aggregate, fit_section_weights
from crc.agents.discipline.features import format_abstract_row, format_chunk
from crc.agents.discipline.hierarchy import softmax
from crc.eval.evaluate_joint import mcnemar
from crc.taxonomy.fields import DISCIPLINE_FIELD_IDS, GLOBAL_LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
CACHE = WORK / "cache" / "documents"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"

STRATEGIES = ["abstract_only", "title_abstract", "mean", "max",
              "weighted_mean", "weighted_geometric"]
NOISE = {"references", "acknowledgements"}
#: A neutral start for the section-weight fit: unlike discipline, there is no
#: prior reason to think the abstract dominates a paper's field.
NEUTRAL_PRIOR = {s: 1.0 for s in ["title", "abstract", "introduction", "background",
                                  "related_work", "methods", "results", "discussion",
                                  "conclusion", "appendix", "body", "other"]}


def ballot_probs(logits: np.ndarray, discipline: str) -> tuple[np.ndarray, list[int]]:
    """Masked softmax over one discipline's fields, exactly as ``predict`` does."""
    allowed = DISCIPLINE_FIELD_IDS[discipline]
    return softmax(np.atleast_2d(logits)[:, allowed], axis=1), list(allowed)


def load_field_model(field_dir: Path):
    from crc.agents.field.predict import FieldClassifier
    return FieldClassifier.load(field_dir)


def cached_logits(name: str, texts: list[str], loader) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npy"
    if path.exists():
        z = np.load(path)
        if len(z) == len(texts):
            return z
    z = loader().chunk_logits(texts, batch_size=64)
    np.save(path, z)
    return z


# ------------------------------------------------------------------ part A
def abstract_formats(df: pd.DataFrame, tag: str, loader) -> dict:
    y = df["field"].map(GLOBAL_LABEL2ID).to_numpy().astype(int)
    formats = {
        "raw": df["text"].tolist(),
        "tagged": [format_abstract_row(t, a) for t, a in zip(df["title"], df["abstract"])],
        "bare": df["abstract"].fillna("").tolist(),
    }
    hits, out = {}, {}
    for name, texts in formats.items():
        z = cached_logits(f"{tag}_fieldtest_{name}", texts, loader)
        pred = np.empty(len(df), dtype=int)
        for i, (row, d) in enumerate(zip(z, df["discipline"])):
            p, allowed = ballot_probs(row, d)
            pred[i] = allowed[int(p.argmax())]
        hits[name] = pred == y
        out[name] = {"accuracy": round(float(hits[name].mean()), 4),
                     "macro_f1": round(float(f1_score(y, pred, average="macro")), 4)}
    for name in ("tagged", "bare"):
        out[name]["mcnemar_vs_raw"] = mcnemar(hits["raw"], hits[name])
    return out


# ------------------------------------------------------------------ part B
def eval_documents(split_frames: dict, tag: str, loader, fmt: str) -> dict:
    titles = {}
    docs_by_split = {}
    for split, sub in split_frames.items():
        t = sub[sub["section"] == "title"]
        titles.update(dict(zip(t["paper_id"], t["text"])))
        if fmt == "tagged":        # what the pipeline sends
            texts = [format_chunk(x, s, titles.get(pid))
                     for x, s, pid in zip(sub["text"], sub["section"], sub["paper_id"])]
        else:                      # training-style: "Title. text", no tag
            texts = [(f"{titles.get(pid)}. {x}" if titles.get(pid) and s != "title" else x)
                     for x, s, pid in zip(sub["text"], sub["section"], sub["paper_id"])]
        z = cached_logits(f"{tag}_chunks_eval_fields_{split}_{fmt}", texts, loader)
        docs = []
        for pid, idx in sub.groupby("paper_id", sort=False).indices.items():
            r0 = sub.iloc[idx[0]]
            p, allowed = ballot_probs(z[idx], r0["discipline"])
            y = allowed.index(int(r0["field_id"]))
            docs.append((p, [str(s) for s in sub["section"].to_numpy()[idx]],
                         sub["n_words"].to_numpy()[idx].astype(float), y))
        docs_by_split[split] = docs

    val = docs_by_split["val"]
    weights = fit_section_weights([d[0] for d in val], [d[1] for d in val],
                                  [d[2] for d in val], np.array([d[3] for d in val]),
                                  prior=NEUTRAL_PRIOR, verbose=True)

    def decide(strategy, d):
        p, secs, nw, _ = d
        if strategy in ("abstract_only", "title_abstract"):
            want = {"abstract"} if strategy == "abstract_only" else {"abstract", "title"}
            m = [i for i, s in enumerate(secs) if s in want] or [0]
            return p[m].mean(axis=0)
        return aggregate(p, secs, nw, strategy, weights)[0]

    res: dict = {"section_weights": {k: round(v, 3) for k, v in weights.items()}}
    hits = {}
    for split, docs in docs_by_split.items():
        y = np.array([d[3] for d in docs])
        res[split] = {"n_papers": len(docs)}
        for strat in STRATEGIES:
            pred = np.array([decide(strat, d).argmax() for d in docs])
            hits[(split, strat)] = pred == y
            res[split][strat] = round(float((pred == y).mean()), 4)
    best = max(STRATEGIES, key=lambda s: (res["val"][s], s == "mean"))
    res["selected_on_val"] = best
    res["test_mcnemar"] = {
        f"{best}_vs_mean": mcnemar(hits[("test", "mean")], hits[("test", best)]),
        f"{best}_vs_abstract_only": mcnemar(hits[("test", "abstract_only")],
                                            hits[("test", best)]),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-dir", default=str(MODELS / "field-scibert-v2"))
    ap.add_argument("--out", default=str(RESULTS / "field_documents_eval.json"))
    args = ap.parse_args()
    fdir = Path(args.field_dir)
    tag = fdir.name
    _m = {}

    def loader():
        if "m" not in _m:
            _m["m"] = load_field_model(fdir)
        return _m["m"]

    report: dict = {"field_model": tag, "conditioning": "oracle (true discipline)"}

    # ---- A: formats, on the field-test papers ------------------------------
    fc = pd.read_parquet(CORPUS_DIR / "fields_corpus.parquet")
    ft = fc[(fc["split"] == "test") & fc["field"].map(lambda f: f in GLOBAL_LABEL2ID)]
    ft = ft.reset_index(drop=True)
    report["abstract_formats"] = abstract_formats(ft, tag, loader)
    print("A · abstract formats (oracle field accuracy on "
          f"{len(ft):,} test papers):")
    for k, v in report["abstract_formats"].items():
        print(f"   {k:7s} acc {v['accuracy']:.4f}  macro-F1 {v['macro_f1']:.4f}  "
              f"{v.get('mcnemar_vs_raw', '')}")

    # ---- B: whole documents ------------------------------------------------
    train_ids = set(pd.read_parquet(CORPUS_DIR / "fields_corpus_st.parquet",
                                    columns=["id", "split"]).query("split == 'train'")["id"])
    pool = pd.read_parquet(CORPUS_DIR / "fields_pool.parquet",
                           columns=["id", "field", "discipline"])
    pool = pool[pool["field"].map(lambda f: f in GLOBAL_LABEL2ID)]
    chunks = pd.read_parquet(CORPUS_DIR / "chunks_eval.parquet",
                             columns=["paper_id", "split", "section", "n_words", "text"])
    chunks = chunks[chunks["split"].isin(["val", "test"]) & ~chunks["section"].isin(NOISE)]
    chunks = chunks.merge(pool.rename(columns={"id": "paper_id"}), on="paper_id")
    chunks = chunks[~chunks["paper_id"].isin(train_ids)]
    chunks["field_id"] = chunks["field"].map(GLOBAL_LABEL2ID).astype(int)
    frames = {s: chunks[chunks["split"] == s].reset_index(drop=True) for s in ("val", "test")}
    print(f"\nB · whole documents: val {frames['val']['paper_id'].nunique():,} / test "
          f"{frames['test']['paper_id'].nunique():,} leak-free field-labelled papers")
    report["documents"] = {}
    for fmt in ("tagged", "untagged"):
        r = eval_documents(frames, tag, loader, fmt)
        report["documents"][fmt] = r
        print(f"   [{fmt}] selected on val: {r['selected_on_val']}")
        for strat in STRATEGIES:
            print(f"      {strat:20s} val {r['val'][strat]:.4f}  test {r['test'][strat]:.4f}")
        print(f"      {r['test_mcnemar']}")
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
