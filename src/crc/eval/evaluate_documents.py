"""Document-level evaluation over the full-text chunk corpus.

This is the experiment that justifies the architecture. Reading the whole
document is only worth its cost if it beats reading the abstract alone, so this
script measures exactly that, on the same papers, with the same model:

  abstract_only      classify just the abstract chunk (what the prototype did)
  title_abstract     title + abstract chunks
  mean               unweighted mean over every chunk
  weighted_mean      section x length x certainty weighting
  weighted_geometric same weights, geometric pooling
  max                the single most confident chunk

Section weights are fitted on the validation papers and then applied unchanged
to test, so the reported test number is not tuned on itself.

**Inference-faithful input (2026-09).** At inference every chunk is formatted
with the paper's title (``format_chunk(text, section, title)``); earlier runs of
this script omitted it, so they scored a slightly different input than the
deployed classifier sees. The title is now recovered from each paper's title
chunk and passed through (``--no-title-prefix`` reproduces the old behaviour).
Chunk probabilities are cached under ``gp_data/cache/documents`` so every
pooling or calibration question can be answered offline afterwards.

Run:
    python -m crc.eval.evaluate_documents --model-dir models/scibert
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from crc.agents.discipline.aggregate import (
    DEFAULT_SECTION_WEIGHTS,
    aggregate,
    fit_section_weights,
)
from crc.agents.discipline.calibrate import expected_calibration_error, softmax
from crc.agents.discipline.features import format_chunk
from crc.taxonomy import DISCIPLINES, LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CHUNKS = WORK / "corpus" / "chunks_v2.parquet"
CACHE = WORK / "cache" / "documents"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"

STRATEGIES = ["abstract_only", "title_abstract", "mean", "max",
              "weighted_mean", "weighted_geometric"]


def score_chunks(model, tok, texts: list[str], device: str, temperature: float,
                 max_length: int, batch_size: int) -> tuple[np.ndarray, float]:
    import torch

    out = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True,
                      max_length=max_length, padding=True,
                      return_tensors="pt").to(device)
            out.append(model(**enc).logits.float().cpu().numpy())
    elapsed = time.perf_counter() - t0
    logits = np.concatenate(out, axis=0) / temperature
    return softmax(logits), elapsed


def paper_titles(df: pd.DataFrame) -> dict:
    """Each paper's title, recovered from its ``title``-section chunk."""
    t = df[df["section"] == "title"]
    return dict(zip(t["paper_id"], t["text"]))


def chunk_texts(df: pd.DataFrame, title_prefix: bool) -> list[str]:
    """Exactly what the deployed classifier feeds the model for each chunk."""
    titles = paper_titles(df) if title_prefix else {}
    return [format_chunk(t, s, titles.get(pid))
            for t, s, pid in zip(df["text"], df["section"], df["paper_id"])]


def group_papers(df: pd.DataFrame, probs: np.ndarray):
    """Yield (paper_id, label, chunk_probs, sections, n_words) per document."""
    idx_by_paper: dict[str, list[int]] = defaultdict(list)
    for i, pid in enumerate(df["paper_id"].to_numpy()):
        idx_by_paper[pid].append(i)
    sections = df["section"].to_numpy()
    words = df["n_words"].to_numpy()
    labels = df["discipline"].map(LABEL2ID).to_numpy()
    for pid, idx in idx_by_paper.items():
        idx_arr = np.array(idx)
        yield (pid, int(labels[idx_arr[0]]), probs[idx_arr],
               [str(s) for s in sections[idx_arr]],
               words[idx_arr].astype(np.float64))


def apply_strategy(name: str, p: np.ndarray, secs: list[str],
                   nw: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    if name == "abstract_only":
        m = [i for i, s in enumerate(secs) if s == "abstract"]
        if not m:
            m = [i for i, s in enumerate(secs) if s == "title"] or [0]
        return p[m].mean(axis=0)
    if name == "title_abstract":
        m = [i for i, s in enumerate(secs) if s in ("abstract", "title")]
        return p[m].mean(axis=0) if m else p[0]
    doc, _ = aggregate(p, secs, nw, name, weights)
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--chunks", default=str(CHUNKS))
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--fit-weights", action="store_true", default=True)
    ap.add_argument("--max-val-papers", type=int, default=1500,
                    help="Cap papers used for weight fitting (it is iterative).")
    ap.add_argument("--no-title-prefix", action="store_true",
                    help="Score chunks without the title, as runs before 2026-09 did.")
    args = ap.parse_args()
    title_prefix = not args.no_title_prefix

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    d = Path(args.model_dir)
    if not d.is_absolute() and not d.exists():
        d = MODELS / d.name
    tag = d.name

    temperature = 1.0
    tpath = d / "temperature.json"
    if tpath.exists():
        temperature = float(json.loads(tpath.read_text()).get("temperature", 1.0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(str(d)).to(device).eval()
    tok = AutoTokenizer.from_pretrained(str(d))
    print(f"model {tag} | T={temperature:.3f} | device={device}")

    chunks = pd.read_parquet(args.chunks)
    print(f"chunk corpus: {len(chunks):,} chunks / "
          f"{chunks['paper_id'].nunique():,} papers")

    report: dict = {"tag": tag, "temperature": temperature,
                    "title_prefix": title_prefix, "splits": {}}
    fitted_weights = dict(DEFAULT_SECTION_WEIGHTS)

    for split in ("val", "test"):
        sub = chunks[chunks["split"] == split].reset_index(drop=True)
        if sub.empty:
            continue
        texts = chunk_texts(sub, title_prefix)
        cache = CACHE / (f"{tag}_{Path(args.chunks).stem}_{split}"
                         f"{'' if title_prefix else '_notitle'}.npz")
        if cache.exists() and len(np.load(cache)["probs"]) == len(sub):
            probs, elapsed = np.load(cache)["probs"], float(np.load(cache)["seconds"])
            print(f"  (chunk probabilities from cache {cache.name})")
        else:
            probs, elapsed = score_chunks(model, tok, texts, device, temperature,
                                          args.max_length, args.batch_size)
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez(cache, probs=probs, seconds=elapsed,
                     paper_id=sub["paper_id"].to_numpy().astype(str))
        docs = list(group_papers(sub, probs))
        n_docs = len(docs)
        print(f"\n=== {split}: {n_docs:,} papers, {len(sub):,} chunks "
              f"({elapsed:.1f}s, {len(sub)/elapsed:.0f} chunks/s) ===")

        # Fit section weights on val only, then reuse on test.
        if split == "val" and args.fit_weights:
            sample = docs[: args.max_val_papers]
            fitted_weights = fit_section_weights(
                [x[2] for x in sample], [x[3] for x in sample],
                [x[4] for x in sample],
                np.array([x[1] for x in sample]),
            )
            top = sorted(fitted_weights.items(), key=lambda kv: -kv[1])[:8]
            print("  fitted section weights (top): " +
                  ", ".join(f"{k}={v:.2f}" for k, v in top))

        y_true = np.array([x[1] for x in docs])
        split_res = {}
        for strat in STRATEGIES:
            P = np.stack([apply_strategy(strat, x[2], x[3], x[4], fitted_weights)
                          for x in docs])
            y_pred = P.argmax(axis=1)
            split_res[strat] = {
                "accuracy": round(float((y_pred == y_true).mean()), 4),
                "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
                "ece": round(expected_calibration_error(P, y_true), 4),
                "mean_confidence": round(float(P.max(axis=1).mean()), 4),
            }

        best = max(split_res, key=lambda k: split_res[k]["macro_f1"])
        print(f"  {'strategy':20s} {'macro-F1':>9s} {'accuracy':>9s} {'ECE':>7s}")
        for strat, r in sorted(split_res.items(),
                               key=lambda kv: -kv[1]["macro_f1"]):
            mark = " <-- best" if strat == best else ""
            print(f"  {strat:20s} {r['macro_f1']:>9.4f} {r['accuracy']:>9.4f} "
                  f"{r['ece']:>7.4f}{mark}")

        # Per-class detail for the winning strategy.
        P = np.stack([apply_strategy(best, x[2], x[3], x[4], fitted_weights)
                      for x in docs])
        print("\n" + classification_report(
            [DISCIPLINES[i] for i in y_true],
            [DISCIPLINES[i] for i in P.argmax(axis=1)],
            labels=DISCIPLINES, digits=4, zero_division=0))

        split_res["_n_papers"] = n_docs
        split_res["_n_chunks"] = int(len(sub))
        split_res["_chunks_per_second"] = round(len(sub) / elapsed, 1)
        split_res["_best"] = best
        report["splits"][split] = split_res

    report["section_weights"] = {k: round(v, 4) for k, v in fitted_weights.items()}
    (d / "section_weights.json").write_text(
        json.dumps(report["section_weights"], indent=2))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"eval_documents_{tag}.json"
    out.write_text(json.dumps(report, indent=2))

    # The headline comparison: does full-document reading earn its cost?
    if "test" in report["splits"]:
        t = report["splits"]["test"]
        ab = t["abstract_only"]["macro_f1"]
        bestf = t[t["_best"]]["macro_f1"]
        print(f"\n{'='*64}")
        print(f"  abstract only        {ab:.4f}")
        print(f"  best full-document   {bestf:.4f}  ({t['_best']})")
        print(f"  delta                {bestf-ab:+.4f}")
        print(f"{'='*64}")
    print(f"\nsaved -> {out}")
    print(f"weights -> {d / 'section_weights.json'}")


if __name__ == "__main__":
    main()
