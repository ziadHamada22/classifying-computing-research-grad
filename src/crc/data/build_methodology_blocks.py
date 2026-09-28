"""Build a BLOCK-level methodology corpus, so Agent 3 can read a whole paper.

The first Agent 3 concatenated each paper's methods and abstract into one string
and fed the first 320 tokens to SciBERT. Measured on the pool, that string
averages 524 words -- roughly 700 tokens -- so **more than half of every paper's
methodological evidence was truncated away before the model saw it**, and because
methods is concatenated *first*, the abstract was usually cut entirely.

That is the same problem Agent 1 already solved. Agent 1 reads the document as
section-tagged chunks, classifies each, then aggregates with weights fitted on
validation; full-document reading was worth +2.62 standard errors there. This
module builds the equivalent input for Agent 3 so the same architecture can be
tried, and so the interesting question can be *measured* rather than assumed:

    which sections of a paper actually reveal its research design?

Two deliberate choices:

**The paper-level splits are inherited unchanged** from
``methodology_corpus.parquet``. Every block of a paper lands in that paper's
split, so no paper is ever split across train and test, and document-level
numbers are directly comparable with the 0.657 baseline on the same papers.

**Train blocks are capped per paper; val and test keep everything.** Training on
every block would take hours for little benefit, but evaluation has to see the
whole document or it cannot answer the question above. Sections that plausibly
mislead (``related_work`` describes *other* people's methods) are deliberately
kept in rather than filtered out, so the fitted weights can show whether they
hurt -- the same way Agent 1's optimiser independently drove ``other`` to zero.

Run:
    python -m crc.data.build_methodology_blocks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS = WORK / "corpus"
CHUNK_FILES = ["chunks_eval.parquet", "chunks_v2.parquet"]

#: Sections carrying running prose about the work itself. ``other`` and ``title``
#: are dropped (headings that did not map, and 6-word fragments); ``appendix`` is
#: dropped as overwhelmingly tables and proofs. Everything else is kept so the
#: weight fit can rank it.
KEEP_SECTIONS = (
    "methods", "abstract", "introduction", "background",
    "related_work", "results", "discussion", "conclusion",
)

#: Order used when capping a paper's training blocks: the two sections the weak
#: labeller itself read come first, then the rest round-robin for diversity.
PRIORITY = ("methods", "abstract", "introduction", "results",
            "conclusion", "background", "discussion", "related_work")


def load_chunks(columns: list[str]) -> pd.DataFrame:
    frames = []
    for name in CHUNK_FILES:
        p = CORPUS / name
        if p.exists():
            frames.append(pd.read_parquet(p, columns=columns))
    if not frames:
        raise FileNotFoundError(f"no chunk files in {CORPUS}")
    ch = pd.concat(frames, ignore_index=True)
    # the two chunk files overlap; identical text in the same section is one block
    return ch.drop_duplicates(subset=["paper_id", "section", "text"])


def cap_paper_blocks(g: pd.DataFrame, cap: int,
                     rng: np.random.Generator) -> pd.DataFrame:
    """Take at most ``cap`` blocks from one paper, spread across its sections.

    Round-robin over sections in :data:`PRIORITY` rather than taking the first
    ``cap`` rows, which would collapse to whichever section happens to appear
    first and reintroduce the truncation bias this module exists to remove.
    """
    if len(g) <= cap:
        return g
    by_sec: dict[str, list] = {}
    for sec, sub in g.groupby("section", sort=False):
        idx = np.array(sub.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        by_sec[sec] = list(idx)

    picked: list = []
    order = [s for s in PRIORITY if s in by_sec] + \
            [s for s in by_sec if s not in PRIORITY]
    while len(picked) < cap and any(by_sec.values()):
        for sec in order:
            if by_sec.get(sec):
                picked.append(by_sec[sec].pop(0))
                if len(picked) == cap:
                    break
    return g.loc[picked]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS / "methodology_corpus.parquet"))
    ap.add_argument("--out", default=str(CORPUS / "methodology_blocks.parquet"))
    ap.add_argument("--train-blocks-per-paper", type=int, default=6,
                    help="Cap for TRAIN only; val/test keep every block.")
    ap.add_argument("--min-words", type=int, default=25,
                    help="Drop very short blocks: too little text to classify.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    papers = pd.read_parquet(
        args.corpus, columns=["paper_id", "design", "discipline", "year", "split"])
    print(f"labelled papers: {len(papers):,}  "
          f"({papers['split'].value_counts().to_dict()})")

    chunks = load_chunks(["paper_id", "section", "n_words", "text"])
    print(f"chunks available: {len(chunks):,}")

    df = chunks.merge(papers, on="paper_id", how="inner")
    print(f"after join to labelled papers: {len(df):,}")

    df = df[df["section"].isin(KEEP_SECTIONS)]
    df = df[df["n_words"] >= args.min_words]
    print(f"after section + length filter: {len(df):,}")

    # cap the training split only
    train = df[df["split"] == "train"]
    kept = [cap_paper_blocks(g, args.train_blocks_per_paper, rng)
            for _, g in train.groupby("paper_id", sort=False)]
    train_capped = (pd.concat(kept, ignore_index=False)
                    if kept else train.iloc[:0])
    out = pd.concat([train_capped, df[df["split"] != "train"]],
                    ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    path = Path(args.out)
    out.to_parquet(path, index=False)

    stats = {
        "n_blocks": int(len(out)),
        "n_papers": int(out["paper_id"].nunique()),
        "train_blocks_per_paper_cap": args.train_blocks_per_paper,
        "min_words": args.min_words,
        "per_split": {k: int(v) for k, v in out["split"].value_counts().items()},
        "papers_per_split": {
            k: int(v) for k, v in
            out.groupby("split")["paper_id"].nunique().items()},
        "per_section": {k: int(v) for k, v in
                        out["section"].value_counts().items()},
        "per_design": {k: int(v) for k, v in out["design"].value_counts().items()},
    }
    (CORPUS / "methodology_blocks_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nblocks -> {path}  ({len(out):,} rows)")
    print(f"\n{'split':8s} {'blocks':>9s} {'papers':>8s} {'blocks/paper':>13s}")
    for s in ("train", "val", "test"):
        sub = out[out["split"] == s]
        if sub.empty:
            continue
        npap = sub["paper_id"].nunique()
        print(f"{s:8s} {len(sub):>9,} {npap:>8,} {len(sub)/npap:>13.1f}")
    print("\nblocks per section:")
    for k, v in out["section"].value_counts().items():
        print(f"  {k:16s} {v:>8,}  {v/len(out):6.1%}")
    print(f"\nstats -> {CORPUS / 'methodology_blocks_stats.json'}")


if __name__ == "__main__":
    main()
