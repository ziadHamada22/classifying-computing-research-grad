"""Pass 3 — build the section-labelled chunk corpus from full text.

The deployed system reads whole documents, so the classifier must be good on
*any* chunk it will meet at inference — a bare title, an abstract, a methods
paragraph, a slab of an article — not just on title+abstract. Training only on
abstracts and then running on full text is exactly the distribution shift that
made full-text inference unsafe in the prototype.

Source is `neuralwork/arxiver`: 138k arXiv papers as LaTeX-derived markdown with
clean headings ('## 2 Methodology'), so section labels come free and no GROBID
is involved. It carries no categories, so it is joined on arXiv id against the
computing pool built by `extract_computing.py`, which supplies the discipline.

Chunks reuse the *same* `chunk_paper` code path the inference-time PDF parser
feeds, so training and inference chunks are drawn from one distribution.

Run:
    python -m crc.data.build_fulltext
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from crc.ingest.chunkers import ChunkConfig, chunk_paper, sections_from_markdown
from crc.ingest.schema import NOISE_SECTIONS, Section

WORK = Path(r"C:\Users\ziada\gp_data")
POOL = WORK / "corpus" / "computing_pool.parquet"
OUT = WORK / "corpus" / "chunks_v2.parquet"

HF_FULLTEXT = "neuralwork/arxiver"

_VERSION_SUFFIX = re.compile(r"v\d+$")


def _norm_id(x) -> str:
    """arxiver ids may carry a version suffix; the metadata snapshot's do not."""
    s = str(x).strip()
    s = s.split("/")[-1] if "/" not in s[:4] else s
    return _VERSION_SUFFIX.sub("", s)


def load_fulltext(cache_dir: Path) -> pd.DataFrame:
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf"))
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_FULLTEXT,
        filename="data/train.parquet",
        repo_type="dataset",
    )
    print(f"full-text parquet: {path}")
    df = pd.read_parquet(path, columns=["id", "title", "abstract", "markdown"])
    df["join_id"] = df["id"].map(_norm_id)
    return df


def build_chunks(df: pd.DataFrame, cfg: ChunkConfig,
                 max_chunks_per_paper: int, drop_noise: bool) -> pd.DataFrame:
    rows: list[dict] = []
    section_counts: Counter[str] = Counter()
    n_no_sections = 0

    for rec in df.itertuples(index=False):
        md = rec.markdown or ""
        if not md.strip():
            continue
        title, sections = sections_from_markdown(md)
        if not sections:
            n_no_sections += 1
            continue
        if drop_noise:
            sections = [s for s in sections if s.section not in NOISE_SECTIONS]
            if not sections:
                continue
        chunks = chunk_paper(sections, title or rec.title, cfg)
        for c in chunks[:max_chunks_per_paper]:
            section_counts[c.section.value] += 1
            rows.append({
                "paper_id": rec.join_id,
                "discipline": rec.discipline,
                "split": rec.split,
                "year": rec.year,
                "primary_category": rec.primary_category,
                "ambiguous": rec.ambiguous,
                "chunk_index": c.index,
                "section": c.section.value,
                "n_words": c.n_words,
                "text": c.text,
            })
    print(f"papers with no recoverable sections: {n_no_sections:,}")
    print("chunk section distribution:")
    for sec, n in section_counts.most_common():
        print(f"  {sec:18s} {n:>8,}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--max-papers-per-discipline", type=int, default=6_000)
    ap.add_argument("--max-chunks-per-paper", type=int, default=24)
    ap.add_argument("--max-chunks-per-discipline", type=int, default=20_000,
                    help="Chunk-level balancing cap; 0 disables.")
    ap.add_argument("--target-words", type=int, default=200)
    ap.add_argument("--keep-noise-sections", action="store_true",
                    help="Keep references/acknowledgements (default: drop).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = pd.read_parquet(args.pool)
    pool["join_id"] = pool["id"].map(_norm_id)
    print(f"computing pool: {len(pool):,}")

    ft = load_fulltext(Path(args.work))
    print(f"full-text papers: {len(ft):,}")

    merged = ft.merge(
        pool[["join_id", "discipline", "primary_category", "ambiguous",
              "year", "margin"]],
        on="join_id", how="inner", suffixes=("", "_pool"),
    )
    print(f"joined (computing only): {len(merged):,}")
    print(merged["discipline"].value_counts().to_string())

    # Train on clean supervision only; ambiguous papers are for evaluation.
    clean = merged[~merged["ambiguous"]]

    # Balance across disciplines so no class dominates the chunk corpus.
    rng_seed = args.seed
    parts = []
    for d, g in clean.groupby("discipline"):
        n = min(len(g), args.max_papers_per_discipline)
        parts.append(g.sample(n=n, random_state=rng_seed))
    balanced = pd.concat(parts, ignore_index=True)
    print(f"\nbalanced paper set: {len(balanced):,}")
    print(balanced["discipline"].value_counts().to_string())

    # Paper-level splits — every chunk of a paper stays in one split, otherwise
    # near-duplicate chunks leak across the boundary and inflate test scores.
    shuffled = balanced.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)
    split_col = []
    for _, g in shuffled.groupby("discipline", sort=False):
        n = len(g)
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        s = ["test"] * n
        s[:n_tr] = ["train"] * n_tr
        s[n_tr:n_tr + n_va] = ["val"] * n_va
        split_col.append(pd.Series(s, index=g.index))
    shuffled["split"] = pd.concat(split_col).reindex(shuffled.index)

    cfg = ChunkConfig(target_words=args.target_words,
                      max_chunks=args.max_chunks_per_paper)
    chunks = build_chunks(shuffled, cfg, args.max_chunks_per_paper,
                          drop_noise=not args.keep_noise_sections)

    # Balancing papers is not enough: arXiv's full-text supply is so skewed that
    # capping papers still leaves ~8:1 at the chunk level (CS ~95k chunks vs SE
    # ~11k). Cap chunks per discipline too, dropping whole papers rather than
    # slicing them, so a paper is never half in and half out of the corpus.
    if args.max_chunks_per_discipline:
        cap = args.max_chunks_per_discipline
        keep_papers: list[str] = []
        for d, g in chunks.groupby("discipline"):
            per_paper = g.groupby("paper_id").size()
            if per_paper.sum() <= cap:
                keep_papers.extend(per_paper.index.tolist())
                continue
            order = per_paper.sample(frac=1.0, random_state=args.seed)
            running = order.cumsum()
            keep_papers.extend(order.index[running <= cap].tolist())
        before = len(chunks)
        chunks = chunks[chunks["paper_id"].isin(set(keep_papers))].reset_index(drop=True)
        print(f"\nchunk-level balancing at {cap:,}/discipline: "
              f"{before:,} -> {len(chunks):,} chunks")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(out, index=False)

    print(f"\nchunk corpus -> {out}  ({len(chunks):,} chunks from "
          f"{chunks['paper_id'].nunique():,} papers)")
    print(chunks.groupby(["discipline", "split"]).size().unstack(fill_value=0).to_string())
    print(f"\nchunks per paper: mean="
          f"{len(chunks)/max(1, chunks['paper_id'].nunique()):.1f}")
    print(f"words per chunk: mean={chunks['n_words'].mean():.0f} "
          f"median={chunks['n_words'].median():.0f} p95={chunks['n_words'].quantile(.95):.0f}")

    stats = {
        "n_chunks": int(len(chunks)),
        "n_papers": int(chunks["paper_id"].nunique()),
        "joined_papers": int(len(merged)),
        "section_distribution": {
            k: int(v) for k, v in chunks["section"].value_counts().items()},
        "per_discipline_papers": {
            k: int(v) for k, v in
            chunks.groupby("discipline")["paper_id"].nunique().items()},
        "splits": {k: int(v) for k, v in chunks["split"].value_counts().items()},
    }
    (out.parent / "chunks_v2_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"stats -> {out.parent / 'chunks_v2_stats.json'}")


if __name__ == "__main__":
    main()
