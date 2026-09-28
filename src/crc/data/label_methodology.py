"""Weakly label a paper's research design from its methods section.

Agent 3's primary axis — research design — is declared in the methods section,
not the abstract or the arXiv category, so this labeller scores each design's
distinctive cue phrases (from `taxonomy/methodology.py`) against the **methods +
abstract** text of each paper. Same philosophy as `label_fields.py`: only label
when the winner is clear; leave the rest ``unlabelled`` and out of training.

The one twist specific to methodology: **Design & Creation and Experiment
overlap** — almost every computing paper both builds something and evaluates it.
So the decision uses *specificity precedence*:

  1. The seven **distinctive** designs (Formal, SLR, Survey, Case Study,
     Simulation, Qualitative, Action Research) have diagnostic cues — if the top
     one fires clearly and is at least as strong as the generic pair, it wins.
  2. Otherwise the **generic** pair is resolved by contribution: strong
     artefact-construction cues -> Design & Creation; evaluation cues without
     construction -> Experiment.
  3. Anything short of a clear winner -> unlabelled.

Worldview and method are filled from each design's prior (the local-LLM
hard-case reviewer refines them later).

Run:
    python -m crc.data.label_methodology --sample 4000   # inspect first
    python -m crc.data.label_methodology                 # full pool
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from crc.taxonomy.methodology import (
    BY_NAME,
    DESIGNS,
    DESIGN_NAMES,
    method_of,
    worldview_of,
)

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS = WORK / "corpus"
CHUNK_FILES = ["chunks_eval.parquet", "chunks_v2.parquet"]
OUT = CORPUS / "methodology_pool.parquet"

GENERIC = {"Design & Creation (Design Science)",
           "Experiment / Empirical Evaluation"}
#: Strongly-diagnostic designs: their cues ("case study", "grounded theory",
#: "systematic literature review", "questionnaire", "we prove") are essentially
#: never written unless that IS the method, so they win on their own cues even
#: when quantitative evaluation cues also fire (a case study still reports
#: numbers). Simulation is deliberately excluded — "simulation" is often just an
#: evaluation technique inside a design-science paper — so it stays gated below.
SEMI = "Simulation & Modelling"
STRONG = [d.name for d in DESIGNS if d.name not in GENERIC and d.name != SEMI]

MIN_SCORE = 2          # need >= this many distinct cues to label at all
STRONG_MARGIN = 1      # top strong design must beat the 2nd strong by this
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").lower())


def score_designs(text: str) -> dict[str, int]:
    """Distinct cue phrases matched per design (substring, case-insensitive)."""
    return {d.name: sum(1 for cue in d.cues if cue in text) for d in DESIGNS}


def decide(scores: dict[str, int]) -> tuple[str | None, float, str | None]:
    """Pick a design or None. Returns (design, margin, runner_up)."""
    strong = sorted(((scores[n], n) for n in STRONG), reverse=True)
    top_s, top_n = strong[0]
    second_s = strong[1][0] if len(strong) > 1 else 0

    dc = scores["Design & Creation (Design Science)"]
    exp = scores["Experiment / Empirical Evaluation"]
    sim = scores[SEMI]
    gen_top = max(dc, exp)

    # 1) A strongly-diagnostic design wins on its own cues (a case study still
    #    reports numbers, so it need not out-score the generic evaluation cues).
    if top_s >= MIN_SCORE and top_s - second_s >= STRONG_MARGIN:
        return top_n, float(top_s - second_s), \
            (strong[1][1] if second_s > 0 else None)

    # 2) Simulation is semi-diagnostic: it wins only if it also beats the
    #    generic build/evaluate cues (else it is just an evaluation technique).
    if sim >= MIN_SCORE and sim >= gen_top and sim > top_s:
        return SEMI, float(sim - gen_top), None

    # 3) Generic pair: construction -> Design & Creation; evaluation-only ->
    #    Experiment.
    if dc >= MIN_SCORE and dc >= exp:
        return "Design & Creation (Design Science)", float(dc - exp), \
            ("Experiment / Empirical Evaluation" if exp else None)
    if exp >= MIN_SCORE and exp > dc:
        return "Experiment / Empirical Evaluation", float(exp - dc), \
            ("Design & Creation (Design Science)" if dc else None)

    return None, 0.0, None


def build_paper_text(g: pd.DataFrame) -> str:
    """Methods + abstract; fall back to abstract + intro when no methods."""
    by_sec: dict[str, list[str]] = {}
    for r in g.itertuples(index=False):
        by_sec.setdefault(r.section, []).append(r.text or "")
    methods = " ".join(by_sec.get("methods", []))
    abstract = " ".join(by_sec.get("abstract", []))
    if methods.strip():
        return _norm(methods + " " + abstract)
    intro = " ".join(by_sec.get("introduction", []))
    return _norm(abstract + " " + intro)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    frames = []
    for name in CHUNK_FILES:
        p = CORPUS / name
        if p.exists():
            frames.append(pd.read_parquet(
                p, columns=["paper_id", "discipline", "split", "year",
                            "primary_category", "section", "text"]))
    chunks = pd.concat(frames, ignore_index=True)
    chunks = chunks.drop_duplicates(subset=["paper_id", "section", "text"])
    print(f"chunks: {len(chunks):,} across {chunks['paper_id'].nunique():,} papers")

    papers = chunks.groupby("paper_id", sort=False)
    ids = list(papers.groups)
    if args.sample:
        import random
        random.seed(42)
        ids = random.sample(ids, min(args.sample, len(ids)))

    rows = []
    for pid in ids:
        g = papers.get_group(pid)
        text = build_paper_text(g)
        scores = score_designs(text)
        design, margin, runner = decide(scores)
        meta = g.iloc[0]
        matched = ([c for c in BY_NAME[design].cues if c in text]
                   if design else [])
        rows.append({
            "paper_id": pid,
            "discipline": meta["discipline"],
            "split": meta["split"],
            "year": meta["year"],
            "design": design,
            "worldview": worldview_of(design) if design else None,
            "method": method_of(design) if design else None,
            "design_score": int(scores[design]) if design else 0,
            "design_margin": round(margin, 2),
            "design_runner_up": runner,
            "matched_cues": ";".join(matched),
            "text": text[:4000],
        })

    df = pd.DataFrame(rows)
    n_lab = df["design"].notna().sum()
    print(f"\nlabelled {n_lab:,} / {len(df):,} ({n_lab/len(df):.1%})")

    print("\n=== design distribution (labelled) ===")
    print(df.loc[df["design"].notna(), "design"].value_counts().to_string())
    print("\n=== worldview / method (labelled) ===")
    print(df.loc[df["design"].notna(), "worldview"].value_counts().to_string())
    print(df.loc[df["design"].notna(), "method"].value_counts().to_string())

    print("\n=== labelled design x discipline ===")
    lab = df[df["design"].notna()]
    if len(lab):
        print(pd.crosstab(lab["design"], lab["discipline"]).to_string())

    if args.sample:
        print("\n=== 12 sample labels ===")
        for r in lab.sample(n=min(12, len(lab)), random_state=1).itertuples():
            print(f"  [{r.design[:26]:26s} s={r.design_score} m={r.design_margin}] "
                  f"{r.discipline[:14]:14s} cues: {r.matched_cues[:70]}")
        print("\n[dry run] not writing.")
        return

    df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df):,} rows)")
    stats = {
        "n_total": int(len(df)), "n_labelled": int(n_lab),
        "coverage": round(float(n_lab / len(df)), 4),
        "designs": {k: int(v) for k, v in
                    df.loc[df["design"].notna(), "design"].value_counts().items()},
    }
    (CORPUS / "methodology_pool_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"stats -> {CORPUS / 'methodology_pool_stats.json'}")


if __name__ == "__main__":
    main()
