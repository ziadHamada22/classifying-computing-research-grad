"""Weakly label papers with a field, within their discipline.

No public corpus carries field-level labels for the six CC2020 disciplines, so
they are derived. Two independent evidence sources vote, and every label records
which one decided it so the result is auditable rather than a black box:

  **category evidence** — the paper's arXiv primary category implies a field
      directly (`cs.CV` -> Computer Vision). High precision, and it covers most
      disciplines, but it is *useless for Software Engineering*, where every
      paper is `cs.SE`.

  **keyword evidence** — distinctive phrases from the field definition matched
      against title + abstract, scored by how many distinct phrases hit and
      weighted toward the title. This is what actually separates the seven SE
      fields, and it breaks ties inside large categories elsewhere.

A paper is only labelled when the winner is clear; otherwise it is marked
``unlabelled`` and kept out of training. That mirrors the discipline level,
where excluding ambiguous papers from training (but keeping them in test) was
what made the labels trustworthy.

Run:
    python -m crc.data.label_fields --sample 5000     # inspect quality first
    python -m crc.data.label_fields                   # full pool
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crc.taxonomy import DISCIPLINES
from crc.taxonomy.fields import FIELDS_BY_DISCIPLINE, Field

WORK = Path(r"C:\Users\ziada\gp_data")
POOL = WORK / "corpus" / "computing_pool.parquet"
OUT = WORK / "corpus" / "fields_pool.parquet"

#: Weight of a category hit relative to one distinct keyword hit. A category is
#: an explicit authorial act, so it counts for more than any single phrase, but
#: not so much that keywords can never overturn it inside a broad category.
CATEGORY_WEIGHT = 3.0
#: A phrase in the title is worth more than the same phrase in the abstract.
TITLE_MULTIPLIER = 2.0
#: Winner must exceed the runner-up by this share of total score to be kept.
MARGIN = 0.20
#: Minimum absolute score; below this there is simply not enough evidence.
#: A single abstract keyword hit scores 1.0 and is far too weak to label on --
#: spot-checking showed those were mostly wrong. Requiring 2.0 means either a
#: category hit, a title phrase, or two distinct abstract phrases.
MIN_SCORE = 2.0
#: Distinct phrases (or a category) that must fire before a keyword-only label
#: is trusted. Without this, one incidental word decides the field.
MIN_KEYWORD_HITS = 2


def _compile(fields: list[Field]) -> list[tuple[Field, re.Pattern[str]]]:
    """One alternation per field; word-boundary anchored where sensible."""
    out = []
    for f in fields:
        parts = []
        for kw in f.keywords:
            esc = re.escape(kw)
            # Allow a trailing word to continue ("encrypt" -> "encryption")
            # only for stems the taxonomy deliberately left short.
            parts.append(esc if " " in kw or len(kw) > 12 else rf"\b{esc}\w*")
        out.append((f, re.compile("|".join(parts), re.IGNORECASE)))
    return out


_PATTERNS: dict[str, list[tuple[Field, re.Pattern[str]]]] = {
    d: _compile(FIELDS_BY_DISCIPLINE[d]) for d in DISCIPLINES
}
_BY_CATEGORY: dict[str, dict[str, Field]] = {
    d: {c: f for f in FIELDS_BY_DISCIPLINE[d] for c in f.categories}
    for d in DISCIPLINES
}


@dataclass
class FieldLabel:
    field: str | None
    score: float
    margin: float
    evidence: str          # "category", "keyword", "both", or "none"
    runner_up: str | None = None

    @property
    def usable(self) -> bool:
        return self.field is not None


def label_one(discipline: str, title: str, abstract: str,
              primary_category: str | None) -> FieldLabel:
    """Score every field in ``discipline`` and pick a winner if one is clear."""
    fields = FIELDS_BY_DISCIPLINE.get(discipline)
    if not fields:
        return FieldLabel(None, 0.0, 0.0, "none")

    scores: dict[str, float] = {f.name: 0.0 for f in fields}
    hits: dict[str, int] = {f.name: 0 for f in fields}
    from_cat: str | None = None

    cat_field = _BY_CATEGORY[discipline].get(primary_category or "")
    if cat_field is not None:
        scores[cat_field.name] += CATEGORY_WEIGHT
        from_cat = cat_field.name

    title = title or ""
    abstract = abstract or ""
    for f, pat in _PATTERNS[discipline]:
        # Count *distinct* matched phrases, not raw occurrences, so one
        # repeated word cannot dominate.
        t_hits = {m.group(0).lower() for m in pat.finditer(title)}
        a_hits = {m.group(0).lower() for m in pat.finditer(abstract)}
        hits[f.name] = len(t_hits | a_hits)
        scores[f.name] += TITLE_MULTIPLIER * len(t_hits) + len(a_hits - t_hits)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_name, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(scores.values())
    # Margin is only meaningful once more than one field is in play. When a
    # single field matched, (top-second)/total is 1.0 — which looks like maximum
    # confidence but is really minimum evidence — so gate on absolute score and
    # hit count instead of letting that pass.
    margin = (top - second) / total if total else 0.0

    enough_evidence = (
        top >= MIN_SCORE
        and (from_cat == top_name or hits[top_name] >= MIN_KEYWORD_HITS)
    )
    if not enough_evidence or margin < MARGIN:
        return FieldLabel(None, top, margin, "none",
                          runner_up=ranked[1][0] if len(ranked) > 1 else None)
    hit_any = hits[top_name] > 0

    if from_cat == top_name and hit_any:
        ev = "both"
    elif from_cat == top_name:
        ev = "category"
    else:
        ev = "keyword"
    return FieldLabel(top_name, top, margin, ev,
                      runner_up=ranked[1][0] if len(ranked) > 1 else None)


def label_frame(df: pd.DataFrame) -> pd.DataFrame:
    labels = [
        label_one(r.discipline, r.title, r.abstract, r.primary_category)
        for r in df.itertuples(index=False)
    ]
    out = df.copy()
    out["field"] = [l.field for l in labels]
    out["field_score"] = [round(l.score, 3) for l in labels]
    out["field_margin"] = [round(l.margin, 4) for l in labels]
    out["field_evidence"] = [l.evidence for l in labels]
    out["field_runner_up"] = [l.runner_up for l in labels]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sample", type=int, default=0,
                    help="Label only N papers (for a quick quality check).")
    ap.add_argument("--year-min", type=int, default=2015)
    ap.add_argument("--year-max", type=int, default=2024)
    args = ap.parse_args()

    pool = pd.read_parquet(args.pool)
    from crc.taxonomy import BY_CATEGORY

    pool = pool[pool["primary_category"].isin(BY_CATEGORY) & ~pool["ambiguous"]]
    pool = pool[(pool["year"] >= args.year_min) & (pool["year"] <= args.year_max)]
    if args.sample:
        pool = pool.sample(n=min(args.sample, len(pool)), random_state=42)
    print(f"labelling {len(pool):,} papers…", flush=True)

    out = label_frame(pool)
    n_lab = out["field"].notna().sum()
    print(f"\nlabelled {n_lab:,} / {len(out):,} ({n_lab/len(out):.1%})")
    print("\nevidence source:")
    print(out.loc[out["field"].notna(), "field_evidence"]
          .value_counts().to_string())

    print("\n=== per discipline ===")
    for d in DISCIPLINES:
        g = out[out["discipline"] == d]
        if not len(g):
            continue
        lab = g[g["field"].notna()]
        print(f"\n{d}  —  {len(lab):,}/{len(g):,} labelled "
              f"({len(lab)/len(g):.0%})")
        vc = lab["field"].value_counts()
        for name, n in vc.items():
            print(f"    {name:46s} {n:>7,}  {n/max(1,len(lab)):5.1%}")
        missing = [f.name for f in FIELDS_BY_DISCIPLINE[d]
                   if f.name not in set(vc.index)]
        if missing:
            print(f"    !! no papers: {missing}")

    if not args.sample:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(outp, index=False)
        print(f"\nwrote {outp}  ({len(out):,} rows)")
        stats = {
            "n_total": int(len(out)),
            "n_labelled": int(n_lab),
            "coverage": round(float(n_lab / len(out)), 4),
            "per_discipline": {
                d: {
                    "n": int((out["discipline"] == d).sum()),
                    "labelled": int(((out["discipline"] == d)
                                     & out["field"].notna()).sum()),
                    "fields": {
                        k: int(v) for k, v in
                        out[(out["discipline"] == d) & out["field"].notna()]
                        ["field"].value_counts().items()},
                }
                for d in DISCIPLINES
            },
        }
        (outp.parent / "fields_pool_stats.json").write_text(
            json.dumps(stats, indent=2))
        print(f"stats -> {outp.parent / 'fields_pool_stats.json'}")


if __name__ == "__main__":
    main()
