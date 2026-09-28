"""External baselines for Agent 1 (discipline), on the SAME v2 test set.

Answers the project brief's promise that the system "must beat existing
classifiers such as the CSO Classifier and TF-IDF baselines". Three off-the-shelf
scholarly classifiers are scored against our CC2020 label space:

  * OpenAlex  — mBERT topic model over ~250M works (Scopus-ASJC hierarchy).
  * S2FOS     — Semantic Scholar's char-ngram SVM field-of-study classifier.
  * CSO       — the Computer Science Ontology classifier (unsupervised, 14k
                topics); scored from a cache produced by `scripts/run_cso.py`.

None of them targets the CC2020 practitioner split, so each foreign label space
is mapped onto our disciplines with a deliberate, conservative, auditable map
(`taxonomy/openalex_map.py`); unmappable labels are coverage loss, never a wrong
answer. Two honest numbers are reported, exactly as `evaluate.py` does for
Agent 1: strict (exact) and co-listed (credited if the mapped discipline is among
those the paper's own arXiv categories point to).

These are OFFLINE research comparisons built from public indexes; they are not
part of the deployed pipeline and do not bear on the "no network at inference"
constraint.

Run (after fetch_openalex / fetch_s2fos / run_cso have populated the caches):
    python -m crc.eval.evaluate_baselines
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, normalized_mutual_info_score

from crc.taxonomy import BY_CATEGORY, DISCIPLINES
from crc.taxonomy.disciplines import DISCIPLINE_ABBR
from crc.taxonomy.openalex_map import is_ambiguous, map_subfield

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "results"

OA_CACHE = CORPUS_DIR / "openalex_test_cache.json"
S2_CACHE = CORPUS_DIR / "s2fos_test_cache.json"
CSO_CACHE = CORPUS_DIR / "cso_test_cache.json"


def allowed_disciplines(mapped_categories) -> set[str]:
    out: set[str] = set()
    if mapped_categories is None:
        return out
    for c in list(mapped_categories):
        m = BY_CATEGORY.get(c)
        if m:
            out.add(m.discipline)
            if m.secondary:
                out.add(m.secondary)
    return out


def _agreement(pred: list, gold: list, allowed: list) -> dict:
    """strict + co-listed agreement over rows where pred is not None."""
    idx = [i for i, p in enumerate(pred) if p is not None]
    if not idx:
        return {"n": 0}
    p = [pred[i] for i in idx]
    g = [gold[i] for i in idx]
    a = [allowed[i] for i in idx]
    strict = np.mean([pi == gi for pi, gi in zip(p, g)])
    co = np.mean([(pi == gi) or (pi in ai) for pi, gi, ai in zip(p, g, a)])
    macro = f1_score(g, p, labels=DISCIPLINES, average="macro", zero_division=0)
    return {"n": len(idx), "strict": round(float(strict), 4),
            "co_listed": round(float(co), 4), "macro_f1": round(float(macro), 4)}


def eval_openalex(df: pd.DataFrame, gold, allowed) -> dict:
    if not OA_CACHE.exists():
        return {"available": False}
    cache = json.loads(OA_CACHE.read_text())
    # NOTE: iterate a plain Python list, not Series.map — df["id"] is pandas
    # StringDtype here and Series.map over it mishandles the dict-lookup closure.
    ids = df["id"].astype(str).tolist()
    subs = [(cache.get(a) or {}).get("subfield") for a in ids]
    n_resolved = sum(1 for s in subs if s is not None)
    pred_all, pred_clean = [], []
    for s in subs:
        d = map_subfield(s)
        pred_all.append(d)
        pred_clean.append(None if (d is not None and is_ambiguous(s)) else d)
    mappable = sum(1 for p in pred_all if p is not None)
    out = {
        "available": True,
        "n": int(len(df)),
        "resolved": int(n_resolved),
        "resolved_pct": round(n_resolved / len(df), 4),
        "mappable": int(mappable),
        "mappable_pct": round(mappable / len(df), 4),
        "all": _agreement(pred_all, gold, allowed),
        "clean": _agreement(pred_clean, gold, allowed),
    }
    # per gold discipline, strict agreement on its mappable papers
    per = {}
    for d in DISCIPLINES:
        m = [i for i in range(len(df)) if gold[i] == d and pred_all[i] is not None]
        if m:
            per[d] = {"n": len(m),
                      "strict": round(float(np.mean([pred_all[i] == d for i in m])), 4)}
    out["per_discipline"] = per
    return out


def eval_s2fos(df: pd.DataFrame, gold) -> dict:
    """S2FOS has no intra-computing labels; quantify that it cannot separate the
    six disciplines rather than pretend it predicts them."""
    if not S2_CACHE.exists():
        return {"available": False}
    cache = json.loads(S2_CACHE.read_text())
    ids = df["id"].astype(str).tolist()          # plain list; see eval_openalex note
    cats = [(cache.get(a) or {}).get("model") for a in ids]
    resolved = np.array([bool(c) for c in cats])
    is_cs = np.array([bool(c) and "Computer Science" in c for c in cats])
    # per gold discipline: fraction tagged Computer Science (should be ~uniform)
    per = {}
    for d in DISCIPLINES:
        m = (np.array(gold) == d) & resolved
        if m.any():
            per[d] = {"n": int(m.sum()),
                      "pct_tagged_CS": round(float(is_cs[m].mean()), 4)}
    # discriminative power: NMI between gold discipline and the "is CS" tag
    nmi = normalized_mutual_info_score(np.array(gold)[resolved], is_cs[resolved].astype(int))
    return {
        "available": True,
        "resolved": int(resolved.sum()),
        "resolved_pct": round(float(resolved.mean()), 4),
        "pct_tagged_CS_overall": round(float(is_cs[resolved].mean()), 4),
        "per_discipline_pct_CS": per,
        "nmi_gold_vs_isCS": round(float(nmi), 4),
        "note": "S2FOS top-level 'Computer Science' spans all six of our "
                "disciplines; near-uniform pct_CS and NMI~0 mean it carries no "
                "6-way discriminative signal by construction.",
    }


def eval_cso(df: pd.DataFrame, gold, allowed) -> dict:
    if not CSO_CACHE.exists():
        return {"available": False,
                "note": "run scripts/run_cso.py in the .venv-baselines first"}
    from crc.taxonomy.cso_map import map_topics
    cache = json.loads(CSO_CACHE.read_text())
    # cache[id] = {"topics": [...]} (raw CSO topics); map to a discipline here so
    # the decision stays in the auditable crc.taxonomy.cso_map. Iterate a plain
    # list (df["id"] is StringDtype; Series.map mishandles the closure).
    ids = df["id"].astype(str).tolist()
    pred = [map_topics(cache[a]["topics"]) if a in cache else None for a in ids]
    mappable = sum(1 for p in pred if p is not None)
    resolved = sum(1 for a in ids if a in cache)   # CSO ran on a subset of test
    out = {"available": True, "n_run": int(resolved), "mappable": int(mappable),
           # coverage is relative to the papers CSO actually ran on, not full test
           "mappable_pct": round(mappable / max(resolved, 1), 4),
           "all": _agreement(pred, gold, allowed)}
    per = {}
    for d in DISCIPLINES:
        m = [i for i in range(len(df)) if gold[i] == d and pred[i] is not None]
        if m:
            per[d] = {"n": len(m),
                      "strict": round(float(np.mean([pred[i] == d for i in m])), 4)}
    out["per_discipline"] = per
    return out


def agent1_reference() -> dict:
    p = RESULTS / "eval_scibert.json"
    if not p.exists():
        return {}
    j = json.loads(p.read_text())
    t = j.get("splits", {}).get("test", {})
    return {"strict": t.get("strict_accuracy"), "co_listed": t.get("co_listed_accuracy"),
            "macro_f1": t.get("strict_macro_f1")}


def main() -> None:
    corpus = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet")
    df = corpus[corpus["split"] == "test"].reset_index(drop=True)
    gold = df["discipline"].tolist()
    allowed = df["mapped_categories"].map(allowed_disciplines).tolist()
    print(f"test papers: {len(df):,} (balanced {len(df)//len(DISCIPLINES):,}/discipline)")

    oa = eval_openalex(df, gold, allowed)
    s2 = eval_s2fos(df, gold)
    cso = eval_cso(df, gold, allowed)
    a1 = agent1_reference()

    print("\n===== OpenAlex (mBERT topic model) =====")
    if oa.get("available"):
        print(f"  resolved {oa['resolved']:,}/{oa['n']:,} ({oa['resolved_pct']:.1%}); "
              f"mappable to a discipline {oa['mappable_pct']:.1%}")
        print(f"  agreement (mappable, clean): strict {oa['clean']['strict']:.3f}  "
              f"co-listed {oa['clean']['co_listed']:.3f}  (n={oa['clean']['n']:,})")
        for d, v in oa["per_discipline"].items():
            print(f"    {DISCIPLINE_ABBR[d]}  n={v['n']:>4}  strict {v['strict']:.3f}")
    else:
        print("  cache missing — run crc.eval.fetch_openalex")

    print("\n===== S2FOS (Semantic Scholar field-of-study) =====")
    if s2.get("available"):
        print(f"  resolved {s2['resolved']:,}; tagged 'Computer Science' {s2['pct_tagged_CS_overall']:.1%} overall")
        print(f"  per-discipline %CS (uniform => no signal):")
        for d, v in s2["per_discipline_pct_CS"].items():
            print(f"    {DISCIPLINE_ABBR[d]}  {v['pct_tagged_CS']:.1%}")
        print(f"  NMI(gold, isCS) = {s2['nmi_gold_vs_isCS']:.4f}  (0 => cannot separate the six)")
    else:
        print("  cache missing — run crc.eval.fetch_s2fos")

    print("\n===== CSO Classifier =====")
    if cso.get("available"):
        print(f"  ran on {cso['n_run']:,} papers (stratified subset); "
              f"mappable {cso['mappable_pct']:.1%}")
        print(f"  agreement (mappable): strict {cso['all']['strict']:.3f}  "
              f"co-listed {cso['all']['co_listed']:.3f}  (n={cso['all']['n']:,})")
        for d, v in cso["per_discipline"].items():
            print(f"    {DISCIPLINE_ABBR[d]}  n={v['n']:>4}  strict {v['strict']:.3f}")
    else:
        print(f"  {cso.get('note', 'cache missing')}")

    print("\n===== vs Agent 1 (SciBERT) on the same test set =====")
    if a1:
        print(f"  Agent 1: strict {a1['strict']}  co-listed {a1['co_listed']}  macro-F1 {a1['macro_f1']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"test_n": int(len(df)), "openalex": oa, "s2fos": s2, "cso": cso,
           "agent1_scibert": a1}
    (RESULTS / "baselines_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {RESULTS / 'baselines_eval.json'}")


if __name__ == "__main__":
    main()
