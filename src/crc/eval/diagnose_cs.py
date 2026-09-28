"""Why is Computer Science Agent 1's weakest class? A reproducible diagnosis.

Computer Science recall (0.742 on the field-test papers, 0.718 on Agent 1's own
test split) is the binding constraint on the whole pipeline, because a CS paper
sent elsewhere cannot get its field. Before trying to raise it, this script
measures *where* the misses come from, from saved logits only (no GPU):

1. **By primary arXiv category.** Recall per CS category, and where the misses go.
2. **Mirrored categories.** arXiv keeps near-duplicate categories on two sides of
   our taxonomy -- ``cs.SD`` (Sound, -> CS) mirrors ``eess.AS`` (Audio and Speech,
   -> CE), and ``cs.CV`` (-> CS) mirrors ``eess.IV`` (-> CE). A paper listed in
   both gets whichever discipline wins the category vote -- decided by which of
   the pair the author listed first (it counts double) and by whatever else is
   co-listed. None of that is in the text, so the label splits roughly evenly
   between CS and CE, the labeller itself flags most of these papers ambiguous,
   and no text classifier can reliably call them.
3. **Ambiguous vs clean papers.** Ambiguous papers are excluded from training but
   kept in test, so this separates a model weakness from a label-definition one.
4. **Error decomposition for CS alone**: co-listed (the predicted discipline is one
   the paper's own categories point to), adjacent, or genuine.

Run:
    python -m crc.eval.diagnose_cs                # SciBERT, test + temporal
    python -m crc.eval.diagnose_cs --tag bert-base-uncased
"""
from __future__ import annotations

from crc.agents.discipline import DEPLOYED_MODEL

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from crc.agents.discipline.hierarchy import softmax
from crc.taxonomy import BY_CATEGORY, DISCIPLINES, LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"

CS = "Computer Science"

#: arXiv category pairs covering the same subject but mapped to different
#: disciplines. Only cross-discipline mirrors matter for the label.
MIRROR_PAIRS = [("cs.SD", "eess.AS"), ("cs.CV", "eess.IV")]


def _cats(x) -> list[str]:
    return [] if x is None else list(x)


def load_split(tag: str, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    if split == "temporal":
        df = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet")
    else:
        df = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet")
        df = df[df["split"] == split]
    df = df.reset_index(drop=True)
    z = np.load(MODELS / tag / f"{split}_predictions.npz")
    assert np.array_equal(z["labels"], df["discipline"].map(LABEL2ID).to_numpy()), \
        f"{tag}/{split}: saved labels do not match the corpus order"
    df["pred"] = [DISCIPLINES[i] for i in softmax(z["logits"]).argmax(1)]
    df["hit"] = df["pred"] == df["discipline"]
    return df, softmax(z["logits"])


def error_kind(row) -> str:
    prim, sec = set(), set()
    for c in _cats(row["mapped_categories"]):
        m = BY_CATEGORY.get(c)
        if m:
            prim.add(m.discipline)
            if m.secondary:
                sec.add(m.secondary)
    if row["pred"] in prim:
        return "co_listed"
    if row["pred"] in sec:
        return "adjacent"
    return "genuine"


def diagnose(df: pd.DataFrame) -> dict:
    cs = df[df["discipline"] == CS]
    out: dict = {"n": int(len(df)), "cs_n": int(len(cs)),
                 "cs_recall": round(float(cs["hit"].mean()), 4)}

    # --- clean vs ambiguous --------------------------------------------------
    for flag, name in ((False, "clean"), (True, "ambiguous")):
        sub = cs[cs["ambiguous"] == flag]
        if len(sub):
            out[f"cs_recall_{name}"] = round(float(sub["hit"].mean()), 4)
            out[f"cs_n_{name}"] = int(len(sub))

    # --- where CS misses go -------------------------------------------------
    miss = cs[~cs["hit"]]
    out["cs_misses_to"] = dict(Counter(miss["pred"]).most_common())
    kinds = Counter(error_kind(r) for _, r in miss.iterrows())
    out["cs_error_kinds"] = {k: int(v) for k, v in kinds.items()}
    out["cs_defensible_share"] = round(
        (kinds["co_listed"] + kinds["adjacent"]) / max(1, len(miss)), 4)

    # --- per primary category ------------------------------------------------
    g = cs.groupby("primary_category").agg(n=("hit", "size"), recall=("hit", "mean"),
                                            ambiguous=("ambiguous", "mean"))
    g = g[g["n"] >= 10].sort_values("recall")
    out["cs_recall_by_primary"] = {
        c: {"n": int(r.n), "recall": round(float(r.recall), 4),
            "ambiguous_share": round(float(r.ambiguous), 4)}
        for c, r in g.iterrows()}

    # --- mirrored categories -------------------------------------------------
    mirrors = {}
    in_any_mirror = np.zeros(len(df), bool)
    for a, b in MIRROR_PAIRS:
        both = df["categories"].map(lambda c, a=a, b=b: a in _cats(c) and b in _cats(c))
        in_any_mirror |= both.to_numpy()
        sub = df[both]
        if not len(sub):
            continue
        # How contested these labels are: the labeller's own vote margin, and
        # the share it already flags as ambiguous.
        mirrors[f"{a}+{b}"] = {
            "n": int(len(sub)),
            "label_split": dict(Counter(sub["discipline"])),
            "ambiguous_share": round(float(sub["ambiguous"].mean()), 4),
            "mean_vote_margin": round(float(sub["margin"].mean()), 4),
            "model_accuracy": round(float(sub["hit"].mean()), 4),
        }
    out["mirrored_pairs"] = mirrors
    cs_miss_mask = (df["discipline"] == CS) & ~df["hit"]
    out["cs_misses_on_mirrored_papers"] = int((cs_miss_mask & in_any_mirror).sum())
    out["cs_misses_total"] = int(cs_miss_mask.sum())
    # Recall if the mirrored papers -- whose label the text cannot determine --
    # are set aside. Not a headline: a diagnostic of how much is label noise.
    keep = (df["discipline"] == CS) & ~in_any_mirror
    out["cs_recall_excluding_mirrored"] = round(float(df.loc[keep, "hit"].mean()), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DEPLOYED_MODEL)
    ap.add_argument("--splits", nargs="+", default=["test", "temporal"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {"model": args.tag, "mirror_pairs": [f"{a}+{b}" for a, b in MIRROR_PAIRS],
              "splits": {}}
    for split in args.splits:
        if not (MODELS / args.tag / f"{split}_predictions.npz").exists():
            print(f"[skip {split}: no saved predictions for {args.tag}]")
            continue
        df, _ = load_split(args.tag, split)
        r = diagnose(df)
        report["splits"][split] = r
        print(f"\n== {args.tag} / {split}: CS recall {r['cs_recall']:.3f} "
              f"(clean {r.get('cs_recall_clean', float('nan')):.3f}, "
              f"ambiguous {r.get('cs_recall_ambiguous', float('nan')):.3f})")
        print(f"   CS misses go to: {r['cs_misses_to']}")
        print(f"   CS error kinds:  {r['cs_error_kinds']}  "
              f"-> defensible {r['cs_defensible_share']:.1%}")
        print(f"   CS misses on mirrored-category papers: "
              f"{r['cs_misses_on_mirrored_papers']} of {r['cs_misses_total']}")
        for k, v in r["mirrored_pairs"].items():
            print(f"   {k:16s} n={v['n']:4d}  labels {v['label_split']}  "
                  f"ambiguous {v['ambiguous_share']:.0%}  "
                  f"margin {v['mean_vote_margin']:.2f}  "
                  f"model acc {v['model_accuracy']:.3f}")
        print(f"   CS recall excluding mirrored papers: "
              f"{r['cs_recall_excluding_mirrored']:.3f}")

    out = Path(args.out) if args.out else RESULTS / f"cs_diagnosis_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
