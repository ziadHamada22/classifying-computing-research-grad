"""Discipline-conditioned self-training to expand Agent 2's weak labels.

The weak labeller in `label_fields.py` covers ~95% of most disciplines through
high-precision arXiv-category evidence, but **Software Engineering sits at 29%**
because arXiv gives SE one category and the seven SE fields are carried by
keywords alone. The 8,638 unlabelled SE papers are exactly the ones whose
abstracts lack the distinctive phrases — the harder, more valuable examples the
v1 head never saw. All four of Agent 2's thinnest classes are SE.

Self-training recovers them. For every *unlabelled* paper we already know the
discipline (from the discipline pipeline), so pseudo-labelling is conditioned:
the v1 field model scores the paper against **only that discipline's fields**.
A pseudo-label is accepted only when two independent views back it:

  * the encoder is very confident on its own (``--conf-high``), OR
  * the encoder is reasonably confident (``--conf-mid``) AND an independent
    keyword vote (the same phrase lists, but with no margin/score gate) picks
    the same field.

Two agreeing views is what buys precision; a per-field cap stops the majority
field (Testing) from swallowing the budget. Accepted labels are marked
``field_evidence="self-train"`` so they stay auditable and separable.

The output corpus keeps v1's val/test split **byte-identical** — pseudo-labels
are added to *train only* — so ``field-scibert`` vs ``field-scibert-v2`` is a
clean, single-variable comparison.

Run:
    python -m crc.data.self_train_fields                 # full run
    python -m crc.data.self_train_fields --sample 4000   # quick dry run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from crc.taxonomy import DISCIPLINES
from crc.taxonomy.fields import (
    DISCIPLINE_FIELD_IDS,
    FIELDS_BY_DISCIPLINE,
    GLOBAL_ID2LABEL,
    N_GLOBAL_FIELDS,
)
# The keyword view — reuse the exact compiled patterns the labeller uses, but
# without its score/margin gate, so it is a genuinely independent second vote.
from crc.data.label_fields import _PATTERNS

WORK = Path(r"C:\Users\ziada\gp_data")
POOL = WORK / "corpus" / "fields_pool.parquet"
CORPUS = WORK / "corpus" / "fields_corpus.parquet"
OUT_CORPUS = WORK / "corpus" / "fields_corpus_st.parquet"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def keyword_vote(discipline: str, title: str, abstract: str) -> str | None:
    """Field with the most distinct keyword hits, or None. No score gate."""
    title, abstract = title or "", abstract or ""
    best, best_n = None, 0
    for f, pat in _PATTERNS[discipline]:
        hits = {m.group(0).lower() for m in pat.finditer(title)}
        hits |= {m.group(0).lower() for m in pat.finditer(abstract)}
        if len(hits) > best_n:
            best, best_n = f.name, len(hits)
    return best


@torch.no_grad()
def encoder_logits(model, tok, texts: list[str], device: str,
                   batch_size: int, max_length: int) -> np.ndarray:
    out = []
    use_amp = device == "cuda"
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, padding=True,
                  max_length=max_length, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(**enc).logits
        out.append(logits.float().cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"    {i:>7,}/{len(texts):,}", flush=True)
    return np.concatenate(out, axis=0) if out else np.zeros((0, N_GLOBAL_FIELDS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "field-scibert"))
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--out", default=str(OUT_CORPUS))
    ap.add_argument("--conf-high", type=float, default=0.90,
                    help="Accept on encoder confidence alone at/above this.")
    ap.add_argument("--conf-mid", type=float, default=0.70,
                    help="Accept at/above this IF the keyword view agrees.")
    ap.add_argument("--cap-per-field", type=int, default=1500,
                    help="Max pseudo-labels added to any one field.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--sample", type=int, default=0,
                    help="Dry run over N unlabelled papers; writes nothing.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    corpus = pd.read_parquet(args.corpus)
    used_ids = set(corpus["id"])
    print(f"v1 corpus: {len(corpus):,} rows "
          f"(train {int((corpus['split']=='train').sum()):,} / "
          f"val {int((corpus['split']=='val').sum()):,} / "
          f"test {int((corpus['split']=='test').sum()):,})")

    pool = pd.read_parquet(args.pool)
    unl = pool[pool["field"].isna() & ~pool["id"].isin(used_ids)].copy()
    unl = unl[unl["discipline"].isin(set(DISCIPLINES))]
    if args.sample:
        unl = unl.sample(n=min(args.sample, len(unl)), random_state=args.seed)
    print(f"unlabelled candidates: {len(unl):,}")
    print(unl["discipline"].value_counts().to_string())

    texts = (unl["title"].fillna("") + ". " + unl["abstract"].fillna("")
             ).str.strip().tolist()

    print("\nloading v1 field model…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device).eval()

    print("scoring unlabelled papers…", flush=True)
    t0 = time.time()
    logits = encoder_logits(model, tok, texts, device,
                            args.batch_size, args.max_length)
    print(f"  scored in {time.time()-t0:.0f}s")

    discs = unl["discipline"].tolist()
    titles = unl["title"].fillna("").tolist()
    abstracts = unl["abstract"].fillna("").tolist()

    pseudo_field = np.empty(len(unl), dtype=object)
    pseudo_conf = np.zeros(len(unl))
    pseudo_route = np.empty(len(unl), dtype=object)   # how it was accepted
    kw_agree = np.zeros(len(unl), dtype=bool)

    for i, d in enumerate(discs):
        allowed = DISCIPLINE_FIELD_IDS[d]
        probs = _softmax(logits[i, allowed])
        j = int(np.argmax(probs))
        top_id = allowed[j]
        conf = float(probs[j])
        name = GLOBAL_ID2LABEL[top_id]
        kw = keyword_vote(d, titles[i], abstracts[i])
        agree = (kw is not None and kw == name)
        kw_agree[i] = agree
        pseudo_conf[i] = conf
        if conf >= args.conf_high:
            pseudo_field[i], pseudo_route[i] = name, "conf-high"
        elif conf >= args.conf_mid and agree:
            pseudo_field[i], pseudo_route[i] = name, "conf-mid+kw"
        else:
            pseudo_field[i], pseudo_route[i] = None, "rejected"

    unl["field_new"] = pseudo_field
    unl["field_conf"] = pseudo_conf.round(4)
    unl["accept_route"] = pseudo_route
    unl["kw_agree"] = kw_agree

    accepted = unl[unl["field_new"].notna()].copy()
    print(f"\naccepted {len(accepted):,} / {len(unl):,} "
          f"({len(accepted)/max(1,len(unl)):.1%})")
    print("\nroute:")
    print(accepted["accept_route"].value_counts().to_string())

    # Per-field cap — keep the highest-confidence ones.
    capped = []
    for name, g in accepted.groupby("field_new", sort=False):
        g = g.sort_values("field_conf", ascending=False).head(args.cap_per_field)
        capped.append(g)
    accepted = pd.concat(capped, ignore_index=True) if capped else accepted
    print(f"\nafter per-field cap ({args.cap_per_field}): {len(accepted):,}")

    print("\n=== pseudo-labels added per discipline / field ===")
    for d in DISCIPLINES:
        g = accepted[accepted["discipline"] == d]
        if g.empty:
            continue
        print(f"\n{d}  +{len(g):,}")
        for name, n in g["field_new"].value_counts().items():
            mc = g.loc[g["field_new"] == name, "field_conf"].mean()
            print(f"    {name:46s} +{n:>5,}   mean_conf {mc:.3f}")

    # Save a sample for eyeballing precision before committing to a retrain.
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample = accepted.sample(n=min(40, len(accepted)), random_state=args.seed)
    sample_out = RESULTS / "selftrain_sample.json"
    sample_out.write_text(json.dumps([
        {"discipline": r.discipline, "field": r.field_new,
         "conf": float(r.field_conf), "route": r.accept_route,
         "title": r.title[:160]}
        for r in sample.itertuples(index=False)
    ], indent=2))
    print(f"\nwrote {len(sample)} spot-check rows -> {sample_out}")

    stats = {
        "n_candidates": int(len(unl)),
        "n_accepted": int(len(accepted)),
        "acceptance_rate": round(len(accepted) / max(1, len(unl)), 4),
        "conf_high": args.conf_high, "conf_mid": args.conf_mid,
        "cap_per_field": args.cap_per_field,
        "added_per_field": {
            f"{r.discipline}/{r.field_new}": 0 for r in accepted.itertuples()
        },
    }
    per_field = accepted.groupby(["discipline", "field_new"]).size()
    stats["added_per_field"] = {f"{d}/{f}": int(n)
                                for (d, f), n in per_field.items()}
    (RESULTS / "selftrain_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"stats -> {RESULTS / 'selftrain_stats.json'}")

    if args.sample:
        print("\n[dry run] --sample set, not writing corpus.")
        return

    # Assemble augmented corpus: v1 rows untouched (all splits), pseudo rows
    # appended to TRAIN only, columns aligned to the v1 corpus schema.
    add = pd.DataFrame({
        "id": accepted["id"].values,
        "title": accepted["title"].values,
        "abstract": accepted["abstract"].values,
        "primary_category": accepted["primary_category"].values,
        "categories": accepted["categories"].values,
        "mapped_categories": accepted["mapped_categories"].values,
        "co_listed": accepted["co_listed"].values,
        "discipline": accepted["discipline"].values,
        "margin": accepted["margin"].values,
        "ambiguous": accepted["ambiguous"].values,
        "year": accepted["year"].values,
        "field": accepted["field_new"].values,
        "field_score": np.nan,
        "field_margin": np.nan,
        "field_evidence": "self-train",
        "field_runner_up": None,
        "split": "train",
    })
    add["text"] = (add["title"].fillna("") + ". "
                   + add["abstract"].fillna("")).str.strip()
    add = add[list(corpus.columns)]  # exact column order

    out = pd.concat([corpus, add], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    print(f"\naugmented corpus -> {args.out}")
    print(f"  rows: {len(corpus):,} -> {len(out):,}  (+{len(add):,} train)")
    print("  val/test rows unchanged (byte-identical split to v1).")
    print("\nSE train growth:")
    for name in FIELD_NAMES(("Software Engineering",)):
        v1n = int(((corpus["discipline"] == "Software Engineering")
                   & (corpus["field"] == name) & (corpus["split"] == "train")).sum())
        addn = int((add["field"] == name).sum())
        print(f"    {name:46s} {v1n:>4,} -> {v1n + addn:>4,}  (+{addn})")


def FIELD_NAMES(disciplines):
    return [f.name for d in disciplines for f in FIELDS_BY_DISCIPLINE[d]]


if __name__ == "__main__":
    main()
