"""Label methodological facets from several independent sources, with no gold set.

The problem this solves: we have no hand-labelled ground truth for research
methodology, and no realistic way to get expert labels. A single cue-based
labeller gives labels but no way to know how good they are -- and when a second,
independent labeller was built (title self-declaration) the two agreed only 43% of
the time, which means the previously reported "accuracy" was agreement with one
noisy source, not correctness.

The standard answer to exactly this is **latent-variable label modelling**
(Dawid & Skene 1979, the statistical core of Snorkel). Given several imperfect
labelling functions that vote independently, their *agreement structure alone*
identifies how accurate each one is, without ever seeing a true label. Sources
that agree with the consensus more often than chance are inferred to be reliable;
sources that agree less are down-weighted. The output is a posterior probability
per facet per paper, and -- the part that matters here -- an **estimated accuracy
for every labelling function**.

**The five labelling functions are document regions**: title, abstract,
introduction, methods, and results+conclusion. Each votes on each facet from its
own text. They are separate evidence: whether an abstract says "we prove" is a
different observation from whether the methods section does.

The independence assumption is approximate and worth stating plainly. The regions
share one cue vocabulary, so a cue list that is systematically wrong about a facet
will bias every region the same way, and Dawid-Skene cannot detect that -- it
measures *reliability*, not *validity*. What it does rule out is the failure mode
we actually observed: one region being far noisier than we assumed.

Run:
    python -m crc.data.label_facets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crc.taxonomy.facets import FACET_KEYS, FACETS, derive_designs, extended_evidence

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS = WORK / "corpus"
CHUNK_FILES = ["chunks_eval.parquet", "chunks_v2.parquet"]

#: One labelling function per document region.
REGIONS: dict[str, tuple[str, ...]] = {
    "title": ("title",),
    "abstract": ("abstract",),
    "introduction": ("introduction", "background"),
    "methods": ("methods",),
    "results_conclusion": ("results", "discussion", "conclusion"),
}
REGION_NAMES = list(REGIONS)

#: A region shorter than this cannot be trusted to have expressed a facet, so it
#: abstains rather than voting "absent".
MIN_WORDS_TO_VOTE = 40
#: The title is short by nature; it votes only for the facet it declares, and
#: never votes "absent".
ABSTAIN, PRESENT, ABSENT = 0, 1, -1


def vote_region(text: str, region: str, n_words: int,
                extended: bool = True) -> dict[str, int]:
    """One labelling function's vote on every facet, from one region's text.

    ``extended=False`` reproduces the v1 cue vocabulary exactly (the yardstick the
    August model was trained and judged on); the default adds vocabulary v2
    (`crc.taxonomy.facets.extended_evidence`).
    """
    text = text or ""
    low = text.lower()
    votes: dict[str, int] = {}
    for f in FACETS:
        if region == "title":
            pat = f.title_re()
            # A title that does not declare a facet is not evidence against it.
            votes[f.key] = (PRESENT if pat and pat.search(low) else ABSTAIN)
            continue
        if n_words < MIN_WORDS_TO_VOTE:
            votes[f.key] = ABSTAIN
            continue
        if any(a in low for a in f.anti_cues):
            votes[f.key] = ABSENT
            continue
        hit = any(c in low for c in f.cues)
        if extended and not hit:
            hit = extended_evidence(f.key, text, low)
        votes[f.key] = PRESENT if hit else ABSENT
    return votes


def dawid_skene(votes: np.ndarray, max_iter: int = 100,
                tol: float = 1e-6, seed: int = 0) -> dict:
    """Binary Dawid-Skene EM over labelling-function votes.

    ``votes`` is ``(n_items, n_lfs)`` with values in {+1, -1, 0=abstain}.

    Returns the posterior P(facet present) per item, plus each labelling
    function's estimated sensitivity and specificity -- accuracy estimates
    obtained without a single ground-truth label.
    """
    n, m = votes.shape
    pos, neg = votes == PRESENT, votes == ABSENT

    # Initialise from majority vote so EM starts somewhere sensible.
    with np.errstate(invalid="ignore"):
        post = np.where(pos.sum(1) + neg.sum(1) > 0,
                        pos.sum(1) / np.maximum(pos.sum(1) + neg.sum(1), 1), 0.5)
    prior = float(np.clip(post.mean(), 0.02, 0.98))
    sens = np.full(m, 0.7)     # P(vote=+1 | y=1)
    spec = np.full(m, 0.7)     # P(vote=-1 | y=0)

    prev = None
    for _ in range(max_iter):
        # --- M step (given current posteriors)
        w1, w0 = post, 1.0 - post
        for j in range(m):
            v1 = w1[pos[:, j]].sum(); v1n = w1[neg[:, j]].sum()
            v0 = w0[neg[:, j]].sum(); v0n = w0[pos[:, j]].sum()
            sens[j] = float(np.clip((v1 + 1.0) / (v1 + v1n + 2.0), 0.01, 0.99))
            spec[j] = float(np.clip((v0 + 1.0) / (v0 + v0n + 2.0), 0.01, 0.99))
        prior = float(np.clip(post.mean(), 0.02, 0.98))

        # --- E step (log space; abstentions contribute nothing)
        l1 = np.full(n, np.log(prior))
        l0 = np.full(n, np.log1p(-prior))
        for j in range(m):
            l1[pos[:, j]] += np.log(sens[j])
            l1[neg[:, j]] += np.log1p(-sens[j])
            l0[pos[:, j]] += np.log1p(-spec[j])
            l0[neg[:, j]] += np.log(spec[j])
        mx = np.maximum(l1, l0)
        post = np.exp(l1 - mx) / (np.exp(l1 - mx) + np.exp(l0 - mx))

        cur = float(np.sum(mx + np.log(np.exp(l1 - mx) + np.exp(l0 - mx))))
        if prev is not None and abs(cur - prev) < tol:
            break
        prev = cur

    return {
        "posterior": post,
        "prior": prior,
        "sensitivity": sens.tolist(),
        "specificity": spec.tolist(),
        # A single readable number per LF: balanced accuracy.
        "estimated_accuracy": ((sens + spec) / 2.0).tolist(),
    }


def load_regions(paper_ids: set[str] | None = None) -> pd.DataFrame:
    """Per-paper text for each region."""
    frames = []
    for name in CHUNK_FILES:
        p = CORPUS / name
        if p.exists():
            frames.append(pd.read_parquet(
                p, columns=["paper_id", "section", "n_words", "text"]))
    ch = pd.concat(frames, ignore_index=True)
    ch = ch.drop_duplicates(subset=["paper_id", "section", "text"])
    if paper_ids is not None:
        ch = ch[ch["paper_id"].isin(paper_ids)]

    sec2region = {s: r for r, secs in REGIONS.items() for s in secs}
    ch["region"] = ch["section"].map(sec2region)
    ch = ch[ch["region"].notna()]
    agg = (ch.groupby(["paper_id", "region"])
             .agg(text=("text", lambda s: " ".join(s.astype(str))),
                  n_words=("n_words", "sum"))
             .reset_index())
    return agg


def collect_votes(paper_ids: set[str], order: list[str] | None = None,
                  extended: bool = True) -> dict[str, np.ndarray]:
    """Every region's vote on every facet: ``{facet: (n_papers, n_regions)}``.

    Rows follow ``order`` (default: sorted ids). Titles come from the metadata pool.
    """
    order = list(order) if order is not None else sorted(paper_ids)
    idx = {p: i for i, p in enumerate(order)}
    votes = {f: np.zeros((len(order), len(REGION_NAMES)), dtype=np.int8)
             for f in FACET_KEYS}
    titles = pd.read_parquet(CORPUS / "computing_pool.parquet", columns=["id", "title"])
    titles = titles[titles["id"].isin(idx)]
    jt = REGION_NAMES.index("title")
    for pid, t in zip(titles["id"], titles["title"].fillna("")):
        for f, v in vote_region(t, "title", 0, extended).items():
            votes[f][idx[pid], jt] = v
    for r in load_regions(set(order)).itertuples(index=False):
        if r.region == "title" or r.paper_id not in idx:
            continue
        j = REGION_NAMES.index(r.region)
        for f, v in vote_region(r.text, r.region, int(r.n_words), extended).items():
            votes[f][idx[r.paper_id], j] = v
    return votes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(CORPUS / "methodology_pool.parquet"))
    ap.add_argument("--out", default=str(CORPUS / "facets_pool_v2.parquet"))
    ap.add_argument("--legacy-cues", action="store_true",
                    help="v1 vocabulary (what facets_pool.parquet was built with)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = pd.read_parquet(args.pool, columns=["paper_id", "discipline", "split",
                                               "year", "design"])
    ids = set(pool["paper_id"])
    print(f"papers: {len(ids):,}  vocabulary: {'v1 (legacy)' if args.legacy_cues else 'v2'}")

    # ---- collect votes: (n_papers, n_regions) per facet; titles are their own LF
    order = sorted(ids)
    n = len(order)
    votes = collect_votes(ids, order=order, extended=not args.legacy_cues)

    # ---- fit a label model per facet
    print("\nfitting Dawid-Skene label models (no ground truth used)\n")
    hdr = f"{'facet':16s} {'prior':>7s} " + " ".join(
        f"{r[:9]:>10s}" for r in REGION_NAMES)
    print(hdr)
    print("-" * len(hdr))
    models, post = {}, {}
    for f in FACET_KEYS:
        m = dawid_skene(votes[f], seed=args.seed)
        models[f] = {k: v for k, v in m.items() if k != "posterior"}
        post[f] = m["posterior"]
        acc = m["estimated_accuracy"]
        print(f"{f:16s} {m['prior']:>7.3f} " +
              " ".join(f"{a:>10.3f}" for a in acc))
    print("\n(values are each region's ESTIMATED balanced accuracy for that "
          "facet, inferred from agreement alone)")

    # ---- assemble output
    out = pd.DataFrame({"paper_id": order})
    out = out.merge(pool, on="paper_id", how="left")
    for f in FACET_KEYS:
        out[f"p_{f}"] = post[f]
        out[f] = post[f] >= 0.5
    derived = [derive_designs({f: bool(row[f]) for f in FACET_KEYS})
               for _, row in out.iterrows()]
    out["derived_design"] = [d[0][0] for d in derived]
    out["derived_rule"] = [d[0][1] for d in derived]
    out["n_compatible_designs"] = [len(d) for d in derived]

    out.to_parquet(args.out, index=False)
    stats_path = Path(args.out).with_name(Path(args.out).stem + "_stats.json")

    stats = {
        "n_papers": int(n),
        "vocabulary": 1 if args.legacy_cues else 2,
        "no_facet_rate": float((out[FACET_KEYS].sum(1) == 0).mean()),
        "regions": REGION_NAMES,
        "label_models": models,
        "facet_prevalence": {f: float(out[f].mean()) for f in FACET_KEYS},
        "derived_design_counts": {k: int(v) for k, v in
                                  out["derived_design"].value_counts().items()},
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\nfacet prevalence (share of papers):")
    for f in FACET_KEYS:
        print(f"  {f:16s} {out[f].mean():6.1%}")

    # the whole point: papers that are more than one thing at once
    multi = int((out["builds"] & out["evaluates"]).sum())
    print(f"\npapers that BOTH build and evaluate: {multi:,} "
          f"({multi/n:.1%}) -- these are the ones a single label had to "
          f"guess between")

    print(f"\nderived design distribution:")
    for k, v in out["derived_design"].value_counts().items():
        print(f"  {k:48s} {v:>7,}  {v/n:6.1%}")

    print(f"\nwrote {args.out}")
    print(f"stats -> {stats_path}")


if __name__ == "__main__":
    main()
