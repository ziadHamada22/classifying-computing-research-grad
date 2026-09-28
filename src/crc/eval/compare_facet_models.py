"""Head-to-head: the August facet model (vocabulary v1) vs the September one (v2).

Changing the cue vocabulary changes the labels *and* the yardstick the gold-free
test uses, so a new model agreeing better with the new cues proves nothing on its
own. The comparison is therefore made three ways, on the same test papers:

1. against the unseen methods / results text read with the **old** cues -- the
   August model's home ground;
2. against the same text read with the **new** cues;
3. against arXiv ``comments`` / ``journal-ref`` signals that no cue list and no
   model ever read (`crc.data.external_signals`) -- the neutral arbiter.

Each model is judged at its own fitted thresholds. Differences in design-level
agreement carry a paired bootstrap 95% interval.

Run:
    python -m crc.eval.compare_facet_models
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from crc.agents.methodology.predict import MODELS, THRESHOLDS_FILE
from crc.data.external_signals import SIGNAL_FACET, load_signals
from crc.data.label_facets import ABSENT, PRESENT, REGION_NAMES
from crc.eval.evaluate_facet_decisions import (
    CACHE,
    UNSEEN,
    designs_of,
    hard,
    unseen_reference,
)
from crc.taxonomy.facets import FACET_KEYS

PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "results"


def load_votes(split: str, vocab: str, pids: np.ndarray) -> dict[str, np.ndarray]:
    p = CACHE / f"votes_{split}_{vocab}_{len(pids)}.npz"
    if not p.exists():
        from crc.data.label_facets import collect_votes
        v = collect_votes(set(pids), order=list(pids), extended=(vocab == "v2"))
        np.savez(p, pids=pids, **v)
    z = np.load(p, allow_pickle=True)
    assert list(z["pids"]) == list(pids)
    return {f: z[f] for f in FACET_KEYS}


def mean_kappa(H: np.ndarray, votes: dict[str, np.ndarray]) -> tuple[float, dict]:
    per = {}
    for k, f in enumerate(FACET_KEYS):
        mv = np.where(H[:, k], PRESENT, ABSENT)
        per[f] = {}
        for reg in UNSEEN:
            rv = votes[f][:, REGION_NAMES.index(reg)]
            m = rv != 0
            per[f][reg] = round(float(cohen_kappa_score(mv[m], rv[m])), 4)
    return float(np.mean([v for d in per.values() for v in d.values()])), per


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=str(MODELS / "methodology-facets"))
    ap.add_argument("--new", default=str(MODELS / "methodology-facets-v2"))
    args = ap.parse_args()

    models = {}
    for tag, d in (("v1 model", Path(args.old)), ("v2 model", Path(args.new))):
        thr = None
        if (d / THRESHOLDS_FILE).exists():
            thr = json.loads((d / THRESHOLDS_FILE).read_text())["thresholds"]
        per_split = {}
        for split in ("val", "test"):
            z = np.load(d / f"{split}_facet_predictions.npz", allow_pickle=True)
            per_split[split] = (z["paper_ids"].astype(str), z["probs"].astype(float))
        models[tag] = (thr, per_split)

    pids = models["v1 model"][1]["test"][0]
    assert list(pids) == list(models["v2 model"][1]["test"][0]), "test sets differ"
    sig = load_signals()
    report: dict = {"test_papers": int(len(pids)), "thresholds": {k: v[0] for k, v in models.items()}}

    # ---- 1 & 2: unseen text, under each vocabulary
    for vocab in ("v1", "v2"):
        votes = load_votes("test", vocab, pids)
        R, ref, has_ref = unseen_reference(votes)
        ref = np.array(ref)
        row, agree = {}, {}
        for tag, (thr, per_split) in models.items():
            H = hard(per_split["test"][1], thr)
            k, per = mean_kappa(H, votes)
            des, nofacet = designs_of(H)
            agree[tag] = (np.array(des)[has_ref] == ref[has_ref]).astype(float)
            row[tag] = {"mean_kappa": round(k, 4), "per_facet_kappa": per,
                        "design_agreement": round(float(agree[tag].mean()), 4),
                        "no_facet_rate": round(float(nofacet.mean()), 4)}
        row["design_agreement_diff_v2_minus_v1_ci95"] = boot_diff(agree["v2 model"], agree["v1 model"])
        report[f"unseen_text_{vocab}_cues"] = row

    # ---- 3: external signals, threshold-free AUC and lift at threshold (val + test)
    ext = {}
    for tag, (thr, per_split) in models.items():
        ids = np.concatenate([per_split[s][0] for s in ("val", "test")])
        P = np.concatenate([per_split[s][1] for s in ("val", "test")])
        H = hard(P, thr)
        s = sig.reindex(ids).fillna(False)
        ext[tag] = {}
        for name, facet in SIGNAL_FACET.items():
            y = s[name].to_numpy().astype(bool)
            if y.sum() < 5:
                continue
            k = FACET_KEYS.index(facet)
            ext[tag][name] = {"facet": facet, "n": int(y.sum()),
                              "auc": round(float(roc_auc_score(y, P[:, k])), 4),
                              "p_given_signal": round(float(H[y, k].mean()), 4),
                              "p_otherwise": round(float(H[~y, k].mean()), 4)}
        y = s["position"].to_numpy().astype(bool)
        ext[tag]["position_no_facet"] = {"n": int(y.sum()),
                                         "share_with_no_facet": round(float((~H[y].any(1)).mean()), 4)}
    report["external_signals_val_test"] = ext

    out = RESULTS / "facet_models_comparison.json"
    out.write_text(json.dumps(report, indent=2))

    for vocab in ("v1", "v2"):
        r = report[f"unseen_text_{vocab}_cues"]
        print(f"\nunseen methods/results text, {vocab} cues:")
        for tag in models:
            print(f"  {tag}: mean kappa {r[tag]['mean_kappa']:.4f} | design agreement "
                  f"{r[tag]['design_agreement']:.4f} | no facet {r[tag]['no_facet_rate']:.3f}")
        print(f"  design agreement v2 - v1, 95% CI: {r['design_agreement_diff_v2_minus_v1_ci95']}")
    print("\narXiv comments / journal-ref signals (val + test), AUC of the facet probability:")
    for name in ext["v1 model"]:
        if name == "position_no_facet":
            continue
        a, b = ext["v1 model"][name], ext["v2 model"][name]
        print(f"  {name:16s}->{a['facet']:13s} n={a['n']:4d}  AUC {a['auc']:.3f} -> {b['auc']:.3f}   "
              f"P(facet|signal) {a['p_given_signal']:.3f} -> {b['p_given_signal']:.3f}")
    print("  position papers with no facet:",
          ext["v1 model"]["position_no_facet"], "->", ext["v2 model"]["position_no_facet"])
    print("saved ->", out)


if __name__ == "__main__":
    main()
