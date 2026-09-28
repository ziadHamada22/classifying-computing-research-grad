"""Agent 3 as deployed in August vs as deployed now, on the same test papers.

Before: `methodology-facets` (vocabulary v1), 0.5 for every facet, and Design &
Creation returned silently whenever no facet fired.
After: `methodology-facets-v2` (vocabulary v2), per-facet thresholds fitted on
validation, and the no-evidence fallback chosen on validation (flagged).

Judged, with a paired bootstrap, against the design implied by each paper's own
unseen methods / results text -- under both cue vocabularies, so the new system is
also tested on the old one's home ground -- plus the no-evidence rate and the
arXiv comment / journal-ref signals.

Run:
    python -m crc.eval.agent3_before_after
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crc.agents.methodology.predict import DEPLOYED_MODEL, MODELS, THRESHOLDS_FILE
from crc.data.external_signals import SIGNAL_FACET, load_signals
from crc.eval.compare_facet_models import boot_diff, load_votes
from crc.eval.evaluate_facet_decisions import designs_of, hard, unseen_reference
from crc.taxonomy.facets import FACET_KEYS, derive_designs

PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "results"
DEFAULT = "Design & Creation (Design Science)"


def system_designs(P: np.ndarray, thr: dict | None, fallback: str | None) -> tuple[np.ndarray, np.ndarray]:
    H = hard(P, thr)
    des, nofacet = designs_of(H)
    des = np.array(des, dtype=object)
    if fallback == "top_facet":
        for i in np.where(nofacet)[0]:
            top = FACET_KEYS[int(np.argmax(P[i]))]
            des[i] = derive_designs({f: f == top for f in FACET_KEYS})[0][0]
    elif fallback is not None:
        raise ValueError("this comparison supports the deployed fallback only")
    return des, nofacet


def main() -> None:
    old_dir, new_dir = MODELS / "methodology-facets", MODELS / DEPLOYED_MODEL
    cfg = json.loads((new_dir / THRESHOLDS_FILE).read_text())
    systems = {
        "August (v1 model, 0.5, silent default)": (old_dir, None, None),
        f"now ({new_dir.name}, fitted thresholds, {cfg['no_evidence_fallback']} fallback)":
            (new_dir, cfg["thresholds"], cfg["no_evidence_fallback"]),
    }
    out: dict = {}
    preds = {}
    for name, (d, thr, fb) in systems.items():
        z = np.load(d / "test_facet_predictions.npz", allow_pickle=True)
        preds[name] = (z["paper_ids"].astype(str), z["probs"].astype(float), thr, fb)
    pids = next(iter(preds.values()))[0]
    assert all(list(p[0]) == list(pids) for p in preds.values())

    for vocab in ("v1", "v2"):
        R, ref, has_ref = unseen_reference(load_votes("test", vocab, pids))
        ref = np.array(ref, dtype=object)
        agree, row = {}, {}
        for name, (_, P, thr, fb) in preds.items():
            des, nofacet = system_designs(P, thr, fb)
            agree[name] = (des[has_ref] == ref[has_ref]).astype(float)
            dc = des == DEFAULT
            row[name] = {"design_agreement_with_unseen_text": round(float(agree[name].mean()), 4),
                         "no_evidence_rate": round(float(nofacet.mean()), 4),
                         "share_of_design_and_creation_with_no_evidence":
                             round(float((nofacet & dc).sum() / max(dc.sum(), 1)), 4)
                             if fb is None else 0.0,
                         "design_counts": pd.Series(des).value_counts().to_dict()}
        a, b = list(agree)
        row["difference_now_minus_august"] = round(float(agree[b].mean() - agree[a].mean()), 4)
        row["difference_ci95"] = boot_diff(agree[b], agree[a])
        row["n_with_unseen_design"] = int(has_ref.sum())
        out[f"judged_with_{vocab}_cues"] = row

    sig = load_signals().reindex(pids).fillna(False)
    ext = {}
    for name, (_, P, thr, fb) in preds.items():
        H = hard(P, thr)
        ext[name] = {}
        for s, f in SIGNAL_FACET.items():
            y = sig[s].to_numpy().astype(bool)
            if y.sum():
                k = FACET_KEYS.index(f)
                ext[name][s] = {"n": int(y.sum()), "facet": f,
                                "p_given_signal": round(float(H[y, k].mean()), 4),
                                "p_otherwise": round(float(H[~y, k].mean()), 4)}
    out["external_signals_test"] = ext

    path = RESULTS / "agent3_before_after.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    for key in ("judged_with_v1_cues", "judged_with_v2_cues"):
        r = out[key]
        print(f"\n{key} ({r['n_with_unseen_design']:,} test papers with an unseen-text design)")
        for name in systems:
            print(f"  {name:62s} agreement {r[name]['design_agreement_with_unseen_text']:.4f} | "
                  f"no evidence {r[name]['no_evidence_rate']:.3f}")
        print(f"  difference {r['difference_now_minus_august']:+.4f}, 95% CI {r['difference_ci95']}")
    print("saved ->", path)


if __name__ == "__main__":
    main()
