"""How Agent 3 turns facet probabilities into a design -- thresholds and no-evidence cases.

Two decisions sit between the facet model and the design it reports, and neither
had been measured:

1. **One 0.5 threshold for every facet**, although facet prevalence runs from 1%
   to 53%. Per-facet thresholds are fitted here on validation (F1 against the
   label model's hard labels, the only labels there are) and judged on test by the
   gold-free yardstick: agreement with the methods / results text the model never
   read, and with arXiv comments / journal-ref signals it never saw either
   (`crc.data.external_signals`).
2. **What to answer when no facet fires.** The taxonomy returned Design & Creation
   by default -- on ~19% of papers, over half of all Design & Creation answers.
   Three replacements are compared, on validation to choose and on test to report:
   the default itself, the design the most probable single facet implies, and the
   designs of the most similar reference papers (`retrieval.py`). The reference is
   the design derived from the paper's own unseen methods and results text.

Nothing here uses a hand-labelled paper. What it establishes is agreement with
independent evidence, not accuracy against truth.

Run:
    python -m crc.eval.evaluate_facet_decisions                    # deployed model
    python -m crc.eval.evaluate_facet_decisions --model-dir models/methodology-facets \
        --facets C:/Users/ziada/gp_data/corpus/facets_pool.parquet  # August model
    python -m crc.eval.evaluate_facet_decisions --deploy     # write thresholds.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from crc.agents.methodology.predict import (
    DEPLOYED_LABELS,
    DEPLOYED_MODEL,
    MODELS,
    THRESHOLDS_FILE,
    MethodologyClassifier,
)
from crc.agents.methodology.retrieval import STORE_FILE, ReferenceStore
from crc.agents.methodology.train_facets import build_corpus
from crc.data.external_signals import SIGNAL_FACET, load_signals
from crc.data.label_facets import ABSENT, PRESENT, REGION_NAMES, collect_votes
from crc.taxonomy.facets import FACET_KEYS, derive_designs

PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "results"
CORPUS = Path(r"C:\Users\ziada\gp_data\corpus")
CACHE = Path(r"C:\Users\ziada\gp_data\cache\facet_decisions")

UNSEEN = ("methods", "results_conclusion")
GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
DEFAULT = "Design & Creation (Design Science)"


def fit_thresholds(probs: np.ndarray, soft: np.ndarray) -> dict[str, float]:
    """Per facet, the threshold maximising F1 against the label model's hard labels."""
    y = soft >= 0.5
    out = {}
    for k, f in enumerate(FACET_KEYS):
        if y[:, k].sum() == 0:
            out[f] = 0.5
            continue
        f1 = [f1_score(y[:, k], probs[:, k] >= t, zero_division=0) for t in GRID]
        out[f] = float(GRID[int(np.argmax(f1))])
    return out


def hard(probs: np.ndarray, thr: dict[str, float] | None) -> np.ndarray:
    t = np.array([(thr or {}).get(f, 0.5) for f in FACET_KEYS])
    return probs >= t[None, :]


def designs_of(H: np.ndarray) -> tuple[list[str], np.ndarray]:
    out = [derive_designs(dict(zip(FACET_KEYS, map(bool, row))))[0] for row in H]
    return [d for d, _ in out], np.array([r.startswith("default") for _, r in out])


def unseen_reference(votes: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Facets the unseen regions assert (either region votes present), and the design."""
    cols = [REGION_NAMES.index(r) for r in UNSEEN]
    R = np.stack([(votes[f][:, cols] == PRESENT).any(1) for f in FACET_KEYS], 1)
    voted = np.stack([(votes[f][:, cols] != 0).any(1) for f in FACET_KEYS], 1).any(1)
    d, default = designs_of(R)
    return R, d, voted & ~default


def region_agreement(H: np.ndarray, votes: dict[str, np.ndarray]) -> dict:
    """Per facet and unseen region: raw agreement and Cohen's kappa with the model."""
    out = {}
    for k, f in enumerate(FACET_KEYS):
        mv = np.where(H[:, k], PRESENT, ABSENT)
        row = {}
        for reg in UNSEEN:
            rv = votes[f][:, REGION_NAMES.index(reg)]
            m = rv != 0
            row[reg] = {"agreement": round(float((mv[m] == rv[m]).mean()), 4),
                        "kappa": round(float(cohen_kappa_score(mv[m], rv[m])), 4),
                        "n": int(m.sum())}
        out[f] = row
    return out


def external(H: np.ndarray, pids: np.ndarray, sig: pd.DataFrame) -> dict:
    s = sig.reindex(pids).fillna(False)
    out = {}
    for name, facet in SIGNAL_FACET.items():
        y = s[name].to_numpy().astype(bool)
        if y.sum() == 0:
            continue
        k = FACET_KEYS.index(facet)
        a, b = float(H[y, k].mean()), float(H[~y, k].mean())
        out[name] = {"facet": facet, "n": int(y.sum()),
                     "p_facet_given_signal": round(a, 4),
                     "p_facet_otherwise": round(b, 4),
                     "lift": round(a / b, 2) if b > 0 else None}
    return out


def embed(clf: MethodologyClassifier, name: str, texts: list[str]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{clf.name}_{name}.npy"
    if p.exists() and len(np.load(p)) == len(texts):
        return np.load(p)
    _, e = clf.encode(texts, batch_size=64)
    np.save(p, e)
    return e


def fallback_designs(P: np.ndarray, H: np.ndarray, E: np.ndarray,
                     store: ReferenceStore) -> dict[str, list[str]]:
    """For the rows with no facet: four candidate answers.

    The majority baseline -- always the commonest design among reference papers
    that had evidence -- is the bar a data-driven fallback has to clear: anything
    that merely leans towards the common class would beat the default without
    reading the paper at all.
    """
    ev = store.designs[store.has_evidence]
    majority = pd.Series(ev).value_counts().index[0]
    default, top_facet, neighbours = [], [], []
    for i in range(len(P)):
        default.append(DEFAULT)
        one = np.zeros(len(FACET_KEYS), bool)
        one[int(np.argmax(P[i]))] = True
        top_facet.append(derive_designs(dict(zip(FACET_KEYS, map(bool, one))))[0][0])
        neighbours.append(store.design_vote(E[i])[0])
    return {"default (Design & Creation)": default,
            f"majority of evidenced references ({majority.split(' (')[0].split(' /')[0]})":
                [majority] * len(P),
            "most probable single facet": top_facet,
            "similar reference papers": neighbours}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODELS / DEPLOYED_MODEL))
    ap.add_argument("--facets", default=str(DEPLOYED_LABELS),
                    help="the label file the model was trained on")
    ap.add_argument("--legacy-cues", action="store_true",
                    help="judge against the pre-2026-09 cue vocabulary")
    ap.add_argument("--deploy", action="store_true")
    args = ap.parse_args()

    d = Path(args.model_dir)
    clf = MethodologyClassifier.load(d)
    if not (d / STORE_FILE).exists():
        raise SystemExit(f"no {STORE_FILE} in {d}: run crc.agents.methodology.retrieval first")
    store = clf.store
    sig = load_signals()
    corpus = build_corpus(Path(args.facets)).set_index("paper_id")

    report: dict = {"model": d.name, "yardstick": "legacy cues" if args.legacy_cues else "current cues"}
    split_data = {}
    for split in ("val", "test"):
        z = np.load(d / f"{split}_facet_predictions.npz", allow_pickle=True)
        pids = z["paper_ids"].astype(str)
        P, S = z["probs"].astype(float), z["soft"].astype(float)
        vc = CACHE / f"votes_{split}_{'v1' if args.legacy_cues else 'v2'}_{len(pids)}.npz"
        CACHE.mkdir(parents=True, exist_ok=True)
        if vc.exists() and list(np.load(vc, allow_pickle=True)["pids"]) == list(pids):
            zc = np.load(vc, allow_pickle=True)
            votes = {f: zc[f] for f in FACET_KEYS}
        else:
            votes = collect_votes(set(pids), order=list(pids), extended=not args.legacy_cues)
            np.savez(vc, pids=pids, **votes)
        E = embed(clf, split, corpus.loc[pids, "text"].tolist())
        split_data[split] = (pids, P, S, votes, E)

    thr = fit_thresholds(split_data["val"][1], split_data["val"][2])
    report["thresholds"] = thr
    print("per-facet thresholds (fitted on val):",
          {f: t for f, t in thr.items()})

    for split, (pids, P, S, votes, E) in split_data.items():
        R, ref_design, has_ref = unseen_reference(votes)
        rows = {}
        for name, t in (("0.5 everywhere", None), ("fitted per facet", thr)):
            H = hard(P, t)
            des, nofacet = designs_of(H)
            des = np.array(des)
            row = {
                "no_facet_rate": round(float(nofacet.mean()), 4),
                "prevalence": {f: round(float(H[:, k].mean()), 4) for k, f in enumerate(FACET_KEYS)},
                "label_model_prevalence": {f: round(float((S[:, k] >= .5).mean()), 4)
                                           for k, f in enumerate(FACET_KEYS)},
                "f1_vs_label_model": {f: round(float(f1_score(S[:, k] >= .5, H[:, k], zero_division=0)), 4)
                                      for k, f in enumerate(FACET_KEYS)},
                "unseen_region": region_agreement(H, votes),
                "external_signals": external(H, pids, sig),
                "design_counts": pd.Series(des).value_counts().to_dict(),
                "design_agreement_with_unseen_text": round(float(
                    (des[has_ref] == np.array(ref_design)[has_ref]).mean()), 4),
                "n_with_unseen_reference": int(has_ref.sum()),
            }
            # the no-evidence rows: which answer agrees best with the unseen text
            m = nofacet & has_ref
            cand = fallback_designs(P[nofacet], H[nofacet], E[nofacet], store)
            ref_nf = np.array(ref_design)[nofacet]
            has_nf = has_ref[nofacet]
            row["no_evidence"] = {
                "n": int(nofacet.sum()), "n_with_unseen_reference": int(m.sum()),
                "reference_design_counts": pd.Series(ref_nf[has_nf]).value_counts().to_dict(),
                "agreement_with_unseen_text": {
                    k: round(float((np.array(v)[has_nf] == ref_nf[has_nf]).mean()), 4)
                    for k, v in cand.items()},
            }
            # full-pipeline design agreement when a fallback replaces the default
            for key, mode in (("similar reference papers", "similar_papers"),
                              ("most probable single facet", "top_facet")):
                des2 = des.copy()
                des2[nofacet] = cand[key]
                row[f"design_agreement_with_unseen_text_fallback_{mode}"] = round(float(
                    (des2[has_ref] == np.array(ref_design)[has_ref]).mean()), 4)
            row["design_agreement_with_unseen_text_with_fallback"] = \
                row["design_agreement_with_unseen_text_fallback_similar_papers"]
            rows[name] = row
        report[split] = rows

        print(f"\n===== {split} ({len(pids):,} papers; {int(has_ref.sum()):,} with an unseen-text design) =====")
        for name, row in rows.items():
            print(f"  [{name}] no facet {row['no_facet_rate']:.3f} | design agrees with unseen text "
                  f"{row['design_agreement_with_unseen_text']:.3f} -> with fallback: similar papers "
                  f"{row['design_agreement_with_unseen_text_fallback_similar_papers']:.3f}, top facet "
                  f"{row['design_agreement_with_unseen_text_fallback_top_facet']:.3f}")
            ne = row["no_evidence"]
            print(f"      no-evidence papers {ne['n']} ({ne['n_with_unseen_reference']} with a reference): "
                  + " | ".join(f"{k} {v:.3f}" for k, v in ne["agreement_with_unseen_text"].items()))
            for f in FACET_KEYS:
                u = row["unseen_region"][f]
                print(f"      {f:14s} prev {row['prevalence'][f]:.3f} (LM {row['label_model_prevalence'][f]:.3f}) "
                      f"F1 {row['f1_vs_label_model'][f]:.3f}  kappa meth {u['methods']['kappa']:.3f} "
                      f"res {u['results_conclusion']['kappa']:.3f}")
            for s_name, e in row["external_signals"].items():
                print(f"      EXT {s_name:16s}->{e['facet']:13s} n={e['n']:3d} "
                      f"{e['p_facet_given_signal']:.3f} vs {e['p_facet_otherwise']:.3f} (lift {e['lift']})")

    # ---- gates, decided on validation
    v = report["val"]
    fb = v["fitted per facet"]["no_evidence"]["agreement_with_unseen_text"]
    # The better of the two paper-reading candidates is used, if it beats the
    # default and the majority answer (which ignore the paper).
    reading = {"similar_papers": fb["similar reference papers"],
               "top_facet": fb["most probable single facet"]}
    best = max(reading, key=reading.get)
    blind = max(a for k, a in fb.items()
                if k not in ("similar reference papers", "most probable single facet"))
    use_fallback = best if reading[best] > blind else None
    k_old = np.mean([v["0.5 everywhere"]["unseen_region"][f][r]["kappa"] for f in FACET_KEYS for r in UNSEEN])
    k_new = np.mean([v["fitted per facet"]["unseen_region"][f][r]["kappa"] for f in FACET_KEYS for r in UNSEEN])
    use_thr = k_new >= k_old
    report["gates"] = {"thresholds_mean_kappa_val": {"0.5": round(float(k_old), 4), "fitted": round(float(k_new), 4)},
                       "use_fitted_thresholds": bool(use_thr),
                       "fallback_val_agreement": fb, "no_evidence_fallback": use_fallback}
    print("\ngates (validation):", json.dumps(report["gates"], indent=1))

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = "" if d.name == "methodology-facets" else f"_{d.name}"
    out = RESULTS / f"facet_decisions_eval{tag}{'_legacy' if args.legacy_cues else ''}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print("saved ->", out)

    if args.deploy:
        cfg = {"thresholds": thr if use_thr else {f: 0.5 for f in FACET_KEYS},
               "fitted_on": "validation split, F1 against the label model's hard labels",
               "no_evidence_fallback": use_fallback,
               "evidence": out.name}
        (d / THRESHOLDS_FILE).write_text(json.dumps(cfg, indent=2))
        print("deployed ->", d / THRESHOLDS_FILE)


if __name__ == "__main__":
    main()
