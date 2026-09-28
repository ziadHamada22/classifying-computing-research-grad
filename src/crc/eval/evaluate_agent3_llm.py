"""Does the proposal's Agent 3 -- a local LLM with retrieved examples -- help?

The proposal specified Agent 3 as "SciBERT + RAG few-shot": retrieve similar
labelled papers, let an LLM name the design with them as examples, and ensemble
that with the fine-tuned model. This measures exactly that against the deployed
facet model, with no gold labels, on evidence none of the systems was built from
where possible.

Systems (all on the same papers):

* ``deployed``  -- methodology-facets-v2, fitted thresholds, flagged fallback;
* ``llm_zero``  -- Qwen2.5-3B, the design rubric and the abstract;
* ``llm_rag``   -- the same plus the 4 most similar reference papers with evidence
  and their designs as prior chat turns (the proposal's RAG few-shot);
* ``*_cc``      -- the LLM's label prior removed first ("contextual calibration",
  Zhao et al. 2021), the prior estimated on 200 validation papers only;
* ``ensemble_*`` -- 0.5 x the deployed design distribution + 0.5 x the LLM's (the
  proposal's ensemble). The weight is fixed in advance, not tuned.

Judged by:

1. **arXiv comment / journal-ref signals** (`data/external_signals.py`) -- read by
   no system: the share of theory-venue papers called Formal, HCI-venue papers
   called a people-based design, and so on, against the same share on a random
   sample (lift). The neutral arbiter, but small.
2. **agreement with the design implied by the unseen methods / results text**, on
   600 random test papers. Caveat stated in the output: that reference is derived
   with the same cue vocabulary the facet model's training labels came from, so it
   favours the facet model; the LLM never saw those cues.
3. design distribution, agreement between systems, latency.
4. a silver standard, if `results/agent3_silver_labels.csv` exists.

LLM outputs are cached per paper (JSONL, resumable).

Run:
    python -m crc.eval.evaluate_agent3_llm              # run what is missing, then analyse
    python -m crc.eval.evaluate_agent3_llm --analyse-only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from crc.agents.methodology.predict import (
    DEPLOYED_LABELS,
    DEPLOYED_MODEL,
    MODELS,
    THRESHOLDS_FILE,
    rule_probability,
)
from crc.agents.methodology.retrieval import STORE_FILE, ReferenceStore
from crc.agents.methodology.train_facets import build_corpus
from crc.data.external_signals import load_signals
from crc.eval.agent3_before_after import system_designs
from crc.eval.compare_facet_models import boot_diff, load_votes
from crc.eval.evaluate_facet_decisions import CACHE as EMB_CACHE
from crc.eval.evaluate_facet_decisions import unseen_reference
from crc.eval.evaluate_joint import mcnemar
from crc.taxonomy.facets import DESIGN_RULES, FACET_KEYS
from crc.taxonomy.methodology import DESIGN_NAMES

PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "results"
CACHE = Path(r"C:\Users\ziada\gp_data\cache\llm_agent3")
SILVER = RESULTS / "agent3_silver_labels.csv"

K_EXAMPLES = 4
N_SAMPLE, N_DEV = 600, 200
ENSEMBLE_W = 0.5
PEOPLE = {"Survey (empirical study of respondents)",
          "Qualitative Field Study / Grounded Theory", "Case Study", "Action Research"}
#: What each external signal says the design should be.
EXPECTED = {
    "theory_venue": {"Formal / Theoretical"},
    "hci_venue": PEOPLE,
    "code_release": {"Design & Creation (Design Science)"},
    "tool_demo": {"Design & Creation (Design Science)"},
    "industry_track": {"Case Study", "Action Research"},
    "survey_venue": {"Systematic Literature Review / Secondary Study"},
    "simulation_venue": {"Simulation & Modelling"},
}


def rule_distribution(P: np.ndarray) -> np.ndarray:
    """The deployed model's design distribution: each design's best rule, normalised."""
    out = np.zeros((len(P), len(DESIGN_NAMES)))
    for i, row in enumerate(P):
        p = dict(zip(FACET_KEYS, row))
        for design, rule in DESIGN_RULES:
            j = DESIGN_NAMES.index(design)
            out[i, j] = max(out[i, j], rule_probability(rule, p))
    s = out.sum(1, keepdims=True)
    return np.where(s > 0, out / np.maximum(s, 1e-12), 1.0 / len(DESIGN_NAMES))


def load_cache(variant: str) -> dict[str, dict]:
    p = CACHE / f"{variant}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["paper_id"]] = r
    return out


def run_llm(variant: str, pids: list[str], texts: dict, emb: dict, store: ReferenceStore) -> None:
    from crc.agents.methodology.llm_design import design_reviewer, retrieved_examples

    CACHE.mkdir(parents=True, exist_ok=True)
    done = load_cache(variant)
    todo = [p for p in pids if p not in done]
    if not todo:
        return
    rv = design_reviewer()
    t0 = time.time()
    with (CACHE / f"{variant}.jsonl").open("a", encoding="utf-8") as fh:
        for n, pid in enumerate(todo, 1):
            ex = (retrieved_examples(store, emb[pid], texts, K_EXAMPLES, exclude=pid)
                  if variant == "rag" else None)
            r = rv.review(texts[pid], examples=ex, max_paper_chars=2000)
            fh.write(json.dumps({"paper_id": pid, "label": r.label, "probs": r.probs,
                                 "n_examples": r.n_examples, "prompt_tokens": r.prompt_tokens,
                                 "latency_ms": r.latency_ms}) + "\n")
            fh.flush()
            if n % 100 == 0:
                print(f"  {variant}: {n}/{len(todo)} ({(time.time() - t0) / n:.2f} s/paper)",
                      flush=True)


def as_matrix(cache: dict, pids: list[str]) -> np.ndarray:
    return np.array([[cache[p]["probs"][d] for d in DESIGN_NAMES] for p in pids])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    d = MODELS / DEPLOYED_MODEL
    cfg = json.loads((d / THRESHOLDS_FILE).read_text())
    store = ReferenceStore.load(d / STORE_FILE)
    texts = build_corpus(DEPLOYED_LABELS).set_index("paper_id")["text"].to_dict()
    sig = load_signals()

    split = {}
    for s in ("val", "test"):
        z = np.load(d / f"{s}_facet_predictions.npz", allow_pickle=True)
        pids = z["paper_ids"].astype(str)
        E = np.load(EMB_CACHE / f"{DEPLOYED_MODEL}_{s}.npy")
        assert len(E) == len(pids)
        split[s] = (pids, z["probs"].astype(float), E)

    rng = np.random.default_rng(1)
    sample = sorted(rng.choice(split["test"][0], N_SAMPLE, replace=False).tolist())
    dev = sorted(np.random.default_rng(0).choice(split["val"][0], N_DEV, replace=False).tolist())
    vt_ids = np.concatenate([split["val"][0], split["test"][0]])
    s_vt = sig.reindex(vt_ids).fillna(False)
    signal_ids = sorted(vt_ids[s_vt[list(EXPECTED)].to_numpy().any(1)].tolist())
    everyone = sorted(set(sample) | set(dev) | set(signal_ids))
    emb = {p: e for s in split.values() for p, e in zip(s[0], s[2])}
    probs = {p: row for s in split.values() for p, row in zip(s[0], s[1])}
    print(f"papers: sample {len(sample)} (test), dev {len(dev)} (val), "
          f"signal papers {len(signal_ids)} (val+test); {len(everyone)} distinct")

    if not args.analyse_only:
        for variant in ("zero", "rag"):
            run_llm(variant, everyone, texts, emb, store)
    caches = {v: load_cache(v) for v in ("zero", "rag")}

    # ---- every system's design distribution and hard answer, per paper
    def systems_for(pids: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        P = np.array([probs[p] for p in pids])
        dep_hard, _ = system_designs(P, cfg["thresholds"], cfg["no_evidence_fallback"])
        dep_dist = rule_distribution(P)
        out = {"deployed": (dep_dist, np.array(dep_hard, dtype=object))}
        for v in ("zero", "rag"):
            L = as_matrix(caches[v], pids)
            out[f"llm_{v}"] = (L, np.array(DESIGN_NAMES, dtype=object)[L.argmax(1)])
            logp = np.log(np.clip(as_matrix(caches[v], dev), 1e-12, 1)).mean(0)
            C = np.log(np.clip(L, 1e-12, 1)) - logp
            C = np.exp(C - C.max(1, keepdims=True))
            C /= C.sum(1, keepdims=True)
            out[f"llm_{v}_cc"] = (C, np.array(DESIGN_NAMES, dtype=object)[C.argmax(1)])
            for tag, M in ((f"ensemble_{v}", L), (f"ensemble_{v}_cc", C)):
                B = (1 - ENSEMBLE_W) * dep_dist + ENSEMBLE_W * M
                out[tag] = (B, np.array(DESIGN_NAMES, dtype=object)[B.argmax(1)])
        return out

    report: dict = {"model": DEPLOYED_MODEL, "llm": "qwen2.5-3b-instruct-q4_k_m",
                    "k_examples": K_EXAMPLES, "ensemble_weight": ENSEMBLE_W,
                    "n_sample": len(sample), "n_dev": len(dev), "n_signal_papers": len(signal_ids)}

    # ---- 1: external signals (lift over the random sample)
    S = systems_for(sample)
    G = systems_for(signal_ids)
    s_sig = sig.reindex(signal_ids).fillna(False)
    ext = {}
    for name, (_, hard_s) in S.items():
        ext[name] = {}
        for s, exp in EXPECTED.items():
            y = s_sig[s].to_numpy().astype(bool)
            if y.sum() == 0:
                continue
            hit = float(np.isin(G[name][1][y], list(exp)).mean())
            base = float(np.isin(hard_s, list(exp)).mean())
            ext[name][s] = {"n": int(y.sum()), "share_expected": round(hit, 4),
                            "base_rate": round(base, 4),
                            "lift": round(hit / base, 2) if base else None}
    report["external_signals"] = ext

    # ---- 2: unseen methods/results text, on the random test sample
    tp = list(split["test"][0])
    votes = load_votes("test", "v2", split["test"][0])
    _, ref, has_ref = unseen_reference(votes)
    pos = [tp.index(p) for p in sample]
    ref_s, has_s = np.array(ref, dtype=object)[pos], has_ref[pos]
    agree = {name: (hard[has_s] == ref_s[has_s]).astype(float) for name, (_, hard) in S.items()}
    report["unseen_text"] = {
        "caveat": "reference derived with the facet labels' own cue vocabulary; favours `deployed`",
        "n": int(has_s.sum()),
        "agreement": {k: round(float(v.mean()), 4) for k, v in agree.items()},
        "vs_deployed": {k: {"diff": round(float(v.mean() - agree["deployed"].mean()), 4),
                            "ci95": boot_diff(v, agree["deployed"]),
                            "mcnemar_p": mcnemar(agree["deployed"].astype(bool),
                                                 v.astype(bool))["p_value"]}
                        for k, v in agree.items() if k != "deployed"},
    }

    # ---- 3: distributions, inter-system agreement, retrieval use, latency
    report["design_distribution"] = {
        name: pd.Series(hard).value_counts(normalize=True).round(4).to_dict()
        for name, (_, hard) in S.items()}
    report["kappa_with_deployed"] = {
        name: round(float(cohen_kappa_score(S["deployed"][1], hard)), 4)
        for name, (_, hard) in S.items() if name != "deployed"}
    report["llm_runtime"] = {
        v: {"mean_latency_ms": round(float(np.mean([caches[v][p]["latency_ms"] for p in everyone])), 1),
            "mean_prompt_tokens": round(float(np.mean([caches[v][p]["prompt_tokens"] for p in everyone])), 1),
            "mean_examples": round(float(np.mean([caches[v][p]["n_examples"] for p in everyone])), 2)}
        for v in caches}
    # how often the RAG answer simply copies the nearest example's design
    rag_copy = []
    for p in sample:
        ex = caches["rag"][p]
        if ex["n_examples"]:
            idx, _ = store.search(emb[p], K_EXAMPLES * 4)
            nearest = next((store.designs[i] for i in idx
                            if store.has_evidence[i] and store.paper_ids[i] != p), None)
            rag_copy.append(ex["label"] == nearest)
    report["rag_answer_equals_nearest_example"] = round(float(np.mean(rag_copy)), 4)

    # ---- 4: silver standard, when present
    if SILVER.exists():
        sv = pd.read_csv(SILVER, dtype=str).dropna(subset=["design"]).set_index("paper_id")
        ids = [p for p in sample if p in sv.index]
        if ids:
            T = systems_for(ids)
            y = sv.loc[ids, "design"].to_numpy()
            report["silver"] = {"n": len(ids), "accuracy": {
                name: round(float((hard == y).mean()), 4) for name, (_, hard) in T.items()}}

    out = RESULTS / "agent3_llm_eval.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    print(f"\nunseen-text agreement on {report['unseen_text']['n']} sampled test papers "
          f"(favours `deployed`, see caveat):")
    for k, v in report["unseen_text"]["agreement"].items():
        extra = report["unseen_text"]["vs_deployed"].get(k)
        print(f"  {k:20s} {v:.3f}" + (f"   diff {extra['diff']:+.3f} CI {extra['ci95']} "
                                       f"McNemar p={extra['mcnemar_p']:.3g}" if extra else ""))
    print("\nexternal signals: share given the expected design (lift over the random sample)")
    names = list(S)
    print("  " + " " * 16 + "".join(f"{n[:14]:>15s}" for n in names))
    for s in EXPECTED:
        if s in ext["deployed"]:
            row = ext["deployed"][s]
            print(f"  {s:14s} n={row['n']:<3d}" + "".join(
                f"{ext[n][s]['share_expected']:>8.2f} ({ext[n][s]['lift'] or 0:>4.1f})" for n in names))
    print("\nkappa with deployed:", report["kappa_with_deployed"])
    print("RAG answer = nearest example's design:", report["rag_answer_equals_nearest_example"])
    print("runtime:", report["llm_runtime"])
    if "silver" in report:
        print("silver accuracy:", report["silver"])
    print("saved ->", out)


if __name__ == "__main__":
    main()
