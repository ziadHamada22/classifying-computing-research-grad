"""Does a local LLM second opinion help on the papers Agent 1 calls contested?

The design reserved one place for a language model: a *second opinion on
contested documents only*, local, offline, never in the path of confident
answers. The conformal layer now defines "contested" precisely -- the deployed
alpha = 0.20 prediction set holds more (or fewer) than one discipline -- so this
measures exactly the intervention the design describes.

Setup, chosen so nothing reported is tuned on itself:

* **Reviewer.** Qwen2.5-3B-Instruct (Q4_K_M GGUF) via llama-cpp on the same 6 GB
  GPU, scoring the six discipline names by log-probability (``llm_review.py``).
* **Variants.** ``zero`` -- the rubric and the paper; ``rag`` -- the same plus the
  ``k`` most similar training papers and their labels as prior chat turns,
  retrieved by SPECTER2 from Agent 1's training split (``retrieval.py``). The
  second is the proposal's "retrieval-augmented few-shot" family.
* **Decision rules.** Blend ``(1 - w) * p_agent1 + w * p_llm``, optionally
  restricted to the conformal set (the reviewer arbitrates among the live
  candidates rather than re-opening the whole question), and optionally with
  the LLM's label prior removed first ("contextual calibration", Zhao et al.
  2021): label scoring is biased toward generic answers -- the 3B model says
  "Computer Science" for most papers -- so its mean log-probability per label,
  estimated on the development set only, is subtracted before the softmax.
* **Development set.** The contested papers in the 2025+ *calibration* half
  (the half the conformal bank was fitted on). Variant, ``w`` and restriction are
  all chosen there, by accuracy.
* **Reported sets.** The contested papers in the 2025+ *evaluation* half, and all
  contested papers of the 2015-2024 test split. Neither influenced any choice.

LLM outputs are cached per paper (JSONL, resumable), so the analysis re-runs in
seconds and an interrupted run continues where it stopped.

Run:
    python -m crc.eval.evaluate_llm_review --sets dev --variants zero,rag
    python -m crc.eval.evaluate_llm_review --sets temporal_eval,test --variants zero,rag
    python -m crc.eval.evaluate_llm_review --analyse-only
"""
from __future__ import annotations

from crc.agents.discipline import DEPLOYED_MODEL

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from crc.agents.discipline.conformal import load_bank, pick_alpha
from crc.agents.discipline.hierarchy import softmax
from crc.eval.evaluate_joint import mcnemar
from crc.taxonomy import DISCIPLINES, LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
CACHE = WORK / "cache" / "llm"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"

W_GRID = [round(w, 2) for w in np.arange(0.0, 1.0001, 0.1)]


# ------------------------------------------------------------------ data
def load_sets(model_dir: Path, alpha: float, seed: int) -> dict:
    cal = pick_alpha(load_bank(model_dir / "conformal.json"), alpha)
    corpus = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet")
    frames = {"test": corpus[corpus["split"] == "test"].reset_index(drop=True),
              "temporal": pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet")
              .reset_index(drop=True)}
    out = {}
    for split, df in frames.items():
        z = np.load(model_dir / f"{split}_predictions.npz")
        y = df["discipline"].map(LABEL2ID).to_numpy()
        assert np.array_equal(z["labels"], y), f"{split} predictions out of order"
        p = softmax(z["logits"])
        s = cal.predict_set(p)
        contested = s.sum(axis=1) != 1
        out[split] = {"df": df, "p": p, "y": y, "set": s, "contested": contested}
    idx = np.random.default_rng(seed).permutation(len(frames["temporal"]))
    half = len(idx) // 2
    calib = np.zeros(len(idx), bool)
    calib[idx[:half]] = True
    t = out["temporal"]
    return {
        "full": out,
        "dev": ("temporal", np.where(t["contested"] & calib)[0]),
        "temporal_eval": ("temporal", np.where(t["contested"] & ~calib)[0]),
        "test": ("test", np.where(out["test"]["contested"])[0]),
    }


def paper_text(row) -> str:
    return f"{row['title']}. {row['abstract']}"


# ------------------------------------------------------------------ running the LLM
def cache_path(variant: str) -> Path:
    return CACHE / f"qwen3b_{variant}.jsonl"


def read_cache(variant: str) -> dict[str, dict]:
    p = cache_path(variant)
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # a run stopped mid-write leaves one truncated line; that paper
                # is simply reviewed again on resume
                continue
            out[r["key"]] = r
    return out


def run(sets: dict, names: list[str], variants: list[str], k: int) -> None:
    from crc.agents.discipline.llm_review import LocalLLMReviewer
    from crc.agents.discipline.retrieval import (
        RetrievalStore,
        Specter2Encoder,
        cached_embeddings,
    )

    CACHE.mkdir(parents=True, exist_ok=True)
    reviewer = LocalLLMReviewer()
    store = RetrievalStore.load_train() if "rag" in variants else None
    q_emb = {}
    if store is not None:
        for split in {sets[n][0] for n in names}:
            df = sets["full"][split]["df"]
            q_emb[split] = cached_embeddings(split, df, Specter2Encoder)

    for variant in variants:
        done = read_cache(variant)
        with cache_path(variant).open("a", encoding="utf-8") as fh:
            for name in names:
                split, rows = sets[name]
                df = sets["full"][split]["df"]
                todo = [i for i in rows if f"{split}:{df.at[i, 'id']}" not in done]
                print(f"[{variant}] {name}: {len(rows):,} contested, {len(todo):,} to run")
                t0 = time.time()
                for n, i in enumerate(todo, 1):
                    row = df.iloc[i]
                    ex = store.examples(q_emb[split][i], k=k) if variant == "rag" else None
                    rv = reviewer.review(paper_text(row), examples=ex)
                    fh.write(json.dumps({
                        "key": f"{split}:{row['id']}", "probs": rv.probs,
                        "prompt_tokens": rv.prompt_tokens, "latency_ms": rv.latency_ms,
                        "n_examples": rv.n_examples,
                        "example_labels": [lab for _, lab in (ex or [])]}) + "\n")
                    if n % 100 == 0:
                        fh.flush()
                        rate = n / (time.time() - t0)
                        print(f"    {n:,}/{len(todo):,}  {rate:.2f} papers/s", flush=True)


# ------------------------------------------------------------------ analysis
def decide(p1: np.ndarray, pl: np.ndarray, s: np.ndarray, w: float,
           restrict: bool) -> np.ndarray:
    blend = (1 - w) * p1 + w * pl
    if restrict:
        # arbitrate only among the live candidates; an empty set means the
        # conformal layer ruled nothing in, so fall back to the full blend
        masked = np.where(s, blend, -1.0)
        return np.where(s.any(axis=1), masked.argmax(1), blend.argmax(1))
    return blend.argmax(1)


def prior_correct(pl: np.ndarray, prior_logp: np.ndarray | None) -> np.ndarray:
    """Remove the LLM's per-label prior (contextual calibration)."""
    if prior_logp is None:
        return pl
    adj = np.log(pl + 1e-12) - prior_logp[None, :]
    out = np.exp(adj - adj.max(axis=1, keepdims=True))
    return out / out.sum(axis=1, keepdims=True)


def analyse(sets: dict, variants: list[str]) -> dict:
    caches = {v: read_cache(v) for v in variants}
    report: dict = {"sets": {}, "selection": None}

    def arrays(name, variant):
        split, rows = sets[name]
        f = sets["full"][split]
        keys = [f"{split}:{f['df'].at[i, 'id']}" for i in rows]
        have = [j for j, kk in enumerate(keys) if kk in caches[variant]]
        if not have:
            return None
        rows = rows[have]
        pl = np.array([[caches[variant][keys[j]]["probs"][d] for d in DISCIPLINES]
                       for j in have])
        lat = [caches[variant][keys[j]]["latency_ms"] for j in have]
        return split, rows, f["p"][rows], pl, f["set"][rows], f["y"][rows], lat

    # ---- label priors, estimated on the development set only ---------------
    priors = {}
    for v in variants:
        a = arrays("dev", v)
        if a is not None:
            priors[v] = np.log(a[3] + 1e-12).mean(axis=0)
    report["llm_label_prior_dev"] = {v: dict(zip(DISCIPLINES, np.round(p, 3).tolist()))
                                     for v, p in priors.items()}

    # ---- choose variant, correction, weight, restriction on development ----
    best = None
    dev_grid = {}
    for v in variants:
        a = arrays("dev", v)
        if a is None:
            continue
        _, _, p1, pl_raw, s, y, _ = a
        for corrected in (False, True):
            pl = prior_correct(pl_raw, priors[v] if corrected else None)
            for restrict in (False, True):
                for w in W_GRID:
                    acc = float((decide(p1, pl, s, w, restrict) == y).mean())
                    dev_grid[f"{v}|corrected={corrected}|w={w}|restrict={restrict}"] = round(acc, 4)
                    key = (acc, -w)          # prefer the smaller weight on ties
                    if best is None or key > best[0]:
                        best = (key, v, w, restrict, corrected)
    if best is None:
        raise SystemExit("no development-set LLM outputs cached; run --sets dev first")
    _, v_star, w_star, r_star, c_star = best
    report["selection"] = {"variant": v_star, "weight": w_star, "restrict_to_set": r_star,
                           "prior_corrected": c_star, "dev_accuracy": best[0][0],
                           "chosen_on": "2025+ calibration half",
                           "note": ("weight 0 means the LLM changes nothing: no rule "
                                    "beat Agent 1 on the development set")}
    report["dev_grid"] = dev_grid
    print(f"selected on dev: variant={v_star}  corrected={c_star}  w={w_star}  "
          f"restrict={r_star}  (dev acc {best[0][0]:.4f})")

    # ---- report every set, every variant ------------------------------------
    for name in ("dev", "temporal_eval", "test"):
        entry = {}
        for v in variants:
            a = arrays(name, v)
            if a is None:
                continue
            split, rows, p1, pl, s, y, lat = a
            plc = prior_correct(pl, priors.get(v))
            a1_hit = p1.argmax(1) == y
            e = {
                "n_contested": int(len(rows)),
                "agent1_accuracy": round(float(a1_hit.mean()), 4),
                "llm_alone_accuracy": round(float((pl.argmax(1) == y).mean()), 4),
                "llm_alone_prior_corrected_accuracy": round(float(
                    (plc.argmax(1) == y).mean()), 4),
                "llm_restricted_to_set_accuracy": round(float(
                    (decide(p1, pl, s, 1.0, True) == y).mean()), 4),
                "llm_share_answering_cs": round(float((pl.argmax(1) == 0).mean()), 4),
                "true_label_in_set": round(float(s[np.arange(len(y)), y].mean()), 4),
                "latency_ms_mean": round(float(np.mean(lat)), 1),
            }
            if v == v_star:
                dec = decide(p1, plc if c_star else pl, s, w_star, r_star)
                e["selected_rule_accuracy"] = round(float((dec == y).mean()), 4)
                e["mcnemar_vs_agent1"] = mcnemar(a1_hit, dec == y)
                # whole-split effect: contested papers re-decided, the rest unchanged
                f = sets["full"][split]
                if name != "dev":
                    base = f["p"].argmax(1)
                    new = base.copy()
                    new[rows] = dec
                    scope = np.ones(len(base), bool)
                    if split == "temporal":           # only the evaluation half
                        idx = np.random.default_rng(42).permutation(len(base))
                        scope[:] = False
                        scope[idx[len(base) // 2:]] = True
                    yy = f["y"][scope]
                    e["whole_split"] = {
                        "n": int(scope.sum()),
                        "agent1_accuracy": round(float((base[scope] == yy).mean()), 4),
                        "with_review_accuracy": round(float((new[scope] == yy).mean()), 4),
                        "agent1_macro_f1": round(float(f1_score(yy, base[scope], average="macro")), 4),
                        "with_review_macro_f1": round(float(f1_score(yy, new[scope], average="macro")), 4),
                        "cs_recall_agent1": round(float((base[scope] == yy)[yy == 0].mean()), 4),
                        "cs_recall_with_review": round(float((new[scope] == yy)[yy == 0].mean()), 4),
                        "reviewed_share": round(float(np.isin(np.where(scope)[0], rows).mean()), 4),
                    }
            entry[v] = e
        report["sets"][name] = entry
        for v, e in entry.items():
            line = (f"{name:14s} {v:5s} n={e['n_contested']:5,}  A1 {e['agent1_accuracy']:.4f}  "
                    f"LLM {e['llm_alone_accuracy']:.4f} (corrected "
                    f"{e['llm_alone_prior_corrected_accuracy']:.4f})  LLM-in-set "
                    f"{e['llm_restricted_to_set_accuracy']:.4f}")
            if "selected_rule_accuracy" in e:
                line += f"  RULE {e['selected_rule_accuracy']:.4f} {e['mcnemar_vs_agent1']}"
            print(line + f"  {e['latency_ms_mean']:.0f} ms")
            if "whole_split" in e:
                ws = e["whole_split"]
                print(f"{'':20s} whole split: acc {ws['agent1_accuracy']:.4f} -> "
                      f"{ws['with_review_accuracy']:.4f}, macro-F1 {ws['agent1_macro_f1']:.4f} -> "
                      f"{ws['with_review_macro_f1']:.4f}, CS recall {ws['cs_recall_agent1']:.4f} -> "
                      f"{ws['cs_recall_with_review']:.4f} ({ws['reviewed_share']:.1%} reviewed)")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODELS / DEPLOYED_MODEL))
    ap.add_argument("--alpha", type=float, default=0.20,
                    help="The deployed 'contested' flag (predict.py's default).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sets", default="dev", help="Comma list: dev,temporal_eval,test")
    ap.add_argument("--variants", default="zero,rag")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "llm_review_eval.json"))
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    sets = load_sets(Path(args.model_dir), args.alpha, args.seed)
    if not args.analyse_only:
        run(sets, [s.strip() for s in args.sets.split(",")], variants, args.k)
    report = analyse(sets, variants)
    report.update({"reviewer": "Qwen2.5-3B-Instruct Q4_K_M (llama-cpp, GPU)",
                   "contested_definition": f"deployed conformal set at alpha={args.alpha} "
                                           f"has size != 1", "k_examples": args.k})
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
