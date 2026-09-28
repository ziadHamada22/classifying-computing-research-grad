"""Report figures for Agent 1: confusion matrices, reliability diagrams, baselines.

The proposal promised confusion matrices and reliability diagrams alongside the
macro-F1 and ECE numbers; v2 had only the numbers. Everything here is drawn from
saved predictions and result files -- no GPU, no re-inference.

  confusion_<tag>_<split>.png     row-normalised (each row = a true discipline, so
                                  the diagonal is per-class recall); counts shown
  reliability_<tag>.png           test vs 2025+ on one plot, with ECE; bars show
                                  how many papers fall in each confidence bin
  baselines.png                   test vs 2025+ macro-F1 for every single model

Palette: the validated reference palette (blue #2a78d6 = test, orange #eb6834 =
2025+; a single-hue blue ramp for magnitudes). Text stays in neutral ink.

Run:
    python -m crc.eval.figures                  # SciBERT + every baseline
    python -m crc.eval.figures --tag scibert
"""
from __future__ import annotations

from crc.agents.discipline import DEPLOYED_MODEL

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from crc.agents.discipline.calibrate import expected_calibration_error  # noqa: E402
from crc.agents.discipline.hierarchy import softmax  # noqa: E402
from crc.taxonomy import DISCIPLINES  # noqa: E402

PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"
FIG = RESULTS / "figures"

SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e4e3df"
TEST_C, TEMPORAL_C = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list(
    "blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"])
SHORT = {"Computer Science": "CS", "Information Systems": "IS",
         "Information Technology": "IT", "Software Engineering": "SE",
         "Computer Engineering": "CE", "Data Science": "DS"}
SPLIT_NAMES = {"test": "Test (2015–2024)", "temporal": "2025+ holdout"}

#: Display names for the baseline chart, in the report's vocabulary.
MODEL_NAMES = {
    "scibert": "SciBERT", "specter2": "SPECTER2", "deberta-v3-base": "DeBERTa-v3",
    "modernbert-base": "ModernBERT", "bert-base-uncased": "BERT (vanilla)",
    "tfidf": "TF-IDF + LogReg", "tfidf-svm": "TF-IDF + SVM",
    "tfidf-xgboost": "TF-IDF + XGBoost", "specter2-knn": "SPECTER2 kNN",
    "specter2-principal": "Principal vectors", "scibert-soft": "SciBERT + soft labels",
    "tfidf-rf": "TF-IDF + Random Forest", "tfidf-nb": "TF-IDF + Naive Bayes",
    "tfidf-knn": "TF-IDF kNN",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _load(tag: str, split: str):
    z = np.load(MODELS / tag / f"{split}_predictions.npz")
    return softmax(z["logits"]), z["labels"]


def confusion(tag: str, split: str) -> Path:
    probs, y = _load(tag, split)
    pred = probs.argmax(1)
    k = len(DISCIPLINES)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    rn = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.6, 4.8), facecolor=SURFACE)
    ax.imshow(rn, cmap=SEQ, vmin=0, vmax=1)
    labels = [SHORT[d] for d in DISCIPLINES]
    ax.set_xticks(range(k), labels)
    ax.set_yticks(range(k), labels)
    ax.set_xlabel("Predicted discipline")
    ax.set_ylabel("True discipline")
    for i in range(k):
        for j in range(k):
            dark = rn[i, j] > 0.55
            ax.text(j, i, f"{rn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=7.5, color="#ffffff" if dark else INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    acc = (pred == y).mean()
    ax.set_title(f"{MODEL_NAMES.get(tag, tag)} — {SPLIT_NAMES[split]}\n"
                 f"row-normalised; diagonal = recall; accuracy {acc:.3f}",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    out = FIG / f"confusion_{tag}_{split}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def _reliability(probs, y, n_bins=15, min_papers=30):
    """Per-bin (confidence, accuracy, count) with the SAME bins as the reported ECE.

    Bins holding fewer than ``min_papers`` papers are dropped from the curve --
    a 12-paper bin's accuracy is noise, not calibration -- but still count in
    the ECE, which is computed by the system's own function so the label on the
    figure is the number in the tables.
    """
    conf = probs.max(1)
    hit = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    xs, accs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() < min_papers:
            continue
        xs.append(conf[m].mean())
        accs.append(hit[m].mean())
        ns.append(m.sum())
    return (np.array(xs), np.array(accs), np.array(ns),
            expected_calibration_error(probs, y, n_bins=n_bins))


def reliability(tag: str) -> Path:
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.4, 5.6), facecolor=SURFACE,
                                  gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    _style(ax)
    _style(axh)
    ax.plot([0, 1], [0, 1], color=INK_3, lw=1, ls=(0, (4, 3)), label="Perfect calibration")
    width = (1 / 15) / 2.4
    for off, split, color in ((-1, "test", TEST_C), (1, "temporal", TEMPORAL_C)):
        f = MODELS / tag / f"{split}_predictions.npz"
        if not f.exists():
            continue
        probs, y = _load(tag, split)
        xs, accs, ns, ece = _reliability(probs, y)
        ax.plot(xs, accs, color=color, lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"{SPLIT_NAMES[split]} (ECE {ece:.3f})")
        axh.bar(xs + off * width / 1.6, ns, width=width, color=color,
                edgecolor=SURFACE, linewidth=1)
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(0.15, 1.0)
    ax.set_ylabel("Accuracy in bin")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    ax.set_title(f"{MODEL_NAMES.get(tag, tag)} — reliability", fontsize=10,
                 color=INK, loc="left")
    axh.set_ylabel("Papers")
    axh.set_xlabel("Predicted confidence (top class)")
    axh.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    out = FIG / f"reliability_{tag}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def reliability_compare(tags: list[str], split: str = "temporal") -> Path:
    """Two models' reliability on one split -- e.g. before/after soft labels."""
    colors = [TEST_C, TEMPORAL_C]
    fig, ax = plt.subplots(figsize=(5.4, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=INK_3, lw=1, ls=(0, (4, 3)), label="Perfect calibration")
    for tag, color in zip(tags, colors):
        probs, y = _load(tag, split)
        xs, accs, _, ece = _reliability(probs, y)
        ax.plot(xs, accs, color=color, lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"{MODEL_NAMES.get(tag, tag)} (ECE {ece:.3f})")
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(0.15, 1.0)
    ax.set_xlabel("Predicted confidence (top class)")
    ax.set_ylabel("Accuracy in bin")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    ax.set_title(f"Calibration on the {SPLIT_NAMES[split]}", fontsize=10,
                 color=INK, loc="left")
    fig.tight_layout()
    out = FIG / f"reliability_compare_{'_vs_'.join(tags)}_{split}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def baselines() -> Path:
    rows = []
    for f in sorted(RESULTS.glob("eval_*.json")):
        if "documents" in f.name:
            continue
        d = json.loads(f.read_text())
        tag = d.get("tag", f.stem[5:])
        if tag not in MODEL_NAMES:
            continue
        s = d.get("splits", {})
        if "test" in s and "temporal_test" in s:
            rows.append((MODEL_NAMES[tag], s["test"]["strict_macro_f1"],
                         s["temporal_test"]["strict_macro_f1"], tag))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.2), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(rows))
    h = 0.36
    ax.barh(y + h / 2 + 0.01, [r[1] for r in rows], height=h, color=TEST_C,
            edgecolor=SURFACE, linewidth=2, label=SPLIT_NAMES["test"])
    ax.barh(y - h / 2 - 0.01, [r[2] for r in rows], height=h, color=TEMPORAL_C,
            edgecolor=SURFACE, linewidth=2, label=SPLIT_NAMES["temporal"])
    ax.set_yticks(y, [r[0] for r in rows])
    for i, r in enumerate(rows):          # label only the test value, at the bar end
        ax.text(r[1] + 0.003, i + h / 2, f"{r[1]:.3f}", va="center", fontsize=7.5,
                color=INK_2)
    ax.set_xlim(0.6, 0.86)
    ax.set_xlabel("Macro-F1 (6 disciplines)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    # Legend above the plot: inside it would sit on a bar-end label.
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=2,
              loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.set_title("Agent 1 — every model on the same test papers", fontsize=10,
                 color=INK, loc="left", pad=22)
    fig.tight_layout()
    out = FIG / "baselines.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def _agent2_sets() -> dict:
    """Agent 2 (field-scibert-v2) under oracle conditioning: test, and the 2025+
    evaluation half the conformal bank was NOT fitted on (same split as
    `evaluate_field_uncertainty.py`)."""
    import pandas as pd

    from crc.data.label_fields import label_frame
    from crc.eval.evaluate_field_uncertainty import (
        CACHE as FIELD_CACHE, CORPUS_DIR, GLOBAL_LABEL2ID, ballot_matrix)

    fdir = MODELS / "field-scibert-v2"
    z = np.load(fdir / "test_predictions.npz", allow_pickle=True)
    out = {"test": (ballot_matrix(z["logits"], z["disciplines"]), z["labels"],
                    np.asarray(z["disciplines"]))}
    t = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet").reset_index(drop=True)
    lab = label_frame(t)
    ok = lab["field"].map(lambda f: f in GLOBAL_LABEL2ID).to_numpy() & lab["field"].notna().to_numpy()
    idx = np.random.default_rng(42).permutation(len(t))
    evalh = np.ones(len(t), bool)
    evalh[idx[: len(t) // 2]] = False
    m = ok & evalh
    Pt = ballot_matrix(np.load(FIELD_CACHE / f"a2_{fdir.name}_temporal.npy"),
                       t["discipline"].to_numpy())
    yt = lab.loc[m, "field"].map(GLOBAL_LABEL2ID).astype(int).to_numpy()
    out["temporal"] = (Pt[m], yt, t["discipline"].to_numpy()[m])
    return out


def _short_field(name: str) -> str:
    s = name.split(" (")[0]
    return s if len(s) <= 30 else s[:29] + "…"


def agent2_confusion() -> Path:
    """One row-normalised confusion matrix per discipline (fields within it)."""
    from crc.taxonomy.fields import DISCIPLINE_FIELD_IDS, GLOBAL_ID2LABEL

    P, y, disc = _agent2_sets()["test"]
    pred = P.argmax(1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6), facecolor=SURFACE)
    for ax, d in zip(axes.ravel(), DISCIPLINES):
        ids = list(DISCIPLINE_FIELD_IDS[d])
        m = disc == d
        k = len(ids)
        pos = {f: i for i, f in enumerate(ids)}
        cm = np.zeros((k, k), dtype=int)
        for t_, p_ in zip(y[m], pred[m]):
            if t_ in pos and p_ in pos:
                cm[pos[t_], pos[p_]] += 1
        rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(rn, cmap=SEQ, vmin=0, vmax=1)
        names = [_short_field(GLOBAL_ID2LABEL[f]) for f in ids]
        ax.set_yticks(range(k), names, fontsize=7.5)
        ax.set_xticks(range(k), [str(i + 1) for i in range(k)], fontsize=7.5)
        for i in range(k):
            for j in range(k):
                if cm[i, j]:
                    ax.text(j, i, f"{rn[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if rn[i, j] > 0.55 else INK_2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK_2, length=0)
        acc = (pred[m] == y[m]).mean()
        ax.set_title(f"{d} — accuracy {acc:.3f} (n = {int(m.sum()):,})", fontsize=9.5,
                     color=INK, loc="left")
        ax.set_xlabel("predicted field (same order as the rows)", fontsize=8, color=INK_2)
    fig.suptitle("Agent 2 — field confusion within each discipline (test, true discipline "
                 "given; row-normalised, diagonal = recall)", fontsize=11, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    out = FIG / "agent2_confusion_test.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return out


def agent2_reliability() -> Path:
    sets = _agent2_sets()
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.4, 5.6), facecolor=SURFACE,
                                  gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    _style(ax)
    _style(axh)
    ax.plot([0, 1], [0, 1], color=INK_3, lw=1, ls=(0, (4, 3)), label="Perfect calibration")
    width = (1 / 15) / 2.4
    for off, split, color in ((-1, "test", TEST_C), (1, "temporal", TEMPORAL_C)):
        P, y, _ = sets[split]
        xs, accs, ns, ece = _reliability(P, y)
        label = SPLIT_NAMES[split] if split == "test" else "2025+ (evaluation half)"
        ax.plot(xs, accs, color=color, lw=2, marker="o", ms=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=f"{label} (ECE {ece:.3f})")
        axh.bar(xs + off * width / 1.6, ns, width=width, color=color, edgecolor=SURFACE,
                linewidth=1)
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(0.15, 1.0)
    ax.set_ylabel("Accuracy in bin")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    ax.set_title("Agent 2 (field-scibert-v2) — reliability, true discipline given",
                 fontsize=10, color=INK, loc="left")
    axh.set_ylabel("Papers")
    axh.set_xlabel("Predicted confidence (top field)")
    axh.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    out = FIG / "agent2_reliability.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def agent3_designs() -> Path:
    """Derived designs, August vs now, with the silent-default share marked."""
    r = json.loads((RESULTS / "agent3_before_after.json").read_text())["judged_with_v2_cues"]
    old, new = [k for k in r if isinstance(r[k], dict) and "design_counts" in r[k]]
    n = sum(r[old]["design_counts"].values())
    order = sorted(r[old]["design_counts"], key=lambda d: -r[old]["design_counts"][d])
    short = {d: d.split(" (")[0].split(" /")[0] for d in order}
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(order))[::-1]
    h = 0.36
    a = np.array([r[old]["design_counts"].get(d, 0) / n for d in order])
    b = np.array([r[new]["design_counts"].get(d, 0) / n for d in order])
    ax.barh(y + h / 2 + 0.01, a, height=h, color=TEMPORAL_C, edgecolor=SURFACE,
            linewidth=2, label="August (0.5 everywhere, silent default)")
    dc = order.index("Design & Creation (Design Science)")
    silent = r[old]["no_evidence_rate"]
    ax.barh(y[dc] + h / 2 + 0.01, silent, height=h, color="none", edgecolor=INK,
            hatch="////", linewidth=0.8, label="…of which no evidence at all")
    ax.barh(y - h / 2 - 0.01, b, height=h, color=TEST_C, edgecolor=SURFACE,
            linewidth=2, label="Now (v2 model, fitted thresholds, flagged fallback)")
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(vb + 0.004, y[i] - h / 2, f"{vb:.1%}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, [short[d] for d in order])
    ax.set_xlabel("Share of 3,045 test papers")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.8, labelcolor=INK_2, loc="lower right")
    ax.set_title("Agent 3 — derived research designs, before and after", fontsize=10,
                 color=INK, loc="left")
    fig.tight_layout()
    out = FIG / "agent3_designs.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def agent3_fallback() -> Path:
    """No-evidence papers: agreement of each candidate answer with the unseen text."""
    r = json.loads((RESULTS / "facet_decisions_eval_methodology-facets-v2.json").read_text())
    names = {"default (Design & Creation)": "Default: Design & Creation (August)",
             "most probable single facet": "Most probable facet (deployed)",
             "similar reference papers": "Similar papers (retrieval)"}
    rows = []
    for split in ("val", "test"):
        ag = r[split]["fitted per facet"]["no_evidence"]["agreement_with_unseen_text"]
        maj = next(k for k in ag if k.startswith("majority"))
        rows.append({**{names[k]: v for k, v in ag.items() if k in names},
                     "Always the commonest design": ag[maj]})
    labels = ["Default: Design & Creation (August)", "Always the commonest design",
              "Similar papers (retrieval)", "Most probable facet (deployed)"]
    fig, ax = plt.subplots(figsize=(7.0, 2.9), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(labels))[::-1]
    h = 0.36
    for j, (split, col) in enumerate((("validation", TEMPORAL_C), ("test", TEST_C))):
        vals = [rows[j][lab] for lab in labels]
        off = h / 2 + 0.01 if j == 0 else -h / 2 - 0.01
        ax.barh(y + off, vals, height=h, color=col, edgecolor=SURFACE, linewidth=2,
                label=split)
        for i, v in enumerate(vals):
            ax.text(v + 0.004, y[i] + off, f"{v:.3f}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.45)
    ax.set_xlabel("Agreement with the paper's own methods/results text")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=2,
              loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.set_title("Agent 3 — no-facet papers: which answer fits best?",
                 fontsize=10, color=INK, loc="left", pad=20)
    fig.tight_layout()
    out = FIG / "agent3_fallback.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DEPLOYED_MODEL)
    ap.add_argument("--agent3", action="store_true", help="only the Agent 3 figures")
    ap.add_argument("--agent2", action="store_true", help="only the Agent 2 figures")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if args.agent2:
        for p in (agent2_confusion(), agent2_reliability()):
            print("wrote", p)
        return
    if args.agent3:
        for p in (agent3_designs(), agent3_fallback()):
            print("wrote", p)
        return
    made = [confusion(args.tag, s) for s in ("test", "temporal")
            if (MODELS / args.tag / f"{s}_predictions.npz").exists()]
    made.append(reliability(args.tag))
    if args.tag != "scibert" and (MODELS / "scibert" / "temporal_predictions.npz").exists():
        made.append(reliability_compare(["scibert", args.tag]))
    made.append(baselines())
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
