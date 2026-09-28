"""Train the multi-label facet model for Agent 3.

Eight independent binary heads instead of one four-way choice. Three reasons that
is the right shape, all measured rather than assumed:

* **40.5% of papers both build and evaluate.** A single label had to guess between
  Design & Creation and Experiment for two papers in five, which is exactly where
  ~52% of the old model's errors lived. Independent heads simply record both.
* **Facets need no expert to check.** "Does this paper prove theorems?" is
  answerable from the abstract by any reader; "is this design science or a case
  study?" is not. That is what makes evaluation possible without a gold set.
* **Coverage doubles.** The cue labeller could only label 63.6% of the pool, so
  the design model trained on 9,383 papers. Facets are labelled for **all 20,343**
  by the label model, because a paper with no recognisable design still has
  observable properties.

Targets are the **soft posteriors** from the Dawid-Skene label model, not hard
0/1 labels. A paper the labelling functions disagreed about arrives as 0.6 rather
than 1.0, so the model is trained to be uncertain exactly where the evidence is.

Input is the abstract, which the previous experiment established as the best
single source for methodology (0.692 against 0.652 for methods, at 100% coverage).

Run:
    python -m crc.agents.methodology.train_facets
"""
from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from crc.taxonomy.facets import FACET_KEYS

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS = WORK / "corpus"
CHUNK_FILES = ["chunks_eval.parquet", "chunks_v2.parquet"]
PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def build_corpus(facets_path: Path) -> pd.DataFrame:
    """Abstract text joined to the label model's facet posteriors."""
    fac = pd.read_parquet(facets_path)
    frames = []
    for name in CHUNK_FILES:
        p = CORPUS / name
        if p.exists():
            frames.append(pd.read_parquet(
                p, columns=["paper_id", "section", "n_words", "text"]))
    ch = pd.concat(frames, ignore_index=True)
    ch = ch.drop_duplicates(subset=["paper_id", "section", "text"])
    ab = ch[ch["section"] == "abstract"]
    ab = (ab.groupby("paper_id")
            .agg(text=("text", lambda s: " ".join(s.astype(str))),
                 n_words=("n_words", "sum"))
            .reset_index())
    df = fac.merge(ab, on="paper_id", how="inner")
    return df[df["n_words"] >= 20].reset_index(drop=True)


class MultiLabelTrainer(Trainer):
    """Binary cross-entropy over independent facet heads, with soft targets.

    Soft targets matter here: the label model returns a probability per facet, and
    rounding it to 0/1 would throw away precisely the uncertainty information that
    the label model exists to produce.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs.logits.float(), labels)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--tag", default="methodology-facets")
    ap.add_argument("--facets", default=str(CORPUS / "facets_pool.parquet"))
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()

    out_dir = MODELS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = build_corpus(Path(args.facets))
    print(f"papers with an abstract: {len(df):,}")
    print(df.groupby("split").size().to_string())
    print("\nfacet prevalence (hard, at 0.5):")
    for f in FACET_KEYS:
        print(f"  {f:16s} {df[f].mean():6.1%}")

    tok = AutoTokenizer.from_pretrained(args.model)
    from datasets import Dataset

    def make(split: str):
        sub = df[df["split"] == split].reset_index(drop=True)
        soft = sub[[f"p_{f}" for f in FACET_KEYS]].to_numpy(dtype=np.float32)
        ds = Dataset.from_dict({"text": sub["text"].tolist(),
                                "labels": soft.tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True,
                                    max_length=args.max_length),
                      batched=True, remove_columns=["text"])

    ds = {s: make(s) for s in ("train", "val", "test")}
    print("\ntokenised:", {k: len(v) for k, v in ds.items()})

    def compute_metrics(p):
        probs = 1.0 / (1.0 + np.exp(-p.predictions))
        y = (p.label_ids >= 0.5).astype(int)
        aps = []
        for k in range(len(FACET_KEYS)):
            if 0 < y[:, k].sum() < len(y):
                aps.append(average_precision_score(y[:, k], probs[:, k]))
        return {"macro_ap": float(np.mean(aps)) if aps else 0.0}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(FACET_KEYS),
        problem_type="multi_label_classification",
        id2label={i: f for i, f in enumerate(FACET_KEYS)},
        label2id={f: i for i, f in enumerate(FACET_KEYS)})

    targs = TrainingArguments(
        output_dir=str(out_dir / "ckpts"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_ap", greater_is_better=True,
        logging_steps=100, fp16=torch.cuda.is_available(),
        report_to=["none"], save_total_limit=1, seed=args.seed,
        dataloader_num_workers=0,
    )

    tok_kw = ("processing_class"
              if "processing_class" in inspect.signature(Trainer.__init__).parameters
              else "tokenizer")
    trainer = MultiLabelTrainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        **{tok_kw: tok})

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    results: dict = {"model": args.model, "tag": args.tag,
                     "facets": FACET_KEYS, "train_seconds": round(train_secs, 1),
                     "args": vars(args)}

    for split in ("val", "test"):
        pred = trainer.predict(ds[split])
        probs = 1.0 / (1.0 + np.exp(-pred.predictions))
        soft = pred.label_ids
        y = (soft >= 0.5).astype(int)
        sub = df[df["split"] == split].reset_index(drop=True)

        per = {}
        for k, f in enumerate(FACET_KEYS):
            pos = int(y[:, k].sum())
            row = {"positives": pos, "prevalence": round(float(y[:, k].mean()), 4)}
            if 0 < pos < len(y):
                row["auc"] = round(float(roc_auc_score(y[:, k], probs[:, k])), 4)
                row["average_precision"] = round(
                    float(average_precision_score(y[:, k], probs[:, k])), 4)
                row["f1_at_0.5"] = round(
                    float(f1_score(y[:, k], probs[:, k] >= 0.5)), 4)
            per[f] = row
        results[split] = {"n": int(len(y)), "per_facet": per,
                          "macro_auc": round(float(np.mean(
                              [v["auc"] for v in per.values() if "auc" in v])), 4)}

        print(f"\n===== {split} =====")
        print(f"  {'facet':16s} {'prev':>7s} {'AUC':>7s} {'AP':>7s} {'F1':>7s}")
        for f, v in per.items():
            print(f"  {f:16s} {v['prevalence']:>7.1%} "
                  f"{v.get('auc', float('nan')):>7.3f} "
                  f"{v.get('average_precision', float('nan')):>7.3f} "
                  f"{v.get('f1_at_0.5', float('nan')):>7.3f}")
        print(f"  macro AUC {results[split]['macro_auc']:.4f}")

        np.savez(out_dir / f"{split}_facet_predictions.npz",
                 probs=probs, soft=soft,
                 paper_ids=sub["paper_id"].to_numpy().astype(str),
                 facets=np.array(FACET_KEYS, dtype=object))

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (RESULTS / f"metrics_{args.tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\ntrained in {train_secs/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
