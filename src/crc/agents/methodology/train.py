"""Train Agent 3 — research-design classifier (the primary methodology axis).

A flat 4-way SciBERT head over the trainable designs (Design & Creation,
Experiment, Formal / Theoretical, Simulation & Modelling). Unlike Agent 2 there
is no masking: methodology is a single label space, not conditioned on
discipline. The input is the paper's methods+abstract text — that is where the
design is declared. Worldview and method are not learned here; they are read from
each design's prior and refined by the local-LLM hard-case reviewer.

The five arXiv-starved human-centric designs are out of scope for this encoder
(see `taxonomy/methodology.TRAINABLE_DESIGNS`).

Run:
    python -m crc.agents.methodology.train
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from crc.taxonomy.methodology import (
    TRAIN_ID2LABEL,
    TRAIN_LABEL2ID,
    TRAINABLE_DESIGNS,
)

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS = WORK / "corpus" / "methodology_corpus.parquet"
PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--tag", default="methodology-scibert")
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
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

    df = pd.read_parquet(args.corpus)
    df = df[df["design"].isin(TRAINABLE_DESIGNS)].copy()
    df["label"] = df["design"].map(TRAIN_LABEL2ID).astype(int)
    print(f"corpus: {len(df):,} papers, {df['design'].nunique()} designs")
    print(df.groupby(["design", "split"]).size().unstack(fill_value=0).to_string())

    tok = AutoTokenizer.from_pretrained(args.model)
    from datasets import Dataset

    def make(split: str):
        sub = df[df["split"] == split]
        ds = Dataset.from_pandas(sub[["text", "label"]].reset_index(drop=True))
        return ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=args.max_length),
            batched=True, remove_columns=["text"])

    ds = {s: make(s) for s in ("train", "val", "test")}
    print("tokenised:", {k: len(v) for k, v in ds.items()})

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {
            "macro_f1": f1_score(p.label_ids, preds, average="macro"),
            "accuracy": float((preds == p.label_ids).mean()),
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(TRAINABLE_DESIGNS),
        id2label=TRAIN_ID2LABEL, label2id=TRAIN_LABEL2ID)

    targs = TrainingArguments(
        output_dir=str(out_dir / "ckpts"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        logging_steps=100, fp16=torch.cuda.is_available(),
        report_to=["none"], save_total_limit=1, seed=args.seed,
        dataloader_num_workers=0,
    )

    tok_kw = ("processing_class"
              if "processing_class" in inspect.signature(Trainer.__init__).parameters
              else "tokenizer")
    trainer = Trainer(
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
                     "train_seconds": round(train_secs, 1), "args": vars(args),
                     "labels": TRAINABLE_DESIGNS}

    for split in ("val", "test"):
        pred = trainer.predict(ds[split])
        y = pred.label_ids
        p = pred.predictions.argmax(-1)
        res = {
            "n": int(len(y)),
            "accuracy": float((p == y).mean()),
            "macro_f1": float(f1_score(y, p, average="macro")),
            "confusion": confusion_matrix(
                y, p, labels=list(range(len(TRAINABLE_DESIGNS)))).tolist(),
        }
        results[split] = res
        print(f"\n===== {split} (n={res['n']:,}) =====")
        print(f"  accuracy {res['accuracy']:.4f}   macro-F1 {res['macro_f1']:.4f}")
        if split == "test":
            print("\n" + classification_report(
                [TRAIN_ID2LABEL[i] for i in y],
                [TRAIN_ID2LABEL[i] for i in p],
                labels=TRAINABLE_DESIGNS, digits=3, zero_division=0))

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (RESULTS / f"metrics_{args.tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\ntrained in {train_secs/60:.1f} min")
    print(f"model   -> {out_dir}")
    print(f"metrics -> {RESULTS / f'metrics_{args.tag}.json'}")


if __name__ == "__main__":
    main()
