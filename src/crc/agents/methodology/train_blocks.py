"""Train Agent 3 at BLOCK level, so the whole paper is read rather than truncated.

The paper-level Agent 3 (`train.py`) concatenated methods+abstract and fed the
first 320 tokens to SciBERT, discarding more than half of the average paper. This
trainer instead classifies each section-tagged block independently, exactly as
Agent 1 does, and leaves document-level aggregation to a separate step.

The split is deliberate. Aggregation has several free choices — which pooling
rule, how to weight sections — and fitting them requires many passes over the
validation set. Saving **block logits** here means all of that happens offline
against saved arrays, so the model is trained once and never re-run while the
aggregation question is explored (the same discipline `ensemble.py` follows).

Block labels are inherited from the paper: every block of a Design-&-Creation
paper is labelled Design & Creation. That is noisy on purpose — a background
paragraph carries little design signal — and it is the noise the fitted section
weights are there to discover and discount. The label is a property of the paper,
so the *document* number is the one that means anything; block accuracy is
reported only as a training diagnostic.

Run:
    python -m crc.agents.methodology.train_blocks
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
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from crc.agents.discipline.features import format_chunk
from crc.taxonomy.methodology import (
    TRAIN_ID2LABEL,
    TRAIN_LABEL2ID,
    TRAINABLE_DESIGNS,
)

WORK = Path(r"C:\Users\ziada\gp_data")
BLOCKS = WORK / "corpus" / "methodology_blocks.parquet"
PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--tag", default="methodology-blocks")
    ap.add_argument("--corpus", default=str(BLOCKS))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--sections", default="",
                    help="Comma-separated sections to keep, e.g. 'abstract'. "
                         "Empty keeps all. Used to test whether one section on "
                         "its own beats reading the whole paper.")
    args = ap.parse_args()

    out_dir = MODELS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(args.corpus)
    df = df[df["design"].isin(TRAINABLE_DESIGNS)].copy()
    if args.sections:
        keep = [s.strip() for s in args.sections.split(",") if s.strip()]
        df = df[df["section"].isin(keep)].copy()
        print(f"restricted to sections {keep}")
    df["label"] = df["design"].map(TRAIN_LABEL2ID).astype(int)
    # Identical formatting to Agent 1, so the section tag is presented the same
    # way the encoder has already been shown to use.
    df["model_text"] = [format_chunk(t, s) for t, s in
                        zip(df["text"], df["section"])]

    print(f"blocks: {len(df):,} over {df['paper_id'].nunique():,} papers")
    print(df.groupby("split").agg(blocks=("text", "size"),
                                  papers=("paper_id", "nunique")).to_string())

    tok = AutoTokenizer.from_pretrained(args.model)
    from datasets import Dataset

    def make(split: str):
        sub = df[df["split"] == split]
        ds = Dataset.from_pandas(
            sub[["model_text", "label"]].rename(columns={"model_text": "text"})
            .reset_index(drop=True))
        return ds.map(lambda b: tok(b["text"], truncation=True,
                                    max_length=args.max_length),
                      batched=True, remove_columns=["text"])

    ds = {s: make(s) for s in ("train", "val", "test")}
    print("tokenised:", {k: len(v) for k, v in ds.items()})

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {"macro_f1": f1_score(p.label_ids, preds, average="macro"),
                "accuracy": float((preds == p.label_ids).mean())}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(TRAINABLE_DESIGNS),
        id2label=TRAIN_ID2LABEL, label2id=TRAIN_LABEL2ID)

    targs = TrainingArguments(
        output_dir=str(out_dir / "ckpts"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
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

    results: dict = {"model": args.model, "tag": args.tag, "level": "block",
                     "train_seconds": round(train_secs, 1), "args": vars(args),
                     "labels": TRAINABLE_DESIGNS}

    for split in ("val", "test"):
        pred = trainer.predict(ds[split])
        logits, labels = pred.predictions, pred.label_ids
        sub = df[df["split"] == split].reset_index(drop=True)
        blk = logits.argmax(-1)

        res = {
            "n_blocks": int(len(labels)),
            "n_papers": int(sub["paper_id"].nunique()),
            "block_accuracy": float((blk == labels).mean()),
            "block_macro_f1": float(f1_score(labels, blk, average="macro")),
            "per_section_block_accuracy": {},
        }
        # Which sections does a block-level model get right? A first, crude read
        # on where the design signal lives; the weighted document fit is the real
        # answer, but this is free and points the same way.
        for sec, g in sub.groupby("section"):
            i = g.index.to_numpy()
            res["per_section_block_accuracy"][sec] = {
                "n": int(len(i)),
                "accuracy": round(float((blk[i] == labels[i]).mean()), 4),
            }
        results[split] = res

        print(f"\n===== {split} (block level) =====")
        print(f"  blocks {res['n_blocks']:,} over {res['n_papers']:,} papers")
        print(f"  block acc {res['block_accuracy']:.4f}  "
              f"macro-F1 {res['block_macro_f1']:.4f}")
        print(f"  {'section':16s} {'n':>7s} {'acc':>7s}")
        for sec, s in sorted(res["per_section_block_accuracy"].items(),
                             key=lambda kv: -kv[1]["accuracy"]):
            print(f"  {sec:16s} {s['n']:>7,} {s['accuracy']:>7.4f}")

        # Everything the aggregation step needs, so the model is never re-run.
        np.savez(out_dir / f"{split}_block_predictions.npz",
                 logits=logits, labels=labels,
                 paper_ids=sub["paper_id"].to_numpy().astype(str),
                 sections=sub["section"].to_numpy().astype(str),
                 n_words=sub["n_words"].to_numpy(),
                 designs=sub["design"].to_numpy().astype(str))

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (RESULTS / f"metrics_{args.tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\ntrained in {train_secs/60:.1f} min")
    print(f"model   -> {out_dir}")
    print("NOTE: block accuracy is a diagnostic only. The label belongs to the "
          "paper, so run crc.eval.evaluate_methodology_blocks for the number "
          "that is comparable with the 0.657 paper-level baseline.")


if __name__ == "__main__":
    main()
