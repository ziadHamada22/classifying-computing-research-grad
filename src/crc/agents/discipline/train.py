"""Train Agent 1 (discipline) — one backbone per invocation.

Defaults are tuned for a 6 GB RTX 3060 Laptop: fp16, small per-device batch with
gradient accumulation to reach a useful effective batch, and max_length 256
(chunks target 200 words ~= 260 subword tokens, so 256 truncates very little).

Two corpus shapes are supported and can be combined:

  --corpus abstract   title+abstract rows (large, clean, cheap)
  --corpus chunk      section-labelled full-text chunks (matches deployment)
  --corpus both       abstract rows plus chunks

Saves val and test logits to .npz so calibration, ensembling and the
document-level aggregator can be fitted afterwards without re-running the model.

Run:
    python -m crc.agents.discipline.train --model microsoft/deberta-v3-base
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
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from crc.agents.discipline.features import build_text_column
from crc.taxonomy import DISCIPLINES, ID2LABEL, LABEL2ID

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
PROJECT = Path(__file__).resolve().parents[4]      # .../GP/system
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def load_corpus(kind: str, max_chunk_rows: int | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if kind in ("abstract", "both"):
        p = CORPUS_DIR / "corpus_v2.parquet"
        df = pd.read_parquet(p)
        df = df[["title", "abstract", "discipline", "split", "primary_category",
                 "ambiguous", "year"]].copy()
        df["source"] = "abstract"
        frames.append(df)
        print(f"abstract corpus: {len(df):,} rows from {p.name}")
    if kind in ("chunk", "both"):
        p = CORPUS_DIR / "chunks_v2.parquet"
        df = pd.read_parquet(p)
        if max_chunk_rows and len(df) > max_chunk_rows:
            # Subsample per paper rather than per row so no paper is dropped
            # entirely and section coverage stays representative.
            df = df.sample(n=max_chunk_rows, random_state=42)
        df["source"] = "chunk"
        frames.append(df)
        print(f"chunk corpus: {len(df):,} rows from {p.name}")
    if not frames:
        raise SystemExit(f"unknown corpus kind {kind!r}")
    out = pd.concat(frames, ignore_index=True)
    out["text_in"] = build_text_column(out)
    out["label"] = out["discipline"].map(LABEL2ID)
    out = out[out["label"].notna()].copy()
    out["label"] = out["label"].astype(int)
    return out


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "accuracy": float((preds == labels).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--tag", default=None, help="Output dir name; defaults from --model.")
    ap.add_argument("--corpus", default="abstract",
                    choices=["abstract", "chunk", "both"])
    ap.add_argument("--max-chunk-rows", type=int, default=400_000)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--label-smoothing", type=float, default=0.05,
                    help="Absorbs residual taxonomy noise; 0 disables.")
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--eval-steps", type=int, default=0,
                    help="0 = evaluate once per epoch.")
    args = ap.parse_args()

    tag = args.tag or args.model.split("/")[-1].replace(".", "-")
    out_dir = MODELS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = load_corpus(args.corpus, args.max_chunk_rows)
    print("\nrows per split:")
    print(df.groupby(["split", "source"]).size().unstack(fill_value=0).to_string())

    tok = AutoTokenizer.from_pretrained(args.model)

    from datasets import Dataset

    def make(split: str) -> "Dataset":
        sub = df[df["split"] == split]
        ds = Dataset.from_pandas(
            sub[["text_in", "label"]].rename(columns={"text_in": "text"})
            .reset_index(drop=True)
        )
        return ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=args.max_length),
            batched=True, remove_columns=["text"],
        )

    ds = {s: make(s) for s in ("train", "val", "test")}
    print("\ntokenised:", {k: len(v) for k, v in ds.items()})

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(DISCIPLINES),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    eval_kwargs = ({"eval_strategy": "steps", "eval_steps": args.eval_steps,
                    "save_steps": args.eval_steps}
                   if args.eval_steps else
                   {"eval_strategy": "epoch", "save_strategy": "epoch"})
    if args.eval_steps:
        eval_kwargs["save_strategy"] = "steps"

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
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to=["none"],
        save_total_limit=1,
        seed=args.seed,
        dataloader_num_workers=0,       # Windows: workers add overhead, not speed
        **eval_kwargs,
    )

    # transformers >=4.48 renamed Trainer's `tokenizer` argument.
    tok_kw = ("processing_class"
              if "processing_class" in inspect.signature(Trainer.__init__).parameters
              else "tokenizer")
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        **{tok_kw: tok},
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    results: dict = {"model": args.model, "tag": tag, "corpus": args.corpus,
                     "train_seconds": round(train_secs, 1), "args": vars(args)}

    for split in ("val", "test"):
        pred = trainer.predict(ds[split])
        logits = pred.predictions
        labels = pred.label_ids
        preds = logits.argmax(-1)
        rep = classification_report(
            [ID2LABEL[i] for i in labels], [ID2LABEL[i] for i in preds],
            labels=DISCIPLINES, digits=4, zero_division=0,
        )
        results[split] = {
            "macro_f1": float(f1_score(labels, preds, average="macro")),
            "weighted_f1": float(f1_score(labels, preds, average="weighted")),
            "accuracy": float((preds == labels).mean()),
            "report": rep,
        }
        np.savez(out_dir / f"{split}_predictions.npz",
                 logits=logits, labels=labels)
        print(f"\n===== {split} =====\n{rep}")

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (RESULTS / f"metrics_{tag}.json").write_text(json.dumps(results, indent=2))

    print(f"\ntrained in {train_secs/60:.1f} min")
    print(f"model   -> {out_dir}")
    print(f"metrics -> {RESULTS / f'metrics_{tag}.json'}")


if __name__ == "__main__":
    main()
