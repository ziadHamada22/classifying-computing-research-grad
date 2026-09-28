"""Train Agent 2 — field within discipline.

One shared encoder, one flat 38-way head. At inference the logits are masked to
the fields of the discipline Agent 1 predicted, so a Computer Science paper is
only ever scored against Computer Science fields. Training is unmasked and joint
across all six disciplines, which is the point: Software Engineering has 3,226
labelled papers against Computer Science's 27,966, and sharing the encoder lets
SE inherit a representation it could never learn alone.

Two evaluation numbers are reported and they answer different questions:

  flat        accuracy over all 38 fields at once — how well the model
              separates the whole space
  conditioned accuracy after masking to the true discipline's fields — the
              number that actually matters, because that is how Agent 2 runs

The gap between them is the value of conditioning, and the conditioned score is
compared against each discipline's majority-class baseline so a head that has
merely learned the prior is visible as such.

Run:
    python -m crc.agents.field.train
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

from crc.taxonomy import DISCIPLINES
from crc.taxonomy.fields import (
    DISCIPLINE_FIELD_IDS,
    GLOBAL_ID2LABEL,
    GLOBAL_LABEL2ID,
    N_GLOBAL_FIELDS,
)

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS = WORK / "corpus" / "fields_corpus.parquet"
PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def conditioned_predictions(logits: np.ndarray,
                            disciplines: list[str]) -> np.ndarray:
    """Argmax restricted to each row's own discipline — how Agent 2 runs."""
    out = np.empty(len(logits), dtype=int)
    for i, d in enumerate(disciplines):
        allowed = DISCIPLINE_FIELD_IDS[d]
        out[i] = allowed[int(np.argmax(logits[i, allowed]))]
    return out


def _build_allowed_matrix() -> "torch.Tensor":
    """(38, 38) bool: allowed[i, j] iff fields i and j share a discipline.

    Each field belongs to exactly one discipline, so a paper's true label alone
    determines which fields are on its ballot — no discipline column needed.
    """
    disc_of = [None] * N_GLOBAL_FIELDS
    for d, ids in DISCIPLINE_FIELD_IDS.items():
        for i in ids:
            disc_of[i] = d
    m = torch.zeros(N_GLOBAL_FIELDS, N_GLOBAL_FIELDS, dtype=torch.bool)
    for i in range(N_GLOBAL_FIELDS):
        for j in range(N_GLOBAL_FIELDS):
            m[i, j] = disc_of[i] == disc_of[j]
    return m


class MaskedTrainer(Trainer):
    """Trainer whose loss is masked to each example's true discipline.

    Design ablation for Agent 2: the logits are restricted to the fields of the
    label's own discipline *before* the softmax, so cross-entropy only ever asks
    "which field within this discipline?" — aligning training with the masked
    inference path. Label smoothing is applied over the allowed fields only, so
    it stays consistent with the mask. Everything else matches the base run.
    """

    def __init__(self, *args, allowed_matrix=None, label_smoothing=0.0, **kw):
        super().__init__(*args, **kw)
        self._allowed = allowed_matrix
        self._ls = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        allowed = self._allowed.to(logits.device)[labels]        # (B, 38)
        neg = torch.finfo(logits.dtype).min
        logp = torch.log_softmax(logits.masked_fill(~allowed, neg), dim=-1)
        nll = -logp.gather(1, labels.unsqueeze(1)).squeeze(1)     # -log p(true)
        n_allowed = allowed.sum(-1).clamp(min=1)
        logp_allowed = torch.where(allowed, logp, torch.zeros_like(logp))
        smooth = -(logp_allowed.sum(-1) / n_allowed)             # mean over allowed
        loss = ((1.0 - self._ls) * nll + self._ls * smooth).mean()
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--tag", default="field-scibert")
    ap.add_argument("--corpus", default=str(CORPUS))
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
    ap.add_argument("--masked-loss", action="store_true",
                    help="Ablation: mask the loss to each label's discipline.")
    args = ap.parse_args()

    out_dir = MODELS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(args.corpus)
    df["label"] = df["field"].map(GLOBAL_LABEL2ID)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    print(f"corpus: {len(df):,} papers, {df['field'].nunique()} fields")
    print(df.groupby(["discipline", "split"]).size().unstack(fill_value=0).to_string())

    tok = AutoTokenizer.from_pretrained(args.model)
    from datasets import Dataset

    def make(split: str):
        sub = df[df["split"] == split]
        ds = Dataset.from_pandas(
            sub[["text", "label"]].reset_index(drop=True))
        return ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=args.max_length),
            batched=True, remove_columns=["text"])

    ds = {s: make(s) for s in ("train", "val", "test")}
    print("tokenised:", {k: len(v) for k, v in ds.items()})

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {
            "flat_macro_f1": f1_score(p.label_ids, preds, average="macro"),
            "flat_accuracy": float((preds == p.label_ids).mean()),
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=N_GLOBAL_FIELDS,
        id2label=GLOBAL_ID2LABEL, label2id=GLOBAL_LABEL2ID)

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
        metric_for_best_model="flat_macro_f1", greater_is_better=True,
        logging_steps=100, fp16=torch.cuda.is_available(),
        report_to=["none"], save_total_limit=1, seed=args.seed,
        dataloader_num_workers=0,
    )

    tok_kw = ("processing_class"
              if "processing_class" in inspect.signature(Trainer.__init__).parameters
              else "tokenizer")
    common = dict(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        **{tok_kw: tok})
    if args.masked_loss:
        print("using MASKED loss (conditioned training ablation)")
        trainer = MaskedTrainer(allowed_matrix=_build_allowed_matrix(),
                                label_smoothing=args.label_smoothing, **common)
    else:
        trainer = Trainer(**common)

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    results: dict = {"model": args.model, "tag": args.tag,
                     "train_seconds": round(train_secs, 1), "args": vars(args)}

    for split in ("val", "test"):
        pred = trainer.predict(ds[split])
        logits, labels = pred.predictions, pred.label_ids
        sub = df[df["split"] == split].reset_index(drop=True)
        discs = sub["discipline"].tolist()

        flat = logits.argmax(-1)
        cond = conditioned_predictions(logits, discs)

        res = {
            "n": int(len(labels)),
            "flat_accuracy": float((flat == labels).mean()),
            "flat_macro_f1": float(f1_score(labels, flat, average="macro")),
            "conditioned_accuracy": float((cond == labels).mean()),
            "conditioned_macro_f1": float(f1_score(labels, cond, average="macro")),
            "per_discipline": {},
        }
        for d in DISCIPLINES:
            m = np.array([x == d for x in discs])
            if not m.any():
                continue
            y, p = labels[m], cond[m]
            counts = np.bincount(y, minlength=N_GLOBAL_FIELDS)
            res["per_discipline"][d] = {
                "n": int(m.sum()),
                "accuracy": float((p == y).mean()),
                "macro_f1": float(f1_score(y, p, average="macro")),
                "majority_baseline": float(counts.max() / m.sum()),
                "n_fields": len(DISCIPLINE_FIELD_IDS[d]),
            }
        results[split] = res

        print(f"\n===== {split} =====")
        print(f"  flat        acc {res['flat_accuracy']:.4f}  "
              f"macro-F1 {res['flat_macro_f1']:.4f}   (all 38 fields)")
        print(f"  conditioned acc {res['conditioned_accuracy']:.4f}  "
              f"macro-F1 {res['conditioned_macro_f1']:.4f}   (masked to discipline)")
        print(f"\n  {'discipline':24s} {'n':>6s} {'acc':>7s} {'macroF1':>8s} "
              f"{'baseline':>9s} {'lift':>7s}")
        for d, s in res["per_discipline"].items():
            print(f"  {d:24s} {s['n']:>6,} {s['accuracy']:>7.4f} "
                  f"{s['macro_f1']:>8.4f} {s['majority_baseline']:>9.4f} "
                  f"{s['accuracy'] - s['majority_baseline']:>+7.4f}")

        if split == "test":
            names = [GLOBAL_ID2LABEL[i] for i in sorted(set(labels) | set(cond))]
            print("\n" + classification_report(
                [GLOBAL_ID2LABEL[i] for i in labels],
                [GLOBAL_ID2LABEL[i] for i in cond],
                labels=names, digits=3, zero_division=0))

        np.savez(out_dir / f"{split}_predictions.npz",
                 logits=logits, labels=labels,
                 disciplines=np.array(discs, dtype=object))

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (RESULTS / f"metrics_{args.tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\ntrained in {train_secs/60:.1f} min")
    print(f"model   -> {out_dir}")
    print(f"metrics -> {RESULTS / f'metrics_{args.tag}.json'}")


if __name__ == "__main__":
    main()
