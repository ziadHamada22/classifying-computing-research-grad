"""Train Agent 1 with ambiguous papers included, as soft (vote-share) targets.

The v2 corpus excludes every ambiguous paper from training -- a paper whose
arXiv categories vote for two disciplines by a narrow margin -- and keeps them
only in test. That was a deliberate labelling decision, and it has a measured
cost: Agent 1 has never seen a boundary case, and on ambiguous test papers its
Computer Science recall is 0.356 against 0.791 on clean ones. The deployed
system meets ambiguous papers constantly (16.7% of test, 28.9% of 2025+).

This trainer tests the direct remedy. It adds ambiguous papers from the unused
part of the 2015-2024 computing pool, and instead of forcing them to one label it
trains on the labeller's own vote distribution: a paper that is 55% CS / 45% CE
by its categories is taught exactly that. Clean papers keep their one-hot target,
with the same 0.05 label smoothing as the original recipe, so the only change is
the added boundary cases.

Everything else is the SciBERT recipe from the bake-off (3 epochs, effective
batch 32, lr 2e-5, fp16, best epoch by val macro-F1), so a difference is
attributable to the data. Validation and test are unchanged, and the added
papers are drawn from outside the discipline corpus, the field corpus and the
2025+ holdout, so no evaluation set gains a leaked paper. Their ids are saved
beside the model for auditing.

``--hard`` adds the same papers with their one-hot consensus label instead: the
control that separates "more boundary data" from "soft targets".

Run:
    python -m crc.agents.discipline.train_soft                 # -> models/scibert-soft
    python -m crc.agents.discipline.train_soft --hard          # -> models/scibert-ambig-hard
"""
from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from crc.agents.discipline.features import build_text_column
from crc.taxonomy import DISCIPLINES, ID2LABEL, LABEL2ID, label_paper

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS_DIR = WORK / "corpus"
PROJECT = Path(__file__).resolve().parents[4]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"

N_CLASSES = len(DISCIPLINES)


def vote_distribution(categories) -> np.ndarray:
    """The labeller's normalised vote over DISCIPLINES for one paper."""
    votes = label_paper(list(categories)).votes
    v = np.array([votes.get(d, 0.0) for d in DISCIPLINES], dtype=np.float64)
    return v / v.sum() if v.sum() > 0 else np.full(N_CLASSES, 1.0 / N_CLASSES)


def smooth(target: np.ndarray, eps: float) -> np.ndarray:
    return (1.0 - eps) * target + eps / N_CLASSES


def sample_ambiguous(per_discipline: int, seed: int) -> pd.DataFrame:
    """Ambiguous 2015-2024 papers no evaluation set or other corpus contains."""
    pool = pd.read_parquet(
        CORPUS_DIR / "computing_pool.parquet",
        columns=["id", "title", "abstract", "categories", "discipline",
                 "ambiguous", "year", "margin"])
    pool = pool[pool["ambiguous"] & pool["year"].between(2015, 2024)
                & pool["discipline"].notna()]
    exclude = set(pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet", columns=["id"])["id"])
    exclude |= set(pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet",
                                   columns=["id"])["id"])
    exclude |= set(pd.read_parquet(CORPUS_DIR / "fields_corpus.parquet",
                                   columns=["id"])["id"])
    pool = pool[~pool["id"].isin(exclude)]
    parts = []
    for d in DISCIPLINES:
        sub = pool[pool["discipline"] == d]
        parts.append(sub.sample(n=min(per_discipline, len(sub)), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    print("ambiguous additions per discipline:",
          out["discipline"].value_counts().reindex(DISCIPLINES).to_dict())
    return out


def soft_target_trainer():
    """A Trainer whose loss is cross-entropy against a target *distribution*.

    Built lazily so importing this module (e.g. for ``vote_distribution`` in the
    tests) does not pull in the whole training stack.
    """
    import torch
    from transformers import Trainer

    class SoftTargetTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            targets = inputs.pop("labels").float()
            outputs = model(**inputs)
            logp = torch.log_softmax(outputs.logits.float(), dim=-1)
            loss = -(targets * logp).sum(dim=-1).mean()
            return (loss, outputs) if return_outputs else loss

    return SoftTargetTrainer


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = np.asarray(p.label_ids)
    if labels.ndim == 2:
        labels = labels.argmax(-1)
    return {"macro_f1": f1_score(labels, preds, average="macro"),
            "accuracy": float((preds == labels).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--hard", action="store_true",
                    help="Control: add the same papers with one-hot consensus labels.")
    ap.add_argument("--ambiguous-per-discipline", type=int, default=1700)
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
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    tag = args.tag or ("scibert-ambig-hard" if args.hard else "scibert-soft")
    out_dir = MODELS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    corpus = pd.read_parquet(CORPUS_DIR / "corpus_v2.parquet")
    temporal = pd.read_parquet(CORPUS_DIR / "corpus_v2_temporal.parquet").reset_index(drop=True)
    extra = sample_ambiguous(args.ambiguous_per_discipline, args.seed)
    (out_dir / "added_ambiguous_ids.json").write_text(json.dumps(extra["id"].tolist()))

    def onehot(labels: np.ndarray) -> np.ndarray:
        return np.eye(N_CLASSES)[labels]

    eps = args.label_smoothing
    train = corpus[corpus["split"] == "train"].reset_index(drop=True)
    y_train = train["discipline"].map(LABEL2ID).to_numpy()
    t_train = smooth(onehot(y_train), eps)
    y_extra = extra["discipline"].map(LABEL2ID).to_numpy()
    t_extra = (smooth(onehot(y_extra), eps) if args.hard else
               smooth(np.stack([vote_distribution(c) for c in extra["categories"]]), eps))
    print(f"train {len(train):,} clean + {len(extra):,} ambiguous "
          f"({'hard' if args.hard else 'soft'} targets); mean max target on the "
          f"ambiguous rows = {t_extra.max(1).mean():.3f}")

    tok = AutoTokenizer.from_pretrained(args.model)
    from datasets import Dataset

    def make(df: pd.DataFrame, targets: np.ndarray) -> "Dataset":
        ds = Dataset.from_dict({"text": build_text_column(df).tolist(),
                                "labels": targets.astype(np.float32).tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True,
                                    max_length=args.max_length),
                      batched=True, remove_columns=["text"])

    train_df = pd.concat([train[["title", "abstract"]], extra[["title", "abstract"]]],
                         ignore_index=True)
    ds_train = make(train_df, np.concatenate([t_train, t_extra]))
    evals = {}
    for name, df in (("val", corpus[corpus["split"] == "val"]),
                     ("test", corpus[corpus["split"] == "test"]),
                     ("temporal", temporal)):
        df = df.reset_index(drop=True)
        y = df["discipline"].map(LABEL2ID).to_numpy()
        evals[name] = (make(df, onehot(y)), y)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=N_CLASSES, id2label=ID2LABEL, label2id=LABEL2ID)
    targs = TrainingArguments(
        output_dir=str(out_dir / "ckpts"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, logging_steps=100,
        fp16=torch.cuda.is_available(), report_to=["none"],
        save_total_limit=1, seed=args.seed, dataloader_num_workers=0,
    )
    tok_kw = ("processing_class"
              if "processing_class" in inspect.signature(Trainer.__init__).parameters
              else "tokenizer")
    trainer = soft_target_trainer()(
        model=model, args=targs, train_dataset=ds_train,
        eval_dataset=evals["val"][0],
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        **{tok_kw: tok},
    )
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0

    results: dict = {"model": args.model, "tag": tag, "corpus": "abstract+ambiguous",
                     "targets": "hard" if args.hard else "soft (vote share)",
                     "n_ambiguous_added": int(len(extra)),
                     "train_seconds": round(secs, 1), "args": vars(args)}
    for name, (ds, y) in evals.items():
        logits = trainer.predict(ds).predictions
        preds = logits.argmax(-1)
        results[name] = {
            "macro_f1": float(f1_score(y, preds, average="macro")),
            "weighted_f1": float(f1_score(y, preds, average="weighted")),
            "accuracy": float((preds == y).mean()),
            "report": classification_report(
                [ID2LABEL[i] for i in y], [ID2LABEL[i] for i in preds],
                labels=DISCIPLINES, digits=4, zero_division=0),
        }
        np.savez(out_dir / f"{name}_predictions.npz", logits=logits, labels=y)
        print(f"\n===== {name} (macro-F1 {results[name]['macro_f1']:.4f}) =====")
        print(results[name]["report"])

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (RESULTS / f"metrics_{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\ntrained in {secs/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
