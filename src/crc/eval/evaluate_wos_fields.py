"""Workstream C — Agent 2 field classifier, external validation on WoS.

Agent 2's field labels are weak supervision; this is their first *external,
human-labelled* check. It is only possible for the subset of WoS areas that
coincide with exactly one of our 38 fields (`WOS_AREA_TO_FIELD`) — ~12 fields
across CS/IS/IT/CE/DS. The seven SE sub-fields are unreachable: WoS has a single
"Software engineering" area and cannot sub-divide it (documented limitation).

**Oracle conditioning.** Agent 2 always runs masked to a discipline. Here we mask
to each paper's *true* home discipline (the one its WoS field belongs to), not to
whatever Agent 1 would predict. That isolates Agent 2's field ability from Agent
1's OOD discipline errors — the question is strictly "given the right discipline,
does Agent 2 pick the right field on non-arXiv text?".

Caveats, stated up front: (1) OOD — WoS is older, other venues, and title-free
where our training rows had titles; (2) the WoS area is a *coarse* label, so a
"miss" can be Agent 2 choosing a legitimate neighbour within the same area rather
than an error. Read agreement as a floor.

Run:
    python -m crc.eval.evaluate_wos_fields
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from crc.taxonomy import DISCIPLINES
from crc.taxonomy.fields import (
    DISCIPLINE_FIELD_IDS,
    FIELDS_BY_DISCIPLINE,
    GLOBAL_ID2LABEL,
    GLOBAL_LABEL2ID,
    N_GLOBAL_FIELDS,
)
from crc.taxonomy.wos_map import WOS_AREA_TO_DISCIPLINE, WOS_AREA_TO_FIELD

WORK = Path(r"C:\Users\ziada\gp_data")
POOL = WORK / "corpus" / "wos_pool.parquet"
PROJECT = Path(__file__).resolve().parents[3]
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@torch.no_grad()
def batched_logits(model, tok, texts, device, batch_size=64, max_length=256):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, padding=True,
                  max_length=max_length, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, N_GLOBAL_FIELDS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--model", default=str(MODELS / "field-scibert-v2"))
    args = ap.parse_args()

    # Consistency check: every clean field belongs to its area's discipline.
    for area, fld in WOS_AREA_TO_FIELD.items():
        assert fld in GLOBAL_LABEL2ID, f"unknown field {fld!r}"
        disc = WOS_AREA_TO_DISCIPLINE[area]
        assert GLOBAL_LABEL2ID[fld] in DISCIPLINE_FIELD_IDS[disc], \
            f"{fld!r} is not in {disc!r}'s fields (area {area!r})"

    df = pd.read_parquet(args.pool)
    df = df[df["wos_area"].isin(WOS_AREA_TO_FIELD)].reset_index(drop=True)
    df["true_field"] = df["wos_area"].map(WOS_AREA_TO_FIELD)
    fields_tested = sorted(set(df["true_field"]))
    print(f"WoS field-validation pool: {len(df):,} papers, "
          f"{len(fields_tested)} distinct fields, "
          f"{df['discipline'].nunique()} disciplines")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device).eval()
    print(f"model: {Path(args.model).name}   device: {device}")

    # Training text was "title. abstract"; WoS has no title -> abstract alone.
    texts = df["abstract"].fillna("").astype(str).tolist()
    t0 = time.perf_counter()
    logits = batched_logits(model, tok, texts, device)
    elapsed = time.perf_counter() - t0

    discs = df["discipline"].tolist()
    pred_field = np.empty(len(df), dtype=object)
    for i, d in enumerate(discs):
        allowed = DISCIPLINE_FIELD_IDS[d]
        j = int(np.argmax(logits[i, allowed]))
        pred_field[i] = GLOBAL_ID2LABEL[allowed[j]]

    true_field = df["true_field"].to_numpy()
    acc = float((pred_field == true_field).mean())
    macro = float(f1_score(true_field, pred_field, average="macro"))
    print(f"\noracle-conditioned agreement: acc {acc:.4f}  macro-F1 {macro:.4f}")
    print(f"throughput {len(df)/elapsed:.1f} docs/s")

    print(f"\n{'field (WoS-derived)':38s} {'disc':4s} {'n':>5s} {'agree':>6s} "
          f"{'baseline':>8s}")
    per_field = {}
    for fld in fields_tested:
        m = true_field == fld
        d = df["discipline"].iloc[np.argmax(m)]
        rec = float((pred_field[m] == fld).mean())
        base = 1.0 / len(DISCIPLINE_FIELD_IDS[d])
        from crc.taxonomy.disciplines import DISCIPLINE_ABBR
        per_field[fld] = {"discipline": d, "n": int(m.sum()),
                          "agreement": round(rec, 4),
                          "chance_baseline": round(base, 4)}
        print(f"  {fld:36s} {DISCIPLINE_ABBR[d]:4s} {int(m.sum()):>5,} "
              f"{rec:>6.3f} {base:>8.3f}")

    # Per WoS area: where within the discipline does Agent 2 send each area?
    print(f"\n{'WoS area':26s} {'->field(target)':34s} {'n':>5s} {'agree':>6s}")
    per_area = {}
    for area, g in df.groupby("wos_area"):
        gp = pred_field[g.index.to_numpy()]
        target = WOS_AREA_TO_FIELD[area]
        rec = float((gp == target).mean())
        per_area[area] = {
            "target_field": target, "discipline": g["discipline"].iloc[0],
            "n": int(len(g)), "agreement": round(rec, 4),
            "pred_distribution": {k: int(v) for k, v in Counter(gp).items()},
        }
        top = ", ".join(f"{k[:22]}:{v}" for k, v in Counter(gp).most_common(2))
        print(f"  {area:24s} {target[:32]:34s} {len(g):>5,} {rec:>6.3f}   {top}")

    out = {
        "model": Path(args.model).name,
        "conditioning": "oracle (true discipline)",
        "n": int(len(df)),
        "fields_tested": fields_tested,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro, 4),
        "docs_per_second": round(len(df) / elapsed, 1),
        "per_field": per_field,
        "per_area": per_area,
        "note": "OOD probe; WoS area is a coarse proxy label, title-free text.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "wos_field_eval.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
