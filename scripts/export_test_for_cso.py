"""Export a stratified test subset (id, title, abstract) to JSON so the CSO
Classifier can be run in the isolated .venv-baselines (which has no pandas).
"""
import json
from pathlib import Path
import pandas as pd

WORK = Path(r"C:\Users\ziada\gp_data")
OUT = WORK / "corpus" / "cso_input.json"
PER_DISCIPLINE = 250

corpus = pd.read_parquet(WORK / "corpus" / "corpus_v2.parquet")
test = corpus[corpus["split"] == "test"]
# unambiguous, stratified — a corroborating estimate, not the headline
clean = test[~test["ambiguous"].astype(bool)]
parts = [g.sample(n=min(PER_DISCIPLINE, len(g)), random_state=11)
         for _, g in clean.groupby("discipline")]
sub = pd.concat(parts, ignore_index=True)
rows = [{"id": str(r["id"]), "title": str(r["title"]), "abstract": str(r["abstract"])}
        for _, r in sub.iterrows()]
OUT.write_text(json.dumps(rows))
print(f"exported {len(rows)} papers -> {OUT}")
print(sub["discipline"].value_counts().to_string())
