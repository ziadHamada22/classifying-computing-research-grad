# Classifying Computing Research

A three-agent NLP system that reads a computing research paper (PDF, Word file or
pasted text) and answers three questions:

| Agent | Question | Answer space | Deployed model |
|---|---|---|---|
| 1 | Which discipline? | 6 Computing Curricula 2020 disciplines | `models/scibert-soft` |
| 2 | Which field? | 38 fields, restricted to the disciplines Agent 1 cannot rule out | `models/field-scibert-v2` |
| 3 | How was the research done? | 8 observable facets, research design derived by written rules | `models/methodology-facets-v2` |

All three are fine-tuned SciBERT models. Every answer comes with a calibrated
prediction set (conformal prediction), so a contested paper is reported as contested.
The system runs offline on a 6 GB laptop GPU, with no language model in the
classification path.

Final-year project, Project Idea 1: *Classifying Computing Research*.

## Headline results

| | Test | 2025+ papers |
|---|---:|---:|
| Agent 1 discipline macro-F1 | 0.811 | 0.734 |
| Agent 1 expected calibration error | 0.025 | 0.035 |
| Agent 2 field accuracy, true discipline given | 0.840 | 0.791 |
| Whole pipeline, field accuracy on predicted disciplines | 0.698 | 0.651 |
| Agent 3 macro AUC against its label model | 0.911 | |

Best off-the-shelf comparator (CSO Classifier): 0.534. Agent 3 has no gold labels and
is evaluated against text the model never reads and against arXiv metadata signals.

## Repository contents

```
src/crc/     the package: taxonomy, ingestion, the three agents, evaluation, CLI, web UI
models/      the three deployed models (weights, tokenizer, calibration files, saved predictions)
data/        the final labelled datasets each agent was trained and evaluated on
results/     one JSON file per experiment, plus figures
```

The model weights (`*.safetensors`) and the datasets (`*.parquet`) are stored with
**Git LFS**. Install Git LFS before cloning (`git lfs install`), or the large files
arrive as small pointer files. See `models/README.md` and `data/README.md`.

## Install

Python 3.12, CUDA GPU recommended (CPU works, slower).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
# all three agents on one document (PDF, DOCX, Markdown, text)
python -m crc.cli pipeline paper.pdf

# the web interface on http://localhost:8000
python -m crc.serve

# parsing and chunking only
python -m crc.cli inspect paper.pdf --show-chunks
```

The input guard refuses what it cannot read (an empty input, a scanned PDF with no
text layer, an unsupported file type) and warns on very short or non-English text.

## Reproducing the numbers

Every experiment's output is in `results/`. Each deployed model folder also holds its
saved predictions on the validation, test and 2025+ papers (`*_predictions.npz`), so
the evaluation scripts in `src/crc/eval/` recompute the reported figures without a GPU.

The training and data-building scripts were run with the datasets in
`C:\Users\ziada\gp_data\corpus`. To retrain, put the files from `data/` in that folder
or change the path constant at the top of the script:

```bash
python -m crc.agents.discipline.train_soft      # Agent 1
python -m crc.agents.field.train                # Agent 2
python -m crc.agents.methodology.train_facets   # Agent 3
```

The optional `--ensemble` mode of the CLI and web UI needs extra models that are not in
this repository; the default mode uses only the three deployed models.
