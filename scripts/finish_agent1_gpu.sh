#!/usr/bin/env bash
# GPU jobs that finish Agent 1 (2026-09-25), run back to back so timings are
# measured on an otherwise idle GPU. Each step logs to gp_data/logs/finish_*.log.
# Run from the system/ directory:  bash scripts/finish_agent1_gpu.sh
set -u
PY=".venv/Scripts/python.exe"
LOG="/c/Users/ziada/gp_data/logs"
CHUNKS="C:/Users/ziada/gp_data/corpus/chunks_eval.parquet"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

step() {  # step <name> <command...>
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] start $name"
  "$@" > "$LOG/finish_$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $name (exit $?)"
}

# 1. the deployed model through the standard evaluator (co-listed, latency)
step eval_soft $PY -m crc.eval.evaluate --model-dir models/scibert-soft

# 2. whole-document evaluation, inference-faithful (title prefix), deployed model
step docs_soft $PY -m crc.eval.evaluate_documents --model-dir models/scibert-soft \
     --chunks "$CHUNKS" --max-val-papers 100000

# 3. retrieval baselines again, now timing the encoder on an idle GPU
step retrieval $PY -m crc.agents.discipline.retrieval

# 4. canonical end-to-end numbers (now with hierarchical F1)
step pipeline_scibert $PY -m crc.eval.evaluate_pipeline --discipline-dir models/scibert \
     --out results/pipeline_eval.json
step pipeline_scibert_deployed $PY -m crc.eval.evaluate_pipeline --discipline-dir models/scibert \
     --use-deployed-bank --out results/pipeline_eval_deployed_scibert.json
step pipeline_soft_deployed $PY -m crc.eval.evaluate_pipeline --discipline-dir models/scibert-soft \
     --use-deployed-bank --out results/pipeline_eval_deployed.json

# 5. the old model's whole-document number, re-measured inference-faithfully
#    (its title-less originals are kept beside it)
cp -n results/eval_documents_scibert.json results/eval_documents_scibert_notitle.json
cp -n models/scibert/section_weights.json models/scibert/section_weights_notitle.json
step docs_scibert $PY -m crc.eval.evaluate_documents --model-dir models/scibert \
     --chunks "$CHUNKS" --max-val-papers 100000

# 6. control: the same ambiguous papers with hard consensus labels
step hard_control $PY -m crc.agents.discipline.train_soft --hard

echo "[$(date +%H:%M:%S)] ALL DONE"
