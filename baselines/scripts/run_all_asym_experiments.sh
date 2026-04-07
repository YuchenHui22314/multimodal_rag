#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/rech/zhangyan/multimodal_rag/baselines"
PYTHON_BIN="/data/rech/zhangyan/qilin/bin/python"
ACCELERATE_BIN="/data/rech/zhangyan/qilin/bin/accelerate"
LOG_DIR="$ROOT/logs"
TRITON_CACHE_DIR="$ROOT/triton_cache"
DOC_EMB_DIR="/part/01/Tmp/zhangyan/qilin_qwen3vl"
WANDB_API_KEY_VALUE='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'

mkdir -p "$LOG_DIR"

declare -a EXPERIMENTS=(
  "query_mlp_mlp|config/asym_query_mlp.yaml|train_asym_query_mlp_eval30_formal_20260406.log|"
  "query_mlp_glu|config/asym_query_mlp.yaml|train_asym_query_glu_eval30_formal_20260406.log|model.proj_type=glu"
  "doc_mlp_mlp|config/asym_doc_mlp.yaml|train_asym_doc_mlp_eval30_formal_20260406.log|"
  "doc_mlp_glu|config/asym_doc_mlp.yaml|train_asym_doc_glu_eval30_formal_20260406.log|model.proj_type=glu"
  "both_mlp_mlp|config/asym_both_mlp.yaml|train_asym_both_mlp_eval30_formal_20260406.log|"
  "both_mlp_glu|config/asym_both_mlp.yaml|train_asym_both_glu_eval30_formal_20260406.log|model.proj_type=glu"
  "lora|config/asym_lora.yaml|train_asym_lora_eval30_formal_20260406.log|"
  "fullft|config/asym_fullft.yaml|train_asym_fullft_eval30_formal_20260406.log|"
)

is_finished() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "Final model saved" "$log_file"
}

wait_for_completion() {
  local log_file="$1"
  local exp_name="$2"
  echo "[runner] Waiting for running experiment to finish: $exp_name"
  while true; do
    if is_finished "$log_file"; then
      echo "[runner] Experiment finished while waiting: $exp_name"
      return 0
    fi
    sleep 20
  done
}

run_experiment() {
  local exp_name="$1"
  local config_path="$2"
  local log_name="$3"
  local extra_override="$4"
  local log_path="$LOG_DIR/$log_name"

  if is_finished "$log_path"; then
    echo "[runner] Skipping completed experiment: $exp_name"
    return 0
  fi

  if [[ -f "$log_path" ]]; then
    wait_for_completion "$log_path" "$exp_name"
    return 0
  fi

  echo "[runner] Starting experiment: $exp_name"
  echo "[runner] Log file: $log_path"

  local -a cmd=(
    "$ACCELERATE_BIN" launch --num_processes 4
    src/train_asymmetric_biencoder.py
    --config "$config_path"
    --override
    "datasets.doc_emb_dir=$DOC_EMB_DIR"
    "log_dir=$LOG_DIR"
    "datasets.batch_size=128"
    "training.num_epochs=30"
  )

  if [[ -n "$extra_override" ]]; then
    cmd+=("$extra_override")
  fi

  (
    cd "$ROOT"
    export WANDB_API_KEY="$WANDB_API_KEY_VALUE"
    export TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    "${cmd[@]}"
  ) 2>&1 | tee -a "$log_path"

  if ! is_finished "$log_path"; then
    echo "[runner] Experiment did not finish cleanly: $exp_name" >&2
    return 1
  fi

  echo "[runner] Completed experiment: $exp_name"
}

for spec in "${EXPERIMENTS[@]}"; do
  IFS="|" read -r exp_name config_path log_name extra_override <<< "$spec"
  run_experiment "$exp_name" "$config_path" "$log_name" "$extra_override"
done

echo "[runner] All experiments processed."
