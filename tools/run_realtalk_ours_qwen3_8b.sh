#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${REALTALK_OURS_API_KEY:-}" ]]; then
  echo "REALTALK_OURS_API_KEY is required" >&2
  exit 2
fi

export REALTALK_OURS_BASE_URL="${REALTALK_OURS_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export REALTALK_OURS_MODEL="qwen3-8b"
export CUDA_VISIBLE_DEVICES=""

dataset_dir="${REALTALK_DATASET_DIR:-dataset}"
output_dir="${REALTALK_OURS_OUTPUT_DIR:-data/realtalk_ours_qwen3_8b}"

python -m src.experiments.realtalk_ours \
  --dataset-dir "$dataset_dir" \
  --output-dir "$output_dir" \
  "$@"
