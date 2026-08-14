#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SOURCE_DIR="${SOURCE_DIR:-data/exp2_v18_reflective_grounding/v18_reflective_grounding_joint_gate}"
OUTPUT_DIR="${OUTPUT_DIR:-data/exp2_v28_controlled}"
CONFIG="${CONFIG:-config.qwen-plus.ini}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_CONFIG_SECTION="${JUDGE_CONFIG_SECTION:-EvaluationAPI}"
GENERATE_WORKERS="${GENERATE_WORKERS:-3}"
JUDGE_WORKERS="${JUDGE_WORKERS:-6}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python -u -m src.experiments.exp2_controlled_state_ablation \
  --phase all \
  --source-dir "$SOURCE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --source-prompt-version v18_reflective_grounding_joint_gate \
  --response-prompt-version v28_repair_grounding_only \
  --conditions scores_only \
  --train-ratio 0.9 \
  --config "$CONFIG" \
  --temperature 0 \
  --generate-workers "$GENERATE_WORKERS" \
  --judge-config-section "$JUDGE_CONFIG_SECTION" \
  --judge-model "$JUDGE_MODEL" \
  --judge-workers "$JUDGE_WORKERS" \
  --eval-device "$EVAL_DEVICE" \
  --eval-batch-size "$EVAL_BATCH_SIZE"
