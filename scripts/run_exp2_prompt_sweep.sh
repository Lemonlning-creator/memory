#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ASSET_SOURCE="${ASSET_SOURCE:-data/exp2_qwen_plus_v5_clean}"
SWEEP_ROOT="${SWEEP_ROOT:-data/exp2_prompt_sweep_v6_v10}"
CONFIG="${CONFIG:-config.qwen-plus.ini}"
DATASET_DIR="${DATASET_DIR:-dataset}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
CASE_SET="${CASE_SET:-all}"
JUDGE_CONFIG_SECTION="${JUDGE_CONFIG_SECTION:-EvaluationAPI}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
REUSE_REFERENCE_CACHE="${REUSE_REFERENCE_CACHE:-1}"

VERSIONS=(
  v10_balanced_surface_act
  v6_last_topic_plain
  v7_recent_style_imitation
  v8_frequency_hard_gate
  v9_evidence_bound_persona
)
REPORT_VERSIONS=(
  v6_last_topic_plain
  v7_recent_style_imitation
  v8_frequency_hard_gate
  v9_evidence_bound_persona
  v10_balanced_surface_act
)

CASES=()
if [[ -n "${CASE_LIST:-}" ]]; then
  IFS=',' read -r -a CASES <<< "$CASE_LIST"
else
  case "$CASE_SET" in
    fast2)
      CASES=(
        Chat_2_Kevin_Elise.json
        Chat_10_Fahim_Muhhamed.json
      )
      ;;
    balanced3)
      CASES=(
        Chat_2_Kevin_Elise.json
        Chat_5_Nicolas_Nebraas.json
        Chat_10_Fahim_Muhhamed.json
      )
      ;;
    all)
      CASES=()
      ;;
    *)
      echo "Unknown CASE_SET=$CASE_SET; use fast2, balanced3, all, or CASE_LIST=a.json,b.json" >&2
      exit 2
      ;;
  esac
fi

if [[ ! -d "$ASSET_SOURCE/cases" ]]; then
  echo "Prepared asset source not found: $ASSET_SOURCE/cases" >&2
  echo "Point ASSET_SOURCE to the completed V5 directory containing current fixed personas and profiles." >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 2
fi

mkdir -p "$SWEEP_ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

CASE_ARGS=()
for case_name in "${CASES[@]}"; do
  CASE_ARGS+=(--case "$case_name")
done

REFERENCE_CACHE_ARGS=()
if [[ "$REUSE_REFERENCE_CACHE" == "1" ]]; then
  REFERENCE_CACHE_ARGS+=(--reuse-reference-cache)
fi

echo "Experiment 2 prompt sweep"
echo "  asset source : $ASSET_SOURCE"
echo "  sweep root  : $SWEEP_ROOT"
echo "  case set    : $CASE_SET"
echo "  cases       : ${CASES[*]:-all 10 conversations}"
echo "  versions    : ${VERSIONS[*]}"
echo "  judge model : $JUDGE_MODEL"
echo "  eval device : $EVAL_DEVICE"
echo "  ref cache   : $REUSE_REFERENCE_CACHE"

failures=()
for version in "${VERSIONS[@]}"; do
  target_dir="$SWEEP_ROOT/$version"
  log_path="$SWEEP_ROOT/${version}.log"
  mkdir -p "$target_dir"

  echo
  echo "[$(date --iso-8601=seconds)] Preparing reusable assets for $version"
  if ! uv run --no-sync python -u -m src.experiments.exp2_prompt_sweep \
    clone-assets \
    --source-dir "$ASSET_SOURCE" \
    --target-dir "$target_dir" \
    --dataset-dir "$DATASET_DIR" \
    --train-ratio "$TRAIN_RATIO" \
    "${CASE_ARGS[@]}" \
    "${REFERENCE_CACHE_ARGS[@]}"; then
    echo "[$version] asset preparation failed; continuing with the next version" >&2
    failures+=("$version:assets")
    continue
  fi

  echo "[$(date --iso-8601=seconds)] Generating and evaluating $version"
  command=(
    uv run --no-sync python -u -m src.experiments.exp2_user_modeling
    --phase generate-evaluate
    --dataset-dir "$DATASET_DIR"
    --train-ratio "$TRAIN_RATIO"
    --config "$CONFIG"
    --prompt-version "$version"
    --output-dir "$target_dir"
    --judge-config-section "$JUDGE_CONFIG_SECTION"
    --eval-device "$EVAL_DEVICE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
  )
  if [[ -n "$JUDGE_MODEL" ]]; then
    command+=(--judge-model "$JUDGE_MODEL")
  fi
  command+=("${CASE_ARGS[@]}")

  if "${command[@]}" 2>&1 | tee -a "$log_path"; then
    echo "[$(date --iso-8601=seconds)] Completed $version"
  else
    echo "[$version] generation/evaluation failed; saved progress will resume on rerun" >&2
    failures+=("$version:run")
  fi

  summary_command=(
    uv run --no-sync python -u -m src.experiments.exp2_prompt_sweep
    summarize
    --sweep-root "$SWEEP_ROOT"
    --baseline-dir "$ASSET_SOURCE"
    --dataset-dir "$DATASET_DIR"
    --train-ratio "$TRAIN_RATIO"
  )
  for sweep_version in "${REPORT_VERSIONS[@]}"; do
    summary_command+=(--version "$sweep_version")
  done
  summary_command+=("${CASE_ARGS[@]}")
  "${summary_command[@]}" || true
done

summary_command=(
  uv run --no-sync python -u -m src.experiments.exp2_prompt_sweep
  summarize
  --sweep-root "$SWEEP_ROOT"
  --baseline-dir "$ASSET_SOURCE"
  --dataset-dir "$DATASET_DIR"
  --train-ratio "$TRAIN_RATIO"
)
for version in "${REPORT_VERSIONS[@]}"; do
  summary_command+=(--version "$version")
done
summary_command+=("${CASE_ARGS[@]}")
"${summary_command[@]}"

echo
echo "Sweep report: $SWEEP_ROOT/prompt_sweep_summary.md"
echo "Machine-readable report: $SWEEP_ROOT/prompt_sweep_summary.json"
if (( ${#failures[@]} > 0 )); then
  echo "Incomplete stages: ${failures[*]}" >&2
  echo "Run this same command again to resume only missing work." >&2
  exit 1
fi
echo "All prompt variants completed successfully."
