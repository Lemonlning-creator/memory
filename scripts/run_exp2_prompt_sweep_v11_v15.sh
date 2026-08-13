#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Prepared assets remain immutable and come from the completed V5 run. The
# comparison baseline is V7 because it was the strongest V6-V10 variant.
ASSET_SOURCE="${ASSET_SOURCE:-data/exp2_qwen_plus_v5_clean}"
BASELINE_DIR="${BASELINE_DIR:-data/exp2_prompt_sweep_v6_v10/v7_recent_style_imitation}"
# Explicit historical best keeps the report comparison auditable. Update this
# path only after a completed full-run version is accepted as the new best.
BEST_FULL_DIR="${BEST_FULL_DIR:-data/exp2_v16_full/v16_v7_selective_followup}"
SWEEP_ROOT="${SWEEP_ROOT:-data/exp2_prompt_sweep_v11_v15_directed}"
CONFIG="${CONFIG:-config.qwen-plus.ini}"
DATASET_DIR="${DATASET_DIR:-dataset}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
CASE_SET="${CASE_SET:-diagnostic3}"
JUDGE_CONFIG_SECTION="${JUDGE_CONFIG_SECTION:-EvaluationAPI}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GENERATE_WORKERS="${GENERATE_WORKERS:-3}"
JUDGE_WORKERS="${JUDGE_WORKERS:-6}"
REUSE_REFERENCE_CACHE="${REUSE_REFERENCE_CACHE:-1}"
UV_BIN="${UV_BIN:-uv}"
VERSIONS_CSV="${VERSIONS_CSV:-}"
FULL_REFERENCE_DIRS_CSV="${FULL_REFERENCE_DIRS_CSV:-}"

# Historical default sweep. Newer wrappers set VERSIONS_CSV explicitly.
DEFAULT_VERSIONS=(
  v11_lexical_fidelity
  v12_reflective_placement
  v13_grounding_precision
  v14_emotion_calibration
  v15_metric_integrated
)
if [[ -n "$VERSIONS_CSV" ]]; then
  IFS=',' read -r -a VERSIONS <<< "$VERSIONS_CSV"
else
  VERSIONS=("${DEFAULT_VERSIONS[@]}")
fi

FULL_REFERENCE_DIRS=()
if [[ -n "$FULL_REFERENCE_DIRS_CSV" ]]; then
  IFS=',' read -r -a FULL_REFERENCE_DIRS <<< "$FULL_REFERENCE_DIRS_CSV"
fi
if (( ${#VERSIONS[@]} == 0 )); then
  echo "No prompt versions selected; set VERSIONS_CSV to a comma-separated list" >&2
  exit 2
fi

CASES=()
if [[ -n "${CASE_LIST:-}" ]]; then
  IFS=',' read -r -a CASES <<< "$CASE_LIST"
else
  case "$CASE_SET" in
    diagnostic3)
      # Selected from the full V7 audit: 24 replies whose four weak metrics,
      # question/emoji rates, and EI false-positive rates closely track the
      # complete 117-reply run while retaining three distinct target speakers.
      CASES=(
        Chat_1_Emi_Elise.json
        Chat_4_Emi_Paola.json
        Chat_9_Fahim_Akib.json
      )
      ;;
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
      echo "Unknown CASE_SET=$CASE_SET; use diagnostic3, fast2, balanced3, all, or CASE_LIST=a.json,b.json" >&2
      exit 2
      ;;
  esac
fi

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
  else
    echo "uv executable not found; set UV_BIN to its absolute path" >&2
    exit 2
  fi
fi
if [[ ! -d "$ASSET_SOURCE/cases" ]]; then
  echo "Prepared asset source not found: $ASSET_SOURCE/cases" >&2
  exit 2
fi
if [[ ! -f "$BASELINE_DIR/evaluation/table2_main_results.json" ]]; then
  echo "Completed V7 baseline not found: $BASELINE_DIR/evaluation/table2_main_results.json" >&2
  exit 2
fi
if [[ ! -f "$BEST_FULL_DIR/evaluation/table2_main_results.json" ]]; then
  echo "Completed post-V7 best result not found: $BEST_FULL_DIR/evaluation/table2_main_results.json" >&2
  echo "Set BEST_FULL_DIR to the accepted historical full-run best directory." >&2
  exit 2
fi
for reference_dir in "${FULL_REFERENCE_DIRS[@]}"; do
  if [[ ! -f "$reference_dir/evaluation/table2_main_results.json" ]]; then
    echo "Completed full-run metric reference not found: $reference_dir/evaluation/table2_main_results.json" >&2
    exit 2
  fi
done
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

FULL_REFERENCE_ARGS=()
for reference_dir in "${FULL_REFERENCE_DIRS[@]}"; do
  FULL_REFERENCE_ARGS+=(--reference-dir "$reference_dir")
done

echo "Experiment 2 prompt sweep"
echo "  asset source : $ASSET_SOURCE"
echo "  V7 baseline  : $BASELINE_DIR"
echo "  post-V7 best : $BEST_FULL_DIR"
echo "  full refs    : ${FULL_REFERENCE_DIRS[*]:-none}"
echo "  sweep root   : $SWEEP_ROOT"
echo "  case set     : $CASE_SET"
echo "  cases        : ${CASES[*]:-all 10 conversations}"
echo "  versions     : ${VERSIONS[*]}"
echo "  candidate    : [API].model in $CONFIG"
echo "  gen workers  : $GENERATE_WORKERS"
echo "  judge model  : $JUDGE_MODEL"
echo "  judge workers: $JUDGE_WORKERS"
echo "  eval device  : $EVAL_DEVICE"

failures=()
completed_versions=()
for version in "${VERSIONS[@]}"; do
  target_dir="$SWEEP_ROOT/$version"
  log_path="$SWEEP_ROOT/${version}.log"
  mkdir -p "$target_dir"

  echo
  echo "[$(date --iso-8601=seconds)] Preparing reusable assets for $version"
  if ! "$UV_BIN" run --no-sync python -u -m src.experiments.exp2_prompt_sweep \
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
    "$UV_BIN" run --no-sync python -u -m src.experiments.exp2_user_modeling
    --phase generate-evaluate
    --dataset-dir "$DATASET_DIR"
    --train-ratio "$TRAIN_RATIO"
    --config "$CONFIG"
    --prompt-version "$version"
    --output-dir "$target_dir"
    --judge-config-section "$JUDGE_CONFIG_SECTION"
    --generate-workers "$GENERATE_WORKERS"
    --eval-device "$EVAL_DEVICE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --judge-workers "$JUDGE_WORKERS"
  )
  if [[ -n "$JUDGE_MODEL" ]]; then
    command+=(--judge-model "$JUDGE_MODEL")
  fi
  command+=("${CASE_ARGS[@]}")

  if "${command[@]}" 2>&1 | tee -a "$log_path"; then
    echo "[$(date --iso-8601=seconds)] Completed $version"
  else
    echo "[$version] generation/evaluation failed; rerun the same command to resume" >&2
    failures+=("$version:run")
  fi

  if [[ -s "$target_dir/evaluation/table2_main_results.json" ]]; then
    completed_versions+=("$version")
  fi

  if (( ${#completed_versions[@]} > 0 )); then
    summary_command=(
      "$UV_BIN" run --no-sync python -u -m src.experiments.exp2_prompt_sweep
      summarize
      --sweep-root "$SWEEP_ROOT"
      --baseline-dir "$BASELINE_DIR"
      --best-dir "$BEST_FULL_DIR"
      --dataset-dir "$DATASET_DIR"
      --train-ratio "$TRAIN_RATIO"
    )
    summary_command+=("${FULL_REFERENCE_ARGS[@]}")
    for completed_version in "${completed_versions[@]}"; do
      summary_command+=(--version "$completed_version")
    done
    summary_command+=("${CASE_ARGS[@]}")
    "${summary_command[@]}" || true
  fi
done

if (( ${#completed_versions[@]} > 0 )); then
  summary_command=(
    "$UV_BIN" run --no-sync python -u -m src.experiments.exp2_prompt_sweep
    summarize
    --sweep-root "$SWEEP_ROOT"
    --baseline-dir "$BASELINE_DIR"
    --best-dir "$BEST_FULL_DIR"
    --dataset-dir "$DATASET_DIR"
    --train-ratio "$TRAIN_RATIO"
  )
  summary_command+=("${FULL_REFERENCE_ARGS[@]}")
  for completed_version in "${completed_versions[@]}"; do
    summary_command+=(--version "$completed_version")
  done
  summary_command+=("${CASE_ARGS[@]}")
  "${summary_command[@]}"
else
  echo "No completed prompt versions are available for summary generation." >&2
fi

echo
echo "Sweep report: $SWEEP_ROOT/prompt_sweep_summary.md"
echo "Machine-readable report: $SWEEP_ROOT/prompt_sweep_summary.json"
if (( ${#failures[@]} > 0 )); then
  echo "Incomplete stages: ${failures[*]}" >&2
  echo "Run this same command again to resume only missing work." >&2
  exit 1
fi
echo "All selected prompt variants completed successfully."
