#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export SWEEP_ROOT="${SWEEP_ROOT:-data/exp2_prompt_sweep_v25}"
export VERSIONS_CSV="${VERSIONS_CSV:-v25_reflective_content_grounding}"
export FULL_REFERENCE_DIRS_CSV="${FULL_REFERENCE_DIRS_CSV:-data/exp2_v18_reflective_grounding/v18_reflective_grounding_joint_gate,data/exp2_prompt_sweep_v23_v24/v23_selected_style_joint_gate}"

exec bash "$PROJECT_ROOT/scripts/run_exp2_prompt_sweep_v11_v15.sh"
