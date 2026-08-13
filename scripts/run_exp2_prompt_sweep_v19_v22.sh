#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export SWEEP_ROOT="${SWEEP_ROOT:-data/exp2_prompt_sweep_v19_v22}"
export VERSIONS_CSV="${VERSIONS_CSV:-v19_reflective_trigger_recall,v20_grounding_specificity_gate,v21_independent_act_decisions,v22_recent_act_imitation}"

exec bash "$PROJECT_ROOT/scripts/run_exp2_prompt_sweep_v11_v15.sh"
