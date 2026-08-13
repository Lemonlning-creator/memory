from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Sequence

from ..llm_client import LLMClient
from ..profile_utils import flatten_static_profile
from ..prompts.exp2_versions import get_exp2_prompt_bundle
from ..utils import load_json, save_json
from .exp2_user_modeling import (
    TABLE2_BASELINES,
    TABLE2_METRICS,
    CasePaths,
    ExperimentCase,
    JsonlStore,
    _progress_line,
    _select_cases,
    aggregate_table2_scores,
    build_cases,
    evaluate_table2,
    read_jsonl,
    save_split_manifest,
)


CONTROLLED_PROTOCOL_VERSION = "exp2_controlled_previous_state_ablation_v1"
DEFAULT_SOURCE_PROMPT_VERSION = "v18_reflective_grounding_joint_gate"
STATE_SCORE_FIELDS = (
    "emotional_reaction",
    "interpretation",
    "exploration",
)
CONDITIONS = (
    "full_state",
    "scores_only",
    "no_state",
    "scores_plus_tone",
)
HIGHER_IS_BETTER = {
    "lexical",
    "semantic",
    "reflective",
    "grounding",
    "sentiment",
    "emotion",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_state(
    previous_empathy_state: Dict[str, Any],
    condition: str,
) -> Dict[str, Any]:
    """Return the only response input that differs across ablation conditions."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown state condition {condition!r}; choose from {CONDITIONS}")
    if not isinstance(previous_empathy_state, dict):
        raise ValueError("source previous_empathy_state must be an object")
    if condition == "full_state":
        return deepcopy(previous_empathy_state)
    if condition == "no_state" or not previous_empathy_state:
        return {}
    if condition == "scores_plus_tone":
        # This condition is intentionally constructed upward from scores_only,
        # not downward from full_state. The abstract tone control is added only
        # after the three numeric fields pass the same strict check.
        selected = _selected_state(previous_empathy_state, "scores_only")
        value = previous_empathy_state.get("activated_tone")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "source previous_empathy_state.activated_tone must be a non-empty "
                "string for scores_plus_tone"
            )
        selected["activated_tone"] = value
        return selected

    selected: Dict[str, Any] = {}
    for field in STATE_SCORE_FIELDS:
        value = previous_empathy_state.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"source previous_empathy_state.{field} must be numeric for scores_only"
            )
        selected[field] = value
    return selected


def _format_static_profile(profile: Dict[str, Any]) -> str:
    """Reproduce StateDrivenCompanionAgent._prompt_context without constructing it."""
    flat_profile = flatten_static_profile(profile)
    profile_lines: list[str] = []
    for _, attrs in flat_profile.items():
        if not isinstance(attrs, dict):
            continue
        for key, value in attrs.items():
            if value:
                profile_lines.append(f"- {key}: {value}")
    return (
        "\n".join(profile_lines)
        if profile_lines
        else json.dumps(flat_profile, ensure_ascii=False)
    )


def _reconstruct_preceding_states(
    predictions: Sequence[Dict[str, Any]],
    understanding_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Map each reply to the state that existed immediately before that reply."""
    state_by_id = {
        str(row["example_id"]): row
        for row in understanding_rows
        if "example_id" in row
    }
    prediction_ids = [str(row["example_id"]) for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise RuntimeError("source predictions contain duplicate example_id values")
    if set(prediction_ids) != set(state_by_id):
        raise RuntimeError(
            "source predictions and user-understanding rows do not contain the same IDs"
        )

    result: Dict[str, Dict[str, Any]] = {}
    for index, example_id in enumerate(prediction_ids):
        if index == 0:
            result[example_id] = {}
            continue
        previous_id = prediction_ids[index - 1]
        previous_state = state_by_id[previous_id].get("core_current_state")
        if not isinstance(previous_state, dict):
            raise RuntimeError(
                f"source state {previous_id} is missing object core_current_state"
            )
        result[example_id] = deepcopy(previous_state)
    return result


def _validate_source_rows(
    case: ExperimentCase,
    paths: CasePaths,
    prompt_version: str,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    required = (
        paths.profile,
        paths.persona,
        paths.runtime_profile,
        paths.predictions,
        paths.understanding,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"controlled source is incomplete for {case.case_id}: {', '.join(missing)}"
        )

    bundle = get_exp2_prompt_bundle(prompt_version)
    predictions = list(read_jsonl(paths.predictions))
    states = list(read_jsonl(paths.understanding))
    if not predictions:
        raise RuntimeError(f"controlled source has no predictions for {case.case_id}")
    mismatched = [
        row
        for row in predictions
        if row.get("prompt_version") != bundle.version
        or row.get("prompt_sha256") != bundle.fingerprint
    ]
    if mismatched:
        raise RuntimeError(
            f"source predictions for {case.case_id} are not {bundle.version}; "
            "choose the actual completed source run"
        )
    for row in predictions:
        audit = row.get("generation_input_audit")
        if not isinstance(audit, dict):
            raise RuntimeError(
                f"source prediction {row.get('example_id')} has no input audit"
            )
        if not isinstance(audit.get("previous_empathy_state"), dict):
            raise RuntimeError(
                f"source prediction {row.get('example_id')} has no previous state"
            )
        if not isinstance(audit.get("relevant_memory"), dict):
            raise RuntimeError(
                f"source prediction {row.get('example_id')} has no relevant memory"
            )
    _reconstruct_preceding_states(predictions, states)
    return predictions, states


def _source_file_manifest(
    source_root: Path,
    cases: Sequence[ExperimentCase],
    prompt_version: str,
) -> Dict[str, Any]:
    files: Dict[str, Any] = {}
    for case in cases:
        paths = CasePaths.for_case(source_root, case)
        predictions, states = _validate_source_rows(case, paths, prompt_version)
        files[case.case_id] = {
            "predictions_sha256": _file_sha256(paths.predictions),
            "understanding_sha256": _file_sha256(paths.understanding),
            "profile_sha256": _file_sha256(paths.profile),
            "persona_sha256": _file_sha256(paths.persona),
            "example_count": len(predictions),
            "state_count": len(states),
        }
    return files


def _condition_manifest(
    *,
    condition: str,
    source_root: Path,
    cases: Sequence[ExperimentCase],
    source_files: Dict[str, Any],
    prompt_version: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    bundle = get_exp2_prompt_bundle(prompt_version)
    return {
        "protocol_version": CONTROLLED_PROTOCOL_VERSION,
        "condition": condition,
        "state_payload": {
            "full_state": "exact previous_empathy_state saved by the source run",
            "scores_only": list(STATE_SCORE_FIELDS),
            "no_state": "empty object",
            "scores_plus_tone": (
                "scores_only plus activated_tone; response_guidance excluded"
            ),
        }[condition],
        "source_root": str(source_root),
        "source_prompt_version": bundle.version,
        "source_prompt_sha256": bundle.fingerprint,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "history_policy": "reuse exact source generation_input_audit.relevant_memory",
        "alignment_policy": "disabled; replay saved source trajectory",
        "current_state_policy": (
            "empty at the first target reply, then the source run's preceding "
            "core_current_state"
        ),
        "cases": [case.case_id for case in cases],
        "source_files": source_files,
    }


def _ensure_condition_manifest(root: Path, expected: Dict[str, Any]) -> None:
    path = root / "controlled_condition_manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = load_json(str(path))
        if existing != expected:
            raise RuntimeError(
                f"existing controlled condition has different inputs: {path}; "
                "use a new --output-dir"
            )
        return
    existing_predictions = list(root.glob("cases/*/generations/predictions.jsonl"))
    if existing_predictions:
        raise RuntimeError(
            f"predictions exist without a controlled manifest under {root}; "
            "use a new --output-dir"
        )
    save_json(str(path), expected)


def _copy_reference_annotations(source: CasePaths, target: CasePaths) -> int:
    """Reuse only reference labels whose cache keys are bound to exact content."""
    if not source.table2_annotations.is_file():
        return 0
    target.table2_annotations.parent.mkdir(parents=True, exist_ok=True)
    target_store = JsonlStore(target.table2_annotations, id_field="annotation_id")
    copied = 0
    for row in read_jsonl(source.table2_annotations):
        annotation_id = str(row.get("annotation_id") or "")
        if (
            row.get("variant") != "reference"
            or not annotation_id
            or not isinstance(row.get("labels"), dict)
            or not row.get("candidate_sha256")
            or not row.get("context_sha256")
        ):
            continue
        if not target_store.contains(annotation_id):
            target_store.append(deepcopy(row))
            copied += 1
    return copied


def _build_response_prompt(
    *,
    bundle: Any,
    prediction: Dict[str, Any],
    profile_text: str,
    persona: Dict[str, Any],
    current_state: Dict[str, Any],
    current_context: Dict[str, Any],
    condition: str,
) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    audit = prediction["generation_input_audit"]
    source_previous_state = deepcopy(audit["previous_empathy_state"])
    state_payload = _selected_state(source_previous_state, condition)
    relevant_memory = deepcopy(audit["relevant_memory"])
    context = {
        "user_input": str(prediction["user_message"]),
        "static_profile": profile_text,
        "current_state": json.dumps(current_state, ensure_ascii=False),
        "current_context": json.dumps(current_context, ensure_ascii=False),
        "persona_config": json.dumps(persona, ensure_ascii=False),
        "relevant_memory": json.dumps(relevant_memory, ensure_ascii=False),
        "previous_empathy_state": json.dumps(state_payload, ensure_ascii=False),
    }
    response_prompt = bundle.response_user.format(**context)
    frozen_input = {
        "user_input": prediction["user_message"],
        "static_profile": profile_text,
        "current_state": current_state,
        "current_context": current_context,
        "persona": persona,
        "relevant_memory": relevant_memory,
        "source_previous_empathy_state": source_previous_state,
        "response_system_sha256": hashlib.sha256(
            bundle.response_system.encode("utf-8")
        ).hexdigest(),
        "response_user_template_sha256": hashlib.sha256(
            bundle.response_user.encode("utf-8")
        ).hexdigest(),
    }
    return response_prompt, frozen_input, state_payload


def _validate_existing_predictions(
    rows: Sequence[Dict[str, Any]],
    source_by_id: Dict[str, Dict[str, Any]],
    condition: str,
    prompt_version: str,
    prompt_sha256: str,
) -> None:
    for row in rows:
        example_id = str(row.get("example_id") or "")
        if example_id not in source_by_id:
            raise RuntimeError(f"unknown resumed example in {condition}: {example_id}")
        if (
            row.get("controlled_protocol_version") != CONTROLLED_PROTOCOL_VERSION
            or row.get("state_condition") != condition
            or row.get("prompt_version") != prompt_version
            or row.get("prompt_sha256") != prompt_sha256
        ):
            raise RuntimeError(
                f"resumed prediction {example_id} does not match condition {condition}; "
                "use a new --output-dir"
            )


def _generate_case(
    *,
    case: ExperimentCase,
    source_root: Path,
    target_root: Path,
    config_path: str,
    prompt_version: str,
    condition: str,
    temperature: float,
    max_tokens: int,
    progress: Dict[str, int],
    progress_lock: Lock,
) -> tuple[str, int]:
    bundle = get_exp2_prompt_bundle(prompt_version)
    source_paths = CasePaths.for_case(source_root, case)
    target_paths = CasePaths.for_case(target_root, case)
    target_paths.ensure_parents()
    predictions, states = _validate_source_rows(case, source_paths, prompt_version)
    source_by_id = {str(row["example_id"]): row for row in predictions}
    preceding_states = _reconstruct_preceding_states(predictions, states)

    profile = load_json(str(source_paths.profile))
    persona = load_json(str(source_paths.persona))
    runtime_profile = load_json(str(source_paths.runtime_profile))
    current_context = runtime_profile.get("context_axis", {})
    if not isinstance(current_context, dict):
        raise RuntimeError(f"source context_axis is not an object for {case.case_id}")
    profile_text = _format_static_profile(profile)

    output = JsonlStore(target_paths.predictions)
    existing = output.read_all()
    _validate_existing_predictions(
        existing,
        source_by_id,
        condition,
        bundle.version,
        bundle.fingerprint,
    )
    llm = LLMClient(config_path)
    generated = 0
    for prediction in predictions:
        example_id = str(prediction["example_id"])
        if output.contains(example_id):
            continue
        response_prompt, frozen_input, state_payload = _build_response_prompt(
            bundle=bundle,
            prediction=prediction,
            profile_text=profile_text,
            persona=persona,
            current_state=preceding_states[example_id],
            current_context=current_context,
            condition=condition,
        )
        generated_reply = llm.chat(
            bundle.response_system,
            response_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ).strip()
        if not generated_reply:
            raise RuntimeError(f"empty generated reply for {example_id}")
        created_at = datetime.now(timezone.utc).isoformat()
        output.append({
            **{
                key: deepcopy(value)
                for key, value in prediction.items()
                if key
                not in {
                    "generated_reply",
                    "generation_input_audit",
                    "model_timing",
                    "created_at",
                }
            },
            "generated_reply": generated_reply,
            "protocol_version": prediction.get("protocol_version"),
            "controlled_protocol_version": CONTROLLED_PROTOCOL_VERSION,
            "generation_policy": "frozen_source_trajectory_response_only",
            "prompt_version": bundle.version,
            "prompt_sha256": bundle.fingerprint,
            "state_condition": condition,
            "generation_temperature": temperature,
            "generation_max_tokens": max_tokens,
            "generation_input_audit": {
                "contains_user_message": True,
                "contains_reference_reply": False,
                "contains_next_user_message": False,
                "relevant_memory": deepcopy(
                    prediction["generation_input_audit"]["relevant_memory"]
                ),
                "source_previous_empathy_state": deepcopy(
                    prediction["generation_input_audit"]["previous_empathy_state"]
                ),
                "previous_empathy_state": deepcopy(state_payload),
                "preceding_current_state": deepcopy(preceding_states[example_id]),
                "frozen_input_sha256": _value_sha256(frozen_input),
                "actual_state_payload_sha256": _value_sha256(state_payload),
                "actual_response_prompt_sha256": hashlib.sha256(
                    response_prompt.encode("utf-8")
                ).hexdigest(),
            },
            "source_prediction_sha256": _value_sha256(prediction),
            "created_at": created_at,
        })
        generated += 1
        with progress_lock:
            progress["completed"] += 1
            print(
                _progress_line(
                    f"Generate {condition}",
                    progress["completed"],
                    progress["total"],
                    detail=case.case_id,
                ),
                flush=True,
            )

    if target_paths.table2_annotations.exists() is False:
        _copy_reference_annotations(source_paths, target_paths)
    return case.case_id, generated


def generate_condition(
    *,
    cases: Sequence[ExperimentCase],
    source_root: Path,
    target_root: Path,
    config_path: str,
    prompt_version: str,
    condition: str,
    temperature: float,
    max_tokens: int,
    workers: int,
) -> Dict[str, int]:
    total = 0
    completed = 0
    for case in cases:
        source_paths = CasePaths.for_case(source_root, case)
        rows = list(read_jsonl(source_paths.predictions))
        total += len(rows)
        target_rows = list(read_jsonl(CasePaths.for_case(target_root, case).predictions))
        completed += len(target_rows)
    progress = {"completed": completed, "total": total}
    progress_lock = Lock()
    print(_progress_line(f"Generate {condition}", completed, total), flush=True)

    results: Dict[str, int] = {}
    worker_count = max(1, min(workers, len(cases)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _generate_case,
                case=case,
                source_root=source_root,
                target_root=target_root,
                config_path=config_path,
                prompt_version=prompt_version,
                condition=condition,
                temperature=temperature,
                max_tokens=max_tokens,
                progress=progress,
                progress_lock=progress_lock,
            ): case.case_id
            for case in cases
        }
        for future in as_completed(futures):
            case_id, count = future.result()
            results[case_id] = count
    return results


def _load_metrics(root: Path) -> Dict[str, Any] | None:
    path = root / "evaluation" / "table2_main_results.json"
    if not path.is_file():
        return None
    data = load_json(str(path))
    return {
        "example_count": data.get("example_count"),
        "speaker_count": data.get("speaker_count"),
        "metrics": data.get("ours", {}),
    }


def _load_metrics_for_cases(
    root: Path,
    cases: Sequence[ExperimentCase],
) -> Dict[str, Any] | None:
    """Load or re-aggregate a completed run over exactly the selected cases."""
    all_scores: list[Dict[str, Any]] = []
    for case in cases:
        path = CasePaths.for_case(root, case).table2_scores
        if not path.is_file():
            return None
        all_scores.extend(load_json(str(path)).get("scores", []))
    if not all_scores:
        return None
    aggregate = aggregate_table2_scores(all_scores)
    return {
        "example_count": aggregate["example_count"],
        "speaker_count": aggregate["speaker_count"],
        "metrics": aggregate["ours"],
    }


def _stat(metrics: Dict[str, Any], metric: str) -> str:
    value = metrics[metric]
    return f"{float(value['mean']):.4f} +/- {float(value['std']):.4f}"


def _load_case_scores(root: Path, cases: Sequence[ExperimentCase]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for case in cases:
        path = CasePaths.for_case(root, case).table2_scores
        if not path.is_file():
            return {}
        for row in load_json(str(path)).get("scores", []):
            result[str(row["example_id"])] = {
                metric: float(row[metric]) for metric in TABLE2_METRICS
            }
    return result


def _paired_comparison(
    left: Dict[str, Dict[str, float]],
    right: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError("condition score files do not contain identical example IDs")
    result: Dict[str, Any] = {}
    for metric in TABLE2_METRICS:
        wins = losses = ties = 0
        deltas: list[float] = []
        direction = 1.0 if metric in HIGHER_IS_BETTER else -1.0
        for example_id in left:
            raw_delta = right[example_id][metric] - left[example_id][metric]
            oriented = direction * raw_delta
            deltas.append(oriented)
            if oriented > 1e-12:
                wins += 1
            elif oriented < -1e-12:
                losses += 1
            else:
                ties += 1
        result[metric] = {
            "oriented_mean_delta_positive_is_better": sum(deltas) / len(deltas),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }
    return result


def _audit_condition_inputs(
    output_root: Path,
    conditions: Sequence[str],
    cases: Sequence[ExperimentCase],
) -> Dict[str, Any]:
    """Prove that every condition replayed the same non-ablated inputs."""
    hashes_by_condition: Dict[str, Dict[str, str]] = {}
    for condition in conditions:
        rows: Dict[str, str] = {}
        for case in cases:
            path = CasePaths.for_case(output_root / condition, case).predictions
            if not path.is_file():
                continue
            for prediction in read_jsonl(path):
                example_id = str(prediction.get("example_id") or "")
                audit = prediction.get("generation_input_audit", {})
                frozen_hash = str(audit.get("frozen_input_sha256") or "")
                if not example_id or not frozen_hash:
                    raise RuntimeError(
                        f"condition {condition} contains a prediction without frozen-input audit"
                    )
                rows[example_id] = frozen_hash
        if rows:
            hashes_by_condition[condition] = rows

    missing_conditions = [
        condition for condition in conditions if condition not in hashes_by_condition
    ]
    if missing_conditions:
        raise RuntimeError(
            "cannot summarize before every requested condition has predictions: "
            + ", ".join(missing_conditions)
        )

    if len(hashes_by_condition) > 1:
        baseline_condition = next(iter(hashes_by_condition))
        baseline = hashes_by_condition[baseline_condition]
        for condition, rows in hashes_by_condition.items():
            if set(rows) != set(baseline):
                raise RuntimeError(
                    f"controlled conditions {baseline_condition} and {condition} "
                    "do not contain identical example IDs"
                )
            mismatched = [
                example_id
                for example_id, frozen_hash in baseline.items()
                if rows[example_id] != frozen_hash
            ]
            if mismatched:
                raise RuntimeError(
                    f"non-ablated inputs differ between {baseline_condition} and "
                    f"{condition}; first mismatch={mismatched[0]}"
                )
    return {
        "verified": True,
        "condition_count": len(hashes_by_condition),
        "example_count": (
            len(next(iter(hashes_by_condition.values())))
            if hashes_by_condition
            else 0
        ),
        "claim": (
            "all conditions have identical frozen_input_sha256 per example; "
            "only previous_empathy_state payload differs"
        ),
    }


def summarize(
    *,
    source_root: Path,
    output_root: Path,
    conditions: Sequence[str],
    cases: Sequence[ExperimentCase],
) -> Dict[str, Any]:
    rows: list[tuple[str, Dict[str, Any]]] = []
    source = _load_metrics_for_cases(source_root, cases)
    if source is not None:
        rows.append(("Source V18 (historical, same cases)", source))
    for condition in conditions:
        result = _load_metrics(output_root / condition)
        if result is not None:
            rows.append((condition, result))

    payload: Dict[str, Any] = {
        "protocol_version": CONTROLLED_PROTOCOL_VERSION,
        "source_root": str(source_root),
        "conditions": list(conditions),
        "results": {
            label: result for label, result in rows
        },
        "paired_comparisons": {},
        "frozen_input_audit": _audit_condition_inputs(
            output_root,
            conditions,
            cases,
        ),
    }
    score_sets = {
        condition: _load_case_scores(output_root / condition, cases)
        for condition in conditions
    }
    if score_sets.get("full_state"):
        for condition in conditions:
            if condition == "full_state" or not score_sets.get(condition):
                continue
            payload["paired_comparisons"][f"{condition}_vs_full_state"] = (
                _paired_comparison(score_sets["full_state"], score_sets[condition])
            )

    output_root.mkdir(parents=True, exist_ok=True)
    save_json(str(output_root / "controlled_state_ablation_summary.json"), payload)

    headers = ["Method", "N"] + [metric.title() for metric in TABLE2_METRICS]
    lines = [
        "# Exp2 Controlled Previous-State Ablation",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for baseline in TABLE2_BASELINES:
        lines.append(
            "| "
            + " | ".join(
                [f"Paper: {baseline['method']}", "-"]
                + [
                    f"{baseline[metric][0]:.4f} +/- {baseline[metric][1]:.4f}"
                    for metric in TABLE2_METRICS
                ]
            )
            + " |"
        )
    for label, result in rows:
        lines.append(
            "| "
            + " | ".join(
                [label, str(result.get("example_count") or "-")]
                + [_stat(result["metrics"], metric) for metric in TABLE2_METRICS]
            )
            + " |"
        )
    lines.extend((
        "",
        "All ablation rows use the exact same source inputs and V18 response prompt. "
        "Only the previous-state payload differs. The historical V18 row is shown "
        "for context and is not the randomized control because it was generated earlier.",
        "",
        "For paired comparisons, positive oriented deltas and more wins are better; "
        "intimacy/empathy directions are inverted before comparison.",
        "",
    ))
    for comparison, metrics in payload["paired_comparisons"].items():
        lines.extend((f"## {comparison}", ""))
        lines.append("| Metric | Oriented mean delta | Wins | Losses | Ties |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for metric in TABLE2_METRICS:
            value = metrics[metric]
            lines.append(
                f"| {metric} | "
                f"{value['oriented_mean_delta_positive_is_better']:.4f} | "
                f"{value['wins']} | {value['losses']} | {value['ties']} |"
            )
        lines.append("")
    (output_root / "controlled_state_ablation_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def _parse_conditions(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--conditions must contain at least one condition")
    unknown = [value for value in values if value not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown conditions {unknown}; choose from {CONDITIONS}")
    if len(values) != len(set(values)):
        raise ValueError("--conditions contains duplicates")
    return values


def _config_model(config_path: str) -> str:
    import configparser

    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    return config.get("API", "model", fallback="unknown")


def run(args: argparse.Namespace) -> None:
    source_root = Path(args.source_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source run does not exist: {source_root}")
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("--output-dir must not contain or overwrite the source run")

    all_cases = build_cases(args.dataset_dir, args.train_ratio)
    cases = _select_cases(all_cases, args.case)
    conditions = _parse_conditions(args.conditions)
    config_path = str(Path(args.config).resolve())
    source_files = _source_file_manifest(
        source_root,
        cases,
        args.source_prompt_version,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    save_split_manifest(cases, output_root / "split_manifest.json", args.train_ratio)

    if args.phase in ("generate", "all"):
        for condition in conditions:
            condition_root = output_root / condition
            manifest = _condition_manifest(
                condition=condition,
                source_root=source_root,
                cases=cases,
                source_files=source_files,
                prompt_version=args.source_prompt_version,
                model=_config_model(config_path),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            _ensure_condition_manifest(condition_root, manifest)
            generated = generate_condition(
                cases=cases,
                source_root=source_root,
                target_root=condition_root,
                config_path=config_path,
                prompt_version=args.source_prompt_version,
                condition=condition,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                workers=args.generate_workers,
            )
            save_json(str(condition_root / "generation_run.json"), {
                "condition": condition,
                "generated_this_run": generated,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })

    if args.phase in ("evaluate", "all"):
        for condition in conditions:
            condition_root = output_root / condition
            manifest_path = condition_root / "controlled_condition_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"condition {condition} has not been generated: {manifest_path}"
                )
            evaluate_table2(
                cases=cases,
                output_dir=condition_root,
                config_path=config_path,
                judge_model=args.judge_model,
                judge_config_section=args.judge_config_section,
                device=args.eval_device,
                batch_size=args.eval_batch_size,
                judge_workers=args.judge_workers,
                prompt_version=args.source_prompt_version,
            )

    if args.phase in ("evaluate", "summarize", "all"):
        summarize(
            source_root=source_root,
            output_root=output_root,
            conditions=conditions,
            cases=cases,
        )
        print(f"summary: {output_root / 'controlled_state_ablation_summary.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled Exp2 replay: freeze one completed run and ablate only "
            "the previous empathy-state payload used by final reply generation."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("generate", "evaluate", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--config", default="config.ini")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument(
        "--source-prompt-version",
        default=DEFAULT_SOURCE_PROMPT_VERSION,
    )
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="Comma-separated subset of: " + ", ".join(CONDITIONS),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--generate-workers", type=int, default=3)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-config-section", default="EvaluationAPI")
    parser.add_argument("--judge-workers", type=int, default=6)
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Repeat to select individual Chat_*.json files or case IDs.",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
