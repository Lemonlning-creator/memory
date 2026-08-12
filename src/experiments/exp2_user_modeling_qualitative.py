from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

from .. import agent as agent_module
from ..agent import StateDrivenCompanionAgent
from ..memory_os_local import MemoryOSLocal
from ..profile_utils import state_axis
from ..utils import load_json, save_json
from .exp2_user_modeling import (
    PROFILE_FIELDS,
    ExperimentCase,
    bubbles_for_sessions,
    build_cases,
)


PROTOCOL = "exp2_fixed_schema_turn_trajectory_from_zero_on_train_split_v3"
SMOOTHING_ALPHA = 0.2
_BASE_PROFILE_EVOLUTION_SYSTEM_PROMPT = agent_module.PROFILE_EVOLUTION_SYSTEM_PROMPT


def _empty_attribute(field: str) -> Dict[str, Any]:
    """Create one unknown attribute using the Bayesian updater's leaf format."""
    return {
        "value": "" if field == "summary" else [],
        "confidence": 0.5,
        "memory_ids": [],
        "evidence": [],
        "bayesian_update": {
            "prior_confidence": 0.5,
            "evidence_strength": "neutral",
            "posterior_confidence": 0.5,
            "update_direction": "unchanged",
        },
    }


def _fixed_empty_runtime_profile() -> Dict[str, Any]:
    """Use exactly the same 21 fields as the one-shot profile extractor."""
    return {
        "state_axis": {
            "static_profile": {
                layer: {field: _empty_attribute(field) for field in fields}
                for layer, fields in PROFILE_FIELDS.items()
            },
            "current_state": {},
            "projected_state": {},
        },
        "context_axis": {
            "current_context": "",
            "context_detail": "",
            "inferred_at_turn": 0,
        },
    }


def _install_fixed_schema_constraint() -> None:
    """Constrain the existing Bayesian updater for this experiment only."""
    schema = {layer: list(fields) for layer, fields in PROFILE_FIELDS.items()}
    agent_module.PROFILE_EVOLUTION_SYSTEM_PROMPT = (
        _BASE_PROFILE_EVOLUTION_SYSTEM_PROMPT
        + "\n\nEXPERIMENT-SPECIFIC FIXED-SCHEMA CONSTRAINT (overrides any rule above "
        "about adding or removing attributes):\n"
        "- The complete static_profile MUST contain exactly the layers and fields "
        f"shown here: {json.dumps(schema, ensure_ascii=False)}.\n"
        "- Never add, remove, rename, move, or nest a field.\n"
        "- Only update the value, confidence, memory_ids, evidence, and "
        "bayesian_update metadata inside an existing field.\n"
        "- Each summary.value MUST be a string. Every other field.value MUST be "
        "a list of strings, matching the one-shot extraction schema.\n"
        "- If no stable evidence supports a field, preserve its empty value and "
        "confidence 0.5.\n"
        "- reasoning.new_attributes and reasoning.removed_attributes MUST both "
        "be empty lists."
    )


def _as_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalise_fixed_static_profile(static_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the agreed schema even if an LLM response drifts from it."""
    normalised: Dict[str, Any] = {}
    for layer, fields in PROFILE_FIELDS.items():
        source_layer = static_profile.get(layer, {})
        if not isinstance(source_layer, dict):
            source_layer = {}
        normalised[layer] = {}
        for field in fields:
            source = source_layer.get(field, {})
            if not isinstance(source, dict) or "value" not in source:
                source = {"value": source}

            raw_value = source.get("value")
            if field == "summary":
                value = raw_value.strip() if isinstance(raw_value, str) else ""
            else:
                value = _as_string_list(raw_value)
            populated = bool(value)

            confidence = source.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            confidence = max(0.0, min(1.0, float(confidence))) if populated else 0.5

            memory_ids = source.get("memory_ids", [])
            evidence = source.get("evidence", [])
            update = source.get("bayesian_update", {})
            normalised[layer][field] = {
                "value": value,
                "confidence": confidence,
                "memory_ids": _as_string_list(memory_ids),
                "evidence": _as_string_list(evidence),
                "bayesian_update": update if isinstance(update, dict) else {},
            }
    return normalised


def _normalise_agent_profile(agent: StateDrivenCompanionAgent) -> None:
    state = state_axis(agent.user_profile)
    state["static_profile"] = _normalise_fixed_static_profile(
        state.get("static_profile", {})
    )
    save_json(agent.profile_path, agent.user_profile)


def _profile_fingerprint(profile: Dict[str, Any]) -> str:
    payload = json.dumps(
        _static_profile(profile),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _field_populated(attribute: Any) -> bool:
    if not isinstance(attribute, dict):
        return bool(attribute)
    value = attribute.get("value")
    return bool(value) if not isinstance(value, list) else any(bool(item) for item in value)


def _fixed_profile_completeness(static_profile: Dict[str, Any]) -> float:
    """Fraction of the fixed one-shot fields that currently contain evidence."""
    total = sum(len(fields) for fields in PROFILE_FIELDS.values())
    populated = sum(
        _field_populated(static_profile.get(layer, {}).get(field))
        for layer, fields in PROFILE_FIELDS.items()
        for field in fields
    )
    return round(populated / total, 4)


def _fixed_profile_entropy(static_profile: Dict[str, Any]) -> float:
    """Mean uncertainty over all fixed fields; unknown fields have entropy 1."""
    entropies: List[float] = []
    for layer, fields in PROFILE_FIELDS.items():
        section = static_profile.get(layer, {})
        for field in fields:
            attribute = section.get(field, {}) if isinstance(section, dict) else {}
            if not _field_populated(attribute):
                entropies.append(1.0)
                continue
            confidence = attribute.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            confidence = max(0.001, min(0.999, float(confidence)))
            entropies.append(
                -confidence * math.log2(confidence)
                - (1.0 - confidence) * math.log2(1.0 - confidence)
            )
    return round(sum(entropies) / len(entropies), 4)


@dataclass(frozen=True)
class QualitativePaths:
    root: Path
    case_root: Path
    runtime_profile: Path
    clean_profile: Path
    persona_stub: Path
    memory_db: Path
    snapshots: Path
    trajectory: Path
    manifest: Path

    @classmethod
    def for_case(
        cls,
        output_dir: str | Path,
        case: ExperimentCase,
    ) -> "QualitativePaths":
        root = Path(output_dir).resolve()
        qualitative = root / "cases" / case.case_id / "qualitative"
        return cls(
            root=root,
            case_root=qualitative,
            runtime_profile=qualitative / "profile_runtime.json",
            clean_profile=qualitative / "user_profile.json",
            persona_stub=qualitative / "persona_stub.json",
            memory_db=qualitative / "memory" / "memory.db",
            snapshots=qualitative / "profile_snapshots",
            trajectory=qualitative / "profile_trajectory.json",
            manifest=qualitative / "trajectory_manifest.json",
        )

    def ensure_parents(self) -> None:
        self.case_root.mkdir(parents=True, exist_ok=True)
        self.memory_db.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots.mkdir(parents=True, exist_ok=True)


def _select_cases(
    cases: Iterable[ExperimentCase],
    selectors: Sequence[str],
) -> List[ExperimentCase]:
    available = list(cases)
    if not selectors:
        return available

    wanted = {selector.lower() for selector in selectors}
    selected = [
        case
        for case in available
        if case.case_id.lower() in wanted
        or Path(case.dataset_path).name.lower() in wanted
    ]
    matched = {
        value
        for case in selected
        for value in (case.case_id.lower(), Path(case.dataset_path).name.lower())
    }
    missing = wanted - matched
    if missing:
        raise ValueError(f"unknown case selectors: {sorted(missing)}")
    return selected


def _model_name(config_path: str) -> str:
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    return config.get("API", "model", fallback="unknown")


def _static_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    return state_axis(profile).get("static_profile", {})


def _clean_one_shot_profile(static_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Export only the fixed fields and values in the one-shot profile format."""
    clean: Dict[str, Any] = {}
    for layer, fields in PROFILE_FIELDS.items():
        section = static_profile.get(layer, {})
        if not isinstance(section, dict):
            section = {}
        clean[layer] = {}
        for field in fields:
            attribute = section.get(field, {})
            value = attribute.get("value") if isinstance(attribute, dict) else attribute
            if field == "summary":
                clean[layer][field] = value.strip() if isinstance(value, str) else ""
            else:
                clean[layer][field] = _as_string_list(value)
    return clean


def _write_clean_profile(runtime_path: Path, clean_path: Path) -> None:
    if not runtime_path.exists():
        raise FileNotFoundError(f"missing runtime profile: {runtime_path}")
    runtime_profile = load_json(str(runtime_path))
    clean_profile = _clean_one_shot_profile(_static_profile(runtime_profile))
    save_json(str(clean_path), clean_profile)


def _point(
    case: ExperimentCase,
    profile: Dict[str, Any],
    turn_index: int,
    session_index: int,
    session_id: str | None,
    session_turn_index: int,
    session_end: bool,
    observed_bubbles: int,
    profile_version: int,
    profile_changed: bool,
    profile_sha256: str,
    finalize_result: Dict[str, Any] | None,
) -> Dict[str, Any]:
    static_profile = _static_profile(profile)
    finalized = finalize_result or {}
    return {
        "case_id": case.case_id,
        "user_speaker": case.user_speaker,
        "turn_index": turn_index,
        "session_index": session_index,
        "session_id": session_id,
        "session_turn_index": session_turn_index,
        "session_end": session_end,
        "observed_bubbles": observed_bubbles,
        "profile_version": profile_version,
        "profile_changed": profile_changed,
        "profile_sha256": profile_sha256,
        "profile_completeness": _fixed_profile_completeness(static_profile),
        "profile_entropy": _fixed_profile_entropy(static_profile),
        "flushed_mid_term_ids": list(finalized.get("flushed_mid_term_ids") or []),
        "long_term_memory_id": finalized.get("long_term_memory_id"),
    }


def _add_turn_progress_and_smoothing(points: List[Dict[str, Any]]) -> None:
    """Add causal visualization fields without changing the raw trajectory."""
    if not points:
        return
    max_turn = max(int(point["turn_index"]) for point in points)
    smooth_completeness: float | None = None
    smooth_entropy: float | None = None
    for point in points:
        turn_index = int(point["turn_index"])
        point["turn_progress"] = turn_index / max_turn if max_turn else 0.0
        completeness = float(point["profile_completeness"])
        entropy = float(point["profile_entropy"])
        if smooth_completeness is None:
            smooth_completeness = completeness
            smooth_entropy = entropy
        else:
            smooth_completeness = (
                SMOOTHING_ALPHA * completeness
                + (1.0 - SMOOTHING_ALPHA) * smooth_completeness
            )
            smooth_entropy = (
                SMOOTHING_ALPHA * entropy
                + (1.0 - SMOOTHING_ALPHA) * smooth_entropy
            )
        point["smoothed_profile_completeness"] = round(smooth_completeness, 6)
        point["smoothed_profile_entropy"] = round(smooth_entropy, 6)


def _save_snapshot(
    path: Path,
    case: ExperimentCase,
    point: Dict[str, Any],
    profile: Dict[str, Any],
) -> None:
    save_json(str(path), {
        "protocol": PROTOCOL,
        "case": asdict(case),
        "point": point,
        "profile": profile,
    })


def _close_memory_manager(manager: Any) -> None:
    client = getattr(manager, "client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _completed_trajectory(
    case: ExperimentCase,
    paths: QualitativePaths,
) -> List[Dict[str, Any]] | None:
    if not paths.manifest.exists() or not paths.trajectory.exists():
        return None
    manifest = load_json(str(paths.manifest))
    if (
        manifest.get("status") != "complete"
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("dataset_sha256") != case.dataset_sha256
        or manifest.get("train_sessions") != list(case.train_sessions)
    ):
        return None
    trajectory = load_json(str(paths.trajectory))
    points = trajectory.get("points")
    if not isinstance(points, list):
        raise ValueError(f"invalid completed trajectory: {paths.trajectory}")
    return points


def _assert_clean_start(paths: QualitativePaths) -> None:
    snapshots = list(paths.snapshots.glob("*.json"))
    if paths.runtime_profile.exists() or paths.memory_db.exists() or snapshots:
        raise RuntimeError(
            "incomplete qualitative state already exists under "
            f"{paths.case_root}. Use a new --output-dir so the from-zero trajectory "
            "cannot accidentally resume from a non-empty profile or memory database."
        )


def run_case_trajectory(
    case: ExperimentCase,
    output_dir: str | Path,
    config_path: str = "config.ini",
) -> List[Dict[str, Any]]:
    """Build one user profile from zero by replaying only the 90% train split."""
    paths = QualitativePaths.for_case(output_dir, case)
    paths.ensure_parents()

    completed = _completed_trajectory(case, paths)
    if completed is not None:
        _write_clean_profile(paths.runtime_profile, paths.clean_profile)
        print(f"[Qualitative] reuse complete trajectory case={case.case_id}")
        return completed
    _assert_clean_start(paths)

    save_json(str(paths.runtime_profile), _fixed_empty_runtime_profile())
    # The core Agent requires a persona file, but persona is never consulted because
    # this protocol observes real dialogue and never generates an agent reply.
    save_json(str(paths.persona_stub), {})
    save_json(str(paths.manifest), {
        "protocol": PROTOCOL,
        "status": "running",
        "dataset_sha256": case.dataset_sha256,
        "train_sessions": list(case.train_sessions),
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    agent: StateDrivenCompanionAgent | None = None
    try:
        _install_fixed_schema_constraint()
        agent = StateDrivenCompanionAgent(
            config_path=config_path,
            profile_path=str(paths.runtime_profile),
            persona_path=str(paths.persona_stub),
            user_name=case.user_speaker,
            modeling_mode="explicit",
            update_mode="bayesian_online",
            exploration_mode="adaptive",
        )
        default_memory = agent.memory_manager
        agent.memory_manager = MemoryOSLocal(
            persist_path=str(paths.memory_db),
            config_path=config_path,
        )
        _close_memory_manager(default_memory)

        chat = load_json(case.dataset_path)
        points: List[Dict[str, Any]] = []
        observed_bubbles = 0
        turn_index = 0
        profile_version = 0
        previous_profile_sha256 = _profile_fingerprint(agent.user_profile)

        initial = _point(
            case=case,
            profile=agent.user_profile,
            turn_index=0,
            session_index=0,
            session_id=None,
            session_turn_index=0,
            session_end=False,
            observed_bubbles=0,
            profile_version=0,
            profile_changed=False,
            profile_sha256=previous_profile_sha256,
            finalize_result=None,
        )
        points.append(initial)
        _save_snapshot(
            paths.snapshots / "000_initial.json",
            case,
            initial,
            agent.user_profile,
        )

        for session_index, session_id in enumerate(case.train_sessions, start=1):
            bubbles = bubbles_for_sessions(chat, [session_id])
            last_point: Dict[str, Any] | None = None
            for session_turn_index, bubble in enumerate(bubbles, start=1):
                if bubble.speaker == case.user_speaker:
                    role = "user"
                elif bubble.speaker == case.agent_speaker:
                    role = "assistant"
                else:
                    raise ValueError(
                        f"unexpected speaker {bubble.speaker!r} in {case.case_id}"
                    )
                agent.observe_dialogue_turn(role, bubble.content)
                observed_bubbles += 1
                turn_index += 1
                session_end = session_turn_index == len(bubbles)
                finalized = agent.finalize_session() if session_end else None
                _normalise_agent_profile(agent)

                profile_sha256 = _profile_fingerprint(agent.user_profile)
                profile_changed = profile_sha256 != previous_profile_sha256
                if profile_changed:
                    profile_version += 1
                point = _point(
                    case=case,
                    profile=agent.user_profile,
                    turn_index=turn_index,
                    session_index=session_index,
                    session_id=session_id,
                    session_turn_index=session_turn_index,
                    session_end=session_end,
                    observed_bubbles=observed_bubbles,
                    profile_version=profile_version,
                    profile_changed=profile_changed,
                    profile_sha256=profile_sha256,
                    finalize_result=finalized,
                )
                points.append(point)
                last_point = point
                if profile_changed:
                    _save_snapshot(
                        paths.snapshots
                        / f"turn_{turn_index:05d}_profile_v{profile_version:03d}.json",
                        case,
                        point,
                        agent.user_profile,
                    )
                previous_profile_sha256 = profile_sha256

            if last_point is not None:
                print(
                    f"[Qualitative] {case.case_id} session={session_index}/"
                    f"{len(case.train_sessions)} turn={turn_index} {session_id} "
                    f"profile_v={last_point['profile_version']} "
                    f"completeness={last_point['profile_completeness']:.4f} "
                    f"entropy={last_point['profile_entropy']:.4f}"
                )

        _add_turn_progress_and_smoothing(points)

        save_json(str(paths.trajectory), {
            "protocol": PROTOCOL,
            "case": asdict(case),
            "data_policy": "from_zero_real_dialogue_train_split_only",
            "reply_policy": "no_generated_agent_replies",
            "turn_definition": "one merged consecutive-speaker dialogue bubble",
            "snapshot_policy": "initial_and_only_when_the_fixed_profile_changes",
            "curve_policy": {
                "raw": "recorded after every observed training turn",
                "smoothing": "causal EWMA for visualization only",
                "smoothing_alpha": SMOOTHING_ALPHA,
            },
            "points": points,
        })
        _write_clean_profile(paths.runtime_profile, paths.clean_profile)
        save_json(str(paths.manifest), {
            "protocol": PROTOCOL,
            "status": "complete",
            "dataset_sha256": case.dataset_sha256,
            "train_sessions": list(case.train_sessions),
            "point_count": len(points),
            "clean_profile_path": str(paths.clean_profile),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        return points
    except Exception:
        save_json(str(paths.manifest), {
            "protocol": PROTOCOL,
            "status": "failed",
            "dataset_sha256": case.dataset_sha256,
            "train_sessions": list(case.train_sessions),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        })
        raise
    finally:
        if agent is not None:
            _close_memory_manager(agent.memory_manager)


def aggregate_curve_points(
    per_case: Dict[str, Sequence[Dict[str, Any]]],
    grid_size: int = 101,
) -> List[Dict[str, Any]]:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    def prepared(source: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = [dict(point) for point in source]
        if not rows:
            return rows
        if any("turn_progress" not in row for row in rows):
            _add_turn_progress_and_smoothing(rows)
        return sorted(rows, key=lambda row: float(row["turn_progress"]))

    def interpolate(rows: Sequence[Dict[str, Any]], x: float, field: str) -> float:
        if not rows:
            raise ValueError("cannot interpolate an empty trajectory")
        if x <= float(rows[0]["turn_progress"]):
            return float(rows[0][field])
        for left, right in zip(rows, rows[1:]):
            left_x = float(left["turn_progress"])
            right_x = float(right["turn_progress"])
            if x <= right_x:
                if right_x == left_x:
                    return float(right[field])
                ratio = (x - left_x) / (right_x - left_x)
                return float(left[field]) + ratio * (
                    float(right[field]) - float(left[field])
                )
        return float(rows[-1][field])

    trajectories = [prepared(points) for points in per_case.values() if points]
    aggregate: List[Dict[str, Any]] = []
    for index in range(grid_size):
        progress = index / (grid_size - 1)
        aggregate.append({
            "turn_progress": progress,
            "turn_progress_percent": progress * 100.0,
            "case_count": len(trajectories),
            "mean_profile_completeness": mean(
                interpolate(rows, progress, "profile_completeness")
                for rows in trajectories
            ),
            "mean_profile_entropy": mean(
                interpolate(rows, progress, "profile_entropy")
                for rows in trajectories
            ),
            "mean_smoothed_profile_completeness": mean(
                interpolate(rows, progress, "smoothed_profile_completeness")
                for rows in trajectories
            ),
            "mean_smoothed_profile_entropy": mean(
                interpolate(rows, progress, "smoothed_profile_entropy")
                for rows in trajectories
            ),
        })
    return aggregate


def write_curves(
    per_case: Dict[str, Sequence[Dict[str, Any]]],
    output_dir: str | Path,
) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    figures = Path(output_dir).resolve() / "qualitative_figures"
    figures.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_curve_points(per_case)
    data_path = figures / "profile_curves.json"
    save_json(str(data_path), {
        "protocol": PROTOCOL,
        "x_axis": {
            "per_case": "global dialogue turn index within the 90% train split",
            "aggregate": "normalized chronological turn progress (0-100%)",
        },
        "fixed_schema": {layer: list(fields) for layer, fields in PROFILE_FIELDS.items()},
        "profile_evolution": "populated fraction of the 21 fixed one-shot extraction fields",
        "profile_entropy": (
            "mean binary entropy across the same 21 fields; empty fields have entropy 1"
        ),
        "smoothing": {
            "method": "causal EWMA",
            "alpha": SMOOTHING_ALPHA,
            "purpose": "visualization only; raw per-turn values remain authoritative",
        },
        "per_case": per_case,
        "aggregate": aggregate,
    })

    single_case_points = (
        list(next(iter(per_case.values()))) if len(per_case) == 1 else []
    )
    if single_case_points:
        x = [int(point["turn_index"]) for point in single_case_points]
        completeness = [float(point["profile_completeness"]) for point in single_case_points]
        entropy = [float(point["profile_entropy"]) for point in single_case_points]
        smooth_completeness = [
            float(point["smoothed_profile_completeness"])
            for point in single_case_points
        ]
        smooth_entropy = [
            float(point["smoothed_profile_entropy"])
            for point in single_case_points
        ]
        x_label = "Training dialogue turn index (90% split)"
    else:
        x = [row["turn_progress_percent"] for row in aggregate]
        completeness = [row["mean_profile_completeness"] for row in aggregate]
        entropy = [row["mean_profile_entropy"] for row in aggregate]
        smooth_completeness = [
            row["mean_smoothed_profile_completeness"] for row in aggregate
        ]
        smooth_entropy = [row["mean_smoothed_profile_entropy"] for row in aggregate]
        x_label = "Normalized training-turn progress (%)"

    def decorate_single_case(metric: str) -> None:
        if not single_case_points:
            return
        for point in single_case_points:
            turn = int(point["turn_index"])
            if point.get("session_end"):
                plt.axvline(turn, linewidth=0.7, alpha=0.12, color="black")
        updates = [point for point in single_case_points if point.get("profile_changed")]
        if updates:
            plt.scatter(
                [int(point["turn_index"]) for point in updates],
                [float(point[metric]) for point in updates],
                s=24,
                zorder=3,
                label="Actual profile update",
            )

    evolution_path = figures / "profile_evolution_curve.png"
    plt.figure(figsize=(7, 4.2))
    plt.plot(x, completeness, linewidth=1, alpha=0.35, label="Raw per-turn mean")
    plt.plot(x, smooth_completeness, linewidth=2.2, label="Causal EWMA trend")
    decorate_single_case("profile_completeness")
    plt.xlabel(x_label)
    plt.ylabel("Mean fixed-field coverage")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(evolution_path, dpi=200)
    plt.close()

    entropy_path = figures / "profile_entropy_curve.png"
    plt.figure(figsize=(7, 4.2))
    plt.plot(
        x,
        entropy,
        linewidth=1,
        alpha=0.35,
        color="#d95f02",
        label="Raw per-turn mean",
    )
    plt.plot(
        x,
        smooth_entropy,
        linewidth=2.2,
        color="#d95f02",
        label="Causal EWMA trend",
    )
    decorate_single_case("profile_entropy")
    plt.xlabel(x_label)
    plt.ylabel("Mean fixed-field profile entropy")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(entropy_path, dpi=200)
    plt.close()

    return {
        "curve_data": str(data_path),
        "profile_evolution": str(evolution_path),
        "profile_entropy": str(entropy_path),
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _select_cases(
        build_cases(args.dataset_dir, args.train_ratio),
        args.case,
    )
    per_case = {
        case.case_id: run_case_trajectory(case, output_dir, args.config)
        for case in cases
    }
    figures = write_curves(per_case, output_dir)
    save_json(str(output_dir / "qualitative_run_manifest.json"), {
        "experiment": "Experiment 2. User Modeling Evaluation",
        "analysis": "Qualitative profile evolution and profile entropy",
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "train_ratio": args.train_ratio,
        "model": _model_name(args.config),
        "data_policy": "from_zero_real_dialogue_train_split_only",
        "reply_policy": "no_generated_agent_replies",
        "cases": [asdict(case) for case in cases],
        "figures": figures,
        "status": "complete",
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 2 qualitative profile trajectories from zero on the "
            "chronological 90% REALTALK train split."
        )
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/exp2_user_modeling")
    parser.add_argument("--config", default="config.ini")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run one named conversation case; repeat to select multiple cases.",
    )
    return parser


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
