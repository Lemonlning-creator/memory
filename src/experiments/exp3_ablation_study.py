"""Experiment 3: component-specific ablations for Deep Empathy.

The three components require different controlled protocols:

* Exp3-A Explicit User Modeling: 90/10 fixed REALTALK evaluation.
* Exp3-B Adaptive Exploration: 50/50 hidden-profile interactive simulation.
* Exp3-C Bayesian Updating: 50/50 chronological REALTALK replay.

This module is intentionally separate from Experiment 2 so its online updating
and simulator behavior cannot change the versioned Experiment 2 protocol.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Sequence

from ..agent import StateDrivenCompanionAgent
from ..epistemic_decay import compute_portrait_entropy, compute_profile_completeness
from ..llm_client import LLMClient
from ..memory_os_local import MemoryOSLocal
from ..metrics import detect_exploration_question
from ..profile_utils import create_empty_profile, state_axis
from ..prompts.exp2_versions import (
    DEFAULT_EXP2_PROMPT_VERSION,
    Exp2PromptBundle,
    exp2_prompt_versions,
    get_exp2_prompt_bundle,
)
from ..utils import load_json, save_json
from .exp2_user_modeling import (
    TABLE2_METRICS,
    CasePaths,
    Exp2JudgeClient,
    ExperimentCase,
    JsonlStore,
    Table2Evaluator,
    _append_real_bubble,
    _assert_consistent_stores,
    _example_by_user_dia,
    _generate_with_parallel_alignment,
    _model_name,
    _resume_position,
    _select_cases,
    _validate_extracted_profile,
    aggregate_table2_scores,
    bubbles_for_sessions,
    build_case_persona,
    build_case_profile,
    build_cases,
    build_reply_examples,
    evaluate_case_table2,
    save_split_manifest,
    session_keys,
    validate_no_leakage,
)
from .exp3_user_simulator import (
    HiddenClaim,
    HiddenProfileUserSimulator,
    aggregate_discovery_results,
    build_hidden_claim_manifest,
    evaluate_hidden_coverage,
    evaluate_profile_discovery,
    hidden_claims_from_manifest,
    validate_simulator_payload,
)
from .extract_profile_persona_en import extract_profile


PROTOCOL_VERSION = "exp3_component_specific_v3"
EXPLICIT_TRAIN_RATIO = 0.9
ONLINE_TRAIN_RATIO = 0.5
DEFAULT_FIXED_OMEGA = 0.5
DEFAULT_SIMULATION_ROUNDS = 20
DEFAULT_SIMULATION_SEEDS = 1
SIMULATOR_ANNOTATION_KEYS = (
    "revealed_claim_ids", "disclosure_strength", "withheld_or_refused",
    "perceived_burden", "burden_reason", "disclosure_decision",
    "disclosure_depth", "trust", "fatigue",
)

TRACK_EXPLICIT = "explicit"
TRACK_EXPLORATION = "exploration"
TRACK_BAYESIAN = "bayesian"
TRACK_ORDER = (TRACK_EXPLICIT, TRACK_EXPLORATION, TRACK_BAYESIAN)


@dataclass(frozen=True)
class ExperimentCondition:
    key: str
    label: str
    track: str
    modeling_mode: str
    exploration_mode: str
    update_mode: str
    uses_explicit_profile: bool
    description: str

    def manifest(self, fixed_omega: float) -> Dict[str, Any]:
        value = asdict(self)
        value["fixed_omega"] = (
            fixed_omega if self.exploration_mode == "fixed_exploration" else None
        )
        return value


CONDITIONS: Dict[str, ExperimentCondition] = {
    "explicit_user_modeling": ExperimentCondition(
        key="explicit_user_modeling",
        label="Explicit User Modeling",
        track=TRACK_EXPLICIT,
        modeling_mode="explicit",
        exploration_mode="adaptive",
        update_mode="static",
        uses_explicit_profile=True,
        description="Use the explicit five-layer profile extracted from the first 90% sessions.",
    ),
    "wo_explicit_user_modeling": ExperimentCondition(
        key="wo_explicit_user_modeling",
        label="w/o Explicit User Modeling",
        track=TRACK_EXPLICIT,
        modeling_mode="self_model",
        exploration_mode="adaptive",
        update_mode="static",
        uses_explicit_profile=False,
        description="Infer transient user state from current dialogue without a persistent user profile.",
    ),
    "adaptive_exploration": ExperimentCondition(
        key="adaptive_exploration",
        label="Adaptive Exploration",
        track=TRACK_EXPLORATION,
        modeling_mode="explicit",
        exploration_mode="adaptive",
        update_mode="bayesian_online",
        uses_explicit_profile=True,
        description="Dynamically allocate exploration using interaction time and profile completeness.",
    ),
    "fixed_exploration": ExperimentCondition(
        key="fixed_exploration",
        label="w/o Adaptive Exploration (Fixed Omega)",
        track=TRACK_EXPLORATION,
        modeling_mode="explicit",
        exploration_mode="fixed_exploration",
        update_mode="bayesian_online",
        uses_explicit_profile=True,
        description="Keep exploration available but use a constant omega budget.",
    ),
    "bayesian_online": ExperimentCondition(
        key="bayesian_online",
        label="Bayesian Online Updating",
        track=TRACK_BAYESIAN,
        modeling_mode="explicit",
        exploration_mode="adaptive",
        update_mode="bayesian_online",
        uses_explicit_profile=True,
        description="Update the initial profile from long-term memories during real-dialogue replay.",
    ),
    "static_profile": ExperimentCondition(
        key="static_profile",
        label="w/o Bayesian Updating (Static Profile)",
        track=TRACK_BAYESIAN,
        modeling_mode="explicit",
        exploration_mode="adaptive",
        update_mode="static",
        uses_explicit_profile=True,
        description="Freeze the same initial profile during real-dialogue replay.",
    ),
}

CONDITION_ORDER = tuple(CONDITIONS)


SELF_MODEL_RESPONSE_ADDENDUM = """

EXPERIMENT ABLATION — NO EXPLICIT USER MODEL:
- No persistent user profile is available in this condition.
- Infer the user's immediate state only from the latest message, recent observed
  dialogue, and your own persona/perspective.
- Do not imply that you remember stable facts, preferences, or traits about the user.
"""

SELF_MODEL_ALIGNMENT_ADDENDUM = """

EXPERIMENT ABLATION — SELF-MODEL-BASED OTHER INFERENCE:
- The explicit USER PROFILE has intentionally been removed.
- Do not reconstruct a persistent five-layer user profile.
- Infer only a transient state from observed dialogue and the agent persona.
"""

REMOVED_PROFILE_MARKER = "[explicit user profile removed for ablation]"


def condition_prompt_bundle(
    base: Exp2PromptBundle,
    condition: ExperimentCondition,
) -> Exp2PromptBundle:
    if condition.uses_explicit_profile:
        return base
    return replace(
        base,
        version=f"{base.version}__no_explicit_user_model",
        response_system=base.response_system + SELF_MODEL_RESPONSE_ADDENDUM,
        alignment_system=base.alignment_system + SELF_MODEL_ALIGNMENT_ADDENDUM,
        description=f"{base.description} Explicit persistent user model removed.",
    )


class Exp3Agent(StateDrivenCompanionAgent):
    """Experiment-only adapter for explicit, exploration, and update ablations."""

    def __init__(
        self,
        *args: Any,
        condition: ExperimentCondition,
        fixed_omega: float = DEFAULT_FIXED_OMEGA,
        **kwargs: Any,
    ) -> None:
        kwargs.update(
            modeling_mode=condition.modeling_mode,
            exploration_mode=condition.exploration_mode,
            update_mode=condition.update_mode,
        )
        super().__init__(*args, **kwargs)
        self.condition = condition
        self.fixed_omega = fixed_omega
        self.prompt_bundle = condition_prompt_bundle(self.prompt_bundle, condition)
        if condition.exploration_mode == "fixed_exploration":
            self.epistemic_tracker.fixed_value = fixed_omega

    def _prompt_context(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
        previous_empathy_state: Dict[str, Any] | None = None,
    ) -> Dict[str, str]:
        context = super()._prompt_context(
            user_input,
            relevant_memory,
            previous_empathy_state=previous_empathy_state,
        )
        if not self.condition.uses_explicit_profile:
            context["static_profile"] = REMOVED_PROFILE_MARKER
        return context

    def _run_empathy_alignment(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.condition.uses_explicit_profile:
            return super()._run_empathy_alignment(user_input, relevant_memory)
        state = state_axis(self.user_profile)
        original_profile = state.get("static_profile", {})
        state["static_profile"] = {}
        try:
            return super()._run_empathy_alignment(user_input, relevant_memory)
        finally:
            state["static_profile"] = original_profile


@dataclass(frozen=True)
class SimulationPaths:
    root: Path
    initial_profile: Path
    runtime_profile: Path
    persona: Path
    memory_db: Path
    dialogue: Path
    profile_snapshots: Path
    final_profile: Path
    generation_summary: Path
    evaluation: Path

    @classmethod
    def for_run(
        cls,
        output_dir: str | Path,
        condition: ExperimentCondition,
        case: ExperimentCase,
        seed_index: int,
    ) -> "SimulationPaths":
        root = (
            condition_root(output_dir, condition)
            / "cases" / case.case_id / "simulations" / f"seed_{seed_index:03d}"
        )
        return cls(
            root=root,
            initial_profile=root / "assets" / "initial_profile_runtime.json",
            runtime_profile=root / "assets" / "runtime_profile.json",
            persona=root / "assets" / "agent_persona.json",
            memory_db=root / "memory" / "memory.db",
            dialogue=root / "dialogue.jsonl",
            profile_snapshots=root / "profile_snapshots",
            final_profile=root / "final_profile_runtime.json",
            generation_summary=root / "generation_summary.json",
            evaluation=root / "evaluation.json",
        )

    def ensure_parents(self) -> None:
        for path in asdict(self).values():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.profile_snapshots.mkdir(parents=True, exist_ok=True)


def select_tracks(values: Iterable[str]) -> list[str]:
    selected = list(values)
    if not selected:
        return list(TRACK_ORDER)
    unknown = sorted(set(selected) - set(TRACK_ORDER))
    if unknown:
        raise ValueError(f"unknown tracks {unknown}; choose from {list(TRACK_ORDER)}")
    wanted = set(selected)
    return [track for track in TRACK_ORDER if track in wanted]


def select_conditions(
    tracks: Sequence[str],
    values: Iterable[str],
) -> Dict[str, list[ExperimentCondition]]:
    requested = list(values)
    unknown = sorted(set(requested) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown conditions {unknown}; choose from {list(CONDITION_ORDER)}")
    requested_set = set(requested)
    result: Dict[str, list[ExperimentCondition]] = {}
    for track in tracks:
        conditions = [
            CONDITIONS[key]
            for key in CONDITION_ORDER
            if CONDITIONS[key].track == track
            and (not requested_set or key in requested_set)
        ]
        if requested_set and not conditions:
            continue
        result[track] = conditions
    selected_keys = {condition.key for values in result.values() for condition in values}
    unused = requested_set - selected_keys
    if unused:
        raise ValueError(
            f"conditions {sorted(unused)} do not belong to selected tracks {list(tracks)}"
        )
    return result


def validate_settings(args: argparse.Namespace) -> None:
    for name in ("explicit_train_ratio", "online_train_ratio"):
        value = float(getattr(args, name))
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0.0 <= args.fixed_omega <= 1.0:
        raise ValueError("fixed_omega must be between 0 and 1")
    if args.sim_rounds < 2:
        raise ValueError("sim_rounds must be at least 2")
    if args.sim_seeds < 1:
        raise ValueError("sim_seeds must be at least 1")


def shared_root(output_dir: str | Path, split_name: str) -> Path:
    return Path(output_dir).resolve() / "shared" / split_name


def track_root(output_dir: str | Path, track: str) -> Path:
    return Path(output_dir).resolve() / "tracks" / track


def condition_root(
    output_dir: str | Path,
    condition: ExperimentCondition,
) -> Path:
    return track_root(output_dir, condition.track) / "conditions" / condition.key


def hidden_case_root(output_dir: str | Path, case: ExperimentCase) -> Path:
    return Path(output_dir).resolve() / "shared" / "hidden_profiles" / "cases" / case.case_id


def scenario_path(
    output_dir: str | Path,
    case: ExperimentCase,
    seed_index: int,
) -> Path:
    return (
        Path(output_dir).resolve() / "shared" / "simulator_scenarios"
        / case.case_id / f"seed_{seed_index:03d}.json"
    )


def _copy_once(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def profile_snapshot(profile: Mapping[str, Any]) -> Dict[str, Any]:
    if "state_axis" not in profile or not isinstance(profile["state_axis"], Mapping):
        raise ValueError("runtime profile snapshot requires state_axis")
    state = profile["state_axis"]
    if "static_profile" not in state or not isinstance(state["static_profile"], Mapping):
        raise ValueError("runtime profile snapshot requires state_axis.static_profile")
    static_profile = state["static_profile"]
    return {
        "static_profile_sha256": _sha256_json(static_profile),
        "profile_completeness": compute_profile_completeness(static_profile),
        "profile_entropy": compute_portrait_entropy(static_profile),
    }


def prepare_common_assets(
    case: ExperimentCase,
    root: str | Path,
    config_path: str,
) -> CasePaths:
    paths = CasePaths.for_case(root, case)
    paths.ensure_parents()
    build_case_persona(case, paths, config_path)
    build_case_profile(case, paths, config_path)
    return paths


def prepare_condition_assets(
    case: ExperimentCase,
    source: CasePaths,
    output_dir: str | Path,
    condition: ExperimentCondition,
    fixed_omega: float,
) -> CasePaths:
    target = CasePaths.for_case(condition_root(output_dir, condition), case)
    target.ensure_parents()
    existing = (target.persona.exists(), target.profile.exists(), target.runtime_profile.exists())
    if any(existing) and not all(existing):
        raise RuntimeError(
            f"partial assets for {condition.key}/{case.case_id}; use a new output directory"
        )
    _copy_once(source.persona, target.persona)
    if condition.uses_explicit_profile:
        _copy_once(source.profile, target.profile)
        _copy_once(source.runtime_profile, target.runtime_profile)
    elif not target.profile.exists():
        save_json(str(target.profile), {})
        save_json(str(target.runtime_profile), create_empty_profile())
    if not target.asset_manifest.exists():
        save_json(str(target.asset_manifest), {
            "protocol_version": PROTOCOL_VERSION,
            "condition": condition.manifest(fixed_omega),
            "case": asdict(case),
            "source_initialization": str(source.case_root),
            "explicit_profile_copied": condition.uses_explicit_profile,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    return target


def prepare_hidden_profile(
    case: ExperimentCase,
    output_dir: str | Path,
    config_path: str,
) -> Dict[str, str]:
    """Build evidence-backed hidden targets H from P* minus the 50% profile P0."""
    root = hidden_case_root(output_dir, case)
    profile_path = root / "hidden_user_profile.json"
    claims_path = root / "hidden_claim_manifest.json"
    context_path = root / "simulator_context.json"
    initial_source = CasePaths.for_case(
        shared_root(output_dir, "online_50_50"), case
    ).profile
    required_paths = (profile_path, claims_path, context_path)
    if all(path.exists() for path in required_paths):
        context = load_json(str(context_path))
        required_context = {
            "protocol_version", "case_id", "dataset_sha256", "user_name",
            "hidden_profile_source", "hidden_target_definition", "style_source",
            "style_examples", "hidden_from_exploration_agent", "all_sessions",
            "initialization_sessions", "held_out_sessions", "initial_profile_path",
            "hidden_claim_manifest_path",
        }
        if not isinstance(context, dict) or set(context) != required_context:
            raise RuntimeError(f"incompatible hidden simulator context for {case.case_id}")
        if (
            context["protocol_version"] != PROTOCOL_VERSION
            or context["case_id"] != case.case_id
            or context["dataset_sha256"] != case.dataset_sha256
            or context["initialization_sessions"] != list(case.train_sessions)
            or context["held_out_sessions"] != list(case.test_sessions)
        ):
            raise RuntimeError(f"stale hidden simulator assets for {case.case_id}")
        hidden_claims_from_manifest(load_json(str(claims_path)))
        return {
            "profile": str(profile_path), "claims": str(claims_path),
            "context": str(context_path),
        }
    if any(path.exists() for path in required_paths):
        raise RuntimeError(f"partial hidden simulator assets for {case.case_id}")
    if not initial_source.exists():
        raise FileNotFoundError(
            f"initial 50% profile missing for {case.case_id}: {initial_source}"
        )

    root.mkdir(parents=True, exist_ok=True)
    chat = load_json(case.dataset_path)
    all_keys = session_keys(chat)
    all_sessions = [chat[key] for key in all_keys]
    llm = LLMClient(config_path)
    hidden_profile = extract_profile(llm, all_sessions, case.user_speaker)
    _validate_extracted_profile(hidden_profile)
    initial_profile = load_json(str(initial_source))
    _validate_extracted_profile(initial_profile)

    held_out_bubbles = bubbles_for_sessions(chat, case.test_sessions)
    heldout_evidence: Dict[str, str] = {}
    for bubble in held_out_bubbles:
        if bubble.speaker != case.user_speaker:
            continue
        evidence_id = "|".join(bubble.dia_ids)
        if not evidence_id:
            raise ValueError(f"empty held-out evidence ID in {case.case_id}")
        if evidence_id in heldout_evidence:
            raise ValueError(f"duplicate held-out evidence ID {evidence_id}")
        heldout_evidence[evidence_id] = bubble.content
    if not heldout_evidence:
        raise ValueError(f"no held-out user evidence for {case.case_id}")
    claim_manifest = build_hidden_claim_manifest(
        llm, case.user_speaker, initial_profile, hidden_profile, heldout_evidence
    )
    claim_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": case.case_id,
        "dataset_sha256": case.dataset_sha256,
        "initialization_sessions": list(case.train_sessions),
        "held_out_sessions": list(case.test_sessions),
        **claim_manifest,
    }
    style_examples = [
        bubble.content
        for bubble in held_out_bubbles
        if bubble.speaker == case.user_speaker
    ][:20]
    save_json(str(profile_path), hidden_profile)
    save_json(str(claims_path), claim_manifest)
    save_json(str(context_path), {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": case.case_id,
        "dataset_sha256": case.dataset_sha256,
        "user_name": case.user_speaker,
        "hidden_profile_source": "all chronological sessions",
        "hidden_target_definition": (
            "semantic new/refinement claims relative to the first-50% profile, "
            "with direct held-out user-message evidence"
        ),
        "style_source": "held-out 50% user messages only",
        "style_examples": style_examples,
        "hidden_from_exploration_agent": True,
        "all_sessions": all_keys,
        "initialization_sessions": list(case.train_sessions),
        "held_out_sessions": list(case.test_sessions),
        "initial_profile_path": str(initial_source),
        "hidden_claim_manifest_path": str(claims_path),
    })
    return {
        "profile": str(profile_path), "claims": str(claims_path),
        "context": str(context_path),
    }


def prepare(
    selected: Mapping[str, Sequence[ExperimentCondition]],
    explicit_cases: Sequence[ExperimentCase],
    online_cases: Sequence[ExperimentCase],
    output_dir: str | Path,
    config_path: str,
    fixed_omega: float,
) -> None:
    if TRACK_EXPLICIT in selected:
        for case in explicit_cases:
            source = prepare_common_assets(
                case, shared_root(output_dir, "explicit_90_10"), config_path
            )
            for condition in selected[TRACK_EXPLICIT]:
                prepare_condition_assets(case, source, output_dir, condition, fixed_omega)

    online_tracks = [track for track in (TRACK_EXPLORATION, TRACK_BAYESIAN) if track in selected]
    if online_tracks:
        for case in online_cases:
            source = prepare_common_assets(
                case, shared_root(output_dir, "online_50_50"), config_path
            )
            for track in online_tracks:
                for condition in selected[track]:
                    prepare_condition_assets(case, source, output_dir, condition, fixed_omega)
            if TRACK_EXPLORATION in selected:
                prepare_hidden_profile(case, output_dir, config_path)


def _restore_resume_state(
    agent: Exp3Agent,
    states: JsonlStore,
    completed_ids: set[str],
) -> int:
    if not completed_ids:
        return 0
    state_rows = states.read_all()
    if not state_rows:
        raise RuntimeError("completed predictions exist but understanding state is empty")
    last = state_rows[-1]
    required = {
        "empathy_state_for_next_turn", "future_understanding", "interaction_count_after"
    }
    missing = required - set(last)
    if missing:
        raise RuntimeError(f"resume state missing required fields: {sorted(missing)}")
    agent.last_empathy_state = deepcopy(last["empathy_state_for_next_turn"])
    agent.last_prediction = deepcopy(last["future_understanding"])
    return int(last["interaction_count_after"])


def run_dataset_replay(
    case: ExperimentCase,
    output_dir: str | Path,
    condition: ExperimentCondition,
    config_path: str,
    prompt_version: str,
    fixed_omega: float,
) -> int:
    """Run fixed REALTALK history; only the human reference reply enters memory."""
    paths = CasePaths.for_case(condition_root(output_dir, condition), case)
    if not paths.persona.exists() or not paths.runtime_profile.exists():
        raise FileNotFoundError(
            f"assets missing for {condition.key}/{case.case_id}; run prepare first"
        )
    chat = load_json(case.dataset_path)
    examples = build_reply_examples(chat, case)
    validate_no_leakage(case, examples)
    example_map = _example_by_user_dia(examples)
    bubbles = bubbles_for_sessions(chat, case.test_sessions)
    predictions = JsonlStore(paths.predictions)
    states = JsonlStore(paths.understanding)
    _assert_consistent_stores(predictions, states)
    prior_rows = predictions.read_all()
    completed_ids = {row["example_id"] for row in prior_rows}
    start_index = _resume_position(bubbles, examples, completed_ids)

    bundle = condition_prompt_bundle(get_exp2_prompt_bundle(prompt_version), condition)
    incompatible = [
        row for row in prior_rows
        if "protocol_version" not in row
        or "condition" not in row
        or "prompt_sha256" not in row
        or row["protocol_version"] != PROTOCOL_VERSION
        or row["condition"] != condition.key
        or row["prompt_sha256"] != bundle.fingerprint
    ]
    if incompatible:
        raise RuntimeError(
            f"incompatible existing outputs for {condition.key}/{case.case_id}"
        )

    agent = Exp3Agent(
        config_path=config_path,
        profile_path=str(paths.runtime_profile),
        persona_path=str(paths.persona),
        user_name=case.user_speaker,
        prompt_version=prompt_version,
        condition=condition,
        fixed_omega=fixed_omega,
    )
    agent.memory_manager = MemoryOSLocal(
        persist_path=str(paths.memory_db), config_path=config_path
    )
    resume_count = _restore_resume_state(agent, states, completed_ids)
    for prior in bubbles[max(0, start_index - 6):start_index]:
        _append_real_bubble(agent, case, prior)
    if completed_ids:
        agent.epistemic_tracker.interaction_count = resume_count

    generated_count = 0
    index = start_index
    while index < len(bubbles):
        user_bubble = bubbles[index]
        reference_bubble = bubbles[index + 1] if index + 1 < len(bubbles) else None
        example = example_map.get(user_bubble.dia_ids)
        is_target = (
            example is not None
            and reference_bubble is not None
            and reference_bubble.dia_ids == example.reference_dia_ids
        )
        if not is_target:
            _append_real_bubble(agent, case, user_bubble)
            agent._run_memory_steps()
            index += 1
            continue

        _append_real_bubble(agent, case, user_bubble)
        relevant_memory = agent.memory_manager.retrieve_relevant_memory(example.user_message)
        before = profile_snapshot(agent.user_profile)
        static_profile = state_axis(agent.user_profile).get("static_profile", {})
        omega = agent.epistemic_tracker.compute(static_profile)
        generated_reply, alignment, previous_empathy, model_timing = (
            _generate_with_parallel_alignment(agent, example.user_message, relevant_memory)
        )

        # Critical control: the model reply is evaluation-only. The recorded
        # REALTALK reply is the sole assistant observation written to memory.
        _append_real_bubble(agent, case, reference_bubble)
        agent._run_memory_steps()
        after = profile_snapshot(agent.user_profile)
        state = state_axis(agent.user_profile)
        created_at = datetime.now(timezone.utc).isoformat()
        shared = {
            "example_id": example.example_id,
            "case_id": case.case_id,
            "session_id": example.session_id,
            "condition": condition.key,
            "condition_label": condition.label,
            "track": condition.track,
            "protocol_version": PROTOCOL_VERSION,
            "prompt_version": bundle.version,
            "prompt_sha256": bundle.fingerprint,
            "omega": omega,
            "interaction_count_after": agent.epistemic_tracker.interaction_count,
            "profile_before": before,
            "profile_after": after,
            "profile_changed": before["static_profile_sha256"] != after["static_profile_sha256"],
            "created_at": created_at,
        }
        states.append({
            **shared,
            "current_understanding": alignment.get("understanding", {}),
            "future_understanding": alignment.get("prediction", {}),
            "exploration_decision": alignment.get("exploration", {}),
            "empathy_state_for_next_turn": deepcopy(agent.last_empathy_state),
            "core_current_state": state.get("current_state", {}),
            "core_projected_state": state.get("projected_state", {}),
        })
        predictions.append({
            **asdict(example),
            **shared,
            "generated_reply": generated_reply,
            "history_policy": "teacher_forcing_real_replies_only",
            "generated_reply_enters_memory": False,
            "reference_reply_enters_memory": True,
            "profile_policy": condition.update_mode,
            "modeling_policy": condition.modeling_mode,
            "exploration_policy": condition.exploration_mode,
            "generation_input_audit": {
                "contains_user_message": True,
                "contains_reference_reply": False,
                "contains_next_user_message": False,
                "contains_explicit_profile": condition.uses_explicit_profile,
                "previous_empathy_state": previous_empathy,
            },
            "model_timing": model_timing,
        })
        generated_count += 1
        index += 2
    return generated_count


def _simulator_assets(
    output_dir: str | Path,
    case: ExperimentCase,
) -> tuple[list[HiddenClaim], Dict[str, Any]]:
    root = hidden_case_root(output_dir, case)
    claims_path = root / "hidden_claim_manifest.json"
    context_path = root / "simulator_context.json"
    if not claims_path.exists() or not context_path.exists():
        raise FileNotFoundError(f"hidden simulator assets missing for {case.case_id}")
    manifest = load_json(str(claims_path))
    context = load_json(str(context_path))
    for field, expected in (
        ("protocol_version", PROTOCOL_VERSION),
        ("case_id", case.case_id),
        ("dataset_sha256", case.dataset_sha256),
    ):
        if manifest[field] != expected or context[field] != expected:
            raise RuntimeError(f"incompatible simulator {field} for {case.case_id}")
    return hidden_claims_from_manifest(manifest), context


def ensure_scenario(
    simulator: HiddenProfileUserSimulator,
    output_dir: str | Path,
    case: ExperimentCase,
    seed_index: int,
) -> Dict[str, Any]:
    path = scenario_path(output_dir, case, seed_index)
    metadata_keys = {
        "protocol_version", "case_id", "dataset_sha256", "seed_index",
        "shared_across_exploration_conditions",
    }
    payload_keys = {
        "user_reply", "revealed_claim_ids", "disclosure_strength",
        "withheld_or_refused", "perceived_burden", "burden_reason",
        "disclosure_decision", "disclosure_depth", "trust", "fatigue",
    }
    if path.exists():
        existing = load_json(str(path))
        if set(existing) != metadata_keys | payload_keys:
            raise RuntimeError(f"incompatible scenario schema for {case.case_id}/seed {seed_index}")
        if (
            existing["protocol_version"] != PROTOCOL_VERSION
            or existing["case_id"] != case.case_id
            or existing["dataset_sha256"] != case.dataset_sha256
            or existing["seed_index"] != seed_index
            or existing["shared_across_exploration_conditions"] is not True
        ):
            raise RuntimeError(f"stale scenario for {case.case_id}/seed {seed_index}")
        validate_simulator_payload(
            {key: existing[key] for key in payload_keys}, set(simulator.claim_by_id)
        )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = simulator.opening_turn()
    save_json(str(path), {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": case.case_id,
        "dataset_sha256": case.dataset_sha256,
        "seed_index": seed_index,
        "shared_across_exploration_conditions": True,
        **payload,
    })
    return load_json(str(path))


def prepare_simulation_run(
    output_dir: str | Path,
    condition: ExperimentCondition,
    case: ExperimentCase,
    seed_index: int,
) -> SimulationPaths:
    source = CasePaths.for_case(condition_root(output_dir, condition), case)
    paths = SimulationPaths.for_run(output_dir, condition, case, seed_index)
    paths.ensure_parents()
    _copy_once(source.runtime_profile, paths.initial_profile)
    _copy_once(source.runtime_profile, paths.runtime_profile)
    _copy_once(source.persona, paths.persona)
    return paths


def run_exploration_simulation(
    case: ExperimentCase,
    output_dir: str | Path,
    condition: ExperimentCondition,
    config_path: str,
    prompt_version: str,
    fixed_omega: float,
    seed_index: int,
    max_rounds: int,
) -> int:
    """Interact with a hidden-profile simulator; generated replies form history."""
    paths = prepare_simulation_run(output_dir, condition, case, seed_index)
    if paths.generation_summary.exists():
        summary = load_json(str(paths.generation_summary))
        if (
            summary["protocol_version"] != PROTOCOL_VERSION
            or summary["case_id"] != case.case_id
            or summary["condition"]["key"] != condition.key
            or summary["seed_index"] != seed_index
            or summary["configured_max_rounds"] != max_rounds
        ):
            raise RuntimeError(
                f"incompatible completed simulation for {condition.key}/{case.case_id}/seed {seed_index}"
            )
        return 0
    hidden_claims, simulator_context = _simulator_assets(output_dir, case)
    initial_profile = load_json(str(paths.initial_profile))
    simulator = HiddenProfileUserSimulator(
        llm=LLMClient(config_path),
        user_name=case.user_speaker,
        initial_profile=initial_profile,
        hidden_claims=hidden_claims,
        style_examples=simulator_context["style_examples"],
        seed_index=seed_index,
    )
    scenario = ensure_scenario(simulator, output_dir, case, seed_index)
    agent = Exp3Agent(
        config_path=config_path,
        profile_path=str(paths.runtime_profile),
        persona_path=str(paths.persona),
        user_name=case.user_speaker,
        prompt_version=prompt_version,
        condition=condition,
        fixed_omega=fixed_omega,
    )
    agent.memory_manager = MemoryOSLocal(
        persist_path=str(paths.memory_db), config_path=config_path
    )
    rows = JsonlStore(paths.dialogue, id_field="turn_id")
    prior_rows = rows.read_all()
    conversation: list[Dict[str, str]] = []
    for row in prior_rows:
        conversation.extend((
            {"speaker": case.user_speaker, "text": row["user_message"]},
            {"speaker": case.agent_speaker, "text": row["agent_reply"]},
        ))

    if prior_rows:
        last = prior_rows[-1]
        for row in prior_rows:
            required_resume_fields = {
                "protocol_version", "case_id", "condition", "seed_index", "turn_index",
                "user_message", "user_private_annotation", "agent_reply",
                "empathy_state_for_next_turn", "future_understanding",
                "interaction_count_after", "profile_snapshot_path",
                "next_user_message", "next_user_annotation",
            }
            missing = required_resume_fields - set(row)
            if missing:
                raise RuntimeError(f"simulation resume row missing {sorted(missing)}")
            if (
                row["protocol_version"] != PROTOCOL_VERSION
                or row["case_id"] != case.case_id
                or row["condition"] != condition.key
                or row["seed_index"] != seed_index
            ):
                raise RuntimeError("incompatible simulation dialogue found during resume")
            annotation = row["user_private_annotation"]
            if not isinstance(annotation, dict) or set(annotation) != set(SIMULATOR_ANNOTATION_KEYS):
                raise RuntimeError("invalid private simulator annotation during resume")
            simulator.disclosed_ids.update(annotation["revealed_claim_ids"])
            simulator.trust = float(annotation["trust"])
            simulator.fatigue = float(annotation["fatigue"])
        if len(prior_rows) >= max_rounds:
            current_user = ""
            current_annotation: Dict[str, Any] = {}
        else:
            if not isinstance(last["next_user_message"], str) or not last["next_user_message"].strip():
                raise RuntimeError("partial simulation lacks next_user_message for resume")
            if not isinstance(last["next_user_annotation"], dict) or set(last["next_user_annotation"]) != set(SIMULATOR_ANNOTATION_KEYS):
                raise RuntimeError("partial simulation has invalid next_user_annotation")
            current_user = last["next_user_message"].strip()
            current_annotation = dict(last["next_user_annotation"])
            simulator.disclosed_ids.update(current_annotation["revealed_claim_ids"])
            simulator.trust = float(current_annotation["trust"])
            simulator.fatigue = float(current_annotation["fatigue"])
        agent.last_empathy_state = deepcopy(last["empathy_state_for_next_turn"])
        agent.last_prediction = deepcopy(last["future_understanding"])
        for row in prior_rows[-3:]:
            agent.memory_manager.append_stm("user", row["user_message"])
            agent.memory_manager.append_stm("assistant", row["agent_reply"])
        agent.epistemic_tracker.interaction_count = int(last["interaction_count_after"])
    else:
        current_user = scenario["user_reply"]
        current_annotation = {key: scenario[key] for key in SIMULATOR_ANNOTATION_KEYS}
        save_json(str(paths.profile_snapshots / "turn_000.json"), initial_profile)

    generated = 0
    for turn_index in range(len(prior_rows) + 1, max_rounds + 1):
        agent.memory_manager.append_stm("user", current_user)
        agent.epistemic_tracker.increment()
        conversation.append({"speaker": case.user_speaker, "text": current_user})
        relevant_memory = agent.memory_manager.retrieve_relevant_memory(current_user)
        before = profile_snapshot(agent.user_profile)
        static_profile = state_axis(agent.user_profile).get("static_profile", {})
        omega = agent.epistemic_tracker.compute(static_profile)
        reply, alignment, previous_empathy, timing = _generate_with_parallel_alignment(
            agent, current_user, relevant_memory
        )
        agent.memory_manager.append_stm("assistant", reply)
        agent.epistemic_tracker.increment()
        agent.last_agent_response = reply
        conversation.append({"speaker": case.agent_speaker, "text": reply})
        agent._run_memory_steps()
        after = profile_snapshot(agent.user_profile)
        snapshot_path = paths.profile_snapshots / f"turn_{turn_index:03d}.json"
        save_json(str(snapshot_path), agent.user_profile)

        next_payload: Dict[str, Any] = {}
        if turn_index < max_rounds:
            next_payload = simulator.respond(conversation, reply)

        rows.append({
            "turn_id": f"{case.case_id}:seed_{seed_index:03d}:turn_{turn_index:03d}",
            "case_id": case.case_id,
            "seed_index": seed_index,
            "turn_index": turn_index,
            "condition": condition.key,
            "protocol_version": PROTOCOL_VERSION,
            "user_message": current_user,
            "user_private_annotation": current_annotation,
            "agent_reply": reply,
            "agent_reply_is_part_of_interactive_history": True,
            "omega": omega,
            "exploration_decision": alignment["exploration"],
            "future_understanding": alignment["prediction"],
            "empathy_state_for_next_turn": deepcopy(agent.last_empathy_state),
            "previous_empathy_state": previous_empathy,
            "interaction_count_after": agent.epistemic_tracker.interaction_count,
            "profile_before": before,
            "profile_after": after,
            "profile_changed": before["static_profile_sha256"] != after["static_profile_sha256"],
            "profile_snapshot_path": str(snapshot_path),
            "next_user_message": next_payload["user_reply"] if next_payload else None,
            "next_user_annotation": (
                {key: next_payload[key] for key in SIMULATOR_ANNOTATION_KEYS}
                if next_payload else None
            ),
            "model_timing": timing,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        generated += 1
        if next_payload:
            current_user = next_payload["user_reply"]
            current_annotation = {key: next_payload[key] for key in SIMULATOR_ANNOTATION_KEYS}

    agent.finalize_session()
    save_json(str(paths.final_profile), agent.user_profile)
    all_rows = rows.read_all()
    if len(all_rows) != max_rounds:
        raise RuntimeError(
            f"simulation ended with {len(all_rows)} rows; expected {max_rounds}"
        )
    annotations = [row["user_private_annotation"] for row in all_rows]
    burdens = [float(value["perceived_burden"]) for value in annotations]
    refusals = [bool(value["withheld_or_refused"]) for value in annotations]
    revealed_ids = sorted({
        str(claim_id)
        for value in annotations
        for claim_id in value["revealed_claim_ids"]
    })
    unknown_revealed = set(revealed_ids) - {claim.claim_id for claim in hidden_claims}
    if unknown_revealed:
        raise RuntimeError(f"simulation revealed unknown hidden claims: {sorted(unknown_revealed)}")
    questions = [detect_exploration_question(str(row["agent_reply"])) for row in all_rows]
    initial_snapshot = profile_snapshot(initial_profile)
    final_snapshot = profile_snapshot(agent.user_profile)
    save_json(str(paths.generation_summary), {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": case.case_id,
        "condition": condition.manifest(fixed_omega),
        "seed_index": seed_index,
        "round_count": len(all_rows),
        "configured_max_rounds": max_rounds,
        "scenario_path": str(scenario_path(output_dir, case, seed_index)),
        "hidden_claim_manifest_path": str(hidden_case_root(output_dir, case) / "hidden_claim_manifest.json"),
        "initial_profile": initial_snapshot,
        "final_profile": final_snapshot,
        "profile_entropy_reduction": round(
            initial_snapshot["profile_entropy"] - final_snapshot["profile_entropy"], 6
        ),
        "revealed_hidden_claim_ids": revealed_ids,
        "revealed_hidden_claim_count": len(revealed_ids),
        "mean_user_burden": mean(burdens),
        "refusal_rate": mean(refusals),
        "exploration_question_rate": mean(questions),
        "generated_agent_replies_enter_history": True,
    })
    return generated


METRIC_DIRECTIONS = {
    "lexical": "higher", "semantic": "higher", "reflective": "higher",
    "grounding": "higher", "sentiment": "higher", "emotion": "higher",
    "intimacy": "lower", "empathy": "lower",
}


def _format_stat(stat: Mapping[str, float]) -> str:
    return f"{float(stat['mean']):.3f} ± {float(stat['std']):.3f}"


def dataset_comparison_payload(
    aggregates: Mapping[str, Mapping[str, Any]],
    conditions: Sequence[ExperimentCondition],
    scores_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    stage_by_example: Mapping[str, str],
    stage_aggregates: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    baseline = aggregates[conditions[0].key]
    rows = []
    for condition in conditions:
        aggregate = aggregates[condition.key]
        degradation = {}
        for metric in TABLE2_METRICS:
            base_value = float(baseline["ours"][metric]["mean"])
            value = float(aggregate["ours"][metric]["mean"])
            degradation[metric] = (
                base_value - value
                if METRIC_DIRECTIONS[metric] == "higher"
                else value - base_value
            )
        rows.append({
            "condition": condition.key,
            "label": condition.label,
            "example_count": aggregate["example_count"],
            "speaker_count": aggregate["speaker_count"],
            "metrics": aggregate["ours"],
            "degradation_vs_first_condition": degradation,
            "early_middle_late": (
                stage_aggregates.get(condition.key, {})
                if stage_aggregates is not None
                else {}
            ),
        })
    paired_comparisons = []
    reference_key = conditions[0].key
    reference_rows = {str(row["example_id"]): row for row in scores_by_condition[reference_key]}
    if len(reference_rows) != len(scores_by_condition[reference_key]):
        raise ValueError(f"duplicate example IDs in {reference_key} scores")
    for condition in conditions[1:]:
        candidate_rows = {
            str(row["example_id"]): row
            for row in scores_by_condition[condition.key]
        }
        if len(candidate_rows) != len(scores_by_condition[condition.key]):
            raise ValueError(f"duplicate example IDs in {condition.key} scores")
        if set(candidate_rows) != set(reference_rows):
            missing = sorted(set(reference_rows) - set(candidate_rows))
            extra = sorted(set(candidate_rows) - set(reference_rows))
            raise ValueError(
                f"paired score IDs differ for {condition.key}; missing={missing}, extra={extra}"
            )
        per_example = []
        for example_id in sorted(reference_rows):
            metric_degradation = {}
            for metric in TABLE2_METRICS:
                reference_value = float(reference_rows[example_id][metric])
                candidate_value = float(candidate_rows[example_id][metric])
                metric_degradation[metric] = (
                    reference_value - candidate_value
                    if METRIC_DIRECTIONS[metric] == "higher"
                    else candidate_value - reference_value
                )
            per_example.append({
                "example_id": example_id,
                "stage": stage_by_example[example_id],
                "degradation": metric_degradation,
            })
        by_stage: Dict[str, Dict[str, float]] = {}
        for stage in ("early", "middle", "late"):
            stage_rows = [row for row in per_example if row["stage"] == stage]
            if stage_rows:
                by_stage[stage] = {
                    metric: mean(row["degradation"][metric] for row in stage_rows)
                    for metric in TABLE2_METRICS
                }
        paired_comparisons.append({
            "reference_condition": reference_key,
            "candidate_condition": condition.key,
            "positive_means_candidate_is_worse": True,
            "mean_paired_degradation": {
                metric: mean(row["degradation"][metric] for row in per_example)
                for metric in TABLE2_METRICS
            },
            "early_middle_late_mean_paired_degradation": by_stage,
            "per_example": per_example,
        })
    return {
        "protocol_version": PROTOCOL_VERSION,
        "temporal_segmentation_required": conditions[0].track == TRACK_BAYESIAN,
        "evaluation_scope": "comparative" if len(conditions) > 1 else "absolute_only",
        "reference_condition": conditions[0].key if len(conditions) > 1 else None,
        "positive_degradation_means_worse": True if len(conditions) > 1 else None,
        "conditions": rows,
        "paired_comparisons": paired_comparisons,
    }


def render_dataset_table(
    title: str,
    aggregates: Mapping[str, Mapping[str, Any]],
    conditions: Sequence[ExperimentCondition],
) -> str:
    headers = (
        "Method", "Lexical ↑", "Semantic ↑", "Reflective ↑", "Grounding ↑",
        "Sentiment ↑", "Emotion ↑", "Intimacy ↓", "Empathy ↓",
    )
    lines = [
        f"# {title}", "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(TABLE2_METRICS)) + " |",
    ]
    for condition in conditions:
        aggregate = aggregates[condition.key]
        lines.append("| " + " | ".join(
            [condition.label]
            + [_format_stat(aggregate["ours"][metric]) for metric in TABLE2_METRICS]
        ) + " |")
    return "\n".join(lines) + "\n"


def evaluate_dataset_tracks(
    selected: Mapping[str, Sequence[ExperimentCondition]],
    explicit_cases: Sequence[ExperimentCase],
    online_cases: Sequence[ExperimentCase],
    output_dir: str | Path,
    config_path: str,
    judge_model: str | None,
    judge_config_section: str,
    device: str,
    batch_size: int,
) -> Dict[str, Dict[str, str]]:
    tracks = [track for track in (TRACK_EXPLICIT, TRACK_BAYESIAN) if track in selected]
    if not tracks:
        return {}
    judge = Exp2JudgeClient(config_path, config_section=judge_config_section, model=judge_model)
    evaluator = Table2Evaluator(judge_llm=judge, device=device, batch_size=batch_size)
    outputs: Dict[str, Dict[str, str]] = {}
    for track in tracks:
        cases = explicit_cases if track == TRACK_EXPLICIT else online_cases
        conditions = selected[track]
        aggregates: Dict[str, Dict[str, Any]] = {}
        stage_aggregates: Dict[str, Dict[str, Any]] = {}
        scores_by_condition: Dict[str, list[Dict[str, Any]]] = {}
        stage_by_example: Dict[str, str] = {}
        reference_case_order: Dict[str, list[str]] = {}
        for condition in conditions:
            scores: list[Dict[str, Any]] = []
            staged_scores: Dict[str, list[Dict[str, Any]]] = {
                "early": [], "middle": [], "late": [],
            }
            for case in cases:
                paths = CasePaths.for_case(condition_root(output_dir, condition), case)
                if not paths.predictions.exists():
                    raise FileNotFoundError(
                        f"missing predictions for {condition.key}/{case.case_id}"
                    )
                case_scores = evaluate_case_table2(case, paths, evaluator)
                case_ids = [str(score["example_id"]) for score in case_scores]
                if condition == conditions[0]:
                    reference_case_order[case.case_id] = case_ids
                elif case_ids != reference_case_order[case.case_id]:
                    raise ValueError(
                        f"ordered paired examples differ for {condition.key}/{case.case_id}"
                    )
                scores.extend(case_scores)
                total = len(case_scores)
                for index, score in enumerate(case_scores):
                    if total == 0:
                        raise ValueError(f"empty evaluation scores for {condition.key}/{case.case_id}")
                    stage_index = min(2, (index * 3) // total)
                    stage = ("early", "middle", "late")[stage_index]
                    staged_scores[stage].append(score)
                    example_id = str(score["example_id"])
                    if example_id in stage_by_example and stage_by_example[example_id] != stage:
                        raise ValueError(f"inconsistent temporal stage for {example_id}")
                    stage_by_example[example_id] = stage
            aggregates[condition.key] = aggregate_table2_scores(scores)
            scores_by_condition[condition.key] = scores
            stage_aggregates[condition.key] = {
                stage: aggregate_table2_scores(rows)
                for stage, rows in staged_scores.items()
                if rows
            }
        result_dir = track_root(output_dir, track) / "evaluation"
        result_dir.mkdir(parents=True, exist_ok=True)
        json_path = result_dir / "comparison.json"
        md_path = result_dir / "comparison.md"
        save_json(
            str(json_path),
            dataset_comparison_payload(
                aggregates, conditions, scores_by_condition,
                stage_by_example, stage_aggregates,
            ),
        )
        title = (
            "Exp3-A: Explicit User Modeling (90/10)"
            if track == TRACK_EXPLICIT
            else "Exp3-C: Bayesian Updating (50/50 Real-Dialogue Replay)"
        )
        md_path.write_text(
            render_dataset_table(title, aggregates, conditions), encoding="utf-8"
        )
        outputs[track] = {"comparison_json": str(json_path), "comparison_markdown": str(md_path)}
    return outputs


def _legacy_render_exploration_table(
    aggregates: Mapping[str, Mapping[str, Any]],
    conditions: Sequence[ExperimentCondition],
) -> str:
    metrics = (
        ("final_precision", "Profile Precision ↑"),
        ("final_recall", "Profile Recall ↑"),
        ("final_f1", "Profile F1 ↑"),
        ("recall_gain", "Recall Gain ↑"),
        ("discovery_efficiency", "Efficiency ↑"),
        ("mean_user_burden", "Burden ↓"),
        ("refusal_rate", "Refusal ↓"),
    )
    lines = [
        "# Exp3-B: Adaptive Exploration (50/50 Hidden-Profile Simulation)", "",
        "| Method | " + " | ".join(label for _, label in metrics) + " |",
        "| --- | " + " | ".join("---:" for _ in metrics) + " |",
    ]
    for condition in conditions:
        aggregate = aggregates[condition.key]
        lines.append("| " + condition.label + " | " + " | ".join(
            _format_stat(aggregate[key]) for key, _ in metrics
        ) + " |")
    return "\n".join(lines) + "\n"


def _legacy_evaluate_exploration_track(
    conditions: Sequence[ExperimentCondition],
    cases: Sequence[ExperimentCase],
    output_dir: str | Path,
    config_path: str,
    judge_model: str | None,
    judge_config_section: str,
    sim_seeds: int,
) -> Dict[str, str]:
    judge = Exp2JudgeClient(config_path, config_section=judge_config_section, model=judge_model)
    aggregates: Dict[str, Dict[str, Any]] = {}
    all_results: Dict[str, list[Dict[str, Any]]] = {}
    for condition in conditions:
        rows = []
        for case in cases:
            hidden_profile, _ = _simulator_assets(output_dir, case)
            for seed_index in range(1, sim_seeds + 1):
                paths = SimulationPaths.for_run(
                    output_dir, condition, case, seed_index
                )
                if paths.evaluation.exists():
                    result = load_json(str(paths.evaluation))
                else:
                    if not paths.generation_summary.exists() or not paths.final_profile.exists():
                        raise FileNotFoundError(
                            f"simulation incomplete for {condition.key}/{case.case_id}/seed {seed_index}"
                        )
                    summary = load_json(str(paths.generation_summary))
                    initial_profile = load_json(str(paths.initial_profile))
                    final_profile = load_json(str(paths.final_profile))
                    discovery = evaluate_profile_discovery(
                        judge, hidden_profile, initial_profile, final_profile
                    )
                    round_count = int(summary["round_count"])
                    result = {
                        "case_id": case.case_id,
                        "condition": condition.key,
                        "seed_index": seed_index,
                        **discovery,
                        "discovery_efficiency": round(
                            discovery["recall_gain"] / round_count if round_count else 0.0,
                            6,
                        ),
                        "mean_user_burden": summary["mean_user_burden"],
                        "refusal_rate": summary["refusal_rate"],
                        "exploration_question_rate": summary["exploration_question_rate"],
                        "profile_entropy_reduction": summary["profile_entropy_reduction"],
                        "round_count": round_count,
                        "revealed_hidden_claim_count": summary["revealed_hidden_claim_count"],
                    }
                    save_json(str(paths.evaluation), result)
                rows.append(result)
        all_results[condition.key] = rows
        aggregates[condition.key] = aggregate_discovery_results(rows)

    result_dir = track_root(output_dir, TRACK_EXPLORATION) / "evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "comparison.json"
    md_path = result_dir / "comparison.md"
    baseline = conditions[0].key
    save_json(str(json_path), {
        "protocol_version": PROTOCOL_VERSION,
        "reference_condition": baseline,
        "aggregates": aggregates,
        "per_run": all_results,
    })
    md_path.write_text(
        render_exploration_table(aggregates, conditions), encoding="utf-8"
    )
    return {"comparison_json": str(json_path), "comparison_markdown": str(md_path)}


def render_exploration_table(
    aggregates: Mapping[str, Mapping[str, Any]],
    conditions: Sequence[ExperimentCondition],
) -> str:
    metrics = (
        ("elicitation_rate", "Elicitation ↑"),
        ("end_to_end_discovery_rate", "End-to-end ↑"),
        ("final_hidden_coverage", "Final coverage ↑"),
        ("discovery_efficiency", "Claims/round ↑"),
        ("coverage_auc", "Coverage AUC ↑"),
        ("mean_user_burden", "Burden ↓"),
        ("refusal_rate", "Refusal ↓"),
    )
    lines = [
        "# Exp3-B: Adaptive Exploration (50/50 Hidden-Profile Simulation)", "",
        "| Method | " + " | ".join(label for _, label in metrics) + " |",
        "| --- | " + " | ".join("---:" for _ in metrics) + " |",
    ]
    for condition in conditions:
        aggregate = aggregates[condition.key]
        lines.append("| " + condition.label + " | " + " | ".join(
            _format_stat(aggregate[key]) for key, _ in metrics
        ) + " |")
    return "\n".join(lines) + "\n"


def _validate_exploration_summary(
    summary: Mapping[str, Any],
    condition: ExperimentCondition,
    case: ExperimentCase,
    seed_index: int,
) -> None:
    required = {
        "protocol_version", "case_id", "condition", "seed_index", "round_count",
        "configured_max_rounds", "scenario_path", "hidden_claim_manifest_path",
        "initial_profile", "final_profile", "profile_entropy_reduction",
        "revealed_hidden_claim_ids", "revealed_hidden_claim_count",
        "mean_user_burden", "refusal_rate", "exploration_question_rate",
        "generated_agent_replies_enter_history",
    }
    if set(summary) != required:
        raise RuntimeError(
            f"generation summary schema mismatch for {condition.key}/{case.case_id}/seed {seed_index}"
        )
    if (
        summary["protocol_version"] != PROTOCOL_VERSION
        or summary["case_id"] != case.case_id
        or summary["condition"]["key"] != condition.key
        or summary["seed_index"] != seed_index
        or summary["generated_agent_replies_enter_history"] is not True
    ):
        raise RuntimeError(
            f"incompatible generation summary for {condition.key}/{case.case_id}/seed {seed_index}"
        )


def _exploration_run_result(
    judge: LLMClient,
    hidden_claims: Sequence[HiddenClaim],
    condition: ExperimentCondition,
    case: ExperimentCase,
    seed_index: int,
    paths: SimulationPaths,
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    initial_profile = load_json(str(paths.initial_profile))
    final_profile = load_json(str(paths.final_profile))
    dialogue = JsonlStore(paths.dialogue, id_field="turn_id").read_all()
    round_count = int(summary["round_count"])
    if round_count < 1 or len(dialogue) != round_count:
        raise RuntimeError("dialogue length does not equal the recorded simulation round count")
    hidden_ids = {claim.claim_id for claim in hidden_claims}
    hidden_hash = _sha256_json([claim.as_dict() for claim in hidden_claims])
    discovery = evaluate_profile_discovery(
        judge, hidden_claims, initial_profile, final_profile
    )
    initial_coverage = evaluate_hidden_coverage(judge, hidden_claims, initial_profile)
    supported_initial = set(initial_coverage["supported_hidden_claim_ids"])
    previous_hash = profile_snapshot(initial_profile)["static_profile_sha256"]
    previous_supported = set(supported_initial)
    curve = [{
        "turn": 0,
        "revealed_claim_ids": [],
        "learned_claim_ids": [],
        "cumulative_revealed_count": 0,
        "cumulative_learned_count": 0,
        "hidden_coverage": 0.0,
    }]
    cumulative_revealed: set[str] = set()
    first_revealed_turn: Dict[str, int] = {}
    first_learned_turn: Dict[str, int] = {}
    for expected_turn, row in enumerate(dialogue, 1):
        if row["turn_index"] != expected_turn:
            raise RuntimeError("simulation dialogue turn indices are not contiguous")
        annotation = row["user_private_annotation"]
        if not isinstance(annotation, dict) or set(annotation) != set(SIMULATOR_ANNOTATION_KEYS):
            raise RuntimeError(f"invalid simulator annotation at turn {expected_turn}")
        turn_revealed = set(annotation["revealed_claim_ids"])
        if turn_revealed - hidden_ids:
            raise RuntimeError(f"unknown revealed claims at turn {expected_turn}")
        for claim_id in turn_revealed:
            first_revealed_turn.setdefault(claim_id, expected_turn)
        cumulative_revealed.update(turn_revealed)

        snapshot_path = Path(row["profile_snapshot_path"])
        expected_path = paths.profile_snapshots / f"turn_{expected_turn:03d}.json"
        if snapshot_path.resolve() != expected_path.resolve() or not snapshot_path.exists():
            raise RuntimeError(f"missing or mismatched profile snapshot at turn {expected_turn}")
        candidate = load_json(str(snapshot_path))
        candidate_hash = profile_snapshot(candidate)["static_profile_sha256"]
        if candidate_hash != row["profile_after"]["static_profile_sha256"]:
            raise RuntimeError(f"profile snapshot hash mismatch at turn {expected_turn}")
        if candidate_hash == previous_hash:
            supported = set(previous_supported)
        else:
            coverage = evaluate_hidden_coverage(judge, hidden_claims, candidate)
            supported = set(coverage["supported_hidden_claim_ids"])
        learned = supported - supported_initial
        newly_learned = learned - (previous_supported - supported_initial)
        for claim_id in newly_learned:
            first_learned_turn.setdefault(claim_id, expected_turn)
        curve.append({
            "turn": expected_turn,
            "revealed_claim_ids": sorted(cumulative_revealed),
            "learned_claim_ids": sorted(learned),
            "cumulative_revealed_count": len(cumulative_revealed),
            "cumulative_learned_count": len(learned),
            "hidden_coverage": round(len(learned) / len(hidden_claims), 6),
        })
        previous_hash = candidate_hash
        previous_supported = supported

    learned_final = previous_supported - supported_initial
    final_judged = set(discovery["judge_annotations"]["hidden_supported_by_final"]) - supported_initial
    if learned_final != final_judged:
        raise RuntimeError("final snapshot coverage disagrees with final discovery judgment")
    elicitation_rate = len(cumulative_revealed) / len(hidden_claims)
    end_to_end = len(learned_final) / len(hidden_claims)
    uptake = (
        len(learned_final & cumulative_revealed) / len(cumulative_revealed)
        if cumulative_revealed else None
    )
    per_layer: Dict[str, Dict[str, Any]] = {}
    for claim in hidden_claims:
        layer = claim.path.split(".", 1)[0]
        layer_row = per_layer.setdefault(layer, {
            "target_claim_ids": [], "revealed_claim_ids": [], "learned_claim_ids": [],
        })
        layer_row["target_claim_ids"].append(claim.claim_id)
        if claim.claim_id in cumulative_revealed:
            layer_row["revealed_claim_ids"].append(claim.claim_id)
        if claim.claim_id in learned_final:
            layer_row["learned_claim_ids"].append(claim.claim_id)
    for layer_row in per_layer.values():
        target_count = len(layer_row["target_claim_ids"])
        layer_row["target_count"] = target_count
        layer_row["elicitation_rate"] = len(layer_row["revealed_claim_ids"]) / target_count
        layer_row["discovery_rate"] = len(layer_row["learned_claim_ids"]) / target_count
    claim_details = [{
        **claim.as_dict(),
        "revealed": claim.claim_id in cumulative_revealed,
        "first_revealed_turn": first_revealed_turn.get(claim.claim_id),
        "learned": claim.claim_id in learned_final,
        "first_learned_turn": first_learned_turn.get(claim.claim_id),
    } for claim in hidden_claims]
    learned_turns = sorted(first_learned_turn.values())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": case.case_id,
        "condition": condition.key,
        "seed_index": seed_index,
        "hidden_claim_manifest_sha256": hidden_hash,
        **discovery,
        "revealed_hidden_claim_ids": sorted(cumulative_revealed),
        "learned_hidden_claim_ids": sorted(learned_final),
        "elicitation_rate": round(elicitation_rate, 6),
        "uptake_rate": None if uptake is None else round(uptake, 6),
        "end_to_end_discovery_rate": round(end_to_end, 6),
        "first_discovery_turn": learned_turns[0] if learned_turns else None,
        "rounds_per_discovery": None if not learned_final else round(round_count / len(learned_final), 6),
        "discovery_efficiency": round(len(learned_final) / round_count, 6),
        "coverage_auc": round(mean(point["hidden_coverage"] for point in curve), 6),
        "completion_curve": curve,
        "per_layer": per_layer,
        "claim_details": claim_details,
        "mean_user_burden": float(summary["mean_user_burden"]),
        "refusal_rate": float(summary["refusal_rate"]),
        "exploration_question_rate": float(summary["exploration_question_rate"]),
        "profile_entropy_reduction": float(summary["profile_entropy_reduction"]),
        "round_count": round_count,
    }


def evaluate_exploration_track(
    conditions: Sequence[ExperimentCondition],
    cases: Sequence[ExperimentCase],
    output_dir: str | Path,
    config_path: str,
    judge_model: str | None,
    judge_config_section: str,
    sim_seeds: int,
) -> Dict[str, str]:
    judge = Exp2JudgeClient(config_path, config_section=judge_config_section, model=judge_model)
    aggregates: Dict[str, Dict[str, Any]] = {}
    all_results: Dict[str, list[Dict[str, Any]]] = {}
    for condition in conditions:
        rows = []
        for case in cases:
            hidden_claims, _ = _simulator_assets(output_dir, case)
            for seed_index in range(1, sim_seeds + 1):
                paths = SimulationPaths.for_run(output_dir, condition, case, seed_index)
                expected_hash = _sha256_json([claim.as_dict() for claim in hidden_claims])
                if paths.evaluation.exists():
                    result = load_json(str(paths.evaluation))
                    if (
                        result["protocol_version"] != PROTOCOL_VERSION
                        or result["case_id"] != case.case_id
                        or result["condition"] != condition.key
                        or result["seed_index"] != seed_index
                        or result["hidden_claim_manifest_sha256"] != expected_hash
                    ):
                        raise RuntimeError(
                            f"stale exploration evaluation for {condition.key}/{case.case_id}/seed {seed_index}"
                        )
                else:
                    if not paths.generation_summary.exists() or not paths.final_profile.exists():
                        raise FileNotFoundError(
                            f"simulation incomplete for {condition.key}/{case.case_id}/seed {seed_index}"
                        )
                    summary = load_json(str(paths.generation_summary))
                    _validate_exploration_summary(summary, condition, case, seed_index)
                    result = _exploration_run_result(
                        judge, hidden_claims, condition, case, seed_index, paths, summary
                    )
                    save_json(str(paths.evaluation), result)
                rows.append(result)
        all_results[condition.key] = rows
        aggregates[condition.key] = aggregate_discovery_results(rows)

    result_dir = track_root(output_dir, TRACK_EXPLORATION) / "evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "comparison.json"
    md_path = result_dir / "comparison.md"
    scope = "comparative" if len(conditions) > 1 else "capability_only"
    save_json(str(json_path), {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_scope": scope,
        "claim_boundary": (
            "Capability-only evidence does not establish Adaptive Exploration's relative advantage."
            if scope == "capability_only"
            else "Relative claims are limited to the explicitly selected conditions."
        ),
        "reference_condition": conditions[0].key if scope == "comparative" else None,
        "aggregates": aggregates,
        "per_run": all_results,
    })
    md_path.write_text(render_exploration_table(aggregates, conditions), encoding="utf-8")
    return {"comparison_json": str(json_path), "comparison_markdown": str(md_path)}


def build_protocol_manifest(
    args: argparse.Namespace,
    selected: Mapping[str, Sequence[ExperimentCondition]],
    explicit_cases: Sequence[ExperimentCase],
    online_cases: Sequence[ExperimentCase],
) -> Dict[str, Any]:
    bundle = get_exp2_prompt_bundle(args.prompt_version)
    track_manifests = {
        TRACK_EXPLICIT: {
            "name": "Exp3-A Explicit User Modeling",
            "split": f"{args.explicit_train_ratio:.0%}/{1-args.explicit_train_ratio:.0%}",
            "protocol": "fixed REALTALK teacher-forcing evaluation",
            "conditions": [condition.manifest(args.fixed_omega) for condition in selected.get(TRACK_EXPLICIT, [])],
            "cases": [asdict(case) for case in explicit_cases],
        },
        TRACK_EXPLORATION: {
            "name": "Exp3-B Adaptive Exploration",
            "split": f"{args.online_train_ratio:.0%}/{1-args.online_train_ratio:.0%}",
            "protocol": "interactive hidden-profile user simulation",
            "hidden_target_definition": (
                "claim-level semantic new/refinement content in P* versus P0; "
                "every target requires held-out user-message evidence"
            ),
            "simulator_architecture": (
                "private disclosure controller followed by a reply renderer that sees "
                "only the claims authorized for that turn"
            ),
            "simulator_disclosure_policy": (
                "no hidden claim in opener; relevant and trust-sensitive; at most two "
                "new claims per turn; may withhold or refuse"
            ),
            "primary_metrics": (
                "elicitation, uptake, end-to-end discovery, correctness, rounds, "
                "completion curve, per-layer coverage"
            ),
            "inference_boundary": (
                "a single selected condition demonstrates exploration capability only; "
                "relative advantage requires an explicit comparator"
            ),
            "rounds": args.sim_rounds,
            "seeds": args.sim_seeds,
            "conditions": [condition.manifest(args.fixed_omega) for condition in selected.get(TRACK_EXPLORATION, [])],
            "cases": [asdict(case) for case in online_cases],
        },
        TRACK_BAYESIAN: {
            "name": "Exp3-C Bayesian Updating",
            "split": f"{args.online_train_ratio:.0%}/{1-args.online_train_ratio:.0%}",
            "protocol": "chronological REALTALK replay",
            "memory_control": "reference reply enters memory; generated reply is evaluation-only",
            "required_temporal_analysis": "paired early/middle/late score deltas",
            "conditions": [condition.manifest(args.fixed_omega) for condition in selected.get(TRACK_BAYESIAN, [])],
            "cases": [asdict(case) for case in online_cases],
        },
    }
    return {
        "experiment": "Experiment 3. Ablation Study",
        "research_question": "How does each proposed component contribute to Deep Empathy?",
        "protocol_version": PROTOCOL_VERSION,
        "model": _model_name(args.config),
        "base_prompt_version": args.prompt_version,
        "tracks": {track: track_manifests[track] for track in selected},
        "condition_prompt_manifests": {
            condition.key: condition_prompt_bundle(bundle, condition).manifest()
            for conditions in selected.values()
            for condition in conditions
        },
    }


def run(args: argparse.Namespace) -> None:
    validate_settings(args)
    tracks = select_tracks(args.track)
    selected = select_conditions(tracks, args.condition)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_explicit_cases = build_cases(args.dataset_dir, args.explicit_train_ratio)
    all_online_cases = build_cases(args.dataset_dir, args.online_train_ratio)
    explicit_cases = _select_cases(all_explicit_cases, args.case)
    online_cases = _select_cases(all_online_cases, args.case)
    save_split_manifest(
        all_explicit_cases, output_dir / "split_explicit_90_10.json", args.explicit_train_ratio
    )
    save_split_manifest(
        all_online_cases, output_dir / "split_online_50_50.json", args.online_train_ratio
    )
    protocol = build_protocol_manifest(args, selected, explicit_cases, online_cases)
    save_json(str(output_dir / "experiment_plan.json"), protocol)

    if args.phase in ("prepare", "all"):
        prepare(
            selected, explicit_cases, online_cases, output_dir,
            args.config, args.fixed_omega,
        )

    generated: Dict[str, Dict[str, int]] = {}
    if args.phase in ("generate", "all"):
        for track, conditions in selected.items():
            generated[track] = {}
            cases = explicit_cases if track == TRACK_EXPLICIT else online_cases
            for condition in conditions:
                condition_total = 0
                if track == TRACK_EXPLORATION:
                    for case in cases:
                        for seed_index in range(1, args.sim_seeds + 1):
                            condition_total += run_exploration_simulation(
                                case, output_dir, condition, args.config,
                                args.prompt_version, args.fixed_omega,
                                seed_index, args.sim_rounds,
                            )
                else:
                    for case in cases:
                        condition_total += run_dataset_replay(
                            case, output_dir, condition, args.config,
                            args.prompt_version, args.fixed_omega,
                        )
                generated[track][condition.key] = condition_total

    evaluation: Dict[str, Any] = {}
    if args.phase in ("evaluate", "all"):
        evaluation.update(evaluate_dataset_tracks(
            selected, explicit_cases, online_cases, output_dir, args.config,
            args.judge_model, args.judge_config_section,
            args.eval_device, args.eval_batch_size,
        ))
        if TRACK_EXPLORATION in selected:
            evaluation[TRACK_EXPLORATION] = evaluate_exploration_track(
                selected[TRACK_EXPLORATION], online_cases, output_dir,
                args.config, args.judge_model, args.judge_config_section,
                args.sim_seeds,
            )

    save_json(str(output_dir / "run_manifest.json"), {
        **protocol,
        "phase": args.phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_this_run": generated,
        "evaluation": evaluation,
        "evaluation_status": "complete" if evaluation else "not_run",
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the three component-specific Experiment 3 ablations."
    )
    parser.add_argument(
        "--phase", choices=("plan", "prepare", "generate", "evaluate", "all"),
        default="plan", help="plan is API-free and writes only protocol manifests",
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/exp3_ablation_study")
    parser.add_argument("--config", default="config.ini")
    parser.add_argument("--explicit-train-ratio", type=float, default=EXPLICIT_TRAIN_RATIO)
    parser.add_argument("--online-train-ratio", type=float, default=ONLINE_TRAIN_RATIO)
    parser.add_argument("--fixed-omega", type=float, default=DEFAULT_FIXED_OMEGA)
    parser.add_argument("--sim-rounds", type=int, default=DEFAULT_SIMULATION_ROUNDS)
    parser.add_argument("--sim-seeds", type=int, default=DEFAULT_SIMULATION_SEEDS)
    parser.add_argument(
        "--prompt-version", choices=exp2_prompt_versions(),
        default=DEFAULT_EXP2_PROMPT_VERSION,
    )
    parser.add_argument(
        "--track", action="append", default=[], choices=TRACK_ORDER,
        help="repeat to select tracks; omitted means all three",
    )
    parser.add_argument(
        "--condition", action="append", default=[], choices=CONDITION_ORDER,
        help="repeat to select conditions within the selected tracks",
    )
    parser.add_argument(
        "--case", action="append", default=[],
        help="repeat to select complete REALTALK conversations",
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-config-section", default="EvaluationAPI")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
