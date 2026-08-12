from __future__ import annotations

import argparse
import configparser
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence

from openai import OpenAI

from ..agent import StateDrivenCompanionAgent
from ..llm_client import LLMClient
from ..memory_os_local import MemoryOSLocal
from ..metrics import compute_rouge_l
from ..persona_schema import (
    PERSONA_SCHEMA_VERSION,
    persona_schema_manifest,
    validate_persona,
)
from ..profile_utils import state_axis
from ..prompts.eval_templates_en import (
    REALTALK_EMPATHY_EVALUATION_SYSTEM_PROMPT,
    REALTALK_GROUNDING_EVALUATION_SYSTEM_PROMPT,
    REALTALK_REFLECTIVE_EVALUATION_SYSTEM_PROMPT,
)
from ..prompts.exp2_versions import (
    DEFAULT_EXP2_PROMPT_VERSION,
    EVALUATION_PROMPT_VERSION,
    exp2_prompt_versions,
    get_exp2_prompt_bundle,
    validate_state_update,
)
from ..utils import load_json, save_json
from .extract_profile_persona_en import extract_persona, extract_profile


SESSION_PATTERN = re.compile(r"session_(\d+)$")
PROFILE_ALGORITHM = "extract_profile_persona_en.extract_profile:one_shot_train_split"
PROTOCOL_VERSION = "exp2_protocol_v2_state_update"
GENERATION_POLICY = "previous_empathy_state_parallel_state_update_v2"
ANNOTATION_CACHE_KEY_VERSION = 2
PROFILE_FIELDS = {
    "core": ("summary", "values", "motivations", "concerns"),
    "regulation": ("summary", "stress_response", "conflict_style", "emotion_regulation"),
    "cognition": ("summary", "thinking_style", "decision_style", "technology_view"),
    "identity": ("summary", "current_stage", "professional_identity", "social_identity"),
    "behavior": ("summary", "learning", "tool_usage", "interests", "interaction_style"),
}

TABLE2_METRICS = (
    "lexical",
    "semantic",
    "reflective",
    "grounding",
    "sentiment",
    "emotion",
    "intimacy",
    "empathy",
)
TABLE2_BASELINES = (
    {
        "method": "w/o fine-tune",
        "lexical": (0.14, 0.04),
        "semantic": (0.76, 0.08),
        "reflective": (0.62, 0.13),
        "grounding": (0.40, 0.13),
        "sentiment": (0.53, 0.22),
        "emotion": (0.43, 0.22),
        "intimacy": (0.06, 0.01),
        "empathy": (1.80, 0.55),
    },
    {
        "method": "w/ fine-tune",
        "lexical": (0.14, 0.05),
        "semantic": (0.78, 0.04),
        "reflective": (0.77, 0.09),
        "grounding": (0.62, 0.08),
        "sentiment": (0.59, 0.18),
        "emotion": (0.46, 0.21),
        "intimacy": (0.07, 0.01),
        "empathy": (1.24, 0.12),
    },
)


# ---------------------------------------------------------------------------
# Dataset protocol
# ---------------------------------------------------------------------------


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_-]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def session_keys(chat: Dict[str, Any]) -> List[str]:
    keys = [
        key
        for key, value in chat.items()
        if SESSION_PATTERN.fullmatch(key) and isinstance(value, list)
    ]
    return sorted(keys, key=lambda key: int(SESSION_PATTERN.fullmatch(key).group(1)))


def split_sessions(
    keys: Sequence[str],
    train_ratio: float = 0.9,
) -> tuple[List[str], List[str]]:
    """Chronologically split each conversation into 90% train and 10% test."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(keys) < 2:
        raise ValueError("at least two sessions are required")
    boundary = min(len(keys) - 1, max(1, math.ceil(len(keys) * train_ratio)))
    return list(keys[:boundary]), list(keys[boundary:])


def message_text(message: Dict[str, Any]) -> str:
    return str(message.get("clean_text") or "").strip()


@dataclass(frozen=True)
class Bubble:
    session_id: str
    speaker: str
    content: str
    dia_ids: tuple[str, ...]


def merge_bubbles(session_id: str, messages: Iterable[Dict[str, Any]]) -> List[Bubble]:
    """Merge adjacent chat bubbles from the same speaker without reordering."""
    merged: List[Bubble] = []
    for message in messages:
        speaker = str(message.get("speaker") or "").strip()
        content = message_text(message)
        if not speaker or not content:
            continue
        dia_id = str(message.get("dia_id") or "")
        if merged and merged[-1].speaker == speaker:
            previous = merged[-1]
            merged[-1] = Bubble(
                session_id=session_id,
                speaker=speaker,
                content=f"{previous.content}\n{content}",
                dia_ids=previous.dia_ids + (dia_id,),
            )
        else:
            merged.append(Bubble(session_id, speaker, content, (dia_id,)))
    return merged


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    dataset_path: str
    user_speaker: str
    agent_speaker: str
    train_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]
    dataset_sha256: str


@dataclass(frozen=True)
class ReplyExample:
    example_id: str
    case_id: str
    session_id: str
    user_speaker: str
    agent_speaker: str
    user_message: str
    reference_reply: str
    user_dia_ids: tuple[str, ...]
    reference_dia_ids: tuple[str, ...]
    next_user_message: str | None
    next_user_session_id: str | None
    next_user_dia_ids: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_from_path(path: str | Path, train_ratio: float = 0.9) -> ExperimentCase:
    source = Path(path).resolve()
    chat = load_json(str(source))
    names = chat.get("name", {})
    user = str(names.get("speaker_1") or "").strip()
    agent = str(names.get("speaker_2") or "").strip()
    if not user or not agent:
        raise ValueError(f"missing speaker mapping in {source}")
    train, test = split_sessions(session_keys(chat), train_ratio)
    case_id = f"{source.stem.lower()}__{slugify(user)}__to__{slugify(agent)}"
    return ExperimentCase(
        case_id=case_id,
        dataset_path=str(source),
        user_speaker=user,
        agent_speaker=agent,
        train_sessions=tuple(train),
        test_sessions=tuple(test),
        dataset_sha256=_sha256(source),
    )


def build_cases(dataset_dir: str | Path, train_ratio: float = 0.9) -> List[ExperimentCase]:
    paths = sorted(Path(dataset_dir).glob("Chat_*.json"), key=lambda path: path.name.lower())
    if not paths:
        raise FileNotFoundError(f"no Chat_*.json files under {dataset_dir}")
    return [case_from_path(path, train_ratio) for path in paths]


def save_split_manifest(
    cases: Sequence[ExperimentCase],
    output_path: str | Path,
    train_ratio: float,
) -> None:
    save_json(str(output_path), {
        "protocol": "chronological_session_split",
        "train_ratio": train_ratio,
        "rounding": "ceil",
        "role_rule": "speaker_1=user, speaker_2=agent (REALTALK target reply role)",
        "cases": [asdict(case) for case in cases],
    })


def bubbles_for_sessions(chat: Dict[str, Any], keys: Iterable[str]) -> List[Bubble]:
    result: List[Bubble] = []
    for key in keys:
        result.extend(merge_bubbles(key, chat[key]))
    return result


def build_reply_examples(chat: Dict[str, Any], case: ExperimentCase) -> List[ReplyExample]:
    """Create U_t -> A_t test examples; A_t is never included in model input."""
    bubbles = bubbles_for_sessions(chat, case.test_sessions)
    examples: List[ReplyExample] = []
    for index in range(len(bubbles) - 1):
        user_bubble = bubbles[index]
        agent_bubble = bubbles[index + 1]
        if (
            user_bubble.speaker != case.user_speaker
            or agent_bubble.speaker != case.agent_speaker
            or user_bubble.session_id != agent_bubble.session_id
        ):
            continue

        next_user = next(
            (candidate for candidate in bubbles[index + 2:] if candidate.speaker == case.user_speaker),
            None,
        )
        example_id = f"{case.case_id}:{user_bubble.session_id}:{'-'.join(user_bubble.dia_ids)}"
        examples.append(ReplyExample(
            example_id=example_id,
            case_id=case.case_id,
            session_id=user_bubble.session_id,
            user_speaker=case.user_speaker,
            agent_speaker=case.agent_speaker,
            user_message=user_bubble.content,
            reference_reply=agent_bubble.content,
            user_dia_ids=user_bubble.dia_ids,
            reference_dia_ids=agent_bubble.dia_ids,
            next_user_message=next_user.content if next_user else None,
            next_user_session_id=next_user.session_id if next_user else None,
            next_user_dia_ids=next_user.dia_ids if next_user else (),
        ))
    return examples


def validate_no_leakage(case: ExperimentCase, examples: Sequence[ReplyExample]) -> None:
    train = set(case.train_sessions)
    test = set(case.test_sessions)
    if train & test:
        raise ValueError(f"overlapping train/test sessions for {case.case_id}")
    if not examples:
        raise ValueError(f"no user->agent reply examples for {case.case_id}")
    for example in examples:
        if example.session_id not in test:
            raise ValueError(f"non-test example {example.example_id}")
        if not example.user_message or not example.reference_reply:
            raise ValueError(f"empty example content {example.example_id}")


# ---------------------------------------------------------------------------
# Paths and resumable output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CasePaths:
    root: Path
    case_root: Path
    persona: Path
    profile: Path
    runtime_profile: Path
    asset_manifest: Path
    memory_db: Path
    predictions: Path
    understanding: Path
    table2_annotations: Path
    table2_scores: Path

    @classmethod
    def for_case(cls, output_dir: str | Path, case: ExperimentCase) -> "CasePaths":
        root = Path(output_dir).resolve()
        case_root = root / "cases" / case.case_id
        assets = case_root / "assets"
        return cls(
            root=root,
            case_root=case_root,
            persona=assets / "agent_persona.json",
            profile=assets / "user_profile.json",
            runtime_profile=assets / "user_profile_runtime.json",
            asset_manifest=assets / "asset_manifest.json",
            memory_db=case_root / "memory" / "memory.db",
            predictions=case_root / "generations" / "predictions.jsonl",
            understanding=case_root / "states" / "user_understanding.jsonl",
            table2_annotations=case_root / "evaluation" / "table2_annotations.jsonl",
            table2_scores=case_root / "evaluation" / "table2_scores.json",
        )

    def ensure_parents(self) -> None:
        for path in (
            self.persona,
            self.profile,
            self.runtime_profile,
            self.asset_manifest,
            self.memory_db,
            self.predictions,
            self.understanding,
            self.table2_annotations,
            self.table2_scores,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)


class JsonlStore:
    """Append-only JSONL output with ID-based resume support."""

    def __init__(self, path: str | Path, id_field: str = "example_id") -> None:
        self.path = Path(path)
        self.id_field = id_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ids = {
            str(row[id_field])
            for row in self.read_all()
            if id_field in row
        }

    def contains(self, item_id: str) -> bool:
        return item_id in self._ids

    def append(self, payload: Dict[str, Any]) -> None:
        item_id = str(payload[self.id_field])
        if item_id in self._ids:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
        self._ids.add(item_id)

    def read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {self.path}:{line_number}") from exc
        return rows


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    return JsonlStore(path).read_all()


# ---------------------------------------------------------------------------
# Training assets
# ---------------------------------------------------------------------------


def build_case_persona(
    case: ExperimentCase,
    paths: CasePaths,
    config_path: str = "config.ini",
) -> Dict[str, Any]:
    """Extract the agent persona from only the training sessions."""
    if paths.persona.exists():
        persona = load_json(str(paths.persona))
        try:
            validate_persona(persona)
        except ValueError as exc:
            raise RuntimeError(
                f"existing persona is not {PERSONA_SCHEMA_VERSION}; use a new "
                "--output-dir so legacy and fixed-schema assets are not mixed"
            ) from exc
        return persona
    chat = load_json(case.dataset_path)
    train_sessions = [chat[session_id] for session_id in case.train_sessions]
    persona = extract_persona(LLMClient(config_path), train_sessions, case.agent_speaker)
    validate_persona(persona)
    save_json(str(paths.persona), persona)
    return persona


def _validate_extracted_profile(profile: Dict[str, Any]) -> None:
    if set(profile) != set(PROFILE_FIELDS):
        raise ValueError(
            "extracted profile top-level fields do not match dataset/lsy_user.json: "
            f"actual={sorted(profile)}"
        )
    for layer, expected_fields in PROFILE_FIELDS.items():
        section = profile.get(layer)
        if not isinstance(section, dict) or set(section) != set(expected_fields):
            actual = sorted(section) if isinstance(section, dict) else type(section).__name__
            raise ValueError(
                f"extracted profile fields do not match dataset/lsy_user.json at {layer}: "
                f"actual={actual}"
            )
        if not isinstance(section["summary"], str):
            raise ValueError(f"{layer}.summary must be a string")
        for field in expected_fields[1:]:
            value = section[field]
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{layer}.{field} must be a list of strings")


def _agent_runtime_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "state_axis": {
            "static_profile": profile,
            "current_state": {},
            "projected_state": {},
        },
        "context_axis": {},
    }


def build_case_profile(
    case: ExperimentCase,
    paths: CasePaths,
    config_path: str = "config.ini",
) -> Dict[str, Any]:
    """Extract one fixed-schema profile from all training sessions at once."""
    paths.ensure_parents()
    if not paths.persona.exists():
        raise FileNotFoundError(f"persona must be built first: {paths.persona}")

    profile_assets = (
        paths.profile.exists(),
        paths.runtime_profile.exists(),
        paths.asset_manifest.exists(),
    )
    if all(profile_assets):
        manifest = load_json(str(paths.asset_manifest))
        if (
            manifest.get("profile_algorithm") == PROFILE_ALGORITHM
            and manifest.get("train_sessions") == list(case.train_sessions)
            and manifest.get("persona_schema_version") == PERSONA_SCHEMA_VERSION
        ):
            profile = load_json(str(paths.profile))
            _validate_extracted_profile(profile)
            return profile
        raise RuntimeError(
            "existing training assets use a different profile/persona protocol; "
            "use a new --output-dir to preserve the earlier assets and results"
        )
    if any(profile_assets):
        raise RuntimeError(
            "partial training assets already exist; use a new --output-dir rather "
            "than overwriting an incomplete or legacy run"
        )

    chat = load_json(case.dataset_path)
    train_sessions = [chat[session_id] for session_id in case.train_sessions]
    profile = extract_profile(
        LLMClient(config_path),
        train_sessions,
        case.user_speaker,
    )
    _validate_extracted_profile(profile)
    save_json(str(paths.profile), profile)
    save_json(str(paths.runtime_profile), _agent_runtime_profile(profile))
    save_json(str(paths.asset_manifest), {
        "case": asdict(case),
        "profile_algorithm": PROFILE_ALGORITHM,
        "profile_schema": "dataset/lsy_user.json",
        "persona_algorithm": "experiments.extract_profile_persona_en.extract_persona",
        "persona_schema_version": PERSONA_SCHEMA_VERSION,
        "persona_schema": persona_schema_manifest(),
        "train_only": True,
        "train_sessions": list(case.train_sessions),
        "profile_path": str(paths.profile),
        "runtime_profile_path": str(paths.runtime_profile),
        "persona_path": str(paths.persona),
    })
    return profile


# ---------------------------------------------------------------------------
# Test reply generation
# ---------------------------------------------------------------------------


def _example_by_user_dia(
    examples: Sequence[ReplyExample],
) -> Dict[tuple[str, ...], ReplyExample]:
    return {example.user_dia_ids: example for example in examples}


def _assert_consistent_stores(predictions: JsonlStore, states: JsonlStore) -> None:
    prediction_ids = {row["example_id"] for row in predictions.read_all()}
    state_ids = {row["example_id"] for row in states.read_all()}
    if prediction_ids != state_ids:
        raise RuntimeError(
            "predictions and understanding JSONL are out of sync; "
            "repair the incomplete last record before resuming"
        )


def _resume_position(
    bubbles: Sequence[Bubble],
    examples: Sequence[ReplyExample],
    completed_ids: set[str],
) -> int:
    if not completed_ids:
        return 0
    ordered_ids = [example.example_id for example in examples]
    completed_prefix = 0
    for example_id in ordered_ids:
        if example_id not in completed_ids:
            break
        completed_prefix += 1
    if set(ordered_ids[:completed_prefix]) != completed_ids:
        raise RuntimeError("resume output is not a contiguous prefix of the test examples")
    last = examples[completed_prefix - 1]
    for index, bubble in enumerate(bubbles):
        if bubble.dia_ids == last.reference_dia_ids:
            return index + 1
    raise RuntimeError(f"cannot locate completed example {last.example_id} in test dialogue")


def _append_real_bubble(
    agent: StateDrivenCompanionAgent,
    case: ExperimentCase,
    bubble: Bubble,
) -> None:
    if bubble.speaker == case.user_speaker:
        role = "user"
    elif bubble.speaker == case.agent_speaker:
        role = "assistant"
    else:
        raise ValueError(f"unexpected speaker {bubble.speaker!r} in {case.case_id}")
    agent.memory_manager.append_stm(role, bubble.content)
    agent.epistemic_tracker.increment()


def _generate_with_parallel_alignment(
    agent: StateDrivenCompanionAgent,
    user_message: str,
    relevant_memory: Dict[str, Any],
) -> tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Generate with the previous state while aligning the current turn in parallel."""
    previous_empathy_state = deepcopy(agent.last_empathy_state)
    # Render from the preceding completed state before the concurrent alignment
    # can write the current turn's state.
    response_prompt = agent._response_prompt(
        user_message,
        relevant_memory,
        previous_empathy_state=previous_empathy_state,
    )
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="exp2-alignment") as executor:
        alignment_future = executor.submit(
            _run_alignment_with_format_retry,
            agent,
            user_message,
            relevant_memory,
        )
        generated_reply = agent.llm.chat(
            agent.prompt_bundle.response_system,
            response_prompt,
            temperature=0.4,
            max_tokens=450,
        ).strip()
        response_model_timing = deepcopy(agent.llm.last_model_timing)
        alignment = alignment_future.result()
    return generated_reply, alignment, previous_empathy_state, response_model_timing


def _valid_alignment_payload(
    agent: StateDrivenCompanionAgent,
    result: Dict[str, Any],
) -> bool:
    """Check only the JSON contract needed by the versioned experiment."""
    if not isinstance(result, dict) or not isinstance(result.get("empathy_state"), dict):
        return False
    if not agent.prompt_bundle.updates_user_state:
        return True
    valid, _ = validate_state_update(result.get("state_update"))
    return valid


def _run_alignment_with_format_retry(
    agent: StateDrivenCompanionAgent,
    user_message: str,
    relevant_memory: Dict[str, Any],
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """Retry only malformed/missing alignment JSON; never silently save an empty state."""
    for attempt in range(1, max_attempts + 1):
        result = agent._run_empathy_alignment(user_message, relevant_memory)
        if _valid_alignment_payload(agent, result):
            return result
        if attempt < max_attempts:
            print(
                f"[Empathy Alignment Retry] invalid JSON contract; "
                f"attempt={attempt + 1}/{max_attempts}"
            )
    raise RuntimeError(
        f"empathy alignment failed to return valid JSON after {max_attempts} attempts"
    )


def run_case_replies(
    case: ExperimentCase,
    paths: CasePaths,
    config_path: str = "config.ini",
    prompt_version: str = DEFAULT_EXP2_PROMPT_VERSION,
) -> int:
    """Generate Ours replies while preserving REALTALK history via teacher forcing."""
    if not paths.profile.exists() or not paths.runtime_profile.exists() or not paths.persona.exists():
        raise FileNotFoundError(f"training assets missing for {case.case_id}; run prepare first")
    persona = load_json(str(paths.persona))
    try:
        validate_persona(persona)
    except ValueError as exc:
        raise RuntimeError(
            f"agent persona for {case.case_id} does not match "
            f"{PERSONA_SCHEMA_VERSION}; run prepare in a new --output-dir"
        ) from exc
    if not paths.asset_manifest.exists():
        raise FileNotFoundError(
            f"asset manifest missing for {case.case_id}; run prepare in a new --output-dir"
        )
    asset_manifest = load_json(str(paths.asset_manifest))
    current_persona_manifest = persona_schema_manifest()
    if (
        asset_manifest.get("persona_schema_version") != PERSONA_SCHEMA_VERSION
        or asset_manifest.get("persona_schema", {}).get("extraction_prompt_sha256")
        != current_persona_manifest["extraction_prompt_sha256"]
    ):
        raise RuntimeError(
            f"training assets for {case.case_id} use an older persona extraction "
            "protocol; run prepare in a new --output-dir"
        )
    # A prompt-version run may reuse only immutable prepared assets in a fresh
    # output directory.  In that case prepare did not create the per-case
    # memory/generation/state/evaluation directories, so generation must own
    # initialization of all of its output parents.
    paths.ensure_parents()

    chat = load_json(case.dataset_path)
    examples = build_reply_examples(chat, case)
    validate_no_leakage(case, examples)
    example_map = _example_by_user_dia(examples)
    bubbles = bubbles_for_sessions(chat, case.test_sessions)

    predictions = JsonlStore(paths.predictions)
    states = JsonlStore(paths.understanding)
    _assert_consistent_stores(predictions, states)
    prediction_rows = predictions.read_all()
    prompt_bundle = get_exp2_prompt_bundle(prompt_version)
    incompatible_rows = [
        row for row in prediction_rows
        if row.get("generation_policy") != GENERATION_POLICY
        or row.get("prompt_version") != prompt_bundle.version
        or row.get("prompt_sha256") != prompt_bundle.fingerprint
    ]
    if incompatible_rows:
        raise RuntimeError(
            "existing predictions were generated with a different protocol or prompt "
            "version; use a new --output-dir so versioned replies are not mixed"
        )
    completed_ids = {row["example_id"] for row in prediction_rows}
    start_index = _resume_position(bubbles, examples, completed_ids)

    agent = StateDrivenCompanionAgent(
        config_path=config_path,
        profile_path=str(paths.runtime_profile),
        persona_path=str(paths.persona),
        user_name=case.user_speaker,
        modeling_mode="explicit",
        update_mode="static",
        exploration_mode="adaptive",
        prompt_version=prompt_bundle.version,
    )
    agent.memory_manager = MemoryOSLocal(
        persist_path=str(paths.memory_db),
        config_path=config_path,
    )

    if completed_ids:
        state_by_id = {row["example_id"]: row for row in states.read_all()}
        ordered_completed = [
            example.example_id for example in examples
            if example.example_id in completed_ids
        ]
        last_state = state_by_id[ordered_completed[-1]]
        if "empathy_state_for_next_turn" not in last_state:
            raise RuntimeError(
                "resume state is missing empathy_state_for_next_turn; use a new "
                "--output-dir to run the current generation policy"
            )
        agent.last_empathy_state = deepcopy(last_state["empathy_state_for_next_turn"])
        agent.last_prediction = deepcopy(last_state.get("future_understanding", {}))

    for bubble in bubbles[max(0, start_index - 6):start_index]:
        _append_real_bubble(agent, case, bubble)

    generated_count = 0
    index = start_index
    while index < len(bubbles):
        bubble = bubbles[index]
        following = bubbles[index + 1] if index + 1 < len(bubbles) else None
        example = example_map.get(bubble.dia_ids)
        is_target = (
            example is not None
            and following is not None
            and following.dia_ids == example.reference_dia_ids
        )
        if not is_target:
            _append_real_bubble(agent, case, bubble)
            agent._run_memory_steps()
            index += 1
            continue

        history_count_before = len(agent.memory_manager.short_term_memory)
        _append_real_bubble(agent, case, bubble)
        relevant_memory = agent.memory_manager.retrieve_relevant_memory(example.user_message)
        generated_reply, alignment, previous_empathy_state, response_model_timing = (
            _generate_with_parallel_alignment(
                agent,
                example.user_message,
                relevant_memory,
            )
        )

        state = state_axis(agent.user_profile)
        created_at = datetime.now(timezone.utc).isoformat()
        states.append({
            "example_id": example.example_id,
            "case_id": case.case_id,
            "session_id": example.session_id,
            "current_user_message": example.user_message,
            "current_understanding": alignment.get("understanding", {}),
            "future_understanding": alignment.get("prediction", {}),
            "empathy_state_for_next_turn": deepcopy(agent.last_empathy_state),
            "core_current_state": state.get("current_state", {}),
            "core_projected_state": state.get("projected_state", {}),
            "next_user_message": example.next_user_message,
            "next_user_session_id": example.next_user_session_id,
            "next_user_dia_ids": list(example.next_user_dia_ids),
            "created_at": created_at,
        })
        predictions.append({
            **asdict(example),
            "generated_reply": generated_reply,
            "protocol_version": PROTOCOL_VERSION,
            "generation_policy": GENERATION_POLICY,
            "prompt_version": prompt_bundle.version,
            "prompt_sha256": prompt_bundle.fingerprint,
            "alignment_execution": "parallel_current_alignment_previous_state_for_response",
            "history_policy": "teacher_forcing_real_replies_only",
            "history_bubbles_before_user": history_count_before,
            "profile_loaded_before_first_test_turn": True,
            "test_profile_policy": "static",
            "generation_input_audit": {
                "contains_user_message": True,
                "contains_reference_reply": False,
                "contains_next_user_message": False,
                "previous_empathy_state": previous_empathy_state,
                "relevant_memory": relevant_memory,
            },
            "model_timing": response_model_timing,
            "created_at": created_at,
        })

        _append_real_bubble(agent, case, following)
        agent._run_memory_steps()
        generated_count += 1
        index += 2

    return generated_count


# ---------------------------------------------------------------------------
# REALTALK Table 2 evaluation
# ---------------------------------------------------------------------------


def _parse_boolean_label(raw: str, metric: str) -> bool:
    normalized = raw.strip().lower().rstrip(".")
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{metric} judge must return exactly True or False; got {raw!r}")


def _parse_json_object(raw: str) -> Dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("judge output must be a JSON object")
    return value


def _normalize_label(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _format_evaluation_context(turns: Sequence[Dict[str, str]]) -> str:
    return "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in turns)


def build_candidate_context(
    chat: Dict[str, Any],
    prediction: Dict[str, Any],
    candidate: str,
) -> List[Dict[str, str]]:
    """Rebuild the official within-session context and replace only the target reply."""
    session_id = str(prediction["session_id"])
    reference_ids = {str(value) for value in prediction["reference_dia_ids"]}
    if not reference_ids:
        raise ValueError(f"missing reference_dia_ids for {prediction['example_id']}")

    turns: List[Dict[str, str]] = []
    located_reference = False
    for message in chat[session_id]:
        if str(message.get("dia_id") or "") in reference_ids:
            located_reference = True
            break
        speaker = str(message.get("speaker") or "").strip()
        text = message_text(message)
        if speaker and text:
            turns.append({"speaker": speaker, "text": text})
    if not located_reference:
        raise ValueError(
            f"cannot locate reference reply for {prediction['example_id']} in {session_id}"
        )
    turns.append({
        "speaker": str(prediction["agent_speaker"]),
        "text": candidate.strip(),
    })
    return turns


class Exp2JudgeClient(LLMClient):
    """Exp2-only LLM client using the independent OpenAI-compatible judge API."""

    def __init__(
        self,
        config_path: str,
        config_section: str,
        model: str | None = None,
    ) -> None:
        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")
        if not config.has_section(config_section):
            raise ValueError(f"missing [{config_section}] in {config_path}")
        api_config = config[config_section]
        self.model = model or api_config.get("model", fallback="gpt-4o-mini")
        self.enable_thinking = False
        self.backend = api_config.get("backend", fallback=config_section).strip()
        self.base_url = api_config.get("base_url", fallback="").strip()
        self.config_section = config_section
        api_key_env = api_config.get("api_key_env", fallback="EVAL_API_KEY").strip()
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"environment variable {api_key_env} is required for [{config_section}]"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url or None,
            timeout=60.0,
        )
        self.last_model_timing = {"first_char_seconds": None}
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
        }

    @property
    def endpoint_identity(self) -> str:
        return f"{self.backend}|{self.base_url}|{self.model}|{self.config_section}"

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.6,
        max_tokens: int | None = None,
    ) -> str:
        """Call an OpenAI-compatible judge without Qwen-only request fields."""
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        response = None
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(**request)
                break
            except Exception as exc:
                last_error = exc
                wait_seconds = attempt * 5
                print(
                    f"[Judge Chat Error] attempt={attempt}/3 "
                    f"wait={wait_seconds}s error={exc}"
                )
                if attempt < 3:
                    time.sleep(wait_seconds)
        if response is None:
            assert last_error is not None
            raise last_error
        self._record_usage(response)
        return response.choices[0].message.content.strip()


class Table2Evaluator:
    """REALTALK Table 2 annotators, loaded only when evaluation is requested."""

    def __init__(
        self,
        judge_llm: LLMClient,
        device: str = "cuda:0",
        sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        emotion_model: str = "cardiffnlp/twitter-roberta-large-emotion-latest",
        intimacy_model: str = "cardiffnlp/twitter-roberta-large-intimacy-latest",
        bertscore_model: str = "roberta-large",
        batch_size: int = 16,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Table 2 evaluation dependencies are missing or mutually incompatible. "
                "Install the pinned torch, transformers, tokenizers, and bert-score versions "
                "documented in README_exp2_user_modeling.md. "
                f"Original import error: {exc}"
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({device}) but torch.cuda.is_available() is False")
        if batch_size < 1:
            raise ValueError("eval batch size must be positive")

        self.judge_llm = judge_llm
        self.device = device
        self.sentiment_model_name = sentiment_model
        self.emotion_model_name = emotion_model
        self.intimacy_model_name = intimacy_model
        self.bertscore_model = bertscore_model
        self.batch_size = batch_size
        if device == "cuda":
            pipeline_device = 0
        elif device.startswith("cuda:"):
            pipeline_device = int(device.split(":", 1)[1])
        else:
            pipeline_device = -1

        def classifier(model_name: str):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            return pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=pipeline_device,
            )

        print(f"[Table 2] loading sentiment model: {sentiment_model}")
        self.sentiment_classifier = classifier(sentiment_model)
        print(f"[Table 2] loading emotion model: {emotion_model}")
        self.emotion_classifier = classifier(emotion_model)
        print(f"[Table 2] loading intimacy model: {intimacy_model}")
        self.intimacy_classifier = classifier(intimacy_model)

    @property
    def fingerprint(self) -> str:
        payload = "|".join((
            EVALUATION_PROMPT_VERSION,
            self.judge_llm.endpoint_identity,
            self.judge_llm.model,
            self.sentiment_model_name,
            self.emotion_model_name,
            self.intimacy_model_name,
            self.bertscore_model,
            REALTALK_REFLECTIVE_EVALUATION_SYSTEM_PROMPT,
            REALTALK_GROUNDING_EVALUATION_SYSTEM_PROMPT,
            REALTALK_EMPATHY_EVALUATION_SYSTEM_PROMPT,
        ))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def _judge_boolean(self, system_prompt: str, context: str, metric: str) -> bool:
        raw = self.judge_llm.chat(
            system_prompt,
            f"Dialogue:\n{context}\n\nLabel the final message.",
            temperature=0.0,
            max_tokens=8,
        )
        return _parse_boolean_label(raw, metric)

    def _judge_empathy(self, context: str) -> Dict[str, int]:
        raw = self.judge_llm.chat(
            REALTALK_EMPATHY_EVALUATION_SYSTEM_PROMPT,
            f"Dialogue:\n{context}\n\nScore the final message.",
            temperature=0.0,
            max_tokens=100,
        )
        parsed = _parse_json_object(raw)
        result: Dict[str, int] = {}
        for field in ("emotional_reaction", "interpretation", "exploration"):
            value = parsed.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"empathy field {field} must be an integer from 0 to 2")
            integer = int(value)
            if value != integer or integer not in (0, 1, 2):
                raise ValueError(f"empathy field {field} must be an integer from 0 to 2")
            result[field] = integer
        return result

    @staticmethod
    def _top_prediction(classifier: Any, text: str) -> Dict[str, Any]:
        result = classifier(text, truncation=True, max_length=512)
        if isinstance(result, list):
            result = result[0]
        if not isinstance(result, dict) or "score" not in result:
            raise ValueError(f"unexpected Hugging Face classifier output: {result!r}")
        return result

    def annotate(self, turns: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        context = _format_evaluation_context(turns)
        text = turns[-1]["text"]
        empathy = self._judge_empathy(context)
        return {
            "reflective": self._judge_boolean(
                REALTALK_REFLECTIVE_EVALUATION_SYSTEM_PROMPT,
                context,
                "reflective",
            ),
            "grounding": self._judge_boolean(
                REALTALK_GROUNDING_EVALUATION_SYSTEM_PROMPT,
                context,
                "grounding",
            ),
            "sentiment": _normalize_label(
                self._top_prediction(self.sentiment_classifier, text)["label"]
            ),
            "emotion": _normalize_label(
                self._top_prediction(self.emotion_classifier, text)["label"]
            ),
            "intimacy": float(
                self._top_prediction(self.intimacy_classifier, text)["score"]
            ),
            "empathy": empathy,
            "empathy_total": sum(empathy.values()),
        }

    def semantic_scores(
        self,
        references: Sequence[str],
        candidates: Sequence[str],
    ) -> List[float]:
        try:
            from bert_score import score as bert_score
        except ImportError as exc:
            raise RuntimeError(
                "bert-score is missing; install it before running --phase evaluate"
            ) from exc
        _, _, f1 = bert_score(
            list(candidates),
            list(references),
            model_type=self.bertscore_model,
            device=self.device,
            batch_size=self.batch_size,
            verbose=True,
        )
        return [float(value) for value in f1.cpu().tolist()]


def _annotation_cache_identity(
    example_id: str,
    variant: str,
    fingerprint: str,
    candidate: str,
    turns: Sequence[Dict[str, str]],
) -> tuple[str, str, str]:
    """Bind cached labels to the exact candidate and evaluated dialogue context."""
    candidate_sha256 = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    context_sha256 = hashlib.sha256(
        _format_evaluation_context(turns).encode("utf-8")
    ).hexdigest()
    annotation_id = ":".join((
        example_id,
        variant,
        fingerprint,
        candidate_sha256,
        context_sha256,
    ))
    return annotation_id, candidate_sha256, context_sha256


def _matching_legacy_annotation(
    rows: Sequence[Dict[str, Any]],
    *,
    example_id: str,
    variant: str,
    fingerprint: str,
    candidate: str,
    context_sha256: str,
) -> Dict[str, Any] | None:
    """Reuse a legacy cache row only after verifying its evaluated content."""
    for row in rows:
        if (
            row.get("example_id") == example_id
            and row.get("variant") == variant
            and row.get("evaluator_fingerprint") == fingerprint
            and row.get("candidate") == candidate
            and row.get("context_sha256") == context_sha256
            and isinstance(row.get("labels"), dict)
        ):
            return row
    return None


def evaluate_case_table2(
    case: ExperimentCase,
    paths: CasePaths,
    evaluator: Table2Evaluator,
) -> List[Dict[str, Any]]:
    predictions = list(read_jsonl(paths.predictions))
    if not predictions:
        raise FileNotFoundError(
            f"no generated replies for {case.case_id}; run --phase generate first"
        )
    chat = load_json(case.dataset_path)
    annotations = JsonlStore(paths.table2_annotations, id_field="annotation_id")
    existing_annotation_rows = annotations.read_all()

    for index, prediction in enumerate(predictions, start=1):
        for variant, field in (
            ("reference", "reference_reply"),
            ("generated", "generated_reply"),
        ):
            candidate = str(prediction[field]).strip()
            turns = build_candidate_context(chat, prediction, candidate)
            annotation_id, candidate_sha256, context_sha256 = (
                _annotation_cache_identity(
                    str(prediction["example_id"]),
                    variant,
                    evaluator.fingerprint,
                    candidate,
                    turns,
                )
            )
            if annotations.contains(annotation_id):
                continue
            legacy = _matching_legacy_annotation(
                existing_annotation_rows,
                example_id=str(prediction["example_id"]),
                variant=variant,
                fingerprint=evaluator.fingerprint,
                candidate=candidate,
                context_sha256=context_sha256,
            )
            if legacy is not None:
                annotations.append({
                    **legacy,
                    "annotation_id": annotation_id,
                    "candidate_sha256": candidate_sha256,
                    "context_sha256": context_sha256,
                    "cache_key_version": ANNOTATION_CACHE_KEY_VERSION,
                    "migrated_from_annotation_id": legacy.get("annotation_id"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
                continue
            print(
                f"[Table 2] {case.case_id} {index}/{len(predictions)} "
                f"{variant}"
            )
            annotations.append({
                "annotation_id": annotation_id,
                "example_id": prediction["example_id"],
                "case_id": case.case_id,
                "agent_speaker": prediction["agent_speaker"],
                "variant": variant,
                "evaluator_fingerprint": evaluator.fingerprint,
                "candidate": candidate,
                "candidate_sha256": candidate_sha256,
                "context_sha256": context_sha256,
                "cache_key_version": ANNOTATION_CACHE_KEY_VERSION,
                "labels": evaluator.annotate(turns),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

    cached = {
        row["annotation_id"]: row
        for row in annotations.read_all()
        if row.get("evaluator_fingerprint") == evaluator.fingerprint
    }
    references = [str(row["reference_reply"]) for row in predictions]
    candidates = [str(row["generated_reply"]) for row in predictions]
    semantic = evaluator.semantic_scores(references, candidates)

    scores: List[Dict[str, Any]] = []
    for prediction, semantic_score in zip(predictions, semantic):
        example_id = str(prediction["example_id"])
        annotation_ids: Dict[str, str] = {}
        for variant, field in (
            ("reference", "reference_reply"),
            ("generated", "generated_reply"),
        ):
            candidate = str(prediction[field]).strip()
            turns = build_candidate_context(chat, prediction, candidate)
            annotation_ids[variant] = _annotation_cache_identity(
                example_id,
                variant,
                evaluator.fingerprint,
                candidate,
                turns,
            )[0]
        reference = cached[annotation_ids["reference"]]["labels"]
        generated = cached[annotation_ids["generated"]]["labels"]
        scores.append({
            "example_id": example_id,
            "case_id": case.case_id,
            "agent_speaker": prediction["agent_speaker"],
            "lexical": compute_rouge_l(
                str(prediction["reference_reply"]), str(prediction["generated_reply"])
            ),
            "semantic": semantic_score,
            "reflective": float(reference["reflective"] == generated["reflective"]),
            "grounding": float(reference["grounding"] == generated["grounding"]),
            "sentiment": float(reference["sentiment"] == generated["sentiment"]),
            "emotion": float(reference["emotion"] == generated["emotion"]),
            "intimacy": abs(float(reference["intimacy"]) - float(generated["intimacy"])),
            "empathy": abs(
                float(reference["empathy_total"]) - float(generated["empathy_total"])
            ),
        })
    save_json(str(paths.table2_scores), {
        "case": asdict(case),
        "evaluator_fingerprint": evaluator.fingerprint,
        "example_count": len(scores),
        "scores": scores,
    })
    return scores


def aggregate_table2_scores(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not scores:
        raise ValueError("cannot aggregate an empty Table 2 score set")
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scores:
        grouped[str(row["agent_speaker"])].append(row)

    per_speaker: List[Dict[str, Any]] = []
    for speaker, rows in sorted(grouped.items()):
        result: Dict[str, Any] = {
            "speaker": speaker,
            "example_count": len(rows),
        }
        for metric in TABLE2_METRICS:
            result[metric] = mean(float(row[metric]) for row in rows)
        per_speaker.append(result)

    ours: Dict[str, Dict[str, float]] = {}
    for metric in TABLE2_METRICS:
        values = [float(row[metric]) for row in per_speaker]
        ours[metric] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }
    return {
        "aggregation": "mean per target speaker, then population mean and std across speakers",
        "example_count": len(scores),
        "speaker_count": len(per_speaker),
        "per_speaker": per_speaker,
        "ours": ours,
    }


def _format_table2_stat(value: Any) -> str:
    if isinstance(value, tuple):
        average, deviation = value
    else:
        average, deviation = value["mean"], value["std"]
    return f"{average:.2f} ± {deviation:.2f}"


def render_table2_markdown(ours: Dict[str, Dict[str, float]]) -> str:
    headers = (
        "Method",
        "Lexical ↑",
        "Semantic ↑",
        "Reflective ↑",
        "Grounding ↑",
        "Sentiment ↑",
        "Emotion ↑",
        "Intimacy ↓",
        "Empathy ↓",
    )
    lines = [
        "# REALTALK Table 2 + Ours",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(TABLE2_METRICS)) + " |",
    ]
    for baseline in TABLE2_BASELINES:
        lines.append(
            "| " + " | ".join(
                [baseline["method"]]
                + [_format_table2_stat(baseline[metric]) for metric in TABLE2_METRICS]
            ) + " |"
        )
    lines.append(
        "| " + " | ".join(
            ["Ours"] + [_format_table2_stat(ours[metric]) for metric in TABLE2_METRICS]
        ) + " |"
    )
    lines.extend((
        "",
        "Lexical/semantic and categorical EI metrics are higher-is-better; "
        "intimacy and empathy are absolute errors and lower-is-better.",
        "",
    ))
    return "\n".join(lines)


def evaluate_table2(
    cases: Sequence[ExperimentCase],
    output_dir: str | Path,
    config_path: str,
    judge_model: str | None,
    judge_config_section: str,
    device: str,
    batch_size: int,
    prompt_version: str = DEFAULT_EXP2_PROMPT_VERSION,
) -> Dict[str, str]:
    prompt_bundle = get_exp2_prompt_bundle(prompt_version)
    for case in cases:
        predictions_path = CasePaths.for_case(output_dir, case).predictions
        predictions = list(read_jsonl(predictions_path)) if predictions_path.exists() else []
        if not predictions:
            raise FileNotFoundError(
                f"no generated replies for {case.case_id}; run --phase generate first"
            )
        mismatched = [
            row for row in predictions
            if (
                row.get("prompt_version", "v1_baseline") != prompt_bundle.version
                or (
                    row.get("prompt_sha256") is not None
                    and row.get("prompt_sha256") != prompt_bundle.fingerprint
                )
            )
        ]
        if mismatched:
            raise RuntimeError(
                f"predictions for {case.case_id} do not match --prompt-version "
                f"{prompt_bundle.version}; select the recorded version instead"
            )

    judge_llm = Exp2JudgeClient(
        config_path,
        config_section=judge_config_section,
        model=judge_model,
    )
    evaluator = Table2Evaluator(
        judge_llm=judge_llm,
        device=device,
        batch_size=batch_size,
    )
    all_scores: List[Dict[str, Any]] = []
    for case in cases:
        paths = CasePaths.for_case(output_dir, case)
        paths.ensure_parents()
        all_scores.extend(evaluate_case_table2(case, paths, evaluator))

    aggregate = aggregate_table2_scores(all_scores)
    evaluation_dir = Path(output_dir).resolve() / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    result_path = evaluation_dir / "table2_main_results.json"
    table_path = evaluation_dir / "table2_main_results.md"
    save_json(str(result_path), {
        "protocol": {
            "experiment_protocol_version": PROTOCOL_VERSION,
            "generation_prompts": prompt_bundle.manifest(),
            "lexical": "ROUGE-L F1",
            "semantic": f"BERTScore F1 ({evaluator.bertscore_model})",
            "categorical": "accuracy against the reference reply label",
            "continuous": "absolute difference from the reference reply score",
            "evaluation_prompt_version": EVALUATION_PROMPT_VERSION,
            "annotation_cache_key_version": ANNOTATION_CACHE_KEY_VERSION,
            "evaluation_prompt_sha256": hashlib.sha256("\n".join((
                REALTALK_REFLECTIVE_EVALUATION_SYSTEM_PROMPT,
                REALTALK_GROUNDING_EVALUATION_SYSTEM_PROMPT,
                REALTALK_EMPATHY_EVALUATION_SYSTEM_PROMPT,
            )).encode("utf-8")).hexdigest(),
            "judge_model": judge_llm.model,
            "judge_backend": judge_llm.backend,
            "judge_base_url": judge_llm.base_url,
            "judge_config_section": judge_config_section,
            "sentiment_model": evaluator.sentiment_model_name,
            "emotion_model": evaluator.emotion_model_name,
            "intimacy_model": evaluator.intimacy_model_name,
            "device": device,
        },
        "paper_baselines": list(TABLE2_BASELINES),
        **aggregate,
    })
    table_path.write_text(render_table2_markdown(aggregate["ours"]), encoding="utf-8")
    return {
        "table2_results": str(result_path),
        "table2_markdown": str(table_path),
    }


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def _select_cases(
    cases: Iterable[ExperimentCase],
    selectors: List[str],
) -> List[ExperimentCase]:
    if not selectors:
        return list(cases)
    wanted = {selector.lower() for selector in selectors}
    selected = [
        case
        for case in cases
        if case.case_id.lower() in wanted or Path(case.dataset_path).name.lower() in wanted
    ]
    missing = wanted - {
        value
        for case in selected
        for value in (case.case_id.lower(), Path(case.dataset_path).name.lower())
    }
    if missing:
        raise ValueError(f"unknown case selectors: {sorted(missing)}")
    return selected


def _model_name(config_path: str) -> str:
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    return config.get("API", "model", fallback="unknown")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_cases = build_cases(args.dataset_dir, args.train_ratio)
    save_split_manifest(all_cases, output_dir / "split_manifest.json", args.train_ratio)
    cases = _select_cases(all_cases, args.case)
    prompt_bundle = get_exp2_prompt_bundle(args.prompt_version)

    generated: Dict[str, int] = {}
    if args.phase in ("prepare", "all"):
        for case in cases:
            paths = CasePaths.for_case(output_dir, case)
            paths.ensure_parents()
            build_case_persona(case, paths, args.config)
            build_case_profile(case, paths, args.config)

    if args.phase in ("generate", "generate-evaluate", "all"):
        for case in cases:
            paths = CasePaths.for_case(output_dir, case)
            generated[case.case_id] = run_case_replies(
                case,
                paths,
                args.config,
                prompt_version=prompt_bundle.version,
            )

    evaluation: Dict[str, str] = {}
    if args.phase in ("evaluate", "generate-evaluate", "all"):
        evaluation = evaluate_table2(
            cases=cases,
            output_dir=output_dir,
            config_path=args.config,
            judge_model=args.judge_model,
            judge_config_section=args.judge_config_section,
            device=args.eval_device,
            batch_size=args.eval_batch_size,
            prompt_version=prompt_bundle.version,
        )

    save_json(str(output_dir / "run_manifest.json"), {
        "experiment": "Experiment 2. User Modeling Evaluation",
        "protocol_version": PROTOCOL_VERSION,
        "generation_prompts": prompt_bundle.manifest(),
        "evaluation_prompt_version": EVALUATION_PROMPT_VERSION,
        "annotation_cache_key_version": ANNOTATION_CACHE_KEY_VERSION,
        "persona_schema": persona_schema_manifest(),
        "research_question": "Does explicit user modeling enable better personalized interactions?",
        "phase": args.phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "train_ratio": args.train_ratio,
        "model": _model_name(args.config),
        "cases": [asdict(case) for case in cases],
        "generated_this_run": generated,
        "evaluation": evaluation,
        "evaluation_status": "complete" if evaluation else "not_run",
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run REALTALK Experiment 2 and its Table 2 evaluation."
    )
    parser.add_argument(
        "--phase",
        choices=("prepare", "generate", "evaluate", "generate-evaluate", "all"),
        default="all",
        help=(
            "prepare assets, generate replies, evaluate existing replies, run "
            "generation followed by evaluation, or run all three stages"
        ),
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/exp2_user_modeling")
    parser.add_argument("--config", default="config.ini")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument(
        "--prompt-version",
        choices=exp2_prompt_versions(),
        default=DEFAULT_EXP2_PROMPT_VERSION,
        help="Immutable generation/alignment prompt bundle recorded with every reply.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override [EvaluationAPI].model; REALTALK uses gpt-4o-mini.",
    )
    parser.add_argument(
        "--judge-config-section",
        default="EvaluationAPI",
        help="Independent config.ini section for the LLM judge backend and key env name.",
    )
    parser.add_argument(
        "--eval-device",
        default="cuda:0",
        help=(
            "Hugging Face and BERTScore device used by evaluate, "
            "generate-evaluate, and all."
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run a full named conversation case; repeat to select multiple cases.",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
