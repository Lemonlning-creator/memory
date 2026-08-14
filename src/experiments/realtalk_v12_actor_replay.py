"""V12 disciplined-natural Actor replay over frozen V9 REALTALK decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .exp1_protocol import REALTALK_PERSONA_SPLITS, protocol_turns, stable_hash
from .operation_checkpoint import OperationCheckpoint
from .personaemp.client import ChatResult
from .realtalk_ours import (
    ALLOWED_MODELS,
    _append_jsonl,
    _backend_from_env,
    _call_with_hard_timeout,
    _normalize_generated_message,
    _target_spoke_in_session,
    _turns_with_session_boundaries,
)


PROTOCOL = "realtalk_task1_ours_v12_disciplined_natural_actor_v1"
ACTOR_VIEW_LOGIC_VERSION = "compact_structured_action_v1"
SOURCE_ACTION_COMPATIBILITY_VERSION = "v9_early_schema_deterministic_projection_v1"
ALL_TEN = tuple(item["speaker"] for item in REALTALK_PERSONA_SPLITS)
SESSIONS = ("session_1", "session_2", "session_3")

GENERATION_SYSTEM_PROMPT = """You are {speaker}. Continue the conversation.
Act as the person represented by the private Self Domain.
Follow the private next-action decision naturally.
Output only the message, not the speaker name."""

GENERATION_USER_PROMPT = """REAL CONVERSATION HISTORY BEFORE YOUR NEXT MESSAGE:
{history}

CURRENT SESSION: {current_session}
YOU HAVE ALREADY SPOKEN IN THIS SESSION: {target_spoke_in_current_session}

PRIVATE BEHAVIORAL SELF DOMAIN:
{behavioral_self_domain}

PRIVATE CURRENT SITUATION:
{situation}

COMPACT PRIVATE ACTION:
{actor_action}

PRIMARY ACTION CONTRACT:
{action_contract}

SELF-REVELATION CONTRACT:
{self_revelation_contract}

QUESTION CONTRACT:
{question_contract}

Produce one natural message whose parts jointly realize the single primary move. A brief supporting component
is optional only when the primary action contract permits it, and it must help complete that same move rather
than become a second social action. Do not add reflection, psychological analysis, confirmation, exploration,
or a request for elaboration merely to make the message sound deeper. Follow the self-revelation and question
contracts literally.

Preserve the target person's ordinary wording, emotional color, cadence, and message scale. Natural warmth,
surprise, agreement, and casual emotion are allowed when they fit this person's style, but do not turn them
into generic reassurance or therapeutic interpretation. Keep unsupported self-expression ordinary and
low-specificity. Reuse concrete entities only when they are visible in the current history. Never turn an
event, plan, place, activity, or anecdote from an earlier session into something the target is currently doing
or has just done. Output one message only."""


def prepare_sample_manifests(
    source_rows: list[dict[str, Any]],
    excluded_ids: set[str],
    output_dir: Path,
    *,
    speakers: tuple[str, ...] = ALL_TEN,
    holdout_per_speaker: int = 8,
) -> dict[str, Any]:
    """Create disjoint deterministic dev30 and holdout80 sample manifests."""
    available = _group_available(source_rows, excluded_ids, speakers)
    dev_rows: list[dict[str, Any]] = []
    for speaker in speakers:
        for session in SESSIONS:
            rows = available[(speaker, session)]
            if not rows:
                raise ValueError(f"no unused development row for {speaker} {session}")
            dev_rows.append(rows[len(rows) // 2])

    dev_ids = {row["result_id"] for row in dev_rows}
    remaining = _group_available(source_rows, excluded_ids | dev_ids, speakers)
    holdout_rows: list[dict[str, Any]] = []
    allocations: dict[str, dict[str, int]] = {}
    for speaker in speakers:
        counts = {session: len(remaining[(speaker, session)]) for session in SESSIONS}
        allocation = _largest_remainder_allocation(counts, holdout_per_speaker)
        allocations[speaker] = allocation
        for session in SESSIONS:
            holdout_rows.extend(
                _select_full_span(remaining[(speaker, session)], allocation[session])
            )

    _validate_selection(dev_rows, speakers, 3)
    _validate_selection(holdout_rows, speakers, holdout_per_speaker)
    holdout_ids = {row["result_id"] for row in holdout_rows}
    if dev_ids & holdout_ids or dev_ids & excluded_ids or holdout_ids & excluded_ids:
        raise ValueError("V12 sample cohorts are not disjoint")

    output_dir.mkdir(parents=True, exist_ok=True)
    dev_path = output_dir / "v12_dev30_sample_ids.json"
    holdout_path = output_dir / "v12_holdout80_sample_ids.json"
    dev_path.write_text(json.dumps([row["result_id"] for row in dev_rows], indent=2), encoding="utf-8")
    holdout_path.write_text(
        json.dumps([row["result_id"] for row in holdout_rows], indent=2), encoding="utf-8"
    )
    manifest = {
        "protocol": "realtalk_v12_disjoint_sample_selection_v1",
        "source_records": len(source_rows),
        "excluded_records": len(excluded_ids),
        "dev_records": len(dev_rows),
        "holdout_records": len(holdout_rows),
        "speakers": list(speakers),
        "sessions": list(SESSIONS),
        "holdout_per_speaker": holdout_per_speaker,
        "holdout_allocations": allocations,
        "dev_ids_sha256": stable_hash([row["result_id"] for row in dev_rows]),
        "holdout_ids_sha256": stable_hash([row["result_id"] for row in holdout_rows]),
        "excluded_ids_sha256": stable_hash(sorted(excluded_ids)),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def run(
    source_dir: Path,
    dataset_dir: Path,
    sample_ids_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_rows = _read_jsonl(source_dir / "predictions.jsonl")
    source_by_id = {row["result_id"]: row for row in source_rows}
    sample_ids = json.loads(sample_ids_path.read_text(encoding="utf-8"))
    if not isinstance(sample_ids, list) or not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample ID manifest must be a non-empty unique list")
    missing = [result_id for result_id in sample_ids if result_id not in source_by_id]
    if missing:
        raise ValueError(f"sample IDs absent from V9 source: {missing[:3]}")
    selected = [source_by_id[result_id] for result_id in sample_ids]

    self_domains = json.loads((source_dir / "self_domains.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((source_dir / "run_manifest.json").read_text(encoding="utf-8"))
    source_model = source_manifest["ours_model"]
    if source_model not in ALLOWED_MODELS:
        raise ValueError(f"V12 source model is not allowed: {source_model}")
    backend = _backend_from_env(source_model)
    if backend.model != source_model:
        raise ValueError(f"V12 replay requires {source_model}, got {backend.model}")

    source_hashes = {
        "predictions": _sha256(source_dir / "predictions.jsonl"),
        "self_domains": _sha256(source_dir / "self_domains.json"),
        "run_manifest": _sha256(source_dir / "run_manifest.json"),
    }
    signature = stable_hash({
        "protocol": PROTOCOL,
        "source_hashes": source_hashes,
        "sample_ids": sample_ids,
        "sample_ids_file_sha256": _sha256(sample_ids_path),
        "generation_system": stable_hash(GENERATION_SYSTEM_PROMPT),
        "generation_user": stable_hash(GENERATION_USER_PROMPT),
        "actor_view_logic_version": ACTOR_VIEW_LOGIC_VERSION,
        "source_action_compatibility_version": SOURCE_ACTION_COMPATIBILITY_VERSION,
        "module_sha256": _sha256(Path(__file__)),
        "model": source_model,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = OperationCheckpoint(output_dir / "checkpoint.json", signature)
    raw_audit = output_dir / "raw_responses.jsonl"
    chat_cache: dict[str, dict[str, dict[str, Any]]] = {}
    output_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []

    for source in selected:
        row = dict(source)
        result_id = row["result_id"]
        try:
            test_chat = row["test_chat"]
            if test_chat not in chat_cache:
                chat = json.loads((dataset_dir / test_chat).read_text(encoding="utf-8"))
                chat_cache[test_chat] = {
                    turn["turn_id"]: turn
                    for turn in protocol_turns(chat, merge_adjacent_bubbles=True)
                }
            by_id = chat_cache[test_chat]
            context = [by_id[turn_id] for turn_id in row["context_turn_ids"]]
            actor_action = _compact_actor_action(row["next_action"], row["situation"])
            envelope = _actor_text_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key=f"v12_actor:{result_id}",
                system_prompt=GENERATION_SYSTEM_PROMPT.format(speaker=row["speaker"]),
                user_prompt=GENERATION_USER_PROMPT.format(
                    history=_turns_with_session_boundaries(context),
                    current_session=row["target_session"],
                    target_spoke_in_current_session=_target_spoke_in_session(
                        context, row["speaker"], row["target_session"]
                    ),
                    behavioral_self_domain=json.dumps(
                        _behavioral_self_domain(self_domains[row["speaker"]]),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    situation=json.dumps(row["situation"], ensure_ascii=False, indent=2),
                    actor_action=json.dumps(actor_action, ensure_ascii=False, indent=2),
                    action_contract=_action_contract(actor_action),
                    self_revelation_contract=_self_revelation_contract(actor_action),
                    question_contract=_question_contract(actor_action),
                ),
                speaker=row["speaker"],
                action=actor_action,
                raw_audit=raw_audit,
            )
            row["v9_generated_message"] = row["generated_message"]
            row["generated_message"] = envelope["data"]
            row["v12_actor_action"] = actor_action
            row["v12_actor_audit"] = envelope["audit"]
            output_rows.append(row)
        except Exception as exc:
            unresolved.append({
                "result_id": result_id,
                "type": type(exc).__name__,
                "error": str(exc),
            })

    _write_jsonl(output_dir / "predictions.jsonl", output_rows)
    _write_jsonl(output_dir / "v9_baseline_predictions.jsonl", selected)
    (output_dir / "sample_ids.json").write_text(
        json.dumps(sample_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "unresolved_errors.json").write_text(
        json.dumps(unresolved, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "complete" if len(output_rows) == len(selected) and not unresolved else "incomplete",
        "protocol": PROTOCOL,
        "source_dir": str(source_dir.resolve()),
        "source_hashes": source_hashes,
        "model": source_model,
        "sample_ids_path": str(sample_ids_path.resolve()),
        "sample_ids_file_sha256": _sha256(sample_ids_path),
        "sample_ids_sha256": stable_hash(sample_ids),
        "records_expected": len(selected),
        "records_complete": len(output_rows),
        "unresolved_errors": len(unresolved),
        "decision_and_domains_regenerated": False,
        "generation_only_replayed": True,
        "thinking_enabled": False,
        "verification_enabled": False,
        "prompt_hashes": {
            "system": stable_hash(GENERATION_SYSTEM_PROMPT),
            "user": stable_hash(GENERATION_USER_PROMPT),
        },
        "actor_view_logic_version": ACTOR_VIEW_LOGIC_VERSION,
        "source_action_compatibility_version": SOURCE_ACTION_COMPATIBILITY_VERSION,
        "module_sha256": _sha256(Path(__file__)),
        "output_predictions_sha256": _sha256(output_dir / "predictions.jsonl"),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "run_signature": signature,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _compact_actor_action(
    action: dict[str, Any], situation: dict[str, Any] | None = None
) -> dict[str, Any]:
    fields = (
        "primary_move", "content_direction", "tone", "message_scale",
        "question_mode", "continuation_move",
    )
    missing = [field for field in fields if field not in action]
    if missing:
        raise ValueError(f"frozen action lacks V12 fields: {missing}")
    primary = action["primary_move"]
    revelation = action.get("self_revelation_mode")
    if revelation is None:
        revelation = (
            "state-only"
            if primary in {"answer", "self-disclose", "topic-shift"}
            else "none"
        )
    missing_information = action.get("missing_information")
    if missing_information is None:
        missing_information = (situation or {}).get("missing_information", "")
    return {
        **{field: action[field] for field in fields},
        "self_revelation_mode": revelation,
        "missing_information": missing_information,
    }


def _action_contract(action: dict[str, Any]) -> str:
    contracts = {
        "answer": "Answer directly. You may add one brief non-reflective qualifier that completes the answer.",
        "acknowledge": "React naturally. You may add one brief same-topic observation without analyzing either person.",
        "self-disclose": "State the intended personal status, view, or update. A same-topic detail is optional.",
        "follow-up": "Ask the one concrete question selected as the primary move; an extremely short lead-in is optional.",
        "open": "Give the intended natural greeting or check-in without adding a second conversational move.",
        "topic-shift": "Introduce the intended target-led topic directly; one short setup detail is optional.",
    }
    primary = action["primary_move"]
    if primary not in contracts:
        raise ValueError(f"unknown primary move: {primary}")
    return contracts[primary]


def _self_revelation_contract(action: dict[str, Any]) -> str:
    mode = action["self_revelation_mode"]
    contracts = {
        "none": "Do not explain the target's inner state, motive, reason, or feeling.",
        "state-only": "A state, choice, plan, preference, or view may be stated, but do not explain why.",
        "brief-reason-or-feeling": "At most one brief natural clause may express a reason or feeling relevant to this action.",
    }
    if mode not in contracts:
        raise ValueError(f"unknown self revelation mode: {mode}")
    return contracts[mode]


def _question_contract(action: dict[str, Any]) -> str:
    primary = action["primary_move"]
    continuation = action["continuation_move"]
    if primary == "follow-up":
        return "Ask exactly one concrete primary question. Do not add another question."
    if continuation == "reciprocal-question":
        detail = str(action["missing_information"]).strip()
        if not detail:
            raise ValueError("reciprocal question lacks missing information")
        return f"After the primary move, ask exactly one short same-slot question about: {detail}"
    return "Do not ask a question and do not use a question mark."


def _validate_actor_message(message: str, action: dict[str, Any]) -> str:
    count = message.count("?")
    expects_one = (
        action["primary_move"] == "follow-up"
        or action["continuation_move"] == "reciprocal-question"
    )
    if expects_one and count != 1:
        raise ValueError(f"V12 action requires exactly one question mark, got {count}")
    if not expects_one and count:
        raise ValueError(f"V12 action forbids question marks, got {count}")
    return message


def _actor_text_call(
    *, checkpoint: OperationCheckpoint, backend: Any, operation_key: str,
    system_prompt: str, user_prompt: str, speaker: str, action: dict[str, Any],
    raw_audit: Path,
) -> dict[str, Any]:
    logical_attempt = {"value": 0}

    def operation() -> ChatResult:
        logical_attempt["value"] += 1
        return _call_with_hard_timeout(
            lambda: backend.chat(
                system_prompt, user_prompt, temperature=0.6, top_p=0.9,
                max_tokens=300, enable_thinking=False,
            ),
            0,
            operation_key,
        )

    def validate(result: ChatResult) -> dict[str, Any]:
        _append_jsonl(raw_audit, {
            "operation_key": operation_key,
            "logical_attempt": logical_attempt["value"],
            "model": result.model,
            "raw_response": result.content,
            "reasoning_content": result.reasoning_content,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_seconds": result.latency_seconds,
            "network_attempts": result.attempts,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
        })
        message = _normalize_generated_message(result.content, speaker)
        _validate_actor_message(message, action)
        return {
            "data": message,
            "audit": {
                "model": result.model,
                "logical_attempts": logical_attempt["value"],
                "network_attempts_last_call": result.attempts,
                "prompt_tokens_last_call": result.prompt_tokens,
                "completion_tokens_last_call": result.completion_tokens,
                "latency_seconds_last_call": result.latency_seconds,
                "thinking_enabled": False,
                "reasoning_sha256": stable_hash(result.reasoning_content),
            },
        }

    return checkpoint.execute(
        operation_key, operation, validate, 3,
        usage_supplier=lambda: dict(getattr(backend, "token_usage", {})),
    )


def _behavioral_self_domain(self_domain: dict[str, Any]) -> dict[str, Any]:
    return {
        "communication_signature": self_domain["communication_signature"],
        "interaction_policy_prior": self_domain["interaction_policy_prior"],
        "affective_social_signature": self_domain["affective_social_signature"],
        "observable_statistics": self_domain["observable_statistics"],
        "calibration_note": "Observed tendencies guide natural style; they are not quotas or hard limits.",
    }


def _group_available(
    rows: list[dict[str, Any]], excluded_ids: set[str], speakers: tuple[str, ...]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups = {(speaker, session): [] for speaker in speakers for session in SESSIONS}
    speaker_lookup = {speaker.casefold(): speaker for speaker in speakers}
    for row in rows:
        speaker = speaker_lookup.get(str(row["speaker"]).casefold())
        session = row["target_session"]
        if speaker is not None and session in SESSIONS and row["result_id"] not in excluded_ids:
            groups[(speaker, session)].append(row)
    for grouped in groups.values():
        grouped.sort(key=lambda row: int(row["message_level_index"]))
    return groups


def _largest_remainder_allocation(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if total < target:
        raise ValueError(f"only {total} rows available for target {target}")
    quotas = {session: target * counts[session] / total for session in SESSIONS}
    allocation = {session: min(counts[session], math.floor(quotas[session])) for session in SESSIONS}
    remaining = target - sum(allocation.values())
    order = sorted(
        SESSIONS,
        key=lambda session: (quotas[session] - math.floor(quotas[session]), counts[session], -SESSIONS.index(session)),
        reverse=True,
    )
    while remaining:
        progressed = False
        for session in order:
            if allocation[session] < counts[session]:
                allocation[session] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError("unable to allocate holdout rows")
    return allocation


def _select_full_span(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count == 0:
        return []
    if count > len(rows):
        raise ValueError("full-span selection exceeds available rows")
    indices = (
        [len(rows) // 2]
        if count == 1
        else [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    )
    if len(indices) != len(set(indices)):
        raise ValueError("full-span selection produced duplicate positions")
    return [rows[index] for index in indices]


def _validate_selection(rows: list[dict[str, Any]], speakers: tuple[str, ...], per_speaker: int) -> None:
    ids = [row["result_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("sample selection contains duplicate IDs")
    for speaker in speakers:
        count = sum(row["speaker"].casefold() == speaker.casefold() for row in rows)
        if count != per_speaker:
            raise ValueError(f"selection has {count} rows for {speaker}; expected {per_speaker}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-samples")
    prepare.add_argument("--source-dir", type=Path, required=True)
    prepare.add_argument("--excluded-sample-ids", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    replay = subparsers.add_parser("run")
    replay.add_argument("--source-dir", type=Path, required=True)
    replay.add_argument("--dataset-dir", type=Path, required=True)
    replay.add_argument("--sample-ids", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-samples":
        rows = _read_jsonl(args.source_dir / "predictions.jsonl")
        excluded = set(json.loads(args.excluded_sample_ids.read_text(encoding="utf-8")))
        result = prepare_sample_manifests(rows, excluded, args.output_dir)
    else:
        result = run(args.source_dir, args.dataset_dir, args.sample_ids, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
