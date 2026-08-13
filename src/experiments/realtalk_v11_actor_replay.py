"""V11 soft-contract Actor replay over frozen V9 REALTALK decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .exp1_protocol import REALTALK_PERSONA_SPLITS, protocol_turns, stable_hash
from .operation_checkpoint import OperationCheckpoint
from .realtalk_ours import (
    ALLOWED_MODELS,
    _backend_from_env,
    _target_spoke_in_session,
    _text_call,
    _turns_with_session_boundaries,
)


PROTOCOL = "realtalk_task1_ours_v11_soft_actor_replay_v1"
FIRST_FIVE = tuple(item["speaker"] for item in REALTALK_PERSONA_SPLITS[:5])
ALL_TEN = tuple(item["speaker"] for item in REALTALK_PERSONA_SPLITS)

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

PRIVATE NEXT ACTION:
{next_action}

SOFT ACTION CONTRACT:
{action_contract}

QUESTION PERMISSION:
{question_permission}

OPTIONAL ADDITION PERMISSION:
{optional_addition_permission}

Speak naturally as the target person. Complete the primary move first. Add at most one brief, same-topic
personal reason, feeling, or immediate reaction only when PRIVATE NEXT ACTION's communicative_intent,
content_direction, or self_expression already explicitly calls for that kind of personal content. Otherwise
realize only the primary move. Do not add a second topic, generic reassurance, therapeutic analysis, advice,
or an interview-like question. Follow QUESTION PERMISSION literally. When it says forbidden, the output must
contain no question and no question mark. An `open` primary move may itself be the one check-in named by its
content_direction, but it gets no second question.

The observable statistics are descriptive style evidence, not hard ceilings or quotas. In particular, a low
reflective-marker rate does not prohibit a brief reason or feeling when it naturally belongs to this action.
Keep any unsupported self-expression ordinary and low-specificity. Reuse concrete entities only when they are
already present in the visible history and still belong to the current exchange. Never turn an event, plan,
place, activity, or anecdote from an earlier session into something the target is currently doing or has just
done. Do not enrich an allowed entity with unsupported venue, location, recipe, weather, timeline, or scene
details. Output one message only."""


def select_fixed_rows(
    rows: list[dict[str, Any]], speakers: Iterable[str], *, per_session: int = 2
) -> list[dict[str, Any]]:
    """Select deterministic full-span points for each speaker and each of three sessions."""
    by_speaker = {speaker.casefold(): [] for speaker in speakers}
    for row in rows:
        key = str(row["speaker"]).casefold()
        if key in by_speaker:
            by_speaker[key].append(row)
    selected: list[dict[str, Any]] = []
    for speaker in speakers:
        speaker_rows = by_speaker[speaker.casefold()]
        if not speaker_rows:
            raise ValueError(f"source is missing speaker {speaker!r}")
        for session in ("session_1", "session_2", "session_3"):
            session_rows = sorted(
                (row for row in speaker_rows if row["target_session"] == session),
                key=lambda row: int(row["message_level_index"]),
            )
            if len(session_rows) < per_session:
                raise ValueError(
                    f"{speaker} {session} has {len(session_rows)} rows; needs {per_session}"
                )
            indices = (
                [len(session_rows) // 2]
                if per_session == 1
                else [
                    round(index * (len(session_rows) - 1) / (per_session - 1))
                    for index in range(per_session)
                ]
            )
            selected.extend(session_rows[index] for index in indices)
    result_ids = [row["result_id"] for row in selected]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("fixed selection produced duplicate result IDs")
    return selected


def run(
    source_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    *,
    speakers: tuple[str, ...],
    per_session: int = 2,
) -> dict[str, Any]:
    source_rows = _read_jsonl(source_dir / "predictions.jsonl")
    selected = select_fixed_rows(source_rows, speakers, per_session=per_session)
    self_domains = json.loads((source_dir / "self_domains.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((source_dir / "run_manifest.json").read_text(encoding="utf-8"))
    source_model = source_manifest["ours_model"]
    if source_model not in ALLOWED_MODELS:
        raise ValueError(f"V11 source model is not allowed: {source_model}")
    backend = _backend_from_env(source_model)
    if backend.model != source_model:
        raise ValueError(f"V11 replay requires {source_model}, got {backend.model}")

    selection = [row["result_id"] for row in selected]
    source_hashes = {
        "predictions": _sha256(source_dir / "predictions.jsonl"),
        "self_domains": _sha256(source_dir / "self_domains.json"),
        "run_manifest": _sha256(source_dir / "run_manifest.json"),
    }
    signature = stable_hash({
        "protocol": PROTOCOL,
        "source_hashes": source_hashes,
        "selection": selection,
        "generation_system": stable_hash(GENERATION_SYSTEM_PROMPT),
        "generation_user": stable_hash(GENERATION_USER_PROMPT),
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
            envelope = _text_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key=f"v11_actor:{result_id}",
                system_prompt=GENERATION_SYSTEM_PROMPT.format(speaker=row["speaker"]),
                user_prompt=GENERATION_USER_PROMPT.format(
                    history=_turns_with_session_boundaries(context),
                    current_session=row["target_session"],
                    target_spoke_in_current_session=_target_spoke_in_session(
                        context, row["speaker"], row["target_session"]
                    ),
                    behavioral_self_domain=json.dumps(
                        _v11_behavioral_self_domain(self_domains[row["speaker"]]),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    situation=json.dumps(row["situation"], ensure_ascii=False, indent=2),
                    next_action=json.dumps(row["next_action"], ensure_ascii=False, indent=2),
                    action_contract=_soft_action_contract(
                        row["next_action"]["primary_move"],
                        row["next_action"].get("continuation_move", "none"),
                    ),
                    question_permission=_question_permission(row["next_action"]),
                    optional_addition_permission=_optional_addition_permission(
                        row["next_action"]
                    ),
                ),
                speaker=row["speaker"],
                max_attempts=3,
                raw_audit=raw_audit,
                enable_thinking=False,
            )
            row["v9_generated_message"] = row["generated_message"]
            row["generated_message"] = envelope["data"]
            row["v11_actor_audit"] = envelope["audit"]
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
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
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
        "speakers": list(speakers),
        "per_session": per_session,
        "selection_mode": "three_sessions_full_span_positions",
        "sample_ids_sha256": stable_hash(selection),
        "records_expected": len(selected),
        "records_complete": len(output_rows),
        "unresolved_errors": len(unresolved),
        "decision_and_domains_regenerated": False,
        "generation_only_replayed": True,
        "prompt_hashes": {
            "system": stable_hash(GENERATION_SYSTEM_PROMPT),
            "user": stable_hash(GENERATION_USER_PROMPT),
        },
        "output_predictions_sha256": _sha256(output_dir / "predictions.jsonl"),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "run_signature": signature,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _v11_behavioral_self_domain(self_domain: dict[str, Any]) -> dict[str, Any]:
    return {
        "communication_signature": self_domain["communication_signature"],
        "interaction_policy_prior": self_domain["interaction_policy_prior"],
        "affective_social_signature": self_domain["affective_social_signature"],
        "observable_statistics": self_domain["observable_statistics"],
        "calibration_note": (
            "Statistics describe observed tendencies. They are not ceilings, quotas, or commands."
        ),
    }


def _soft_action_contract(primary_move: str, continuation_move: str) -> str:
    contracts = {
        "open": "Begin with the intended greeting or check-in.",
        "self-disclose": "Contribute the intended self-focused update or view.",
        "answer": "Answer the latest question directly.",
        "acknowledge": "Give the intended concise reaction to the partner content.",
        "follow-up": "Ask the one relevant question chosen as the primary move.",
        "topic-shift": "Introduce the intended target-led topic directly.",
    }
    if primary_move not in contracts:
        raise ValueError(f"unknown primary move: {primary_move}")
    if continuation_move not in {"none", "reciprocal-question"}:
        raise ValueError(f"unknown continuation move: {continuation_move}")
    suffix = (
        " Then ask exactly one short reciprocal question about the same conversational slot."
        if continuation_move == "reciprocal-question"
        else " Do not append an additional question."
    )
    return contracts[primary_move] + suffix


def _question_permission(action: dict[str, Any]) -> str:
    primary = action["primary_move"]
    continuation = action.get("continuation_move", "none")
    if continuation == "reciprocal-question":
        return "exactly_one_same_slot_reciprocal_question"
    if primary == "follow-up":
        return "exactly_one_primary_follow_up_question"
    if primary == "open" and any(
        marker in " ".join(
            str(action.get(field, ""))
            for field in ("communicative_intent", "content_direction")
        ).casefold()
        for marker in ("check-in", "check in", "inquiry", "how are", "wellbeing")
    ):
        return "at_most_one_opening_check_in_question"
    return "forbidden_no_question_mark"


def _optional_addition_permission(action: dict[str, Any]) -> str:
    if action["primary_move"] == "self-disclose":
        return "one_brief_same_topic_personal_reason_feeling_or_reaction"
    description = " ".join(
        str(action.get(field, ""))
        for field in ("communicative_intent", "content_direction", "self_expression")
    ).casefold()
    markers = (
        "personal take", "personal response", "personal touch", "personal experience",
        "personal status", "own view", "own opinion", "own evening", "share own",
        "reason", "feeling", "motivation",
    )
    return (
        "one_brief_same_topic_personal_reason_feeling_or_reaction"
        if any(marker in description for marker in markers)
        else "forbidden_realize_primary_move_only"
    )


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
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort", choices=("first5", "second5", "all10"), required=True)
    args = parser.parse_args()
    speakers = {
        "first5": FIRST_FIVE,
        "second5": ALL_TEN[5:],
        "all10": ALL_TEN,
    }[args.cohort]
    print(json.dumps(run(
        args.source_dir, args.dataset_dir, args.output_dir, speakers=speakers
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
