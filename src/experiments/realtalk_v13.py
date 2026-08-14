"""Progressive REALTALK V13 runner with transferable Self Domain decisions."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .exp1_protocol import REALTALK_PERSONA_SPLITS, select_realtalk_splits, stable_hash
from .operation_checkpoint import OperationCheckpoint
from .realtalk_ours import (
    DOMAIN_MAX_TOKENS,
    USER_DOMAIN_SCHEMA,
    USER_DOMAIN_SYSTEM_PROMPT,
    USER_DOMAIN_USER_TEMPLATE,
    RealTalkOursConfig,
    _backend_from_env,
    _call_with_hard_timeout,
    _checkpoint_unresolved,
    _checkpoint_usage,
    _failure,
    _normalize_generated_message,
    _prepare_dataset,
    _profile_activation_whitelist,
    _repository_commit,
    _run_preflight,
    _safe_host,
    _structured_call,
    _target_spoke_in_session,
    _turns_with_ids,
    _turns_with_session_boundaries,
    _validate_decision_profile_activation,
    _validate_user_domain_evidence,
    _write_json,
    _write_jsonl,
    _append_jsonl,
)
from .realtalk_ours_schemas import empty_user_domain, normalize_user_domain
from .realtalk_v13_schemas import (
    TURN_TRIGGERS,
    V13_DECISION_SCHEMA,
    V13_SELF_DOMAIN_SCHEMA,
    normalize_v13_decision,
    normalize_v13_self_domain,
)


PROTOCOL = "realtalk_task1_ours_agentic_v13_3_progressive_v1"
MODEL = "deepseek-v4-flash"
GATES = (6, 18, 30, 60, 120, 519)
SELF_MAX_TOKENS = 4000
USER_MAX_TOKENS = 4000

SELF_SYSTEM = """Compile a private cross-partner Self Domain for persona simulation.
Infer the target's stable voice and behavior from the complete Ca conversation, while separating portable
patterns from behavior that may be specific to this Ca partner. Current Cb history will be authoritative if
it conflicts with a Ca identity fact or interaction pattern. Absence in three sessions is uncertainty, not an
absolute prohibition. Describe what is observed rather than an ideal conversational style. Copy both supplied
statistics objects exactly. Return only the strict schema."""

SELF_USER = """TARGET SPEAKER: {speaker}
SOURCE: first three consecutive sessions of the paper-assigned Ca conversation.

COMPLETE LOSSLESSLY MERGED CA CONVERSATION:
{history}

DETERMINISTIC GLOBAL STATISTICS (copy exactly):
{global_stats}

DETERMINISTIC CONDITIONAL STATISTICS (copy exactly):
{conditional_stats}

For each conditional behavior trigger, explain the observed surface pattern and its confidence. Mark patterns
that may depend on this Ca partner instead of presenting them as universal personality rules."""

DECISION_SYSTEM = """You are a private Decision Agent for persona simulation. Decide what the target person is
most likely to do next in this real conversation. The goal is to act as the target, not to produce an ideal
assistant response.

Use the Self Domain as a cross-partner prior. Give current Cb history priority over Ca facts and partner-specific
Ca habits. Use at most two currently relevant User Domain facts. Scan the complete visible history before deciding:
a multipart question may remain partly unanswered after an intervening reply, and the newest partner turn is not
necessarily the only open obligation. Record the exact visible source turn ID for the obligation. An
answer-current-question or answer-earlier-unanswered-question source must contain a literal question mark in the
visible transcript. Then choose one
primary move and at most one same-slot companion move. A real message may answer and briefly
self-disclose, react and reciprocate, or answer and return the same question; do not force every turn into a
single sterile sentence.

The primary move is what the target mainly does. The companion move is either none, a brief reaction,
self-disclosure, or ask. If a question is planned, exactly one of primary_move or companion_move must be ask;
otherwise neither may be ask. turn_obligation describes the conversational obligation and need not duplicate
the primary move label word-for-word, but an unanswered question must still use answer as the primary move.

Choose reflection_depth=surface for facts, plans, preferences, reactions, and ordinary feelings. Choose
brief-reflective only when the visible turn naturally calls for a reason, motivation, or self-observation and
the target's observed behavior supports it. Reflection is not conversational polish.

Choose a question only when the target would naturally ask at this exact point. Being asked a question creates an
answer obligation, not automatic permission to ask one back. Use a reciprocal question only when the same-slot
exchange is already established and the target's conditional behavior supports returning it. Distinguish that
from clarification and a genuine follow-up. Do not ask merely to increase engagement or deepen the dialogue.
If the latest partner message directly asks the target a question, the alignment cannot be self-led: answering
that obligation requires at least balanced alignment. With no history, use self-led and open.

The relational register controls surface style; it does not license extra reassurance, concern, praise, or
psychological interpretation beyond the selected action and the target's observed pattern.

lambda_trace records how much this action adapts to the current partner: self-led 0.00-0.35, balanced 0.36-0.70,
partner-adaptive 0.71-1.00. It must agree with orientation and materially agree with the chosen policy. Select
the relational register and message shape from current interaction plus conditional Self behavior. Return only
the strict schema."""

DECISION_USER = """TARGET SPEAKER: {speaker}
PARTNER: {partner}

FIXED CROSS-PARTNER SELF DOMAIN:
{self_domain}

CURRENT FIVE-LAYER USER DOMAIN:
{user_domain}

COMPLETE REAL HISTORY BEFORE THE TARGET MESSAGE:
{history}

CURRENT SESSION: {current_session}
TARGET HAS SPOKEN IN THIS SESSION: {target_spoke}
LATEST PARTNER MESSAGE: {latest_partner}

EXACT USER DOMAIN ACTIVATION WHITELIST:
{whitelist}

Only copy relevant_user_domain entries verbatim from the whitelist. Submit one internally consistent policy."""

ACTOR_SYSTEM = """You are {speaker}. Continue the conversation.
Act as the person represented by the private Self Domain.
Follow the private behavior policy naturally.
Output only the message, not the speaker name."""

ACTOR_USER = """COMPLETE REAL CONVERSATION HISTORY BEFORE YOUR NEXT MESSAGE:
{history}

CURRENT SESSION: {current_session}
YOU HAVE ALREADY SPOKEN IN THIS SESSION: {target_spoke}

PRIVATE TRANSFERABLE SELF VIEW:
{self_view}

PRIVATE CURRENT SITUATION:
{situation}

PRIVATE BEHAVIOR POLICY:
{policy}

EXECUTION CONTRACT:
{contract}

Produce one natural target-person message. Complete the primary move and only the licensed companion move.
Follow the question plan exactly. Surface expression may state facts, status, preferences, plans, or ordinary
feelings without explaining motives. Brief-reflective expression may contain at most one concise reason or
self-observation. Match the selected relational register and the target's observed message shape. Current Cb
history overrides old Ca details; never present a Ca event as current. Do not add an unplanned second topic,
question, psychological analysis, generic reassurance, or therapeutic language."""


@dataclass(frozen=True)
class V13Config:
    dataset_dir: str
    output_dir: str
    v9_predictions: str
    v9_judge_scored: str
    v9_local_scored: str
    gate: int = 6
    fresh: bool = False
    operation_max_attempts: int = 3
    model_call_timeout_seconds: int = 240
    model: str = MODEL


def run(config: V13Config, backend: Any | None = None) -> dict[str, Any]:
    if config.gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}")
    if config.model != MODEL:
        raise ValueError(f"V13 fixes the model to {MODEL}")
    output = Path(config.output_dir).resolve()
    if config.fresh and config.gate != 6:
        raise ValueError("a fresh V13 protocol must start at gate 6")
    if config.fresh and output.exists() and any(output.iterdir()):
        raise ValueError("fresh V13 output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    base_config = RealTalkOursConfig(
        dataset_dir=config.dataset_dir,
        output_dir=config.output_dir,
        profile_sessions=3,
        test_sessions=3,
        compute_local_metrics=False,
        decision_thinking=False,
        model=config.model,
    )
    splits = select_realtalk_splits(config.dataset_dir)
    dataset_manifest, prepared = _prepare_dataset(base_config, splits)
    source_rows = _read_rows(Path(config.v9_predictions))
    judge_rows = _read_rows(Path(config.v9_judge_scored))
    local_rows = _read_rows(Path(config.v9_local_scored))
    manifests = prepare_gate_manifests(
        source_rows, judge_rows, local_rows,
        output / "gate_manifests.json",
    )
    selected_ids = manifests["gates"][str(config.gate)]
    selected = set(selected_ids)
    point_index = _point_index(prepared)
    missing = selected - set(point_index)
    if missing:
        raise ValueError(f"gate contains unknown dataset IDs: {sorted(missing)[:3]}")

    backend = backend or _backend_from_env(config.model)
    preflight = _run_preflight(
        output, backend, config.model, False, config.model_call_timeout_seconds
    )
    signature = stable_hash({
        "protocol": PROTOCOL,
        "model": config.model,
        "dataset": dataset_manifest["source_files_aggregate_sha256"],
        "v9_predictions": _sha256(Path(config.v9_predictions)),
        "gate_manifest": _sha256(output / "gate_manifests.json"),
        "prompts": _prompt_hashes(),
        "schemas": {
            "self": stable_hash(V13_SELF_DOMAIN_SCHEMA),
            "decision": stable_hash(V13_DECISION_SCHEMA),
            "user": stable_hash(USER_DOMAIN_SCHEMA),
        },
        "implementation_sources": _implementation_source_hashes(),
    })
    checkpoint = OperationCheckpoint(output / "checkpoint.json", signature)
    raw_audit = output / "raw_responses.jsonl"

    for speaker_data in prepared:
        speaker_points = [
            point for point in speaker_data["points"]
            if _result_id(speaker_data["speaker"], point) in selected
        ]
        if not speaker_points:
            continue
        speaker = speaker_data["speaker"]
        speaker_id = _speaker_id(speaker)
        global_stats = _global_statistics(speaker_data["profile"]["turns"], speaker)
        conditional_stats = conditional_statistics(speaker_data["profile"]["turns"], speaker)
        try:
            self_envelope = _structured_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key=f"v13:self:{speaker_id}",
                system_prompt=SELF_SYSTEM,
                user_prompt=SELF_USER.format(
                    speaker=speaker,
                    history=_turns_with_ids(speaker_data["profile"]["turns"]),
                    global_stats=_json(global_stats),
                    conditional_stats=_json(conditional_stats),
                ),
                schema=V13_SELF_DOMAIN_SCHEMA,
                normalizer=lambda value, gs=global_stats, cs=conditional_stats: _validate_self_stats(
                    normalize_v13_self_domain(value), gs, cs
                ),
                max_tokens=SELF_MAX_TOKENS,
                max_attempts=config.operation_max_attempts,
                raw_audit=raw_audit,
                enable_thinking=False,
                hard_timeout_seconds=config.model_call_timeout_seconds,
            )
            self_domain = self_envelope["data"]
        except Exception as exc:
            checkpoint.store_excluded_result(
                f"v13:self:{speaker_id}", _failure("self_domain", speaker, None, exc)
            )
            continue

        for point in speaker_points:
            result_id = _result_id(speaker, point)
            try:
                user_domain, completed_updates = _user_domain_for_point(
                    checkpoint, backend, raw_audit, config, speaker_data, point
                )
                partner_turns = [
                    turn for turn in point["context_turns"]
                    if turn["speaker"].casefold() == speaker_data["partner"].casefold()
                ]
                latest_partner = _turns_with_ids(partner_turns[-1:])
                latest_text = partner_turns[-1]["content"] if partner_turns else ""
                decision_envelope = _structured_call(
                    checkpoint=checkpoint,
                    backend=backend,
                    operation_key=f"v13:decision:{result_id}",
                    system_prompt=DECISION_SYSTEM,
                    user_prompt=DECISION_USER.format(
                        speaker=speaker,
                        partner=speaker_data["partner"],
                        self_domain=_json(self_domain),
                        user_domain=_json(user_domain),
                        history=_turns_with_session_boundaries(point["context_turns"]),
                        current_session=point["target_session"],
                        target_spoke=_target_spoke_in_session(
                            point["context_turns"], speaker, point["target_session"]
                        ),
                        latest_partner=latest_partner,
                        whitelist=_profile_activation_whitelist(user_domain),
                    ),
                    schema=V13_DECISION_SCHEMA,
                    normalizer=lambda value, ud=user_domain, hist=point["context_turns"], latest=latest_text: _validate_decision(
                        _validate_decision_profile_activation(
                            normalize_v13_decision(value), ud
                        ),
                        hist, latest,
                    ),
                    max_tokens=1500,
                    max_attempts=config.operation_max_attempts,
                    raw_audit=raw_audit,
                    enable_thinking=False,
                    hard_timeout_seconds=config.model_call_timeout_seconds,
                )
                decision = decision_envelope["data"]
                generation = _actor_call(
                    checkpoint=checkpoint,
                    backend=backend,
                    operation_key=f"v13:generation:{result_id}",
                    speaker=speaker,
                    system_prompt=ACTOR_SYSTEM.format(speaker=speaker),
                    user_prompt=ACTOR_USER.format(
                        history=_turns_with_session_boundaries(point["context_turns"]),
                        current_session=point["target_session"],
                        target_spoke=_target_spoke_in_session(
                            point["context_turns"], speaker, point["target_session"]
                        ),
                        self_view=_json(_actor_self_view(self_domain)),
                        situation=_json(decision["situation"]),
                        policy=_json(decision["behavior_policy"]),
                        contract=_actor_contract(decision["behavior_policy"]),
                    ),
                    policy=decision["behavior_policy"],
                    max_attempts=config.operation_max_attempts,
                    raw_audit=raw_audit,
                    timeout=config.model_call_timeout_seconds,
                )
                checkpoint.data["failures"].pop(f"sample:{result_id}", None)
                checkpoint.store_result(result_id, {
                    "result_id": result_id,
                    "speaker": speaker,
                    "partner": speaker_data["partner"],
                    "train_chat": speaker_data["split"]["train_chat"],
                    "test_chat": speaker_data["split"]["test_chat"],
                    "profile_sessions": list(speaker_data["profile"]["sessions"]),
                    "test_sessions": list(point["test_sessions"]),
                    "target_session": point["target_session"],
                    "message_level_index": point["message_level_index"],
                    "target_turn_id": point["target"]["turn_id"],
                    "context_turn_ids": [turn["turn_id"] for turn in point["context_turns"]],
                    "context_hash": point["history_hash"],
                    "context_truncated": point["context_truncated"],
                    "ground_truth": point["target_message"],
                    "generated_message": generation["data"],
                    "self_domain_hash": stable_hash(self_domain),
                    "user_domain": user_domain,
                    "user_domain_completed_session_updates": completed_updates,
                    "situation": decision["situation"],
                    "relevant_user_domain": decision["relevant_user_domain"],
                    "alignment": decision["alignment"],
                    "behavior_policy": decision["behavior_policy"],
                    "operation_audit": {
                        "decision": decision_envelope["audit"],
                        "generation": generation["audit"],
                    },
                })
            except Exception as exc:
                checkpoint.store_excluded_result(
                    result_id, _failure("sample", speaker, result_id, exc)
                )

    results_by_id = {
        row["result_id"]: row for row in checkpoint.result_values()
        if row.get("result_id") in selected
    }
    results = [results_by_id[result_id] for result_id in selected_ids if result_id in results_by_id]
    v9_by_id = {row["result_id"]: row for row in source_rows}
    gate_dir = output / "gates" / f"gate{config.gate}"
    gate_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(gate_dir / "v13_predictions.jsonl", results)
    _write_jsonl(
        gate_dir / "v9_predictions.jsonl",
        [v9_by_id[result_id] for result_id in selected_ids],
    )
    unresolved = _checkpoint_unresolved(checkpoint)
    _write_json(gate_dir / "unresolved_errors.json", unresolved)
    complete = len(results) == config.gate and not unresolved
    _write_json(gate_dir / "generation_summary.json", {
        "status": "complete" if complete else "incomplete",
        "protocol": PROTOCOL,
        "gate": config.gate,
        "expected": config.gate,
        "records": len(results),
        "unresolved": len(unresolved),
        "lambda_distribution": _lambda_distribution(results),
        "created_at_utc": _now(),
    })
    _write_json(output / "run_manifest.json", {
        "protocol": PROTOCOL,
        "model": config.model,
        "implementation_repository_commit": _repository_commit(),
        "active_gate": config.gate,
        "gate_sequence": list(GATES),
        "prompt_hashes": _prompt_hashes(),
        "schema_hashes": {
            "self": stable_hash(V13_SELF_DOMAIN_SCHEMA),
            "decision": stable_hash(V13_DECISION_SCHEMA),
            "user": stable_hash(USER_DOMAIN_SCHEMA),
        },
        "implementation_source_hashes": _implementation_source_hashes(),
        "dataset_manifest": dataset_manifest,
        "stage_thinking": {"self": False, "user": False, "decision": False, "actor": False},
        "omega_enabled": False,
        "future_user_state_enabled": False,
        "verification_or_reranking_enabled": False,
        "history_compression_enabled": False,
        "generated_output_rollout": False,
        "preflight": preflight,
        "checkpoint_usage": _checkpoint_usage(checkpoint, backend),
        "run_signature": signature,
        "created_at_utc": _now(),
    })
    _write_json(output / "self_domains.json", _cached_self_domains(checkpoint))
    return {
        "generation_complete": complete,
        "gate": config.gate,
        "records": len(results),
        "unresolved": unresolved,
        "gate_dir": str(gate_dir),
        "lambda_distribution": _lambda_distribution(results),
    }


def prepare_gate_manifests(
    predictions: list[dict[str, Any]],
    judged: list[dict[str, Any]],
    local: list[dict[str, Any]],
    path: Path,
) -> dict[str, Any]:
    pred = {row["result_id"]: row for row in predictions}
    judge = {row["result_id"]: row for row in judged}
    local_by_id = {row["result_id"]: row for row in local}
    if set(pred) != set(judge) or set(pred) != set(local_by_id) or len(pred) != 519:
        raise ValueError("V13 gate sources must align on all 519 V9 records")
    intimacy_values = sorted(
        row["local_metrics"]["intimacy_absolute_difference"] for row in local
    )
    intimacy_p90 = intimacy_values[math.ceil(0.90 * len(intimacy_values)) - 1]
    items = []
    for result_id, row in pred.items():
        score = judge[result_id]
        local_score = local_by_id[result_id]["local_metrics"]["intimacy_absolute_difference"]
        classes = {
            "reflect": score["metrics"]["reflectiveness_accuracy"] == 0,
            "ground": score["metrics"]["grounding_accuracy"] == 0,
            "intimacy": local_score >= intimacy_p90,
        }
        classes["clean"] = (
            not classes["reflect"] and not classes["ground"]
            and local_score <= statistics.median(intimacy_values)
        )
        items.append({
            "result_id": result_id,
            "speaker": row["speaker"],
            "session": row["target_session"],
            "message_level_index": row["message_level_index"],
            "classes": classes,
            "error_count": sum(classes[key] for key in ("reflect", "ground", "intimacy")),
        })
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in items:
        by_cell.setdefault((item["speaker"], item["session"]), []).append(item)
    expected_cells = [
        (split["speaker"], session)
        for split in REALTALK_PERSONA_SPLITS
        for session in ("session_1", "session_2", "session_3")
    ]
    if set(by_cell) != set(expected_cells):
        raise ValueError("V13 source does not cover every speaker-session cell")
    category_cycle = ("reflect", "ground", "intimacy", "clean")
    first: list[dict[str, Any]] = []
    cell_ranked: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, cell in enumerate(expected_cells):
        ranked = sorted(
            by_cell[cell], key=lambda item: (-item["error_count"], _id_hash(item["result_id"]))
        )
        desired = category_cycle[index % len(category_cycle)]
        preferred = [item for item in ranked if item["classes"][desired]]
        chosen = preferred[0] if preferred else ranked[0]
        first.append(chosen)
        cell_ranked[cell] = [chosen] + [item for item in ranked if item is not chosen]
    gate30 = _ordered_ids(first)
    gate6 = _select_gate6(first)
    gate18 = _select_gate18(first, gate6)
    gates: dict[str, list[str]] = {
        "6": gate6,
        "18": gate18,
        "30": gate30,
    }
    for size, per_cell in ((60, 2), (120, 4)):
        selected = list(itertools.chain.from_iterable(
            cell_ranked[cell][:per_cell] for cell in expected_cells
        ))
        if any(len(cell_ranked[cell]) < per_cell for cell in expected_cells):
            raise ValueError(f"not enough rows for gate {size}")
        gates[str(size)] = _ordered_ids(selected)
    gates["519"] = _ordered_ids(items)
    for smaller, larger in zip(GATES, GATES[1:]):
        if not set(gates[str(smaller)]).issubset(gates[str(larger)]):
            raise ValueError(f"gate {smaller} is not nested in gate {larger}")
    manifest = {
        "protocol": "realtalk_v13_nested_gate_selection_v1",
        "intimacy_p90": intimacy_p90,
        "source_hashes": {
            "predictions": stable_hash(predictions),
            "judge": stable_hash(judged),
            "local": stable_hash(local),
        },
        "gates": gates,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("existing V13 gate manifest differs from deterministic reconstruction")
    else:
        _write_json(path, manifest)
    return manifest


def conditional_statistics(turns: list[dict[str, Any]], speaker: str) -> dict[str, Any]:
    grouped = {trigger: [] for trigger in TURN_TRIGGERS}
    by_session: dict[str, list[dict[str, Any]]] = {}
    for turn in turns:
        by_session.setdefault(turn["session_id"], []).append(turn)
    for session_turns in by_session.values():
        for index, turn in enumerate(session_turns):
            if turn["speaker"].casefold() != speaker.casefold():
                continue
            previous = session_turns[index - 1] if index else None
            trigger = _turn_trigger(previous)
            grouped[trigger].append(turn)
    return {trigger: _turn_statistics(grouped[trigger]) for trigger in TURN_TRIGGERS}


def _turn_trigger(previous: dict[str, Any] | None) -> str:
    if previous is None:
        return "session-opening"
    text = previous["content"].casefold()
    if re.search(r"\b(bye|good night|talk (?:to you )?later|see you|speak soon)\b", text):
        return "after-closing"
    if "?" in text:
        return "after-direct-question"
    if re.search(r"\b(i|i'm|im|i've|ive|my|me|we|our)\b", text):
        return "after-partner-disclosure"
    return "after-partner-statement"


def _turn_statistics(turns: list[dict[str, Any]]) -> dict[str, Any]:
    if not turns:
        return {
            "observations": 0,
            "question_rate": 0.0,
            "first_person_rate": 0.0,
            "reflective_marker_rate": 0.0,
            "mean_characters": 0.0,
            "median_characters": 0.0,
            "median_merged_bubbles": 0.0,
        }
    texts = [turn["content"] for turn in turns]
    lengths = [len(text) for text in texts]
    bubbles = [len(turn.get("message_indices", [])) or 1 for turn in turns]
    return {
        "observations": len(turns),
        "question_rate": round(sum("?" in text for text in texts) / len(texts), 6),
        "first_person_rate": round(sum(bool(re.search(r"\b(i|i'm|im|i've|ive|my|me)\b", text.casefold())) for text in texts) / len(texts), 6),
        "reflective_marker_rate": round(sum(bool(re.search(r"\b(i think|i feel|i realize|because|makes me|i guess|i should)\b", text.casefold())) for text in texts) / len(texts), 6),
        "mean_characters": round(statistics.mean(lengths), 6),
        "median_characters": round(statistics.median(lengths), 6),
        "median_merged_bubbles": round(statistics.median(bubbles), 6),
    }


def _global_statistics(turns: list[dict[str, Any]], speaker: str) -> dict[str, Any]:
    target = [turn for turn in turns if turn["speaker"].casefold() == speaker.casefold()]
    stats = _turn_statistics(target)
    return {
        "target_message_count": stats.pop("observations"),
        "mean_characters": stats["mean_characters"],
        "median_characters": stats["median_characters"],
        "question_rate": stats["question_rate"],
        "first_person_rate": stats["first_person_rate"],
        "reflective_marker_rate": stats["reflective_marker_rate"],
        "evaluative_opener_rate": round(sum(bool(re.match(r"\s*(that|it) (sounds|is|was)|\s*(great|nice|wow)", turn["content"].casefold())) for turn in target) / len(target), 6),
        "median_merged_bubbles": stats["median_merged_bubbles"],
    }


def _validate_self_stats(value: dict[str, Any], global_stats: dict[str, Any], conditional: dict[str, Any]) -> dict[str, Any]:
    if value["observable_statistics"] != global_stats:
        raise ValueError("Self Domain changed deterministic global statistics")
    if value["conditional_statistics"] != conditional:
        raise ValueError("Self Domain changed deterministic conditional statistics")
    return value


def _validate_decision(value: dict[str, Any], history: list[dict[str, Any]], latest_partner: str) -> dict[str, Any]:
    situation = value["situation"]
    alignment = value["alignment"]
    policy = value["behavior_policy"]
    has_history = bool(history)
    open_obligation = situation["open_obligation"]
    source_turn_id = situation["obligation_source_turn_id"]
    visible_turn_ids = {turn["turn_id"] for turn in history}
    visible_turns = {turn["turn_id"]: turn for turn in history}
    if open_obligation in {"none", "open-session"}:
        if source_turn_id:
            raise ValueError(f"{open_obligation} must not cite an obligation source turn")
    elif source_turn_id not in visible_turn_ids:
        raise ValueError("obligation_source_turn_id must cite an exact visible history turn")
    if open_obligation.startswith("answer-") and source_turn_id in visible_turns:
        if "?" not in visible_turns[source_turn_id]["content"]:
            raise ValueError("an answer obligation source must contain a visible question mark")
    if open_obligation == "answer-current-question" and history:
        if source_turn_id != history[-1]["turn_id"]:
            raise ValueError("answer-current-question must cite the latest visible turn")
    if open_obligation == "answer-earlier-unanswered-question" and history:
        if source_turn_id == history[-1]["turn_id"]:
            raise ValueError("answer-earlier-unanswered-question must cite an earlier visible turn")
    if open_obligation.startswith("answer-") and policy["primary_move"] != "answer":
        raise ValueError("an unanswered question requires answer as the primary move")
    if not has_history:
        if policy["primary_move"] != "open" or situation["turn_obligation"] != "open":
            raise ValueError("empty history requires open obligation and primary move")
        if open_obligation != "open-session":
            raise ValueError("empty history requires open-session")
        if alignment["orientation"] != "self-led":
            raise ValueError("empty history requires self-led orientation")
    if latest_partner and "?" in latest_partner:
        if alignment["orientation"] == "self-led":
            raise ValueError("a direct partner question cannot use self-led alignment")
        if open_obligation != "answer-current-question":
            raise ValueError("a current direct question requires answer-current-question")
    return value


def _actor_contract(policy: dict[str, Any]) -> str:
    question = policy["question_plan"]
    question_rule = (
        "Do not ask any question and do not use a question mark."
        if question == "none"
        else f"Ask exactly one {question} question about: {policy['question_target']}."
    )
    reflection_rule = (
        "Do not explain motives or analyze inner meaning."
        if policy["reflection_depth"] == "surface"
        else "Include at most one brief reason, motivation, or self-observation."
    )
    return (
        f"Primary move: {policy['primary_move']}. Companion move: {policy['companion_move']}. "
        f"{question_rule} {reflection_rule} Message shape: {policy['message_shape']}."
    )


def _actor_call(*, checkpoint: OperationCheckpoint, backend: Any, operation_key: str,
                speaker: str, system_prompt: str, user_prompt: str, policy: dict[str, Any],
                max_attempts: int, raw_audit: Path, timeout: int) -> dict[str, Any]:
    logical_attempt = {"value": 0}
    def operation():
        logical_attempt["value"] += 1
        return _call_with_hard_timeout(
            lambda: backend.chat(system_prompt, user_prompt, temperature=0.6, top_p=0.9,
                                 max_tokens=300, enable_thinking=False),
            timeout, operation_key,
        )
    def validate(result):
        _append_jsonl(raw_audit, {
            "operation_key": operation_key,
            "logical_attempt": logical_attempt["value"],
            "model": result.model,
            "raw_response": result.content,
            "reasoning_content": result.reasoning_content,
            "recorded_at_utc": _now(),
        })
        message = _normalize_generated_message(result.content, speaker)
        expected_questions = 0 if policy["question_plan"] == "none" else 1
        if message.count("?") != expected_questions:
            raise ValueError(
                f"actor question count {message.count('?')} != planned {expected_questions}"
            )
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
        operation_key, operation, validate, max_attempts,
        usage_supplier=lambda: dict(getattr(backend, "token_usage", {})),
    )


def _user_domain_for_point(checkpoint: OperationCheckpoint, backend: Any, raw_audit: Path,
                           config: V13Config, speaker_data: dict[str, Any], point: dict[str, Any]):
    domain = empty_user_domain()
    allowed: set[str] = set()
    completed = []
    for session in point["test_sessions"]:
        if session == point["target_session"]:
            break
        turns = speaker_data["test_turns_by_session"][session]
        partner_ids = {
            turn["turn_id"] for turn in turns
            if turn["speaker"].casefold() == speaker_data["partner"].casefold()
        }
        allowed |= partner_ids
        envelope = _structured_call(
            checkpoint=checkpoint,
            backend=backend,
            operation_key=f"v13:user:{_speaker_id(speaker_data['speaker'])}:after:{session}",
            system_prompt=USER_DOMAIN_SYSTEM_PROMPT,
            user_prompt=USER_DOMAIN_USER_TEMPLATE.format(
                speaker=speaker_data["speaker"],
                partner=speaker_data["partner"],
                previous_domain=_json(domain),
                completed_session=_turns_with_ids(turns),
            ),
            schema=USER_DOMAIN_SCHEMA,
            normalizer=lambda value, ids=set(allowed): _validate_user_domain_evidence(
                normalize_user_domain(value), ids
            ),
            max_tokens=USER_MAX_TOKENS,
            max_attempts=config.operation_max_attempts,
            raw_audit=raw_audit,
            enable_thinking=False,
            hard_timeout_seconds=config.model_call_timeout_seconds,
        )
        domain = envelope["data"]
        completed.append(session)
    return domain, completed


def _actor_self_view(domain: dict[str, Any]) -> dict[str, Any]:
    return {
        "communication_signature": domain["communication_signature"],
        "interaction_policy_prior": domain["interaction_policy_prior"],
        "affective_social_signature": domain["affective_social_signature"],
        "portable_patterns": domain["cross_partner_transfer"]["portable_patterns"],
        "conditional_behavior": domain["conditional_behavior"],
        "observable_statistics": domain["observable_statistics"],
        "conditional_statistics": domain["conditional_statistics"],
    }


def _select_gate6(items: list[dict[str, Any]]) -> list[str]:
    categories = ("reflect", "reflect", "ground", "ground", "intimacy", "intimacy")
    ordered = sorted(items, key=lambda item: _id_hash(item["result_id"]))
    match: list[str] | None = None
    def search(index: int, chosen: list[dict[str, Any]]) -> bool:
        nonlocal match
        if index == len(categories):
            if len({item["speaker"] for item in chosen}) < 4:
                return False
            if {item["session"] for item in chosen} != {"session_1", "session_2", "session_3"}:
                return False
            match = _ordered_ids(chosen)
            return True
        category = categories[index]
        for item in ordered:
            if item in chosen or not item["classes"][category]:
                continue
            if search(index + 1, chosen + [item]):
                return True
        return False
    search(0, [])
    if match is None:
        raise ValueError("could not construct constrained gate 6")
    return match


def _select_gate18(items: list[dict[str, Any]], gate6: list[str]) -> list[str]:
    selected = [item for item in items if item["result_id"] in set(gate6)]
    unused = [item for item in sorted(items, key=lambda x: _id_hash(x["result_id"])) if item not in selected]
    for category in ("reflect", "ground", "intimacy", "clean"):
        added = 0
        for item in list(unused):
            if item["classes"][category]:
                selected.append(item); unused.remove(item); added += 1
                if added == 3:
                    break
        if added < 3:
            raise ValueError(f"gate 18 lacks three unused {category} rows")
    if len({item["speaker"] for item in selected}) < 6:
        raise ValueError("gate 18 must cover at least six speakers")
    return _ordered_ids(selected)


def _ordered_ids(items: list[dict[str, Any]]) -> list[str]:
    speaker_order = {item["speaker"]: index for index, item in enumerate(REALTALK_PERSONA_SPLITS)}
    return [item["result_id"] for item in sorted(
        items,
        key=lambda item: (
            speaker_order[item["speaker"]],
            int(item["session"].split("_")[1]),
            int(item["message_level_index"]),
        ),
    )]


def _point_index(prepared: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        _result_id(item["speaker"], point): point
        for item in prepared for point in item["points"]
    }


def _result_id(speaker: str, point: dict[str, Any]) -> str:
    return f"{_speaker_id(speaker)}:{point['sample_id']}"


def _speaker_id(speaker: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", speaker.casefold()).strip("_")


def _id_hash(result_id: str) -> str:
    return hashlib.sha256(result_id.encode("utf-8")).hexdigest()


def _cached_self_domains(checkpoint: OperationCheckpoint) -> dict[str, Any]:
    result = {}
    for key, item in checkpoint.data["operations"].items():
        if key.startswith("v13:self:") and item.get("status") == "complete":
            result[key.split(":", 2)[2]] = item["value"]["data"]
    return result


def _lambda_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: 0 for key in ("self-led", "balanced", "partner-adaptive")}
    traces = []
    for row in rows:
        counts[row["alignment"]["orientation"]] += 1
        traces.append(row["alignment"]["lambda_trace"])
    return {"counts": counts, "mean": round(statistics.mean(traces), 6) if traces else None}


def _prompt_hashes() -> dict[str, str]:
    return {
        name: stable_hash(value) for name, value in {
            "self_system": SELF_SYSTEM,
            "self_user": SELF_USER,
            "decision_system": DECISION_SYSTEM,
            "decision_user": DECISION_USER,
            "actor_system": ACTOR_SYSTEM,
            "actor_user": ACTOR_USER,
        }.items()
    }


def _implementation_source_hashes() -> dict[str, str]:
    return {
        "runner": _sha256(Path(__file__)),
        "schemas": _sha256(Path(__file__).with_name("realtalk_v13_schemas.py")),
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> V13Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--v9-predictions", required=True)
    parser.add_argument("--v9-judge-scored", required=True)
    parser.add_argument("--v9-local-scored", required=True)
    parser.add_argument("--gate", type=int, choices=GATES, default=6)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--operation-max-attempts", type=int, default=3)
    parser.add_argument("--model-call-timeout-seconds", type=int, default=240)
    return V13Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
