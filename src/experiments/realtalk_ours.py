"""Protocol-aligned REALTALK Task 1 evaluation for the Deep Empathy method."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .exp1_protocol import (
    REALTALK_PERSONA_SPLITS,
    build_message_level_points,
    build_profile_corpus,
    format_turns,
    message_speakers,
    select_realtalk_splits,
    stable_hash,
)
from .exp2_generation import (
    bertscore_runtime_metadata,
    compute_bertscore_f1,
)
from .operation_checkpoint import OperationCheckpoint
from .personaemp.client import (
    ChatBackend,
    ChatResult,
    OpenAICompatibleChatBackend,
)
from .realtalk_evaluator import RealTalkLabelEvaluator
from .realtalk_ours_schemas import (
    ALIGNMENT_SCHEMA,
    SELF_DOMAIN_SCHEMA,
    USER_DOMAIN_SCHEMA,
    Normalizer,
    empty_user_domain,
    normalize_alignment,
    normalize_self_domain,
    normalize_user_domain,
)
from ..metrics import compute_rouge_l


EXPECTED_MODEL = "qwen3-8b"
OFFICIAL_REALTALK_COMMIT = "b903e06a9770bf4e5fe9018c3e132889666d3b4a"
EXPECTED_FULL_TARGETS = 1076
EXPECTED_SPEAKER_TARGETS = {
    "Emi": 52,
    "Nicolas": 247,
    "Kevin": 55,
    "Akib": 157,
    "Muhhamed": 75,
    "Nebraas": 137,
    "Paola": 31,
    "Vanessa": 182,
    "elise": 55,
    "Fahim Khan": 85,
}
PAPER_TABLE2_ROWS = {
    "w/o fine-tune": {
        "lexical": "0.14 +/- 0.04",
        "semantic": "0.76 +/- 0.08",
        "reflective": "0.62 +/- 0.13",
        "grounding": "0.40 +/- 0.13",
        "sentiment": "0.53 +/- 0.22",
        "emotion": "0.43 +/- 0.22",
        "intimacy": "0.06 +/- 0.01",
        "empathy": "1.80 +/- 0.55",
    },
    "w/ fine-tune": {
        "lexical": "0.14 +/- 0.05",
        "semantic": "0.78 +/- 0.04",
        "reflective": "0.77 +/- 0.09",
        "grounding": "0.62 +/- 0.08",
        "sentiment": "0.59 +/- 0.18",
        "emotion": "0.46 +/- 0.21",
        "intimacy": "0.07 +/- 0.01",
        "empathy": "1.24 +/- 0.12",
    },
}

SELF_DOMAIN_SYSTEM_PROMPT = """You build an evidence-grounded Self Domain for a persona-simulation agent.
The target speaker is the person the model must imitate later. Infer only durable identity, style, and
behavioral tendencies supported by that target speaker's own utterances. The other speaker's utterances
provide conversational context but are never evidence about the target. Do not turn one-off moods,
hypotheticals, quoted material, jokes, or the partner's facts into stable target attributes. Keep uncertainty
explicit and return exactly the requested JSON structure."""

SELF_DOMAIN_USER_TEMPLATE = """TARGET SPEAKER: {speaker}
SOURCE: the first {session_count} sessions of the target's paper-assigned Ca conversation.

TARGET-SPEAKER UTTERANCES (the only factual evidence about {speaker}):
{target_evidence}

FULL MERGED CONVERSATION (context only; partner facts are not {speaker}'s facts):
{full_context}

Build the fixed Self Domain. It must help reproduce {speaker}'s identity, conversational style, initiative,
emotional response style, and boundaries without inventing facts. Empty arrays are valid when evidence is
insufficient. Interests and identity require explicit first-person support or a repeated target behavior.
Uncertainties must concern {speaker}, never the Ca partner."""

USER_DOMAIN_SYSTEM_PROMPT = """You update a five-layer User Domain for a conversation partner.
Use only the partner's already-observed utterances as evidence about that partner. Target-speaker messages
may clarify context but must never become partner facts. Preserve supported prior facts, revise conflicts,
remove unsupported claims, consolidate duplicates, and keep evidence turn IDs. This is causal state: no
future message or ground truth is available. Empty layers are expected when evidence is thin. A greeting,
polite question, or other single speech act can support an immediate behavior observation at most; it does
not establish a core value, identity, stable cognition, or regulation pattern. Do not restate the same
observation across multiple layers merely to fill the schema. Return exactly the requested JSON structure."""

USER_DOMAIN_USER_TEMPLATE = """TARGET SPEAKER (the agent): {speaker}
PARTNER (the modeled user): {partner}

PREVIOUS USER DOMAIN:
{previous_domain}

NEWLY OBSERVED PARTNER TURNS:
{new_turns}

VISIBLE REAL CONVERSATION HISTORY:
{history}

Update the partner's five layers: core concerns/values, regulation patterns, cognition/communication,
identity/life facts, and behavior/preferences. Every fact requires at least one supplied partner turn ID."""

ALIGNMENT_SYSTEM_PROMPT = """You are the adaptive alignment stage of a persona-simulation agent.
First infer the partner's current and short-term likely state from causal evidence. Then choose an explicit
adaptive lambda_t in [0,1]: low values preserve the target speaker's habitual behavior; high values adapt
more strongly to the partner's present needs. The target speaker's identity remains a hard constraint at
every value. Finally produce exactly one Behavior Policy for the next message.

This benchmark asks for the target human's likely message, not an ideal assistant response. Do not force
empathy, advice, questions, positivity, or personalization. Match the target's demonstrated style and use
partner adaptation only where evidence supports it. Orientation must match lambda_t: [0,.25)
self-dominant, [.25,.5) self-leaning, [.5,.75) user-leaning, [.75,1] strongly-user-oriented. Return exactly
the requested JSON structure.

Self Domain identity facts and interests are background knowledge, not topics that must appear. In the
Behavior Policy, self_domain_expression should normally select tone, phrasing, initiative, and response
length. It must not direct the generator to mention an interest, activity, location, possession, plan, or
anecdote unless the visible Cb history has already made that exact content relevant to the current exchange.

If no partner turn is visible, there is no User State evidence: use empty strings for all current/future
semantic fields, high uncertainty, lambda_t=0, self-dominant orientation, no personalization, and a simple
target-style conversation opener. Never infer a partner need merely because a conversation is starting."""

ALIGNMENT_USER_TEMPLATE = """TARGET SPEAKER: {speaker}
PARTNER: {partner}

FIXED SELF DOMAIN:
{self_domain}

CURRENT USER DOMAIN:
{user_domain}

PREVIOUS USER STATE:
{previous_state}

REAL CAUSAL HISTORY BEFORE THE TARGET MESSAGE:
{history}

LATEST PARTNER QUERY/TURN (may be empty):
{current_query}

Infer state, choose lambda_t, and produce one concrete policy for {speaker}'s next message."""

GENERATION_SYSTEM_TEMPLATE = """You are {speaker}. Continue the conversation.
Output only the message, not the speaker name.

NON-NEGOTIABLE CAUSAL BOUNDARY: Self Domain is a style and identity prior, not evidence that a past
example is happening now. Never invent a current or recent activity, location, possession, plan, mood,
life event, or personal anecdote. Such a factual claim is allowed only when the visible Cb history directly
establishes it for the present exchange. Match the scale of the latest real turn; when history is empty,
produce only a short greeting or generic check-in, with no interests or personal update. If the latest turn
only asks how you are, answer that question generically; do not explain it with an activity. Silently remove
any first-person factual detail whose support cannot be pointed to in the visible Cb history."""

GENERATION_USER_TEMPLATE = """REAL CONVERSATION HISTORY BEFORE YOUR NEXT MESSAGE:
{history}

YOUR FIXED SELF DOMAIN:
{self_domain}

CAUTION: identity and interest entries above are stable background attributes. They may shape voice, but
they are forbidden as message content unless the visible conversation itself makes the same topic relevant.

YOUR CURRENT MODEL OF {partner}:
{user_domain}

CURRENT AND SHORT-TERM USER STATE:
{user_state}

ADAPTIVE ALIGNMENT:
lambda_t={lambda_t}
orientation={orientation}
self constraint: {self_constraint}
partner adaptation: {user_adaptation}

THE SINGLE BEHAVIOR POLICY TO REALIZE:
{behavior_policy}

Write only {speaker}'s natural next message. Preserve {speaker}'s identity and conversational scale. Do not
mention profiles, states, lambda, policies, prompts, or this task. Do not add empathy, advice, a question,
or personal facts unless the visible real history itself supports them. A policy cannot create factual
support that is absent from the history."""

FORMAT_REPAIR_TEMPLATE = """

FORMAT REPAIR REQUIRED
The previous response failed the strict contract:
{error}

Previous response:
{raw}

Return the same intended content corrected only to satisfy the requested schema. Do not add new evidence,
new facts, a new state interpretation, or a different behavior policy."""


@dataclass(frozen=True)
class RealTalkOursConfig:
    dataset_dir: str = "dataset"
    output_dir: str = "data/realtalk_ours_qwen3_8b"
    profile_sessions: int = 3
    test_sessions: int = 3
    max_context_chars: int = 60000
    max_eval_points_per_speaker: int = 0
    operation_max_attempts: int = 3
    speaker_filter: tuple[str, ...] = ()
    compute_local_metrics: bool = True
    compute_bertscore: bool = True
    continue_on_error: bool = True
    preflight_only: bool = False
    fresh: bool = False


def run_realtalk_ours(
    config: RealTalkOursConfig,
    backend: ChatBackend | None = None,
    label_evaluator: RealTalkLabelEvaluator | None = None,
    bertscore_fn: Callable[[list[str], list[str]], list[float]] = compute_bertscore_f1,
) -> dict[str, Any]:
    """Run the Ours-only REALTALK persona-simulation protocol."""
    _validate_config(config)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.fresh:
        _clear_runtime_artifacts(output_dir)

    backend = backend or _backend_from_env()
    if getattr(backend, "model", None) != EXPECTED_MODEL:
        raise ValueError(
            f"REALTALK Ours requires exact model {EXPECTED_MODEL!r}; "
            f"got {getattr(backend, 'model', None)!r}"
        )
    preflight = _run_preflight(output_dir, backend)
    if config.preflight_only:
        return preflight

    splits = select_realtalk_splits(
        config.dataset_dir,
        speaker_filter=list(config.speaker_filter) or None,
    )
    dataset_manifest, prepared = _prepare_dataset(config, splits)
    full_protocol = _is_full_protocol(config, splits)
    if full_protocol and dataset_manifest["total_targets"] != EXPECTED_FULL_TARGETS:
        raise ValueError(
            f"full REALTALK reconstruction requires {EXPECTED_FULL_TARGETS} targets; "
            f"found {dataset_manifest['total_targets']}"
        )

    signature = _run_signature(config, backend, dataset_manifest)
    checkpoint = OperationCheckpoint(output_dir / "checkpoint.json", signature)
    raw_audit = output_dir / "raw_responses.jsonl"
    self_domains: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []

    for speaker_data in prepared:
        speaker = speaker_data["speaker"]
        speaker_id = _speaker_id(speaker)
        try:
            self_envelope = _structured_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key=f"self_domain:{speaker_id}",
                system_prompt=SELF_DOMAIN_SYSTEM_PROMPT,
                user_prompt=SELF_DOMAIN_USER_TEMPLATE.format(
                    speaker=speaker,
                    session_count=config.profile_sessions,
                    target_evidence=_turns_with_ids([
                        turn
                        for turn in speaker_data["profile"]["turns"]
                        if turn["speaker"].casefold() == speaker.casefold()
                    ]),
                    full_context=_turns_with_ids(speaker_data["profile"]["turns"]),
                ),
                schema=SELF_DOMAIN_SCHEMA,
                normalizer=normalize_self_domain,
                max_tokens=1800,
                max_attempts=config.operation_max_attempts,
                raw_audit=raw_audit,
            )
        except Exception as exc:
            failure = _failure("self_domain", speaker, None, exc)
            unresolved.append(failure)
            checkpoint.store_excluded_result(f"self:{speaker_id}", failure)
            if not config.continue_on_error:
                raise
            continue

        self_domain = self_envelope["data"]
        self_domains[speaker] = self_domain
        user_domain = empty_user_domain()
        previous_user_state: dict[str, Any] = {}
        observed_partner_turn_ids: set[str] = set()

        for point in speaker_data["points"]:
            result_id = f"{speaker_id}:{point['sample_id']}"
            visible_partner_turns = [
                turn
                for turn in point["context_turns"]
                if turn["speaker"].casefold() == speaker_data["partner"].casefold()
            ]
            new_partner_turns = [
                turn
                for turn in visible_partner_turns
                if turn["turn_id"] not in observed_partner_turn_ids
            ]
            durable_partner_turns = [
                turn for turn in new_partner_turns
                if _has_durable_user_domain_evidence(turn["content"])
            ]
            try:
                update_audit: dict[str, Any]
                if durable_partner_turns:
                    update_envelope = _structured_call(
                        checkpoint=checkpoint,
                        backend=backend,
                        operation_key=f"user_domain:{result_id}",
                        system_prompt=USER_DOMAIN_SYSTEM_PROMPT,
                        user_prompt=USER_DOMAIN_USER_TEMPLATE.format(
                            speaker=speaker,
                            partner=speaker_data["partner"],
                            previous_domain=_json(user_domain),
                            new_turns=_turns_with_ids(durable_partner_turns),
                            history=_turns_with_ids(point["context_turns"]),
                        ),
                        schema=USER_DOMAIN_SCHEMA,
                        normalizer=lambda value: _validate_user_domain_evidence(
                            normalize_user_domain(value),
                            observed_partner_turn_ids
                            | {turn["turn_id"] for turn in new_partner_turns},
                        ),
                        max_tokens=1800,
                        max_attempts=config.operation_max_attempts,
                        raw_audit=raw_audit,
                    )
                    user_domain = update_envelope["data"]
                    update_audit = {
                        "mode": "updated",
                        "new_partner_turn_ids": [
                            turn["turn_id"] for turn in durable_partner_turns
                        ],
                        "operation": update_envelope["audit"],
                    }
                else:
                    update_audit = {
                        "mode": (
                            "reused_no_durable_partner_evidence"
                            if new_partner_turns
                            else "reused_no_new_partner_evidence"
                        ),
                        "new_partner_turn_ids": [
                            turn["turn_id"] for turn in new_partner_turns
                        ],
                        "operation": None,
                    }
                observed_partner_turn_ids.update(
                    turn["turn_id"] for turn in new_partner_turns
                )

                current_query = (
                    visible_partner_turns[-1]["content"]
                    if visible_partner_turns
                    else ""
                )
                alignment_envelope = _structured_call(
                    checkpoint=checkpoint,
                    backend=backend,
                    operation_key=f"alignment:{result_id}",
                    system_prompt=ALIGNMENT_SYSTEM_PROMPT,
                    user_prompt=ALIGNMENT_USER_TEMPLATE.format(
                        speaker=speaker,
                        partner=speaker_data["partner"],
                        self_domain=_json(self_domain),
                        user_domain=_json(user_domain),
                        previous_state=_json(previous_user_state),
                        history=_turns_with_ids(point["context_turns"]),
                        current_query=current_query,
                    ),
                    schema=ALIGNMENT_SCHEMA,
                    normalizer=lambda value: _normalize_alignment_for_context(
                        normalize_alignment(value),
                        set(observed_partner_turn_ids),
                    ),
                    max_tokens=1600,
                    max_attempts=config.operation_max_attempts,
                    raw_audit=raw_audit,
                )
                alignment_result = alignment_envelope["data"]
                alignment_envelope["audit"]["deterministic_no_evidence_adapter"] = (
                    not observed_partner_turn_ids
                )
                previous_user_state = alignment_result["user_state"]
                generation_envelope = _text_call(
                    checkpoint=checkpoint,
                    backend=backend,
                    operation_key=f"generation:{result_id}",
                    system_prompt=GENERATION_SYSTEM_TEMPLATE.format(speaker=speaker),
                    user_prompt=GENERATION_USER_TEMPLATE.format(
                        speaker=speaker,
                        partner=speaker_data["partner"],
                        history=format_turns(point["context_turns"]),
                        self_domain=_json(_generation_self_domain(self_domain)),
                        user_domain=_json(_compact_user_domain(user_domain)),
                        user_state=_json(alignment_result["user_state"]),
                        lambda_t=alignment_result["alignment"]["lambda_t"],
                        orientation=alignment_result["alignment"]["orientation"],
                        self_constraint=alignment_result["alignment"]["self_constraint"],
                        user_adaptation=alignment_result["alignment"]["user_adaptation"],
                        behavior_policy=_json(alignment_result["behavior_policy"]),
                    ),
                    speaker=speaker,
                    max_attempts=config.operation_max_attempts,
                    raw_audit=raw_audit,
                )
                result = {
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
                    "context_turn_ids": [
                        turn["turn_id"] for turn in point["context_turns"]
                    ],
                    "context_hash": point["history_hash"],
                    "context_truncated": point["context_truncated"],
                    "ground_truth": point["target_message"],
                    "generated_message": generation_envelope["data"],
                    "self_domain_hash": stable_hash(self_domain),
                    "user_domain": user_domain,
                    "user_domain_update": update_audit,
                    "user_state": alignment_result["user_state"],
                    "alignment": alignment_result["alignment"],
                    "behavior_policy": alignment_result["behavior_policy"],
                    "operation_audit": {
                        "alignment": alignment_envelope["audit"],
                        "generation": generation_envelope["audit"],
                    },
                }
                checkpoint.data["failures"].pop(f"sample:{result_id}", None)
                checkpoint.store_result(result_id, result)
            except Exception as exc:
                failure = _failure("sample", speaker, result_id, exc)
                unresolved.append(failure)
                checkpoint.store_excluded_result(result_id, failure)
                if not config.continue_on_error:
                    raise

    results = sorted(
        checkpoint.result_values(),
        key=lambda item: (
            _split_order(item["speaker"]),
            int(item["message_level_index"]),
        ),
    )
    unresolved = _checkpoint_unresolved(checkpoint)
    _write_generation_outputs(
        output_dir,
        results,
        self_domains,
        unresolved,
        dataset_manifest,
        config,
        backend,
        preflight,
        signature,
        checkpoint,
    )
    expected = dataset_manifest["total_targets"]
    generation_complete = len(results) == expected and not unresolved
    if generation_complete:
        _write_json(output_dir / "GENERATION_COMPLETE", {
            "completed_at_utc": _now(),
            "records": len(results),
            "model": EXPECTED_MODEL,
            "run_signature": signature,
        })
    else:
        _remove(output_dir / "GENERATION_COMPLETE")

    local_summary: dict[str, Any] | None = None
    if config.compute_local_metrics and generation_complete:
        evaluator = label_evaluator or RealTalkLabelEvaluator()
        local_summary = _run_local_metrics(
            output_dir,
            results,
            checkpoint,
            evaluator,
            config,
            bertscore_fn,
        )
        _write_json(output_dir / "LOCAL_METRICS_COMPLETE", {
            "completed_at_utc": _now(),
            "records": len(results),
            "metrics": [
                "rouge_l",
                "bertscore_f1" if config.compute_bertscore else None,
                "sentiment_accuracy",
                "emotion_accuracy",
                "intimacy_absolute_difference",
            ],
            "run_signature": signature,
        })
    else:
        _remove(output_dir / "LOCAL_METRICS_COMPLETE")

    _remove(output_dir / "PIPELINE_COMPLETE")
    _write_json(output_dir / "GPT_EVALUATION_PENDING.json", {
        "required_model": "gpt-4o-mini",
        "pending_metrics": [
            "reflectiveness_accuracy",
            "grounding_accuracy",
            "empathy_absolute_difference",
        ],
        "pipeline_complete_allowed": False,
        "reason": "No verified gpt-4o-mini evaluation endpoint was configured for this run.",
    })
    _write_partial_report(output_dir, local_summary, dataset_manifest)
    return {
        "generation_complete": generation_complete,
        "records": len(results),
        "expected_records": expected,
        "unresolved": unresolved,
        "local_metrics": local_summary,
        "output_dir": str(output_dir),
    }


def _prepare_dataset(
    config: RealTalkOursConfig,
    splits: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = Path(config.dataset_dir)
    prepared = []
    files: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in splits:
        train_path = dataset / split["train_chat"]
        test_path = dataset / split["test_chat"]
        train_chat = _load_json(train_path)
        test_chat = _load_json(test_path)
        speaker = next(
            value
            for value in message_speakers(test_chat)
            if value.casefold() == split["speaker"].casefold()
        )
        partner = next(
            value
            for value in message_speakers(test_chat)
            if value.casefold() != speaker.casefold()
        )
        profile = build_profile_corpus(
            train_chat,
            speaker,
            profile_sessions=config.profile_sessions,
            merge_adjacent_bubbles=False,
        )
        points = build_message_level_points(
            test_chat,
            speaker,
            test_sessions=config.test_sessions,
            max_context_chars=config.max_context_chars,
            max_eval_points=config.max_eval_points_per_speaker,
            merge_adjacent_bubbles=False,
        )
        if not points:
            raise ValueError(f"no test targets for {speaker}")
        counts[split["speaker"]] = len(points)
        for path in (train_path, test_path):
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        prepared.append({
            "split": dict(split),
            "speaker": speaker,
            "partner": partner,
            "profile": profile,
            "points": points,
        })
    manifest = {
        "dataset": "REALTALK public preprocessed conversations",
        "official_repository_commit": OFFICIAL_REALTALK_COMMIT,
        "table8_speaker_specific_splits": [dict(item) for item in splits],
        "profile_sessions": config.profile_sessions,
        "test_sessions": config.test_sessions,
        "sample_unit": "original message bubble M_t",
        "merge_consecutive_same_speaker_within_session": False,
        "generated_outputs_are_never_rolled_into_history": True,
        "source_files_sha256": dict(sorted(files.items())),
        "source_files_aggregate_sha256": stable_hash(dict(sorted(files.items()))),
        "targets_by_speaker": counts,
        "total_targets": sum(counts.values()),
        "paper_reported_exact_target_count": None,
        "reconstructed_target_count": True,
    }
    return manifest, prepared


def _run_preflight(output_dir: Path, backend: ChatBackend) -> dict[str, Any]:
    path = output_dir / "preflight.json"
    if path.exists():
        value = _load_json(path)
        if (
            value.get("model") == EXPECTED_MODEL
            and value.get("enable_thinking") is False
            and value.get("minimal_generation_succeeded") is True
        ):
            return value
    if bool(getattr(backend, "enable_thinking", False)):
        raise ValueError("qwen3-8b thinking must be disabled")
    available_fn = getattr(backend, "available_models", None)
    available = list(available_fn()) if callable(available_fn) else [backend.model]
    if EXPECTED_MODEL not in available:
        raise ValueError(
            f"configured credential cannot access {EXPECTED_MODEL}; visible={available[:20]}"
        )
    response = backend.chat(
        "Return exactly READY.",
        "READY",
        temperature=0.0,
        top_p=0.9,
        max_tokens=8,
    )
    if "READY" not in response.content.upper():
        raise ValueError(f"minimal model preflight failed: {response.content!r}")
    value = {
        "checked_at_utc": _now(),
        "model": backend.model,
        "base_url_host": _safe_host(getattr(backend, "base_url", "injected-test-backend")),
        "enable_thinking": bool(getattr(backend, "enable_thinking", False)),
        "model_visible": True,
        "minimal_generation_succeeded": True,
        "minimal_generation_attempts": response.attempts,
    }
    _write_json(path, value)
    return value


def _structured_call(
    *,
    checkpoint: OperationCheckpoint,
    backend: ChatBackend,
    operation_key: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    normalizer: Normalizer,
    max_tokens: int,
    max_attempts: int,
    raw_audit: Path,
) -> dict[str, Any]:
    repair = {"raw": "", "error": ""}
    logical_attempt = {"value": 0}

    def operation() -> ChatResult:
        logical_attempt["value"] += 1
        prompt = user_prompt
        if repair["error"]:
            prompt += FORMAT_REPAIR_TEMPLATE.format(
                error=repair["error"], raw=repair["raw"][:12000]
            )
        return backend.chat(
            system_prompt,
            prompt,
            temperature=0.2,
            top_p=0.9,
            max_tokens=max_tokens,
            response_schema=schema,
        )

    def validate(result: ChatResult) -> dict[str, Any]:
        _append_jsonl(raw_audit, {
            "operation_key": operation_key,
            "logical_attempt": logical_attempt["value"],
            "model": result.model,
            "raw_response": result.content,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_seconds": result.latency_seconds,
            "network_attempts": result.attempts,
            "recorded_at_utc": _now(),
        })
        try:
            parsed = json.loads(result.content)
            data = normalizer(parsed)
        except Exception as exc:
            repair["raw"] = result.content
            repair["error"] = f"{type(exc).__name__}: {exc}"
            raise
        return {
            "data": data,
            "audit": {
                "model": result.model,
                "logical_attempts": logical_attempt["value"],
                "network_attempts_last_call": result.attempts,
                "prompt_tokens_last_call": result.prompt_tokens,
                "completion_tokens_last_call": result.completion_tokens,
                "latency_seconds_last_call": result.latency_seconds,
                "schema": schema["name"],
            },
        }

    return checkpoint.execute(
        operation_key,
        operation,
        validate,
        max_attempts,
        usage_supplier=lambda: dict(getattr(backend, "token_usage", {})),
    )


def _text_call(
    *,
    checkpoint: OperationCheckpoint,
    backend: ChatBackend,
    operation_key: str,
    system_prompt: str,
    user_prompt: str,
    speaker: str,
    max_attempts: int,
    raw_audit: Path,
) -> dict[str, Any]:
    logical_attempt = {"value": 0}

    def operation() -> ChatResult:
        logical_attempt["value"] += 1
        return backend.chat(
            system_prompt,
            user_prompt,
            temperature=0.6,
            top_p=0.9,
            max_tokens=300,
        )

    def validate(result: ChatResult) -> dict[str, Any]:
        _append_jsonl(raw_audit, {
            "operation_key": operation_key,
            "logical_attempt": logical_attempt["value"],
            "model": result.model,
            "raw_response": result.content,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_seconds": result.latency_seconds,
            "network_attempts": result.attempts,
            "recorded_at_utc": _now(),
        })
        return {
            "data": _normalize_generated_message(result.content, speaker),
            "audit": {
                "model": result.model,
                "logical_attempts": logical_attempt["value"],
                "network_attempts_last_call": result.attempts,
                "prompt_tokens_last_call": result.prompt_tokens,
                "completion_tokens_last_call": result.completion_tokens,
                "latency_seconds_last_call": result.latency_seconds,
            },
        }

    return checkpoint.execute(
        operation_key,
        operation,
        validate,
        max_attempts,
        usage_supplier=lambda: dict(getattr(backend, "token_usage", {})),
    )


def _run_local_metrics(
    output_dir: Path,
    results: list[dict[str, Any]],
    checkpoint: OperationCheckpoint,
    evaluator: RealTalkLabelEvaluator,
    config: RealTalkOursConfig,
    bertscore_fn: Callable[[list[str], list[str]], list[float]],
) -> dict[str, Any]:
    scored = []
    for result in results:
        result_id = result["result_id"]
        reference_labels = checkpoint.execute(
            f"local_labels:reference:{result_id}",
            lambda text=result["ground_truth"]: evaluator.annotate(text),
            _normalize_local_labels,
            config.operation_max_attempts,
        )
        candidate_labels = checkpoint.execute(
            f"local_labels:candidate:{result_id}",
            lambda text=result["generated_message"]: evaluator.annotate(text),
            _normalize_local_labels,
            config.operation_max_attempts,
        )
        scored.append({
            **result,
            "local_labels": {
                "reference": reference_labels,
                "candidate": candidate_labels,
            },
            "local_metrics": {
                "rouge_l": compute_rouge_l(
                    result["ground_truth"], result["generated_message"]
                ),
                "sentiment_accuracy": float(
                    reference_labels["sentiment"] == candidate_labels["sentiment"]
                ),
                "emotion_accuracy": float(
                    reference_labels["emotion"] == candidate_labels["emotion"]
                ),
                "intimacy_absolute_difference": round(
                    abs(reference_labels["intimacy"] - candidate_labels["intimacy"]),
                    6,
                ),
            },
        })
    if config.compute_bertscore:
        values = checkpoint.execute(
            "local_metrics:bertscore_batch",
            lambda: bertscore_fn(
                [item["ground_truth"] for item in scored],
                [item["generated_message"] for item in scored],
            ),
            lambda value: _normalize_bertscore(value, len(scored)),
            config.operation_max_attempts,
        )
        for item, value in zip(scored, values):
            item["local_metrics"]["bertscore_f1"] = value
    summary = _aggregate_local_metrics(scored)
    summary["classifier_metadata"] = evaluator.metadata()
    summary["bertscore_metadata"] = (
        bertscore_runtime_metadata() if config.compute_bertscore else None
    )
    _write_jsonl(output_dir / "results_with_local_metrics.jsonl", scored)
    _write_json(output_dir / "local_metrics_summary.json", summary)
    return summary


def _aggregate_local_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = list(results[0]["local_metrics"]) if results else []
    by_speaker: dict[str, dict[str, float]] = {}
    for speaker in dict.fromkeys(item["speaker"] for item in results):
        speaker_results = [item for item in results if item["speaker"] == speaker]
        by_speaker[speaker] = {
            metric: round(statistics.mean(
                item["local_metrics"][metric] for item in speaker_results
            ), 6)
            for metric in metric_names
        }
    macro = {}
    for metric in metric_names:
        values = [scores[metric] for scores in by_speaker.values()]
        macro[metric] = {
            "mean": round(statistics.mean(values), 6),
            "std_population": round(statistics.pstdev(values), 6),
            "display": (
                f"{statistics.mean(values):.2f} +/- {statistics.pstdev(values):.2f}"
            ),
        }
    micro = {
        metric: round(statistics.mean(
            item["local_metrics"][metric] for item in results
        ), 6)
        for metric in metric_names
    }
    return {
        "aggregation_for_table2": "speaker_macro_mean_and_population_std",
        "speaker_count": len(by_speaker),
        "message_count": len(results),
        "by_speaker": by_speaker,
        "speaker_macro": macro,
        "message_micro": micro,
    }


def _write_generation_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    self_domains: dict[str, dict[str, Any]],
    unresolved: list[dict[str, Any]],
    dataset_manifest: dict[str, Any],
    config: RealTalkOursConfig,
    backend: ChatBackend,
    preflight: dict[str, Any],
    signature: str,
    checkpoint: OperationCheckpoint,
) -> None:
    _write_jsonl(output_dir / "predictions.jsonl", results)
    _write_json(output_dir / "self_domains.json", self_domains)
    _write_json(output_dir / "unresolved_errors.json", unresolved)
    _write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    _write_json(output_dir / "run_manifest.json", {
        "created_at_utc": _now(),
        "protocol": "realtalk_task1_ours_explicit_modeling_v1",
        "comparison_status": "protocol_aligned_not_runtime_identical",
        "paper_persona_simulation_model_disclosed": False,
        "ours_model": EXPECTED_MODEL,
        "all_ours_stages_use_same_model": True,
        "enable_thinking": False,
        "training_or_finetuning": False,
        "omega_enabled": False,
        "verification_or_rewrite_enabled": False,
        "response_length_restriction": None,
        "generated_output_rollout": False,
        "decoding": {
            "self_domain": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1800},
            "user_domain": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1800},
            "alignment": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1600},
            "generation": {"temperature": 0.6, "top_p": 0.9, "max_tokens": 300},
            "seed": None,
        },
        "structured_logical_attempts": config.operation_max_attempts,
        "config": asdict(config),
        "dataset_manifest_sha256": stable_hash(dataset_manifest),
        "prompt_hashes": _prompt_hashes(),
        "schema_hashes": {
            "self_domain": stable_hash(SELF_DOMAIN_SCHEMA),
            "user_domain": stable_hash(USER_DOMAIN_SCHEMA),
            "alignment": stable_hash(ALIGNMENT_SCHEMA),
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preflight": preflight,
        "run_signature": signature,
        "token_usage": _checkpoint_usage(checkpoint=checkpoint, backend=backend),
        "gpt_evaluation_status": "pending_verified_gpt-4o-mini_endpoint",
        "pipeline_complete": False,
    })


def _write_partial_report(
    output_dir: Path,
    local_summary: dict[str, Any] | None,
    dataset_manifest: dict[str, Any],
) -> None:
    ours = {
        "lexical": _metric_display(local_summary, "rouge_l"),
        "semantic": _metric_display(local_summary, "bertscore_f1"),
        "reflective": "pending gpt-4o-mini",
        "grounding": "pending gpt-4o-mini",
        "sentiment": _metric_display(local_summary, "sentiment_accuracy"),
        "emotion": _metric_display(local_summary, "emotion_accuracy"),
        "intimacy": _metric_display(local_summary, "intimacy_absolute_difference"),
        "empathy": "pending gpt-4o-mini",
    }
    payload = {
        "status": "partial_local_metrics_only",
        "paper_rows": PAPER_TABLE2_ROWS,
        "ours": ours,
        "disclosure": (
            "Protocol-aligned comparison on reconstructed public REALTALK Task 1 points. "
            "The paper does not disclose its persona-simulation base model; Ours uses qwen3-8b."
        ),
        "reconstructed_targets": dataset_manifest["total_targets"],
    }
    _write_json(output_dir / "table2_partial.json", payload)
    lines = [
        "# REALTALK Table 2 + Ours (Partial)",
        "",
        "This is a protocol-aligned comparison on reconstructed public REALTALK Task 1 points. ",
        "The paper does not disclose its persona-simulation base model; Ours uses `qwen3-8b`.",
        "",
        "| Method | Lexical | Semantic | Reflective | Grounding | Sentiment | Emotion | Intimacy | Empathy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in [*PAPER_TABLE2_ROWS.items(), ("Ours", ours)]:
        lines.append(
            "| " + method + " | " + " | ".join(
                row[key]
                for key in (
                    "lexical", "semantic", "reflective", "grounding",
                    "sentiment", "emotion", "intimacy", "empathy",
                )
            ) + " |"
        )
    lines.extend([
        "",
        f"Reconstructed targets: {dataset_manifest['total_targets']}.",
        "The three GPT-scored metrics remain pending and `PIPELINE_COMPLETE` is intentionally absent.",
    ])
    (output_dir / "REPORT_PARTIAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generation_self_domain(value: dict[str, Any]) -> dict[str, Any]:
    """Expose style and constraints while keeping background facts in planning."""
    return {
        "persona": value["persona"],
        "behavior_policy_prior": value["behavior_policy_prior"],
        "hard_constraints": value["hard_constraints"],
        "uncertainties": value["uncertainties"],
        "omitted_from_generation": (
            "Identity facts and interests remain available to Alignment but are not "
            "message-content evidence."
        ),
    }


def _has_durable_user_domain_evidence(content: str) -> bool:
    """Separate stable-profile evidence from greetings and generic check-ins."""
    normalized = re.sub(r"[^a-z0-9' ]+", " ", content.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    low_information = (
        r"(?:hi|hey|hello)(?: there)?",
        r"(?:hi|hey|hello)(?: there)? how are you",
        r"how are you",
        r"how(?:'s| is) it going",
        r"what(?:'s| is) up",
        r"good (?:morning|afternoon|evening|night)",
        r"nice to (?:meet|see) you",
    )
    if any(re.fullmatch(pattern, normalized) for pattern in low_information):
        return False
    generic_checkin_words = {
        "a", "am", "and", "are", "day", "doing", "far", "fine", "going",
        "good", "great", "hello", "hey", "hi", "how", "i", "im", "is",
        "it", "m", "morning", "night", "so", "thanks", "thank", "the",
        "there", "today", "up", "well", "what's", "you", "your",
    }
    informative_words = [
        word for word in normalized.split()
        if word not in generic_checkin_words
    ]
    return len(informative_words) >= 2


def _compact_user_domain(value: dict[str, Any]) -> dict[str, Any]:
    return {
        layer: [
            {"statement": fact["statement"], "confidence": fact["confidence"]}
            for fact in value[layer]
        ]
        for layer in ("core", "regulation", "cognition", "identity", "behavior")
    }


def _normalize_generated_message(value: str, speaker: str) -> str:
    if not isinstance(value, str):
        raise ValueError("generated message must be text")
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.I).strip()
    text = re.sub(rf"^{re.escape(speaker)}\s*:\s*", "", text, flags=re.I).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    if not text:
        raise ValueError("generated message must not be empty")
    if text.startswith("{") or text.startswith("["):
        raise ValueError("generated message leaked structured output")
    leaked = ("lambda_t", "behavior_policy", "user_domain", "self_domain")
    if any(marker in text.casefold() for marker in leaked):
        raise ValueError("generated message leaked internal state")
    return text


def _validate_user_domain_evidence(
    value: dict[str, Any], allowed_turn_ids: set[str]
) -> dict[str, Any]:
    for layer in ("core", "regulation", "cognition", "identity", "behavior"):
        for fact in value[layer]:
            evidence = set(fact["evidence_turn_ids"])
            if not evidence:
                raise ValueError(f"{layer} fact has no evidence turn IDs")
            unknown = evidence - allowed_turn_ids
            if unknown:
                raise ValueError(
                    f"{layer} fact cites unobserved or non-partner turns: {sorted(unknown)}"
                )
    return value


def _normalize_alignment_for_context(
    value: dict[str, Any], allowed_turn_ids: set[str]
) -> dict[str, Any]:
    if not allowed_turn_ids:
        value = {
            **value,
            "user_state": {
                "current": {
                    "emotion": "",
                    "emotional_intensity": "low",
                    "intent": "",
                    "main_need": "",
                    "interaction_expectation": "",
                    "evidence_turn_ids": [],
                    "uncertainty": "high",
                },
                "future": {
                    "likely_reaction": "",
                    "response_risk": "",
                    "desired_transition": "",
                    "uncertainty": "high",
                },
            },
            "alignment": {
                "lambda_t": 0.0,
                "orientation": "self-dominant",
                "lambda_basis": "No partner evidence is visible.",
                "self_constraint": "Preserve the target speaker's demonstrated style.",
                "user_adaptation": "None until partner evidence is observed.",
            },
            "behavior_policy": {
                "response_objective": "Produce a simple target-style conversation opener.",
                "perspective_taking": "None without partner evidence.",
                "emotion_alignment": "Do not infer or manufacture a partner emotion.",
                "personalization": "none",
                "self_domain_expression": "Use style only; do not turn profile facts into a current event.",
                "directness": "medium",
                "guidance": "none",
                "question_policy": "optional",
                "tone": value["behavior_policy"]["tone"],
                "avoid": list(dict.fromkeys([
                    *value["behavior_policy"]["avoid"],
                    "invented current or recent personal events",
                    "unsupported partner needs",
                ])),
            },
        }
    else:
        policy = value["behavior_policy"]
        value = {
            **value,
            "behavior_policy": {
                **policy,
                "self_domain_expression": (
                    "Use demonstrated tone, phrasing, initiative, and conversational scale only. "
                    "Do not introduce identity facts, interests, activities, locations, plans, or "
                    "anecdotes unless the visible Cb history directly supports that content."
                ),
                "avoid": list(dict.fromkeys([
                    *policy["avoid"],
                    "unsupported first-person factual details",
                    "turning stable profile facts into current or recent events",
                ])),
            },
        }
    evidence = set(value["user_state"]["current"]["evidence_turn_ids"])
    unknown = evidence - allowed_turn_ids
    if unknown:
        raise ValueError(
            f"current user state cites unobserved turns: {sorted(unknown)}"
        )
    return value


def _normalize_local_labels(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"emotion", "sentiment", "intimacy"}:
        raise ValueError("local labels must contain exactly emotion, sentiment, intimacy")
    intimacy = value["intimacy"]
    if isinstance(intimacy, bool) or not isinstance(intimacy, (int, float)):
        raise ValueError("intimacy must be numeric")
    intimacy = float(intimacy)
    if not 0 <= intimacy <= 1:
        raise ValueError("intimacy must be in [0,1]")
    return {
        "emotion": str(value["emotion"]).strip().lower(),
        "sentiment": str(value["sentiment"]).strip().lower(),
        "intimacy": intimacy,
    }


def _normalize_bertscore(value: Any, expected: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError("BERTScore output does not align with predictions")
    normalized = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("BERTScore values must be numeric")
        score = float(item)
        if not -1 <= score <= 1:
            raise ValueError("BERTScore values must be in [-1,1]")
        normalized.append(round(score, 6))
    return normalized


def _backend_from_env() -> OpenAICompatibleChatBackend:
    def env(name: str, fallback: str = "") -> str:
        return os.getenv(f"REALTALK_OURS_{name}", fallback).strip()

    return OpenAICompatibleChatBackend(
        api_key=env("API_KEY", os.getenv("API_KEY", "")),
        base_url=env("BASE_URL", os.getenv("BASE_URL", "")),
        model=env("MODEL", EXPECTED_MODEL),
        timeout_seconds=float(env("TIMEOUT_SECONDS", "180")),
        max_attempts=int(env("NETWORK_MAX_ATTEMPTS", "6")),
        enable_thinking=False,
    )


def _run_signature(
    config: RealTalkOursConfig,
    backend: ChatBackend,
    dataset_manifest: dict[str, Any],
) -> str:
    cfg = asdict(config)
    for key in ("output_dir", "fresh", "continue_on_error", "preflight_only"):
        cfg.pop(key, None)
    return stable_hash({
        "config": cfg,
        "model": backend.model,
        "thinking": bool(getattr(backend, "enable_thinking", False)),
        "dataset_manifest": dataset_manifest,
        "prompts": _prompt_hashes(),
        "schemas": {
            "self": stable_hash(SELF_DOMAIN_SCHEMA),
            "user": stable_hash(USER_DOMAIN_SCHEMA),
            "alignment": stable_hash(ALIGNMENT_SCHEMA),
        },
        "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })


def _prompt_hashes() -> dict[str, str]:
    return {
        "self_domain_system": stable_hash(SELF_DOMAIN_SYSTEM_PROMPT),
        "self_domain_user": stable_hash(SELF_DOMAIN_USER_TEMPLATE),
        "user_domain_system": stable_hash(USER_DOMAIN_SYSTEM_PROMPT),
        "user_domain_user": stable_hash(USER_DOMAIN_USER_TEMPLATE),
        "alignment_system": stable_hash(ALIGNMENT_SYSTEM_PROMPT),
        "alignment_user": stable_hash(ALIGNMENT_USER_TEMPLATE),
        "generation_system": stable_hash(GENERATION_SYSTEM_TEMPLATE),
        "generation_user": stable_hash(GENERATION_USER_TEMPLATE),
        "format_repair": stable_hash(FORMAT_REPAIR_TEMPLATE),
    }


def _validate_config(config: RealTalkOursConfig) -> None:
    if config.profile_sessions != 3 or config.test_sessions != 3:
        raise ValueError("main REALTALK Ours protocol requires exactly three Ca and Cb sessions")
    if config.max_context_chars < 0:
        raise ValueError("max_context_chars must be non-negative")
    if config.operation_max_attempts != 3:
        raise ValueError("structured logical attempts are fixed at three")
    if config.max_eval_points_per_speaker < 0:
        raise ValueError("max_eval_points_per_speaker must be non-negative")


def _is_full_protocol(
    config: RealTalkOursConfig, splits: list[dict[str, str]]
) -> bool:
    return (
        not config.speaker_filter
        and config.max_eval_points_per_speaker == 0
        and len(splits) == len(REALTALK_PERSONA_SPLITS)
    )


def _checkpoint_unresolved(checkpoint: OperationCheckpoint) -> list[dict[str, Any]]:
    return [
        {"operation_key": key, **value}
        for key, value in sorted(checkpoint.data["failures"].items())
    ]


def _checkpoint_usage(
    checkpoint: OperationCheckpoint | None,
    backend: ChatBackend,
) -> dict[str, int]:
    if checkpoint is None:
        return {
            key: int(value)
            for key, value in getattr(backend, "token_usage", {}).items()
        }
    total = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    for operation in checkpoint.data["operations"].values():
        for key in total:
            total[key] += int(operation.get("token_usage", {}).get(key, 0))
    return total


def _failure(
    stage: str, speaker: str, result_id: str | None, exc: Exception
) -> dict[str, Any]:
    return {
        "stage": stage,
        "speaker": speaker,
        "result_id": result_id,
        "error_type": type(exc).__name__,
        "error": str(exc)[:1000],
        "recorded_at_utc": _now(),
    }


def _turns_with_ids(turns: Iterable[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{turn['turn_id']}] {turn['speaker']}: {turn['content']}" for turn in turns
    )


def _metric_display(summary: dict[str, Any] | None, metric: str) -> str:
    if not summary:
        return "pending"
    value = summary.get("speaker_macro", {}).get(metric)
    return value["display"] if value else "not computed"


def _split_order(speaker: str) -> int:
    names = [item["speaker"].casefold() for item in REALTALK_PERSONA_SPLITS]
    return names.index(speaker.casefold())


def _speaker_id(speaker: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", speaker.casefold()).strip("_")


def _safe_host(base_url: str) -> str:
    match = re.match(r"https?://([^/]+)", base_url)
    return match.group(1) if match else base_url


def _clear_runtime_artifacts(output_dir: Path) -> None:
    for name in (
        "checkpoint.json",
        "checkpoint.json.tmp",
        "raw_responses.jsonl",
        "predictions.jsonl",
        "results_with_local_metrics.jsonl",
        "self_domains.json",
        "unresolved_errors.json",
        "dataset_manifest.json",
        "run_manifest.json",
        "local_metrics_summary.json",
        "table2_partial.json",
        "REPORT_PARTIAL.md",
        "GENERATION_COMPLETE",
        "LOCAL_METRICS_COMPLETE",
        "PIPELINE_COMPLETE",
        "GPT_EVALUATION_PENDING.json",
        "preflight.json",
    ):
        _remove(output_dir / name)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _remove(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> RealTalkOursConfig:
    parser = argparse.ArgumentParser(
        description="Run REALTALK Task 1 Ours with fixed qwen3-8b"
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/realtalk_ours_qwen3_8b")
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--max-eval-points-per-speaker", type=int, default=0)
    parser.add_argument("--speaker", action="append", dest="speaker_filter")
    parser.add_argument("--skip-local-metrics", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    return RealTalkOursConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_context_chars=args.max_context_chars,
        max_eval_points_per_speaker=args.max_eval_points_per_speaker,
        speaker_filter=tuple(args.speaker_filter or ()),
        compute_local_metrics=not args.skip_local_metrics,
        compute_bertscore=not args.skip_bertscore,
        continue_on_error=not args.stop_on_error,
        preflight_only=args.preflight_only,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    print(json.dumps(run_realtalk_ours(parse_args()), ensure_ascii=False, indent=2))
