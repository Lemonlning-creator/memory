"""Revised Experiment 2: current and future user-modeling evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from ...epistemic_decay import (
    compute_portrait_entropy,
    compute_profile_completeness,
)
from ...llm_client import LLMClient
from ...prompts.templates_en import (
    PROFILE_EXTRACTION_SYSTEM_PROMPT,
    PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
)
from ...utils import load_json
from ..exp1_metrics import classification_report, speaker_macro_report
from ..exp1_protocol import (
    build_message_level_points,
    build_profile_corpus,
    canonical_speaker,
    message_speakers,
    select_realtalk_splits,
    stable_hash,
)
from ..exp1_schema import (
    EMOTION_LABELS,
    SENTIMENT_LABELS,
)
from ..exp2_generation import compute_bertscore_f1, compute_response_scores
from ..experiment_utils import robust_parse_json
from ..operation_checkpoint import OperationCheckpoint
from ..realtalk_evaluator import RealTalkLabelEvaluator
from ..result_provenance import build_run_manifest
from .schemas import (
    CURRENT_STATE_SCHEMA,
    EMPATHY_SCHEMA,
    GROUNDING_SCHEMA,
    REFLECTIVENESS_SCHEMA,
    TOPIC_REFERENCE_SCHEMA,
    normalize_current_state,
    normalize_empathy,
    normalize_grounding,
    normalize_reflectiveness,
    normalize_topic_reference,
)

ZERO_SHOT = "realtalk_zero_shot"
FINE_TUNED = "realtalk_fine_tuned"
OURS = "ours"
GENERATED_METHODS = (ZERO_SHOT, OURS)

# These are paper-reported reference values, not outputs of this runner.
REALTALK_TABLE2_REFERENCE = {
    ZERO_SHOT: {
        "lexical": 0.14,
        "semantic": 0.76,
        "reflective": 0.62,
        "grounding": 0.40,
        "sentiment": 0.53,
        "emotion": 0.43,
        "intimacy_ad": 0.06,
        "empathy_ad": 1.80,
    },
    FINE_TUNED: {
        "lexical": 0.14,
        "semantic": 0.78,
        "reflective": 0.77,
        "grounding": 0.62,
        "sentiment": 0.59,
        "emotion": 0.46,
        "intimacy_ad": 0.07,
        "empathy_ad": 1.24,
    },
}

REALTALK_TABLE8_PERSONA_CONSISTENCY = {
    "Emi": 0.21,
    "Nicolas": 0.09,
    "Kevin": 0.22,
    "Akib": 0.07,
    "Muhhamed": 0.11,
    "Nebraas": 0.20,
    "Paola": 0.17,
    "Vanessa": 0.18,
    "elise": 0.14,
    "Fahim Khan": 0.12,
}

PROFILE_LAYERS = ("core", "regulation", "cognition", "identity", "behavior")

CURRENT_STATE_SYSTEM_PROMPT = """Infer the state expressed by the user's current observed message.

Use the prior conversation only as context. Do not predict a future turn.
Topic must be a concise description of the current message's main subject."""

TOPIC_REFERENCE_SYSTEM_PROMPT = """Identify the concise main subject of the current observed message. Use prior dialogue only to resolve references. Do not infer a different hidden topic."""

REFLECTIVENESS_SYSTEM_PROMPT = """Determine whether the speaker's current message is reflective using the REALTALK Appendix C self-awareness definition.

Reflective language explicitly expresses self-observation, perspective-taking, or intentionality: the speaker analyzes their own thoughts, feelings, actions, effects on others, motivations, or goals. A preference, plan, factual statement, emotional phrase, or conversational reaction alone is not reflective."""

GROUNDING_SYSTEM_PROMPT = """Determine whether the speaker's current message demonstrates grounding using the REALTALK Appendix C motivation definition.

Grounding actively builds mutual understanding through a clarifying question, a follow-up inquiry about information the partner already shared, a confirmation check, or a request to expand shared information. Agreement, a new invitation, a topic change, a rhetorical question, or a generic conversation starter alone is not grounding."""

EMPATHY_SYSTEM_PROMPT = """Score empathy in the current message using the REALTALK Appendix C EPITOME rubric.

Emotional reaction:
- 0: no warmth, compassion, concern, or similar feeling directed toward the partner's disclosed experience.
- 1: such partner-directed feeling is implied.
- 2: such partner-directed feeling is explicitly expressed.

Interpretation:
- 0: no understanding of the partner's disclosed experience or feelings.
- 1: plausible inference, paraphrase, reflection, or description showing understanding.
- 2: deep, explicit understanding and validation, potentially using multiple forms.

Exploration:
- 0: no attempt to explore the partner's disclosed experience or feelings.
- 1: a generic attempt to explore it.
- 2: a specific question or inquiry delving into that particular experience or feeling.

Do not award empathy merely because the message is friendly, emotional, self-disclosing, proposes an activity, or contains a question. Empathy must respond to an experience or feeling the partner actually disclosed in the prior dialogue."""

EMPATHY_SCORING_BOUNDARY = """Only experiences or feelings disclosed by the partner in the prior conversation can support emotional_reaction or interpretation. The current speaker's own self-disclosure cannot count as understanding the partner."""

EMPATHY_EMPTY_HISTORY_BOUNDARY = """No prior partner message exists in this sample. Therefore emotional_reaction and interpretation must both be 0. A generic invitation for the partner to share may receive exploration=1, but exploration=2 requires a specific prior partner disclosure."""

PROFILE_PRIVATE_CONTEXT_BOUNDARY = """END PRIVATE PROFILE.
The profile is private background, not shared history. Never use "we", "our", "remember", or similar wording to imply that the current partner joined or discussed a profile event unless the current conversation above establishes it."""

PROFILE_EMPTY_CONTEXT_BOUNDARY = """No current-partner conversation history is available. Produce a brief, natural opening message consistent with the speaker's style, without mentioning specific profile events, people, places, or stored memories, and without inventing a current or recent activity."""

# REALTALK Appendix D.1 prompt. The baseline receives no additional
# user-model instruction.
REALTALK_CONTINUATION_SYSTEM_PROMPT = """You are {speaker}. Continue the conversation.
Output only the message, not the speaker name."""

OURS_CONTINUATION_SYSTEM_PROMPT = """You are {speaker}. Continue the conversation.
Output only the message, not the speaker name.

Internalize the supplied five-layer user profile when deciding what this person
would naturally say next. Preserve their conversational style and do not mention
the profile or explain your reasoning. Treat profile evidence as the speaker's
private background, not as shared history with the current partner. Do not claim
that the partner joined, witnessed, or discussed a profile event unless the
current conversation establishes it, and do not force unrelated profile details
into the reply. The final {speaker} cue identifies your own speaking role, not
the addressee; never address yourself as {speaker}."""

CURRENT_MAX_TOKENS = 1024
CONTINUATION_MAX_TOKENS = 300


@dataclass
class Exp2UserModelingConfig:
    dataset_dir: str = "dataset"
    output_dir: str = "data/exp2_user_modeling"
    profile_sessions: int = 3
    test_sessions: int = 3
    max_context_chars: int = 60000
    profile_max_tokens: int = 16000
    max_eval_points_per_speaker: int = 0
    operation_max_attempts: int = 3
    chat_filter: list[str] | None = None
    speaker_filter: list[str] | None = None
    fine_tuned_predictions: str | None = None
    compute_bertscore: bool = False
    continue_on_error: bool = True
    fresh: bool = False


def run_user_modeling_evaluation(
    config: Exp2UserModelingConfig,
    llm: LLMClient | None = None,
    label_evaluator: RealTalkLabelEvaluator | None = None,
) -> dict[str, Any]:
    """Run both revised Exp2 tracks over the same causal REALTALK points."""
    splits = select_realtalk_splits(
        config.dataset_dir, config.chat_filter, config.speaker_filter
    )
    if not splits:
        raise ValueError("no REALTALK speaker splits matched the configuration")
    llm = llm or LLMClient()
    evaluator = label_evaluator or RealTalkLabelEvaluator()
    imported_fine_tuned = _load_fine_tuned_predictions(
        config.fine_tuned_predictions
    )

    output_dir = Path(config.output_dir)
    checkpoint_path = output_dir / "checkpoint.json"
    if config.fresh and checkpoint_path.exists():
        checkpoint_path.unlink()
    signature = _run_signature(
        config, llm, evaluator.metadata(), splits, imported_fine_tuned
    )
    checkpoint = OperationCheckpoint(checkpoint_path, signature)
    started = perf_counter()
    run_failures: list[dict[str, Any]] = []
    profile_evolution: dict[str, Any] = {}

    dataset_dir = Path(config.dataset_dir)
    for split in splits:
        train_file = dataset_dir / split["train_chat"]
        test_file = dataset_dir / split["test_chat"]
        train_chat = load_json(str(train_file))
        test_chat = load_json(str(test_file))
        train_speaker = canonical_speaker(train_chat, split["speaker"])
        target_speaker = canonical_speaker(test_chat, split["speaker"])
        if train_speaker.casefold() != target_speaker.casefold():
            raise ValueError(
                f"speaker mismatch across split: {train_speaker!r} "
                f"vs {target_speaker!r}"
            )
        partner_speaker = next(
            speaker for speaker in message_speakers(test_chat)
            if speaker.casefold() != target_speaker.casefold()
        )
        (
            profile_corpus,
            profile,
            profile_key,
            profile_snapshots,
        ) = _build_profile_evolution(
            checkpoint,
            llm,
            config,
            split,
            train_chat,
            target_speaker,
        )
        profile_evolution[target_speaker] = {
            "speaker": target_speaker,
            "source_chat": train_file.name,
            "mode": "independent_prefix_rebuild_not_bayesian_update",
            "snapshots": profile_snapshots,
        }
        points = build_message_level_points(
            test_chat,
            target_speaker,
            test_sessions=config.test_sessions,
            max_context_chars=config.max_context_chars,
            max_eval_points=config.max_eval_points_per_speaker,
        )
        print(
            f"[Exp2 User Modeling] speaker={target_speaker} "
            f"train={train_file.name} test={test_file.name} points={len(points)}"
        )

        for index, point in enumerate(points, start=1):
            result_id = (
                f"{_speaker_id(target_speaker)}:{test_file.stem}:"
                f"{point['sample_id']}"
            )
            if result_id in checkpoint.data["results"]:
                print(f"  [{index}/{len(points)}] resume {point['eval_id']}")
                continue
            sample_started = perf_counter()
            print(f"  [{index}/{len(points)}] evaluate {point['eval_id']}")
            try:
                reference = _cached_message_ei(
                    checkpoint,
                    llm,
                    evaluator,
                    config,
                    f"reference:{result_id}",
                    target_speaker,
                    point["context_text"],
                    point["target_message"],
                    include_topic=True,
                )

                current = {}
                future = {}
                for method in GENERATED_METHODS:
                    current_prediction = _cached_current_prediction(
                        checkpoint,
                        llm,
                        config,
                        result_id,
                        method,
                        point["context_text"],
                        point["target_message"],
                        profile if method == OURS else None,
                    )
                    current[method] = {
                        "prediction": current_prediction,
                        "scores": _current_scores(current_prediction, reference),
                    }

                    generated = _cached_continuation(
                        checkpoint,
                        llm,
                        config,
                        result_id,
                        method,
                        target_speaker,
                        point["context_text"],
                        profile if method == OURS else None,
                    )
                    generated_ei = _cached_message_ei(
                        checkpoint,
                        llm,
                        evaluator,
                        config,
                        f"generated:{result_id}:{method}",
                        target_speaker,
                        point["context_text"],
                        generated,
                        include_topic=False,
                    )
                    future[method] = {
                        "generated_message": generated,
                        "message_ei": generated_ei,
                        "scores": _future_scores(
                            generated,
                            point["target_message"],
                            generated_ei,
                            reference,
                        ),
                    }

                fine_tuned_text = imported_fine_tuned.get(result_id)
                if fine_tuned_text:
                    fine_tuned_ei = _cached_message_ei(
                        checkpoint,
                        llm,
                        evaluator,
                        config,
                        f"generated:{result_id}:{FINE_TUNED}",
                        target_speaker,
                        point["context_text"],
                        fine_tuned_text,
                        include_topic=False,
                    )
                    future[FINE_TUNED] = {
                        "generated_message": fine_tuned_text,
                        "message_ei": fine_tuned_ei,
                        "scores": _future_scores(
                            fine_tuned_text,
                            point["target_message"],
                            fine_tuned_ei,
                            reference,
                        ),
                        "source": "imported_external_fine_tuned_prediction",
                    }

                checkpoint.store_result(result_id, {
                    "result_id": result_id,
                    "speaker": target_speaker,
                    "partner_speaker": partner_speaker,
                    "train_chat_file": train_file.name,
                    "test_chat_file": test_file.name,
                    "eval_id": point["eval_id"],
                    "message_level_index": point["message_level_index"],
                    "target_session": point["target_session"],
                    "target_message": point["target_message"],
                    "target_dia_ids": point["target"].get("dia_ids", []),
                    "reference": reference,
                    "current_understanding": current,
                    "future_understanding": future,
                    "profile": {
                        "source": "fixed_cross_conversation_ca",
                        "train_sessions": profile_corpus["sessions"],
                        "history_hash": profile_corpus["history_hash"],
                        "cache_key": profile_key,
                        "characters": len(
                            json.dumps(profile, ensure_ascii=False)
                        ),
                        "portrait_entropy": compute_portrait_entropy(profile),
                    },
                    "context": {
                        "source": "all_prior_merged_turns_in_selected_cb_segment",
                        "test_sessions": point["test_sessions"],
                        "semantic_turns": len(point["context_turns"]),
                        "characters": len(point["context_text"]),
                        "truncated": point["context_truncated"],
                        "history_hash": point["history_hash"],
                        "target_visible_current_track": True,
                        "target_visible_future_track": False,
                    },
                    "sample_elapsed_seconds": round(
                        perf_counter() - sample_started, 3
                    ),
                    "completed_at_utc": _now(),
                    "status": "complete",
                })
            except Exception as exc:
                failure = {
                    "result_id": result_id,
                    "speaker": target_speaker,
                    "eval_id": point["eval_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "failed_at_utc": _now(),
                }
                checkpoint.store_excluded_result(result_id, failure)
                run_failures.append(failure)
                if not config.continue_on_error:
                    raise

    results = sorted(
        checkpoint.result_values(), key=lambda item: item["result_id"]
    )
    if config.compute_bertscore:
        _add_batched_bertscore(checkpoint, results, config)
    summary = aggregate_results(results)
    summary["profile_evolution"] = {
        speaker: [
            {
                key: snapshot[key]
                for key in (
                    "session_count",
                    "sessions",
                    "portrait_entropy",
                    "profile_completeness",
                    "characters",
                )
            }
            for snapshot in value["snapshots"]
        ]
        for speaker, value in sorted(profile_evolution.items())
    }
    summary.update({
        "elapsed_seconds": round(perf_counter() - started, 3),
        "failed_samples_excluded_this_run": run_failures,
        "token_usage": _checkpoint_token_usage(checkpoint),
        "official_table2_reference": REALTALK_TABLE2_REFERENCE,
        "official_reference_is_not_merged_with_run_results": True,
        "fine_tuned_predictions_loaded": bool(imported_fine_tuned),
    })
    manifest = build_run_manifest(
        {**asdict(config), "fresh": False}, getattr(llm, "model", None)
    )
    manifest.update({
        "experiment": "Experiment 2 - User Modeling Evaluation",
        "run_signature": signature,
        "tracks": {
            "current_understanding": {
                "target_visible": True,
                "metrics": [
                    "emotion_accuracy",
                    "sentiment_accuracy",
                    "topic_consistency",
                ],
                "status": "paper-specific extension; not REALTALK Table 2",
            },
            "future_understanding": {
                "target_visible": False,
                "unit": "next merged speaker message",
                "metrics": [
                    "rouge_l",
                    "bertscore_f1",
                    "reflectiveness_accuracy",
                    "grounding_accuracy",
                    "sentiment_accuracy",
                    "emotion_accuracy",
                    "intimacy_absolute_difference",
                    "empathy_absolute_difference",
                ],
                "status": "REALTALK persona-simulation alignment",
            },
        },
        "methods_generated_in_run": list(GENERATED_METHODS),
        "fine_tuned_policy": (
            "optional external per-sample predictions; absent predictions are "
            "reported as unavailable and never synthesized"
        ),
        "label_evaluator": evaluator.metadata(),
        "schemas": {
            "current_state": CURRENT_STATE_SCHEMA,
            "topic_reference": TOPIC_REFERENCE_SCHEMA,
            "reflectiveness": REFLECTIVENESS_SCHEMA,
            "grounding": GROUNDING_SCHEMA,
            "empathy": EMPATHY_SCHEMA,
        },
        "message_ei_evaluation": {
            "mode": "independent_realtalk_appendix_c_judgments",
            "llm_judgments": [
                "topic_reference",
                "reflectiveness",
                "grounding",
                "epitome_empathy",
            ],
            "fixed_classifiers": ["emotion", "sentiment", "intimacy"],
        },
        "bertscore": {
            "enabled": config.compute_bertscore,
            "model": "roberta-large",
            "num_layers": 17,
            "raw_response_pairs_preserved": True,
        },
        "profile_evolution": {
            "mode": "independent Ca prefix rebuild",
            "session_prefixes": list(
                range(1, config.profile_sessions + 1)
            ),
            "bayesian_updating_enabled": False,
            "output": "profile_evolution.json",
        },
        "prompt_hashes": _prompt_hashes(),
        "realtalk_speaker_splits": splits,
    })
    _write_outputs(
        output_dir, results, summary, manifest, profile_evolution
    )
    return summary


def _cached_current_prediction(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp2UserModelingConfig,
    result_id: str,
    method: str,
    history: str,
    target_message: str,
    profile: dict[str, Any] | None,
) -> dict[str, str]:
    profile_text = (
        json.dumps(profile, ensure_ascii=False, indent=2) if profile else ""
    )
    prompt = (
        f"PRIOR CONVERSATION:\n{history or '(none)'}\n\n"
        f"CURRENT OBSERVED MESSAGE:\n{target_message}\n"
    )
    if profile_text:
        prompt += f"\nFIVE-LAYER USER PROFILE:\n{profile_text}\n"
    key = "current:" + stable_hash({
        "result_id": result_id,
        "method": method,
        "prompt": prompt,
        "system": CURRENT_STATE_SYSTEM_PROMPT,
        "schema": CURRENT_STATE_SCHEMA,
        "model": getattr(llm, "model", None),
    })
    return checkpoint.execute(
        key,
        lambda: json.loads(llm.chat(
            CURRENT_STATE_SYSTEM_PROMPT,
            prompt,
            temperature=0.0,
            max_tokens=CURRENT_MAX_TOKENS,
            response_schema=CURRENT_STATE_SCHEMA,
        )),
        normalize_current_state,
        config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )


def _cached_message_ei(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    evaluator: RealTalkLabelEvaluator,
    config: Exp2UserModelingConfig,
    source_id: str,
    speaker: str,
    history: str,
    message: str,
    *,
    include_topic: bool,
) -> dict[str, Any]:
    prompt = (
        f"CONVERSATION HISTORY:\n{history or '(none)'}\n\n"
        f"CURRENT OBSERVED MESSAGE BY {speaker}:\n{message}\n\n"
        "Judge only the current observed message, using its prior context."
    )
    common_key = {
        "source_id": source_id,
        "prompt": prompt,
        "model": getattr(llm, "model", None),
    }

    fixed_labels = checkpoint.execute(
        "message_labels:" + stable_hash({
            **common_key,
            "classifier": evaluator.metadata(),
            "message": message,
        }),
        lambda: evaluator.annotate(message),
        _normalize_fixed_labels,
        config.operation_max_attempts,
    )
    reflective = _cached_ei_judgment(
        checkpoint,
        llm,
        config,
        common_key,
        "reflectiveness",
        REFLECTIVENESS_SYSTEM_PROMPT,
        REFLECTIVENESS_SCHEMA,
        normalize_reflectiveness,
        prompt,
    )
    grounding = _cached_ei_judgment(
        checkpoint,
        llm,
        config,
        common_key,
        "grounding",
        GROUNDING_SYSTEM_PROMPT,
        GROUNDING_SCHEMA,
        normalize_grounding,
        prompt,
    )
    empathy_prompt = prompt + (
        "\n\nSCORING BOUNDARY:\n" + EMPATHY_SCORING_BOUNDARY
    )
    if not history:
        empathy_prompt += "\n" + EMPATHY_EMPTY_HISTORY_BOUNDARY
    empathy = _cached_ei_judgment(
        checkpoint,
        llm,
        config,
        common_key,
        "empathy",
        EMPATHY_SYSTEM_PROMPT,
        EMPATHY_SCHEMA,
        normalize_empathy,
        empathy_prompt,
    )
    result = {
        **fixed_labels,
        **reflective,
        **grounding,
        "empathy": empathy,
    }
    if include_topic:
        result.update(_cached_ei_judgment(
            checkpoint,
            llm,
            config,
            common_key,
            "topic",
            TOPIC_REFERENCE_SYSTEM_PROMPT,
            TOPIC_REFERENCE_SCHEMA,
            normalize_topic_reference,
            prompt,
        ))
    return _normalize_message_ei(result, require_topic=include_topic)


def _cached_ei_judgment(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp2UserModelingConfig,
    common_key: dict[str, Any],
    metric: str,
    system_prompt: str,
    response_schema: dict[str, Any],
    validator: Callable[[Any], Any],
    user_prompt: str,
) -> dict[str, Any]:
    return checkpoint.execute(
        f"message_{metric}:" + stable_hash({
            **common_key,
            "system": system_prompt,
            "schema": response_schema,
        }),
        lambda: json.loads(llm.chat(
            system_prompt,
            user_prompt,
            temperature=0.0,
            max_tokens=CURRENT_MAX_TOKENS,
            response_schema=response_schema,
        )),
        validator,
        config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )


def _normalize_fixed_labels(value: Any) -> dict[str, Any]:
    required = {"emotion", "sentiment", "intimacy"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("fixed labels must contain emotion, sentiment, intimacy")
    emotion = str(value["emotion"]).strip().lower()
    sentiment = str(value["sentiment"]).strip().lower()
    intimacy = value["intimacy"]
    if emotion not in EMOTION_LABELS:
        raise ValueError(f"unsupported emotion label: {emotion}")
    if sentiment not in SENTIMENT_LABELS:
        raise ValueError(f"unsupported sentiment label: {sentiment}")
    if isinstance(intimacy, bool) or not isinstance(intimacy, (int, float)):
        raise ValueError("intimacy must be numeric")
    intimacy = float(intimacy)
    if not 0 <= intimacy <= 1:
        raise ValueError("intimacy must be in [0, 1]")
    return {
        "emotion": emotion,
        "sentiment": sentiment,
        "intimacy": intimacy,
    }


def _normalize_message_ei(
    value: Any, *, require_topic: bool
) -> dict[str, Any]:
    required = {
        "emotion",
        "sentiment",
        "intimacy",
        "reflective",
        "grounding",
        "empathy",
    }
    if require_topic:
        required.add("topic")
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"message EI must contain exactly {sorted(required)}")
    normalized = {
        **_normalize_fixed_labels({
            field: value[field]
            for field in ("emotion", "sentiment", "intimacy")
        }),
        **normalize_reflectiveness({"reflective": value["reflective"]}),
        **normalize_grounding({"grounding": value["grounding"]}),
        "empathy": normalize_empathy(value["empathy"]),
    }
    if require_topic:
        normalized.update(normalize_topic_reference({"topic": value["topic"]}))
    return normalized


def _cached_continuation(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp2UserModelingConfig,
    result_id: str,
    method: str,
    speaker: str,
    history: str,
    profile: dict[str, Any] | None,
) -> str:
    system = (
        REALTALK_CONTINUATION_SYSTEM_PROMPT
        if method == ZERO_SHOT
        else OURS_CONTINUATION_SYSTEM_PROMPT
    ).format(speaker=speaker)
    prompt = f"{history}\n" if history else ""
    if profile:
        prompt += (
            "FIVE-LAYER USER PROFILE:\n"
            + json.dumps(profile, ensure_ascii=False, indent=2)
            + "\n"
            + PROFILE_PRIVATE_CONTEXT_BOUNDARY
            + "\n"
        )
        if not history:
            prompt += PROFILE_EMPTY_CONTEXT_BOUNDARY + "\n"
    prompt += speaker
    key = "continuation:" + stable_hash({
        "result_id": result_id,
        "method": method,
        "prompt": prompt,
        "system": system,
        "model": getattr(llm, "model", None),
    })
    return checkpoint.execute(
        key,
        lambda: llm.chat(
            system,
            prompt,
            temperature=0.6,
            max_tokens=CONTINUATION_MAX_TOKENS,
        ),
        _normalize_generated_message,
        config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )


def _normalize_generated_message(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("generated next message must not be empty")
    return text


def _current_scores(
    prediction: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float]:
    return {
        "emotion_accuracy": float(
            prediction["emotion"] == reference["emotion"]
        ),
        "sentiment_accuracy": float(
            prediction["sentiment"] == reference["sentiment"]
        ),
        "topic_consistency": round(
            _topic_overlap(prediction["topic"], reference["topic"]), 6
        ),
    }


def _future_scores(
    candidate_text: str,
    reference_text: str,
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, float]:
    return compute_response_scores(
        reference_text=reference_text,
        candidate_text=candidate_text,
        reference_ei=reference,
        candidate_ei=candidate,
    )


def _topic_overlap(prediction: str, reference: str) -> float:
    predicted = set(re.findall(r"[\w']+", prediction.casefold()))
    expected = set(re.findall(r"[\w']+", reference.casefold()))
    return len(predicted & expected) / max(len(expected), 1)


def _add_batched_bertscore(
    checkpoint: OperationCheckpoint,
    results: list[dict[str, Any]],
    config: Exp2UserModelingConfig,
) -> None:
    locations: list[tuple[dict[str, Any], str, str, str]] = []
    for result in results:
        reference = result["target_message"]
        for method, method_data in sorted(result["future_understanding"].items()):
            candidate = method_data.get("generated_message")
            if candidate:
                locations.append((
                    method_data["scores"],
                    reference,
                    candidate,
                    f"{result['result_id']}:{method}",
                ))
    if not locations:
        return
    payload = [
        {
            "id": item[3],
            "reference": item[1],
            "candidate": item[2],
        }
        for item in locations
    ]
    key = "bertscore_batch:" + stable_hash(payload)

    def validate(values: Any) -> list[float]:
        if not isinstance(values, list) or len(values) != len(locations):
            raise ValueError("BERTScore batch result does not align with responses")
        normalized = [float(value) for value in values]
        if any(not -1.0 <= value <= 1.0 for value in normalized):
            raise ValueError("BERTScore value must be in [-1, 1]")
        return normalized

    values = checkpoint.execute(
        key,
        lambda: compute_bertscore_f1(
            [item[1] for item in locations],
            [item[2] for item in locations],
        ),
        validate,
        config.operation_max_attempts,
    )
    for (scores, _, _, _), value in zip(locations, values):
        scores["bertscore_f1"] = round(value, 6)
    checkpoint.save()


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "num_eval_points": 0,
            "num_speakers": 0,
            "current_understanding": {},
            "future_understanding": {},
        }
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_speaker[result["speaker"]].append(result)

    current_methods = sorted({
        method
        for result in results
        for method in result["current_understanding"]
    })
    future_methods = sorted({
        method
        for result in results
        for method in result["future_understanding"]
    })
    return {
        "num_eval_points": len(results),
        "num_speakers": len(by_speaker),
        "primary_aggregation": "speaker_macro",
        "current_understanding": {
            method: _aggregate_current(results, method)
            for method in current_methods
        },
        "future_understanding": {
            method: _aggregate_future(results, method)
            for method in future_methods
        },
        "persona_consistency": {
            "status": "dataset_diagnostic_not_method_ranking",
            "definition": (
                "REALTALK absolute cross-conversation EI difference; official "
                "speaker-level values from Table 8"
            ),
            "by_speaker": {
                speaker: REALTALK_TABLE8_PERSONA_CONSISTENCY[speaker]
                for speaker in sorted(by_speaker)
            },
            "speaker_macro": _mean(
                REALTALK_TABLE8_PERSONA_CONSISTENCY[speaker]
                for speaker in by_speaker
            ),
        },
        "portrait_entropy": {
            "status": "fixed_Ca_profile_diagnostic",
            "by_speaker": {
                speaker: _mean(
                    result["profile"]["portrait_entropy"]
                    for result in speaker_results
                )
                for speaker, speaker_results in sorted(by_speaker.items())
            },
        },
    }


def _build_profile_evolution(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp2UserModelingConfig,
    split: dict[str, str],
    train_chat: dict[str, Any],
    user_speaker: str,
) -> tuple[dict[str, Any], dict[str, Any], str, list[dict[str, Any]]]:
    snapshots: list[dict[str, Any]] = []
    final_corpus: dict[str, Any] | None = None
    final_profile: dict[str, Any] | None = None
    final_key = ""
    for session_count in range(1, config.profile_sessions + 1):
        corpus = build_profile_corpus(
            train_chat, user_speaker, session_count
        )
        profile, cache_key = _cached_profile(
            checkpoint,
            llm,
            config,
            split,
            corpus,
            user_speaker,
            "explicit",
        )
        snapshots.append({
            "session_count": session_count,
            "sessions": corpus["sessions"],
            "history_hash": corpus["history_hash"],
            "cache_key": cache_key,
            "characters": len(json.dumps(profile, ensure_ascii=False)),
            "portrait_entropy": compute_portrait_entropy(profile),
            "profile_completeness": compute_profile_completeness(profile),
            "profile": profile,
        })
        final_corpus = corpus
        final_profile = profile
        final_key = cache_key
    assert final_corpus is not None and final_profile is not None
    return final_corpus, final_profile, final_key, snapshots


def _cached_profile(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp2UserModelingConfig,
    split: dict[str, str],
    profile_corpus: dict[str, Any],
    user_speaker: str,
    _profile_type: str = "explicit",
) -> tuple[dict[str, Any], str]:
    prompt_hash = stable_hash(
        PROFILE_EXTRACTION_SYSTEM_PROMPT
        + PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE
    )
    key = ":".join((
        "exp2_profile",
        _speaker_id(user_speaker),
        Path(split["train_chat"]).stem,
        f"sessions_{len(profile_corpus['sessions'])}",
        profile_corpus["history_hash"],
        str(getattr(llm, "model", "unknown")),
        prompt_hash,
    ))

    def operation() -> dict[str, Any]:
        return robust_parse_json(llm.chat(
            PROFILE_EXTRACTION_SYSTEM_PROMPT.format(user_name=user_speaker),
            PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
                user_name=user_speaker,
                corpus=profile_corpus["text"],
            ),
            temperature=0.3,
            max_tokens=config.profile_max_tokens,
        ))

    return checkpoint.execute(
        key,
        operation,
        _normalize_explicit_profile,
        config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    ), key


def _normalize_explicit_profile(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("error"):
        raise ValueError("explicit profile is invalid")
    missing = [
        layer for layer in PROFILE_LAYERS
        if not isinstance(value.get(layer), dict)
    ]
    if missing:
        raise ValueError(f"explicit profile is missing layers: {missing}")
    return value


def _aggregate_current(
    results: list[dict[str, Any]], method: str
) -> dict[str, Any]:
    available = [
        result for result in results
        if method in result["current_understanding"]
    ]
    emotion_records = _classification_records(
        available, method, "current_understanding", "emotion"
    )
    sentiment_records = _classification_records(
        available, method, "current_understanding", "sentiment"
    )
    return {
        "speaker_macro": {
            "emotion_accuracy": speaker_macro_report(
                emotion_records, EMOTION_LABELS
            )["accuracy"],
            "sentiment_accuracy": speaker_macro_report(
                sentiment_records, SENTIMENT_LABELS
            )["accuracy"],
            "topic_consistency": _speaker_macro_score(
                available, "current_understanding", method, "topic_consistency"
            ),
        },
        "micro": {
            "emotion": classification_report(
                [item["reference"] for item in emotion_records],
                [item["prediction"] for item in emotion_records],
                EMOTION_LABELS,
            ),
            "sentiment": classification_report(
                [item["reference"] for item in sentiment_records],
                [item["prediction"] for item in sentiment_records],
                SENTIMENT_LABELS,
            ),
            "topic_consistency": _mean(
                result["current_understanding"][method]["scores"][
                    "topic_consistency"
                ]
                for result in available
            ),
        },
        "num_evaluations": len(available),
    }


def _aggregate_future(
    results: list[dict[str, Any]], method: str
) -> dict[str, Any]:
    available = [
        result for result in results
        if method in result["future_understanding"]
    ]
    emotion_records = _classification_records(
        available, method, "future_understanding", "emotion"
    )
    metric_names = [
        "rouge_l",
        "reflectiveness_accuracy",
        "grounding_accuracy",
        "sentiment_accuracy",
        "emotion_accuracy",
        "intimacy_absolute_difference",
        "empathy_absolute_difference",
        "lexical_overlap",
        "style_similarity",
    ]
    if any(
        "bertscore_f1" in result["future_understanding"][method]["scores"]
        for result in available
    ):
        metric_names.insert(1, "bertscore_f1")
    return {
        "speaker_macro": {
            metric: _speaker_macro_score(
                available,
                "future_understanding",
                method,
                metric,
            )
            for metric in metric_names
        },
        "micro": {
            "emotion": classification_report(
                [item["reference"] for item in emotion_records],
                [item["prediction"] for item in emotion_records],
                EMOTION_LABELS,
            ),
            **{
                metric: _mean(
                    result["future_understanding"][method]["scores"][metric]
                    for result in available
                )
                for metric in metric_names
            },
        },
        "num_evaluations": len(available),
    }


def _classification_records(
    results: list[dict[str, Any]],
    method: str,
    track: str,
    field: str,
) -> list[dict[str, str]]:
    records = []
    for result in results:
        if track == "current_understanding":
            prediction = result[track][method]["prediction"][field]
        else:
            prediction = result[track][method]["message_ei"][field]
        records.append({
            "speaker": result["speaker"],
            "chat_file": result["test_chat_file"],
            "reference": result["reference"][field],
            "prediction": prediction,
        })
    return records


def _speaker_macro_score(
    results: list[dict[str, Any]],
    track: str,
    method: str,
    metric: str,
) -> float:
    by_speaker: dict[str, list[float]] = defaultdict(list)
    for result in results:
        by_speaker[result["speaker"]].append(
            float(result[track][method]["scores"][metric])
        )
    return round(_mean(_mean(values) for values in by_speaker.values()), 6)


def _load_fine_tuned_predictions(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"fine-tuned predictions not found: {source}")
    if source.suffix.casefold() == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        value = json.loads(source.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("predictions", [])
    predictions = {}
    for row in rows:
        result_id = str(row.get("result_id", "")).strip()
        message = str(
            row.get("generated_message") or row.get("prediction") or ""
        ).strip()
        if not result_id or not message:
            raise ValueError(
                "each fine-tuned prediction needs result_id and generated_message"
            )
        if result_id in predictions:
            raise ValueError(f"duplicate fine-tuned result_id: {result_id}")
        predictions[result_id] = message
    return predictions


def _run_signature(
    config: Exp2UserModelingConfig,
    llm: LLMClient,
    evaluator_metadata: dict[str, Any],
    splits: list[dict[str, str]],
    fine_tuned_predictions: dict[str, str],
) -> str:
    payload = asdict(config)
    for key in ("output_dir", "continue_on_error", "fresh"):
        payload.pop(key, None)
    payload.update({
        "model": getattr(llm, "model", None),
        "evaluator": evaluator_metadata,
        "splits": splits,
        "fine_tuned_prediction_hash": stable_hash(fine_tuned_predictions),
        "prompt_hashes": _prompt_hashes(),
        "source_hash": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    })
    return stable_hash(payload)


def _prompt_hashes() -> dict[str, str]:
    return {
        "current_state": stable_hash(CURRENT_STATE_SYSTEM_PROMPT),
        "topic_reference": stable_hash(TOPIC_REFERENCE_SYSTEM_PROMPT),
        "reflectiveness": stable_hash(REFLECTIVENESS_SYSTEM_PROMPT),
        "grounding": stable_hash(GROUNDING_SYSTEM_PROMPT),
        "empathy": stable_hash(EMPATHY_SYSTEM_PROMPT),
        "empathy_scoring_boundary": stable_hash(
            EMPATHY_SCORING_BOUNDARY + EMPATHY_EMPTY_HISTORY_BOUNDARY
        ),
        "realtalk_zero_shot": stable_hash(
            REALTALK_CONTINUATION_SYSTEM_PROMPT
        ),
        "ours_continuation": stable_hash(OURS_CONTINUATION_SYSTEM_PROMPT),
        "profile_private_context_boundary": stable_hash(
            PROFILE_PRIVATE_CONTEXT_BOUNDARY + PROFILE_EMPTY_CONTEXT_BOUNDARY
        ),
        "five_layer_profile": stable_hash(
            PROFILE_EXTRACTION_SYSTEM_PROMPT
            + PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE
        ),
    }


def _checkpoint_token_usage(
    checkpoint: OperationCheckpoint,
) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    for operation in checkpoint.data["operations"].values():
        for key in total:
            total[key] += int(operation.get("token_usage", {}).get(key, 0))
    return total


def _write_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest: dict[str, Any],
    profile_evolution: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    for name, value in (
        ("summary.json", summary),
        ("run_manifest.json", manifest),
        ("profile_evolution.json", profile_evolution),
    ):
        (output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _speaker_id(speaker: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", speaker.casefold()).strip("_")


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return round(
        sum(collected) / len(collected), 6
    ) if collected else 0.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> Exp2UserModelingConfig:
    parser = argparse.ArgumentParser(
        description="Run revised REALTALK user-modeling evaluation"
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/exp2_user_modeling")
    parser.add_argument("--profile-sessions", type=int, default=3)
    parser.add_argument("--test-sessions", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--profile-max-tokens", type=int, default=16000)
    parser.add_argument("--max-eval-points-per-speaker", type=int, default=0)
    parser.add_argument("--operation-max-attempts", type=int, default=3)
    parser.add_argument("--chat", action="append", dest="chat_filter")
    parser.add_argument("--speaker", action="append", dest="speaker_filter")
    parser.add_argument("--fine-tuned-predictions")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    return Exp2UserModelingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        profile_sessions=args.profile_sessions,
        test_sessions=args.test_sessions,
        max_context_chars=args.max_context_chars,
        profile_max_tokens=args.profile_max_tokens,
        max_eval_points_per_speaker=args.max_eval_points_per_speaker,
        operation_max_attempts=args.operation_max_attempts,
        chat_filter=args.chat_filter,
        speaker_filter=args.speaker_filter,
        fine_tuned_predictions=args.fine_tuned_predictions,
        compute_bertscore=args.bertscore,
        continue_on_error=not args.stop_on_error,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    run_user_modeling_evaluation(parse_args())
