"""Strict schemas for the REALTALK V13 progressive persona actor."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .realtalk_ours_schemas import (
    CONFIDENCE,
    ORIENTATIONS,
    PROFILE_LAYERS,
    SELF_DOMAIN_SCHEMA,
    _boolean,
    _enum,
    _exact_object,
    _integer,
    _number,
    _string_array_schema,
    _strings,
    _text,
    normalize_self_domain,
)


TURN_TRIGGERS = (
    "session-opening",
    "after-direct-question",
    "after-partner-disclosure",
    "after-partner-statement",
    "after-closing",
)
TURN_OBLIGATIONS = (
    "answer",
    "react",
    "self-update",
    "ask",
    "open",
    "close",
    "topic-shift",
)
PRIMARY_MOVES = TURN_OBLIGATIONS
COMPANION_MOVES = (
    "none",
    "brief-reaction",
    "self-disclose",
    "reciprocal-question",
)
QUESTION_PLANS = (
    "none",
    "opening-check-in",
    "reciprocal",
    "clarify",
    "follow-up",
)
OPEN_OBLIGATIONS = (
    "none",
    "open-session",
    "answer-current-question",
    "answer-earlier-unanswered-question",
    "respond-current-disclosure",
    "continue-current-topic",
)


def _conditional_stat_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "observations": {"type": "integer", "minimum": 0},
            "question_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "first_person_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "reflective_marker_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "mean_characters": {"type": "number", "minimum": 0},
            "median_characters": {"type": "number", "minimum": 0},
            "median_merged_bubbles": {"type": "number", "minimum": 0},
        },
        "required": [
            "observations",
            "question_rate",
            "first_person_rate",
            "reflective_marker_rate",
            "mean_characters",
            "median_characters",
            "median_merged_bubbles",
        ],
        "additionalProperties": False,
    }


V13_SELF_DOMAIN_SCHEMA = deepcopy(SELF_DOMAIN_SCHEMA)
V13_SELF_DOMAIN_SCHEMA["name"] = "realtalk_ours_v13_self_domain_v1"
V13_SELF_DOMAIN_SCHEMA["schema"]["properties"].update({
    "cross_partner_transfer": {
        "type": "object",
        "properties": {
            "portable_patterns": _string_array_schema(),
            "ca_partner_specific_patterns": _string_array_schema(),
            "current_history_override": {"type": "string"},
        },
        "required": [
            "portable_patterns",
            "ca_partner_specific_patterns",
            "current_history_override",
        ],
        "additionalProperties": False,
    },
    "conditional_behavior": {
        "type": "array",
        "minItems": len(TURN_TRIGGERS),
        "maxItems": len(TURN_TRIGGERS),
        "items": {
            "type": "object",
            "properties": {
                "trigger": {"type": "string", "enum": list(TURN_TRIGGERS)},
                "observed_pattern": {"type": "string"},
                "question_tendency": {"type": "string"},
                "reflection_tendency": {"type": "string"},
                "message_shape": {"type": "string"},
                "confidence": {"type": "string", "enum": list(CONFIDENCE)},
            },
            "required": [
                "trigger",
                "observed_pattern",
                "question_tendency",
                "reflection_tendency",
                "message_shape",
                "confidence",
            ],
            "additionalProperties": False,
        },
    },
    "conditional_statistics": {
        "type": "object",
        "properties": {
            trigger: _conditional_stat_schema() for trigger in TURN_TRIGGERS
        },
        "required": list(TURN_TRIGGERS),
        "additionalProperties": False,
    },
})
V13_SELF_DOMAIN_SCHEMA["schema"]["required"].extend([
    "cross_partner_transfer",
    "conditional_behavior",
    "conditional_statistics",
])


V13_DECISION_SCHEMA = {
    "name": "realtalk_ours_v13_decision_v1",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "situation": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "partner_move": {"type": "string"},
                    "turn_obligation": {"type": "string", "enum": list(TURN_OBLIGATIONS)},
                    "open_obligation": {"type": "string", "enum": list(OPEN_OBLIGATIONS)},
                    "obligation_source_turn_id": {"type": "string"},
                    "explicit_affect": {"type": "string"},
                    "support_request": {"type": "boolean"},
                    "uncertainty": {"type": "string", "enum": list(CONFIDENCE)},
                },
                "required": [
                    "topic",
                    "partner_move",
                    "turn_obligation",
                    "open_obligation",
                    "obligation_source_turn_id",
                    "explicit_affect",
                    "support_request",
                    "uncertainty",
                ],
                "additionalProperties": False,
            },
            "relevant_user_domain": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "layer": {"type": "string", "enum": list(PROFILE_LAYERS)},
                        "value": {"type": "string"},
                    },
                    "required": ["layer", "value"],
                    "additionalProperties": False,
                },
            },
            "alignment": {
                "type": "object",
                "properties": {
                    "orientation": {"type": "string", "enum": list(ORIENTATIONS)},
                    "lambda_trace": {"type": "number", "minimum": 0, "maximum": 1},
                    "decision_basis": {"type": "string"},
                },
                "required": ["orientation", "lambda_trace", "decision_basis"],
                "additionalProperties": False,
            },
            "behavior_policy": {
                "type": "object",
                "properties": {
                    "primary_move": {"type": "string", "enum": list(PRIMARY_MOVES)},
                    "companion_move": {"type": "string", "enum": list(COMPANION_MOVES)},
                    "reflection_depth": {
                        "type": "string",
                        "enum": ["surface", "brief-reflective"],
                    },
                    "question_plan": {"type": "string", "enum": list(QUESTION_PLANS)},
                    "question_target": {"type": "string"},
                    "relational_register": {
                        "type": "string",
                        "enum": ["casual-neutral", "familiar-warm", "intimate-supportive"],
                    },
                    "message_shape": {
                        "type": "string",
                        "enum": ["short", "typical", "multi-bubble"],
                    },
                    "content_direction": {"type": "string"},
                    "tone": {"type": "string"},
                },
                "required": [
                    "primary_move",
                    "companion_move",
                    "reflection_depth",
                    "question_plan",
                    "question_target",
                    "relational_register",
                    "message_shape",
                    "content_direction",
                    "tone",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["situation", "relevant_user_domain", "alignment", "behavior_policy"],
        "additionalProperties": False,
    },
}


def normalize_v13_self_domain(value: Any) -> dict[str, Any]:
    root = _exact_object(value, V13_SELF_DOMAIN_SCHEMA["schema"], "self_domain")
    legacy_keys = SELF_DOMAIN_SCHEMA["schema"]["required"]
    result = normalize_self_domain({key: root[key] for key in legacy_keys})
    transfer_schema = V13_SELF_DOMAIN_SCHEMA["schema"]["properties"]["cross_partner_transfer"]
    transfer = _exact_object(root["cross_partner_transfer"], transfer_schema, "cross_partner_transfer")
    result["cross_partner_transfer"] = {
        "portable_patterns": _strings(transfer["portable_patterns"], "portable_patterns"),
        "ca_partner_specific_patterns": _strings(
            transfer["ca_partner_specific_patterns"], "ca_partner_specific_patterns"
        ),
        "current_history_override": _text(
            transfer["current_history_override"], "current_history_override"
        ),
    }
    item_schema = V13_SELF_DOMAIN_SCHEMA["schema"]["properties"]["conditional_behavior"]["items"]
    if not isinstance(root["conditional_behavior"], list):
        raise ValueError("conditional_behavior must be an array")
    seen: set[str] = set()
    behaviors = []
    for index, raw in enumerate(root["conditional_behavior"]):
        item = _exact_object(raw, item_schema, f"conditional_behavior[{index}]")
        trigger = _enum(item["trigger"], TURN_TRIGGERS, f"conditional_behavior[{index}].trigger")
        if trigger in seen:
            raise ValueError(f"duplicate conditional behavior trigger: {trigger}")
        seen.add(trigger)
        behaviors.append({
            "trigger": trigger,
            "observed_pattern": _text(item["observed_pattern"], "observed_pattern"),
            "question_tendency": _text(item["question_tendency"], "question_tendency"),
            "reflection_tendency": _text(item["reflection_tendency"], "reflection_tendency"),
            "message_shape": _text(item["message_shape"], "message_shape"),
            "confidence": _enum(item["confidence"], CONFIDENCE, "confidence"),
        })
    if seen != set(TURN_TRIGGERS):
        raise ValueError("conditional_behavior must cover every trigger exactly once")
    result["conditional_behavior"] = behaviors
    stats_schema = V13_SELF_DOMAIN_SCHEMA["schema"]["properties"]["conditional_statistics"]
    stats = _exact_object(root["conditional_statistics"], stats_schema, "conditional_statistics")
    result["conditional_statistics"] = {
        trigger: _normalize_conditional_stat(stats[trigger], trigger)
        for trigger in TURN_TRIGGERS
    }
    return result


def _normalize_conditional_stat(value: Any, trigger: str) -> dict[str, Any]:
    schema = _conditional_stat_schema()
    item = _exact_object(value, schema, f"conditional_statistics.{trigger}")
    return {
        "observations": _integer(
            item["observations"], f"conditional_statistics.{trigger}.observations"
        ),
        **{
            key: _number(item[key], f"conditional_statistics.{trigger}.{key}")
            for key in (
                "question_rate",
                "first_person_rate",
                "reflective_marker_rate",
                "mean_characters",
                "median_characters",
                "median_merged_bubbles",
            )
        },
    }


def normalize_v13_decision(value: Any) -> dict[str, Any]:
    schema = V13_DECISION_SCHEMA["schema"]
    root = _exact_object(value, schema, "decision")
    situation = _exact_object(root["situation"], schema["properties"]["situation"], "situation")
    alignment = _exact_object(root["alignment"], schema["properties"]["alignment"], "alignment")
    policy = _exact_object(
        root["behavior_policy"], schema["properties"]["behavior_policy"], "behavior_policy"
    )
    relevant = []
    fact_schema = schema["properties"]["relevant_user_domain"]["items"]
    if not isinstance(root["relevant_user_domain"], list) or len(root["relevant_user_domain"]) > 2:
        raise ValueError("relevant_user_domain must contain at most two facts")
    for index, raw in enumerate(root["relevant_user_domain"]):
        item = _exact_object(raw, fact_schema, f"relevant_user_domain[{index}]")
        relevant.append({
            "layer": _enum(item["layer"], PROFILE_LAYERS, "relevant layer"),
            "value": _text(item["value"], "relevant value"),
        })
    normalized = {
        "situation": {
            "topic": _text(situation["topic"], "situation.topic", allow_empty=True),
            "partner_move": _text(situation["partner_move"], "situation.partner_move", allow_empty=True),
            "turn_obligation": _enum(
                situation["turn_obligation"], TURN_OBLIGATIONS, "situation.turn_obligation"
            ),
            "open_obligation": _enum(
                situation["open_obligation"], OPEN_OBLIGATIONS, "situation.open_obligation"
            ),
            "obligation_source_turn_id": _text(
                situation["obligation_source_turn_id"],
                "situation.obligation_source_turn_id",
                allow_empty=True,
            ),
            "explicit_affect": _text(
                situation["explicit_affect"], "situation.explicit_affect", allow_empty=True
            ),
            "support_request": _boolean(situation["support_request"], "support_request"),
            "uncertainty": _enum(situation["uncertainty"], CONFIDENCE, "uncertainty"),
        },
        "relevant_user_domain": relevant,
        "alignment": {
            "orientation": _enum(alignment["orientation"], ORIENTATIONS, "orientation"),
            "lambda_trace": _number(alignment["lambda_trace"], "lambda_trace"),
            "decision_basis": _text(alignment["decision_basis"], "decision_basis"),
        },
        "behavior_policy": {
            "primary_move": _enum(policy["primary_move"], PRIMARY_MOVES, "primary_move"),
            "companion_move": _enum(policy["companion_move"], COMPANION_MOVES, "companion_move"),
            "reflection_depth": _enum(
                policy["reflection_depth"], ("surface", "brief-reflective"), "reflection_depth"
            ),
            "question_plan": _enum(policy["question_plan"], QUESTION_PLANS, "question_plan"),
            "question_target": _text(policy["question_target"], "question_target", allow_empty=True),
            "relational_register": _enum(
                policy["relational_register"],
                ("casual-neutral", "familiar-warm", "intimate-supportive"),
                "relational_register",
            ),
            "message_shape": _enum(
                policy["message_shape"], ("short", "typical", "multi-bubble"), "message_shape"
            ),
            "content_direction": _text(policy["content_direction"], "content_direction"),
            "tone": _text(policy["tone"], "tone"),
        },
    }
    _validate_policy(normalized)
    return normalized


def _validate_policy(value: dict[str, Any]) -> None:
    alignment = value["alignment"]
    orientation = alignment["orientation"]
    trace = alignment["lambda_trace"]
    ranges = {
        "self-led": (0.0, 0.35),
        "balanced": (0.36, 0.70),
        "partner-adaptive": (0.71, 1.0),
    }
    low, high = ranges[orientation]
    if not low <= trace <= high:
        raise ValueError(f"lambda_trace {trace} does not match {orientation}")
    policy = value["behavior_policy"]
    question_plan = policy["question_plan"]
    target = policy["question_target"]
    if question_plan == "none" and target:
        raise ValueError("question_target must be empty when no question is planned")
    if question_plan != "none" and not target:
        raise ValueError("a planned question requires question_target")
    if policy["primary_move"] == "ask" and question_plan not in {"clarify", "follow-up"}:
        raise ValueError("ask primary move requires clarify or follow-up")
    if question_plan in {"clarify", "follow-up"} and policy["primary_move"] != "ask":
        raise ValueError("clarify or follow-up requires ask primary move")
    if policy["companion_move"] == "reciprocal-question" and question_plan != "reciprocal":
        raise ValueError("reciprocal companion requires reciprocal question plan")
    if question_plan == "reciprocal" and policy["companion_move"] != "reciprocal-question":
        raise ValueError("reciprocal question plan requires reciprocal companion")
    if question_plan == "opening-check-in" and policy["primary_move"] != "open":
        raise ValueError("opening check-in requires open primary move")
    if value["situation"]["turn_obligation"] != policy["primary_move"]:
        raise ValueError("turn_obligation must equal primary_move")
