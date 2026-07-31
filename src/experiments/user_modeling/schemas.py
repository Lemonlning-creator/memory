"""Strict contracts used by the revised Experiment 2."""
from __future__ import annotations

from typing import Any

from ..exp1_schema import EMOTION_LABELS, SENTIMENT_LABELS

CURRENT_STATE_SCHEMA: dict[str, Any] = {
    "name": "exp2_current_user_understanding",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "emotion": {"type": "string", "enum": list(EMOTION_LABELS)},
            "sentiment": {"type": "string", "enum": list(SENTIMENT_LABELS)},
            "topic": {"type": "string", "minLength": 1, "maxLength": 160},
        },
        "required": ["emotion", "sentiment", "topic"],
        "additionalProperties": False,
    },
}

TOPIC_REFERENCE_SCHEMA: dict[str, Any] = {
    "name": "exp2_reference_topic",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "minLength": 1, "maxLength": 160},
        },
        "required": ["topic"],
        "additionalProperties": False,
    },
}

REFLECTIVENESS_SCHEMA: dict[str, Any] = {
    "name": "exp2_realtalk_reflectiveness",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reflective": {"type": "boolean"},
        },
        "required": ["reflective"],
        "additionalProperties": False,
    },
}

GROUNDING_SCHEMA: dict[str, Any] = {
    "name": "exp2_realtalk_grounding",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "grounding": {"type": "boolean"},
        },
        "required": ["grounding"],
        "additionalProperties": False,
    },
}

EMPATHY_SCHEMA: dict[str, Any] = {
    "name": "exp2_realtalk_epitome_empathy",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "emotional_reaction": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2,
            },
            "interpretation": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2,
            },
            "exploration": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2,
            },
        },
        "required": [
            "emotional_reaction",
            "interpretation",
            "exploration",
        ],
        "additionalProperties": False,
    },
}


def normalize_current_state(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "emotion", "sentiment", "topic"
    }:
        raise ValueError("current-state output must contain emotion, sentiment, topic")
    emotion = str(value["emotion"]).strip().lower()
    sentiment = str(value["sentiment"]).strip().lower()
    topic = str(value["topic"]).strip()
    if emotion not in EMOTION_LABELS:
        raise ValueError(f"unsupported emotion label: {emotion}")
    if sentiment not in SENTIMENT_LABELS:
        raise ValueError(f"unsupported sentiment label: {sentiment}")
    if not topic:
        raise ValueError("topic must not be empty")
    return {"emotion": emotion, "sentiment": sentiment, "topic": topic}


def normalize_topic_reference(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"topic"}:
        raise ValueError("topic reference must contain exactly one topic field")
    topic = str(value["topic"]).strip()
    if not topic:
        raise ValueError("topic must not be empty")
    return {"topic": topic}


def normalize_reflectiveness(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict) or set(value) != {"reflective"}:
        raise ValueError("reflectiveness result must contain exactly reflective")
    if not isinstance(value["reflective"], bool):
        raise ValueError("reflective must be boolean")
    return {"reflective": value["reflective"]}


def normalize_grounding(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict) or set(value) != {"grounding"}:
        raise ValueError("grounding result must contain exactly grounding")
    if not isinstance(value["grounding"], bool):
        raise ValueError("grounding must be boolean")
    return {"grounding": value["grounding"]}


def normalize_empathy(value: Any) -> dict[str, int]:
    fields = {"emotional_reaction", "interpretation", "exploration"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("empathy result must contain all EPITOME components")
    normalized = {}
    for field in fields:
        score = value[field]
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError(f"{field} must be an integer")
        if not 0 <= score <= 2:
            raise ValueError(f"{field} must be in [0, 2]")
        normalized[field] = score
    return normalized
