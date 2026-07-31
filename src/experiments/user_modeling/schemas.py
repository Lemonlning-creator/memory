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
