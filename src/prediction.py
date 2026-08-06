"""Future-state prediction module for Experiment 2 / RQ2."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .experiments.exp2_schema import (
    FUTURE_STATE_RESPONSE_SCHEMA,
    normalize_future_state,
)
from .experiments.exp1_schema import EMOTION_LABELS, SENTIMENT_LABELS
from .llm_client import LLMClient


FUTURE_STATE_MAX_TOKENS = 4096
SCHEMA_REPAIR_ATTEMPTS = 2

TAXONOMY_REPAIR_SCHEMA = {
    "name": "exp2_future_state_taxonomy_repair",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "emotion_index": {
                "type": "integer",
                "enum": list(range(len(EMOTION_LABELS))),
            },
            "sentiment_index": {
                "type": "integer",
                "enum": list(range(len(SENTIMENT_LABELS))),
            },
        },
        "required": ["emotion_index", "sentiment_index"],
        "additionalProperties": False,
    },
}


# The core task wording is retained from the original Experiment 2.
PREDICTION_SYSTEM_PROMPT = """You predict the user's next conversational state.

Given the dialogue (and optionally a user profile), infer the most likely state in the user's next turn.

Requirements:
- Predict exactly one emotion.
- Intimacy means the disclosure level expressed in the next message itself,
  not general relationship closeness or how much is known in the profile.
"""

_CONTEXT_BLOCKS = {
    "recent_exchange": """MOST RECENT OBSERVED EXCHANGE:
{recent_exchange}

""",
    "history": """CONVERSATION HISTORY ({n_turns} observed turns):
{conversation_history}

""",
    "profile": """USER BACKGROUND:
{user_profile}

""",
    "state": """CURRENT STATE DERIVED FROM OBSERVED HISTORY:
{current_state}

""",
}

_OUTPUT_INSTRUCTION = """Predict the user's state in their next message."""

_MODE_BLOCKS = {
    "llm_only": ["recent_exchange"],
    "dialogue_history": ["history"],
    "user_profile": ["history", "profile"],
    "full_framework": ["history", "profile", "state"],
}
_VALID_MODES = set(_MODE_BLOCKS)


def build_user_prompt(
    mode: str,
    recent_exchange: str = "",
    conversation_history: str = "",
    n_turns: int = 0,
    user_profile: str = "",
    current_state: str = "",
) -> str:
    """Assemble the original four conditions without mode-specific hints."""
    parts = []
    for block_name in _MODE_BLOCKS[mode]:
        if block_name == "recent_exchange" and not recent_exchange:
            continue
        if block_name == "history" and not conversation_history:
            continue
        if block_name == "state" and not current_state:
            continue
        template = _CONTEXT_BLOCKS[block_name]
        parts.append(template.format(
            recent_exchange=recent_exchange,
            conversation_history=conversation_history,
            n_turns=n_turns,
            user_profile=user_profile,
            current_state=current_state,
        ))
    parts.append(_OUTPUT_INSTRUCTION)
    return "".join(parts)


class FutureStatePredictor:
    """Predict the next user state from caller-selected causal observations."""

    def __init__(self, llm: LLMClient, mode: str = "full_framework"):
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown prediction mode: {mode}. Must be one of {_VALID_MODES}"
            )
        self.llm = llm
        self.mode = mode
        self.last_provenance = _empty_prediction_provenance()

    def predict(
        self,
        recent_exchange: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        current_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.last_provenance = _empty_prediction_provenance()
        history = conversation_history or []
        recent = recent_exchange or []
        recent_text = "\n".join(
            f"{turn['speaker']}: {turn['content']}" for turn in recent
        )
        history_text = "\n".join(
            f"{turn['speaker']}: {turn['content']}" for turn in history
        )
        profile_text = (
            json.dumps(user_profile, ensure_ascii=False, indent=2)
            if user_profile else ""
        )
        state_text = (
            json.dumps(current_state, ensure_ascii=False, indent=2)
            if current_state else ""
        )
        user_prompt = build_user_prompt(
            mode=self.mode,
            recent_exchange=recent_text,
            conversation_history=history_text,
            n_turns=len(history),
            user_profile=profile_text,
            current_state=state_text,
        )
        attempt_prompt = user_prompt
        last_error: Exception | None = None
        for repair_attempt in range(SCHEMA_REPAIR_ATTEMPTS + 1):
            raw = self.llm.chat(
                PREDICTION_SYSTEM_PROMPT,
                attempt_prompt,
                temperature=0.0,
                max_tokens=FUTURE_STATE_MAX_TOKENS,
                response_schema=FUTURE_STATE_RESPONSE_SCHEMA,
            )
            try:
                result = json.loads(raw)
                normalized = normalize_future_state(result)
                self.last_provenance["schema_repair_attempts"] = repair_attempt
                return normalized
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if repair_attempt >= SCHEMA_REPAIR_ATTEMPTS:
                    break
                canonical = self._canonicalize_taxonomy_labels(
                    raw,
                    schema_repair_attempts=repair_attempt,
                )
                if canonical is not None:
                    return canonical
                attempt_prompt = _schema_repair_prompt(user_prompt, raw, exc)
        assert last_error is not None
        raise ValueError(
            "strict future-state response remained invalid after schema repairs: "
            f"{last_error}"
        ) from last_error

    def _canonicalize_taxonomy_labels(
        self,
        invalid_output: str,
        *,
        schema_repair_attempts: int,
    ) -> Dict[str, Any] | None:
        """Map only out-of-taxonomy labels while preserving the prediction."""
        try:
            value = json.loads(invalid_output)
        except json.JSONDecodeError:
            return None
        required = {
            "future_emotion",
            "future_sentiment",
            "future_intimacy",
            "future_topic",
        }
        if not isinstance(value, dict) or set(value) != required:
            return None

        emotion = str(value["future_emotion"]).strip().lower()
        sentiment = str(value["future_sentiment"]).strip().lower()
        if emotion in EMOTION_LABELS and sentiment in SENTIMENT_LABELS:
            return None

        prompt = (
            "Map the already predicted labels to the closest entries in the "
            "fixed REALTALK taxonomy. Do not reconsider the conversation, "
            "topic, or intimacy. Select indices only.\n\n"
            f"Predicted emotion label: {emotion}\n"
            f"Emotion indices: {_indexed_labels(EMOTION_LABELS)}\n\n"
            f"Predicted sentiment label: {sentiment}\n"
            f"Sentiment indices: {_indexed_labels(SENTIMENT_LABELS)}"
        )
        raw = self.llm.chat(
            "You are a deterministic taxonomy adapter.",
            prompt,
            temperature=0.0,
            max_tokens=256,
            response_schema=TAXONOMY_REPAIR_SCHEMA,
        )
        try:
            mapping = json.loads(raw)
            emotion_index = mapping["emotion_index"]
            sentiment_index = mapping["sentiment_index"]
            if isinstance(emotion_index, bool) or isinstance(sentiment_index, bool):
                return None
            value["future_emotion"] = EMOTION_LABELS[int(emotion_index)]
            value["future_sentiment"] = SENTIMENT_LABELS[int(sentiment_index)]
            normalized = normalize_future_state(value)
        except (KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError):
            return None
        print(
            "[future-state taxonomy repair] "
            f"emotion={emotion!r}->{normalized['emotion']!r} "
            f"sentiment={sentiment!r}->{normalized['sentiment']!r}"
        )
        self.last_provenance = {
            "taxonomy_repaired": True,
            "schema_repair_attempts": schema_repair_attempts,
            "repair_reason": "out_of_taxonomy_label",
            "raw_invalid_labels": {
                "emotion": emotion,
                "sentiment": sentiment,
            },
            "mapped_labels": {
                "emotion": normalized["emotion"],
                "sentiment": normalized["sentiment"],
            },
        }
        return normalized


def _empty_prediction_provenance() -> Dict[str, Any]:
    return {
        "taxonomy_repaired": False,
        "schema_repair_attempts": 0,
        "repair_reason": None,
        "raw_invalid_labels": None,
        "mapped_labels": None,
    }


def _schema_repair_prompt(
    original_prompt: str,
    invalid_output: str,
    error: Exception,
) -> str:
    """Retry only invalid structured output without silently mapping labels."""
    return (
        f"{original_prompt}\n\n"
        "VALIDATION RETRY:\n"
        f"The previous structured result failed validation: {error}\n"
        f"Previous result: {invalid_output}\n"
        "Return a corrected result for the same prediction. "
        "Do not add new fields or explanations.\n"
        f"Allowed emotion labels: {', '.join(EMOTION_LABELS)}.\n"
        f"Allowed sentiment labels: {', '.join(SENTIMENT_LABELS)}."
    )


def _indexed_labels(labels: tuple[str, ...]) -> str:
    return ", ".join(f"{index}={label}" for index, label in enumerate(labels))


def compute_prediction_error(
    prediction: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, float]:
    """Compute REALTALK-style per-attribute prediction outcomes."""
    pred_words = set(str(prediction.get("topic", "")).lower().split())
    gt_words = set(str(ground_truth.get("topic", "")).lower().split())
    topic_overlap = (
        len(pred_words & gt_words) / len(gt_words)
        if pred_words and gt_words else 0.0
    )
    return {
        "emotion_accuracy": float(
            prediction.get("emotion") == ground_truth.get("emotion")
        ),
        "sentiment_accuracy": float(
            prediction.get("sentiment") == ground_truth.get("sentiment")
        ),
        "intimacy_absolute_difference": round(
            abs(
                float(prediction.get("intimacy", 0.0))
                - float(ground_truth.get("intimacy", 0.0))
            ),
            4,
        ),
        "topic_consistency": round(topic_overlap, 4),
    }
