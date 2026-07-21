"""
Future State Prediction Module (Experiment 2 / RQ2)

Predicts the user's future emotional state based on:
  - Current session dialogue
  - User profile (optional)
  - Empathy alignment reasoning (optional)

Supports 4 experimental conditions:
  1. llm_only        — Zero-shot LLM (no user info)
  2. dialogue_history — Only conversation history
  3. user_profile    — Profile + history
  4. full_framework  — Full context with anti-neutral emphasis
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .utils import parse_json


# ---------------------------------------------------------------------------
# Prediction prompts — single template, conditional context blocks
# ---------------------------------------------------------------------------

PREDICTION_SYSTEM_PROMPT = """You are the future-state prediction module of a long-term companion agent. Predict the state most likely to continue into the user's next conversational turn.

Evidence priority:
1. The user's latest message: its explicit affect, wording, topic, and level of self-disclosure.
2. The immediately preceding dialogue trajectory.
3. Current state, when available.
4. The user profile as a calibrated prior for resolving ambiguity, never as a substitute for current evidence.

Reason silently in this order:
- Identify the emotion and sentiment expressed or strongly implied by the latest message.
- Decide whether the next turn is likely to continue, intensify, or shift that state. Continuation is the default unless context gives a concrete reason for change.
- Predict the topic at the same semantic granularity as the latest message (a specific 2-5 word phrase).
- Estimate intimacy from observable self-disclosure: 0.0-0.2 formal, 0.3-0.4 casual, 0.5-0.6 friendly/personal, 0.7-0.8 vulnerable/close, 0.9-1.0 deeply intimate.

Requirements:
- Predict exactly one canonical emotion.
- Prefer a specific supported emotion over neutral; use neutral only for genuinely affect-flat content.
- Do not let generic profile traits override a clear current signal.
- Confidence must reflect evidence strength, not optimism.

Emotion must be one of:
joy, sadness, anger, fear, surprise, disgust,
trust, anticipation, amusement, guilt, curiosity, neutral.

Return JSON only:
{
  "future_emotion": "...",
  "future_sentiment": "positive|negative|neutral",
  "future_intimacy": 0.0,
  "future_topic": "...",
  "confidence": 0.0
}"""

# Context block templates — assembled dynamically based on mode
_CONTEXT_BLOCKS = {
    "history": """CONVERSATION HISTORY (last {n_turns} turns):
{conversation_history}

""",
    "profile": """USER BACKGROUND (for reference only — focus on conversation tone):
{user_profile}

""",
    "state": """CURRENT STATE:
{current_state}

""",
}

# Per-mode anti-neutral hints (empty = no extra hint)
_MODE_HINTS = {
    "llm_only": "",
    "dialogue_history": "",
    "user_profile": "Focus on the conversation tone, NOT the profile, for your prediction.",
    "full_framework": "FULL-FRAMEWORK INFERENCE: Integrate the five-layer profile as a prior with the latest-message evidence. Infer current state from the latest message when CURRENT STATE is empty. Predict continuity by default: preserve the latest supported emotion, sentiment, intimacy band, and specific topic unless the recent trajectory clearly indicates a shift. Use profile layers only to resolve ambiguity and calibrate likely support needs; never average a clear emotion into neutral.",
}

_OUTPUT_INSTRUCTION = """USER'S LATEST MESSAGE:
"{user_message}"

Predict the state most likely to carry into the user's next turn. The latest message is the strongest evidence. Output JSON:
{{
  "future_emotion": "emotion label",
  "future_sentiment": "positive/negative/neutral",
  "future_intimacy": 0.0,
  "future_topic": "short topic phrase (2-5 words)",
  "confidence": 0.0
}}"""

# Which context blocks each mode includes
_MODE_BLOCKS = {
    "llm_only": [],
    "dialogue_history": ["history"],
    "user_profile": ["history", "profile"],
    "full_framework": ["history", "profile", "state"],
}


def build_user_prompt(
    mode: str,
    user_message: str,
    conversation_history: str = "",
    n_turns: int = 0,
    user_profile: str = "",
    current_state: str = "",
) -> str:
    """Build a prediction prompt by assembling context blocks for the given mode."""
    parts = []

    for block_name in _MODE_BLOCKS[mode]:
        template = _CONTEXT_BLOCKS[block_name]
        parts.append(template.format(
            conversation_history=conversation_history,
            n_turns=n_turns,
            user_profile=user_profile,
            current_state=current_state,
        ))

    hint = _MODE_HINTS.get(mode, "")
    if hint:
        parts.append(hint + "\n\n")

    parts.append(_OUTPUT_INSTRUCTION.format(user_message=user_message))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Prediction module
# ---------------------------------------------------------------------------

_VALID_MODES = set(_MODE_BLOCKS.keys())


class FutureStatePredictor:
    """Predicts user's future emotional/conversational state."""

    def __init__(self, llm: LLMClient, mode: str = "full_framework"):
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown prediction mode: {mode}. Must be one of {_VALID_MODES}")
        self.llm = llm
        self.mode = mode

    def predict(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        current_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Predict the user's future state."""
        # Format conversation history
        history_text = ""
        n_turns = 0
        if conversation_history:
            n_turns = min(len(conversation_history), 20)
            recent = conversation_history[-n_turns:]
            history_text = "\n".join(f"{t['speaker']}: {t['content']}" for t in recent)

        # Format profile
        profile_text = ""
        if user_profile:
            from .profile_utils import flatten_static_profile, state_axis
            if "state_axis" in user_profile:
                sp = state_axis(user_profile).get("static_profile", {})
                profile_text = json.dumps(flatten_static_profile(sp), ensure_ascii=False, indent=2)[:2000]
            else:
                profile_text = json.dumps(user_profile, ensure_ascii=False, indent=2)[:2000]

        # Format current state
        state_text = ""
        if current_state:
            state_text = json.dumps(current_state, ensure_ascii=False, indent=2)[:500]

        # Build prompt
        user_prompt = build_user_prompt(
            mode=self.mode,
            user_message=user_message,
            conversation_history=history_text,
            n_turns=n_turns,
            user_profile=profile_text,
            current_state=state_text,
        )

        try:
            result = parse_json(self.llm.chat(
                PREDICTION_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.3,
                max_tokens=500,
            ))
            if isinstance(result, dict):
                return result
            return {"error": "Prediction returned non-dict", "raw": str(result)[:200]}
        except Exception as e:
            return {"error": str(e)}


def compute_prediction_error(
    prediction: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute prediction error between predicted and actual future state."""
    errors: Dict[str, Any] = {}

    # Emotion accuracy (semantic similarity-based)
    pred_emotion = prediction.get("future_emotion", "").lower().strip()
    gt_emotion = ground_truth.get("actual_emotion", "").lower().strip()
    from .metrics import compute_emotion_similarity
    emo_sim = compute_emotion_similarity(pred_emotion, gt_emotion)
    errors["emotion_match"] = 1.0 if emo_sim >= 0.5 else emo_sim

    # Sentiment accuracy
    pred_sentiment = prediction.get("future_sentiment", "").lower().strip()
    gt_sentiment = ground_truth.get("actual_sentiment", "").lower().strip()
    errors["sentiment_match"] = 1.0 if pred_sentiment == gt_sentiment else 0.0

    # Intimacy error
    pred_intimacy = float(prediction.get("future_intimacy", 0.5))
    gt_intimacy = float(ground_truth.get("actual_intimacy", 0.5))
    errors["intimacy_error"] = abs(pred_intimacy - gt_intimacy)

    # Topic overlap
    pred_topic = prediction.get("future_topic", "").lower().strip()
    gt_topic = ground_truth.get("actual_topic", "").lower().strip()
    pred_words = set(pred_topic.split())
    gt_words = set(gt_topic.split())
    overlap = len(pred_words & gt_words) / len(gt_words) if pred_words and gt_words else 0.0
    errors["topic_overlap"] = round(overlap, 3)

    # Composite error
    errors["total_error"] = round(
        (1 - errors["emotion_match"]) * 0.3
        + (1 - errors["sentiment_match"]) * 0.3
        + errors["intimacy_error"] * 0.2
        + (1 - errors["topic_overlap"]) * 0.2,
        4
    )

    return errors
