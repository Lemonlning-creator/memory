"""
Future State Prediction Module (Experiment 2 / RQ2)

Predicts the user's future emotional state based on:
  - Current session dialogue
  - User profile (optional)
  - Empathy alignment reasoning (optional)

Supports 4 experimental conditions:
  1. llm_only        — Zero-shot LLM (no user info)
  2. dialogue_history — Only conversation history
  3. user_profile    — Profile + history, no prediction module
  4. full_framework  — Full deep empathy framework with structured prediction
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .utils import parse_json


# ---------------------------------------------------------------------------
# Prediction prompts
# ---------------------------------------------------------------------------

PREDICTION_SYSTEM_PROMPT = """You are a user state prediction module. Your task is to predict the user's future emotional and conversational state based on the available information.

You must predict:
1. Future Emotion: What emotion will the user likely feel next?
   MUST use one of: joy, sadness, anger, fear, surprise, disgust, trust, anticipation, amusement, guilt, curiosity, neutral
2. Future Sentiment: positive / negative / neutral
3. Future Intimacy: How intimate/personal will the next exchange be? (0.0 = very distant/formal, 1.0 = very intimate/personal)
4. Future Topic: What topic will the user likely discuss next? (short phrase, 2-5 words)

Be specific and concise. Output ONLY valid JSON."""

PREDICTION_USER_PROMPT_TEMPLATES = {
    "llm_only": """The user sent the following message:
"{user_message}"

Predict the user's next emotional state with NO other information.
Output JSON:
{{
  "future_emotion": "emotion label",
  "future_sentiment": "positive/negative/neutral",
  "future_intimacy": 0.0,
  "future_topic": "predicted topic",
  "confidence": 0.0
}}""",

    "dialogue_history": """CONVERSATION HISTORY (last {n_turns} turns):
{conversation_history}

USER'S LATEST MESSAGE:
"{user_message}"

Based on the conversation history, predict the user's next emotional state.
Output JSON:
{{
  "future_emotion": "emotion label",
  "future_sentiment": "positive/negative/neutral",
  "future_intimacy": 0.0,
  "future_topic": "predicted topic",
  "confidence": 0.0
}}""",

    "user_profile": """CONVERSATION HISTORY (last {n_turns} turns):
{conversation_history}

USER'S LATEST MESSAGE:
"{user_message}"

USER PROFILE:
{user_profile}

Based on the conversation history AND user profile, predict the user's next emotional state.
Output JSON:
{{
  "future_emotion": "emotion label",
  "future_sentiment": "positive/negative/neutral",
  "future_intimacy": 0.0,
  "future_topic": "predicted topic",
  "confidence": 0.0
}}""",

    "full_framework": """CONVERSATION HISTORY (last {n_turns} turns):
{conversation_history}

USER'S LATEST MESSAGE:
"{user_message}"

USER PROFILE:
{user_profile}

CURRENT STATE:
{current_state}

Based on all available information, predict the user's next emotional state.
Focus on what is most likely — do not overthink.

IMPORTANT: Future emotion MUST be one of: joy, sadness, anger, fear, surprise, disgust, trust, anticipation, amusement, guilt, curiosity, neutral

Output JSON:
{{
  "future_emotion": "one emotion from the list above",
  "future_sentiment": "positive/negative/neutral",
  "future_intimacy": 0.0,
  "future_topic": "short topic phrase (2-5 words)"
}}"""
}


# ---------------------------------------------------------------------------
# Prediction module
# ---------------------------------------------------------------------------

class FutureStatePredictor:
    """Predicts user's future emotional/conversational state.

    Supports 4 experimental conditions for RQ2.
    """

    def __init__(self, llm: LLMClient, mode: str = "full_framework"):
        if mode not in PREDICTION_USER_PROMPT_TEMPLATES:
            raise ValueError(f"Unknown prediction mode: {mode}")
        self.llm = llm
        self.mode = mode

    def predict(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        empathy_reasoning: Optional[Dict[str, Any]] = None,
        current_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Predict the user's future state.

        Args:
            user_message: The user's latest message.
            conversation_history: List of {speaker, content} dicts.
            user_profile: The user's profile (5-layer or flat).
            empathy_reasoning: Output from empathy alignment reasoning.
            current_state: The user's current state snapshot.

        Returns:
            Dict with keys: future_emotion, future_sentiment, future_intimacy,
            future_topic, confidence, and optionally reasoning.
        """
        # Format conversation history
        history_text = ""
        n_turns = 0
        if conversation_history:
            n_turns = min(len(conversation_history), 20)
            recent = conversation_history[-n_turns:]
            lines = [f"{t['speaker']}: {t['content']}" for t in recent]
            history_text = "\n".join(lines)

        # Format profile
        profile_text = ""
        if user_profile:
            from .profile_utils import flatten_static_profile, state_axis
            if "state_axis" in user_profile:
                sp = state_axis(user_profile).get("static_profile", {})
                profile_text = json.dumps(flatten_static_profile(sp), ensure_ascii=False, indent=2)[:2000]
            else:
                profile_text = json.dumps(user_profile, ensure_ascii=False, indent=2)[:2000]

        # Format empathy reasoning
        reasoning_text = ""
        if empathy_reasoning:
            reasoning_text = json.dumps(empathy_reasoning, ensure_ascii=False, indent=2)[:1500]

        # Format current state
        state_text = ""
        if current_state:
            state_text = json.dumps(current_state, ensure_ascii=False, indent=2)[:500]

        # Build prompt based on mode
        template = PREDICTION_USER_PROMPT_TEMPLATES[self.mode]
        user_prompt = template.format(
            user_message=user_message,
            conversation_history=history_text,
            n_turns=n_turns,
            user_profile=profile_text,
            empathy_reasoning=reasoning_text,
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
    """Compute prediction error between predicted and actual future state.

    Args:
        prediction: Dict with future_emotion, future_sentiment, future_intimacy, future_topic.
        ground_truth: Dict with actual_emotion, actual_sentiment, actual_intimacy, actual_topic.

    Returns:
        Dict with per-dimension errors and total error score.
    """
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

    # Intimacy error (MSE-like, lower is better)
    pred_intimacy = float(prediction.get("future_intimacy", 0.5))
    gt_intimacy = float(ground_truth.get("actual_intimacy", 0.5))
    errors["intimacy_error"] = abs(pred_intimacy - gt_intimacy)

    # Topic overlap (simple keyword overlap for now; can be upgraded to embedding similarity)
    pred_topic = prediction.get("future_topic", "").lower().strip()
    gt_topic = ground_truth.get("actual_topic", "").lower().strip()
    pred_words = set(pred_topic.split())
    gt_words = set(gt_topic.split())
    if pred_words and gt_words:
        overlap = len(pred_words & gt_words) / len(gt_words)
    else:
        overlap = 0.0
    errors["topic_overlap"] = round(overlap, 3)

    # Composite error (lower is better): average of (1 - accuracy) for categorical + continuous errors
    errors["total_error"] = round(
        (1 - errors["emotion_match"]) * 0.3
        + (1 - errors["sentiment_match"]) * 0.3
        + errors["intimacy_error"] * 0.2
        + (1 - errors["topic_overlap"]) * 0.2,
        4
    )

    return errors
