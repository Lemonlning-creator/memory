"""
Shared utilities for Experiments 1-5.

Provides common functions:
  - Data loading and case selection
  - Profile extraction (explicit / flat / self-model)
  - Ground-truth annotation (emotion, sentiment, topic, intimacy)
  - Result aggregation and saving
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..llm_client import LLMClient
from ..utils import load_json, save_json, parse_json
from ..prompts.prompt_loader import (
    PROFILE_EXTRACTION_SYSTEM_PROMPT,
    PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
    FLAT_PROFILE_EXTRACTION_SYSTEM_PROMPT,
    FLAT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
    SELF_MODEL_SYSTEM_PROMPT,
    SELF_MODEL_USER_PROMPT_TEMPLATE,
    EMOTION_SENTIMENT_EXTRACTION_SYSTEM_PROMPT,
    EMOTION_SENTIMENT_EXTRACTION_USER_PROMPT_TEMPLATE,
    TOPIC_EXTRACTION_SYSTEM_PROMPT,
    TOPIC_EXTRACTION_USER_PROMPT_TEMPLATE,
    INTIMACY_EXTRACTION_SYSTEM_PROMPT,
    INTIMACY_EXTRACTION_USER_PROMPT_TEMPLATE,
    PROFILE_CONSISTENCY_SYSTEM_PROMPT,
    PROFILE_CONSISTENCY_USER_PROMPT_TEMPLATE,
    PERSONA_CONSISTENCY_SYSTEM_PROMPT,
    PERSONA_CONSISTENCY_USER_PROMPT_TEMPLATE,
)
from .persona_simulation import (
    detect_speakers,
    flatten_messages,
    session_keys,
    format_conversation_history,
    condense_profile,
    condense_persona,
)


def robust_parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON with fallback for common LLM formatting errors."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = re.sub(r'"\s*\n\s*"', '",\n"', text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return {"error": "JSON parse failed", "raw": text[:300]}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chat_files(dataset_dir: str, chat_filter: Optional[List[str]] = None) -> List[Path]:
    """Load chat file paths from dataset directory."""
    dataset_path = Path(dataset_dir)
    chat_files = sorted(dataset_path.glob("Chat_*.json"))
    if chat_filter:
        chat_files = [f for f in chat_files if f.stem in chat_filter]
    return chat_files


def get_session_messages(chat: Dict[str, Any], session_key: str) -> List[Dict[str, Any]]:
    """Get all messages from a specific session."""
    return chat.get(session_key, [])


def split_into_sessions(chat: Dict[str, Any]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Split chat into list of (session_key, messages) tuples."""
    keys = session_keys(chat)
    return [(k, chat[k]) for k in keys]


# ---------------------------------------------------------------------------
# Profile extraction methods
# ---------------------------------------------------------------------------

def build_corpus(chat: Dict[str, Any], speaker: str) -> str:
    """Build a text corpus of one speaker's messages from a conversation."""
    turns = flatten_messages(chat)
    lines = []
    for t in turns:
        if t["speaker"] == speaker:
            lines.append(f"{speaker}: {t['content']}")
        else:
            lines.append(f"Partner: {t['content']}")
    return "\n".join(lines[-200:])


def extract_explicit_profile(llm: LLMClient, chat: Dict[str, Any], user_speaker: str) -> Dict[str, Any]:
    """Extract the standard 5-layer hierarchical profile (our method)."""
    corpus = build_corpus(chat, user_speaker)
    user_prompt = PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        user_name=user_speaker, corpus=corpus
    )
    raw = llm.chat(
        PROFILE_EXTRACTION_SYSTEM_PROMPT.format(user_name=user_speaker),
        user_prompt,
        temperature=0.3,
        max_tokens=3000,
    )
    return robust_parse_json(raw)


def extract_flat_profile(llm: LLMClient, chat: Dict[str, Any], user_speaker: str) -> Dict[str, Any]:
    """Extract a flat (non-hierarchical) profile (baseline for Exp 1)."""
    corpus = build_corpus(chat, user_speaker)
    user_prompt = FLAT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        user_name=user_speaker, corpus=corpus
    )
    raw = llm.chat(
        FLAT_PROFILE_EXTRACTION_SYSTEM_PROMPT.format(user_name=user_speaker),
        user_prompt,
        temperature=0.3,
        max_tokens=3000,
    )
    return robust_parse_json(raw)


def infer_self_model(
    llm: LLMClient,
    user_message: str,
    conversation_history: str,
    persona: Dict[str, Any],
) -> Dict[str, Any]:
    """Use Self-model Other Modeling to infer user state (baseline for Exp 1).

    The agent projects its own persona onto the user instead of building
    an explicit user model.
    """
    persona_text = condense_persona(persona)
    user_prompt = SELF_MODEL_USER_PROMPT_TEMPLATE.format(
        conversation_history=conversation_history[:3000],
        user_message=user_message,
        agent_persona=persona_text,
    )
    raw = llm.chat(
        SELF_MODEL_SYSTEM_PROMPT.format(agent_persona=persona_text),
        user_prompt,
        temperature=0.3,
        max_tokens=500,
    )
    return robust_parse_json(raw)


# ---------------------------------------------------------------------------
# Ground-truth annotation
# ---------------------------------------------------------------------------

def extract_emotion_sentiment(llm: LLMClient, message: str) -> Dict[str, Any]:
    """Extract emotion and sentiment from a message (for ground truth annotation)."""
    user_prompt = EMOTION_SENTIMENT_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        user_message=message
    )
    raw = llm.chat(
        EMOTION_SENTIMENT_EXTRACTION_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.1,
        max_tokens=200,
    )
    return robust_parse_json(raw)


def extract_topic(llm: LLMClient, message: str) -> Dict[str, Any]:
    """Extract the main topic from a message."""
    user_prompt = TOPIC_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        user_message=message
    )
    raw = llm.chat(
        TOPIC_EXTRACTION_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.1,
        max_tokens=200,
    )
    return robust_parse_json(raw)


def extract_intimacy(llm: LLMClient, message: str) -> Dict[str, Any]:
    """Extract intimacy level from a message."""
    user_prompt = INTIMACY_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        message=message
    )
    raw = llm.chat(
        INTIMACY_EXTRACTION_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.1,
        max_tokens=200,
    )
    return robust_parse_json(raw)


# ---------------------------------------------------------------------------
# Consistency evaluation
# ---------------------------------------------------------------------------

def evaluate_profile_consistency(
    llm: LLMClient,
    profile_a: Dict[str, Any],
    profile_b: Dict[str, Any],
    source_a: str = "conversation_a",
    source_b: str = "conversation_b",
    speaker_name: str = "user",
) -> Dict[str, Any]:
    """Evaluate consistency between two profiles using LLM judge."""
    user_prompt = PROFILE_CONSISTENCY_USER_PROMPT_TEMPLATE.format(
        source_a=source_a,
        profile_a_json=json.dumps(profile_a, ensure_ascii=False, indent=2),
        source_b=source_b,
        profile_b_json=json.dumps(profile_b, ensure_ascii=False, indent=2),
        speaker_name=speaker_name,
    )
    raw = llm.chat(
        PROFILE_CONSISTENCY_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.1,
        max_tokens=1000,
    )
    return robust_parse_json(raw)


# ---------------------------------------------------------------------------
# Evaluation point construction
# ---------------------------------------------------------------------------

def build_eval_points_at_sessions(
    chat: Dict[str, Any],
    agent_speaker: str,
    user_speaker: str,
    min_context_sessions: int = 2,
) -> List[Dict[str, Any]]:
    """Build evaluation points at session boundaries.

    Each eval point: agent has observed sessions 1..N,
    must predict/understand user's state at session N+1.
    """
    turns = flatten_messages(chat)
    sessions = session_keys(chat)
    eval_points: List[Dict[str, Any]] = []

    for boundary_idx in range(1, len(sessions)):
        context_sessions = sessions[:boundary_idx]
        target_session = sessions[boundary_idx]

        if boundary_idx < min_context_sessions:
            continue

        # Get first user message in target session
        target_msg = None
        for turn in turns:
            if turn["session_id"] == target_session and turn["speaker"] == user_speaker:
                target_msg = turn
                break

        if target_msg is None:
            continue

        # Context: all messages before the target message
        context_turns = [
            t for t in turns
            if t["session_index"] < target_msg["session_index"]
            or (t["session_index"] == target_msg["session_index"]
                and t["message_index"] < target_msg["message_index"])
        ]

        # Also get user messages in target session for ground truth annotation
        target_session_user_msgs = [
            t for t in turns
            if t["session_id"] == target_session and t["speaker"] == user_speaker
        ]

        eval_points.append({
            "eval_id": f"session_boundary_{boundary_idx}",
            "boundary_idx": boundary_idx,
            "context_sessions": context_sessions,
            "target_session": target_session,
            "target_message": target_msg["content"],
            "target_session_user_msgs": [m["content"] for m in target_session_user_msgs[:5]],
            "context_turns": context_turns,
        })

    return eval_points


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_experiment_results(
    output_dir: str,
    experiment_name: str,
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[str, str]:
    """Save experiment results and summary to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = output_path / f"{experiment_name}_results.json"
    save_json(str(results_path), {
        "results": results,
        "config": config,
    })

    summary_path = output_path / f"{experiment_name}_summary.json"
    save_json(str(summary_path), summary)

    return str(results_path), str(summary_path)
