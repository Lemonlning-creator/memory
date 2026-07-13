"""
Experiments 1b & 1c: State Axis + Context Axis Validation (Ablation)

1b - STATE AXIS: Validates that the "current state" captures transient
     emotional states that change over time, distinct from the stable profile.
     Method: Extract current-state snapshots at multiple points in each conversation.
     Show that (a) states vary significantly across time points, and
     (b) states correlate with the emotional content of nearby messages.

1c - CONTEXT AXIS: Validates that different conversation contexts surface
     context-specific traits. Method: Extract context-specific profiles from
     different segments of each conversation. Show that the dominant context
     and context-specific traits vary meaningfully across segments.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..llm_client import LLMClient
from ..utils import load_json, save_json
from ..prompts.prompt_loader import (
    CURRENT_STATE_EXTRACTION_SYSTEM_PROMPT,
    CURRENT_STATE_EXTRACTION_USER_PROMPT_TEMPLATE,
    CONTEXT_PROFILE_EXTRACTION_SYSTEM_PROMPT,
    CONTEXT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
)
from .persona_simulation import detect_speakers, flatten_messages, session_keys


def _parse_json(raw: str) -> Dict[str, Any]:
    """Robustly parse LLM JSON output."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r'"\s*\n\s*"', '",\n"', raw)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            return {"error": str(e), "raw": raw[:300]}


# ---------------------------------------------------------------------------
# Experiment 1b: State Axis Validation
# ---------------------------------------------------------------------------

def extract_current_state(
    llm: LLMClient,
    user_speaker: str,
    recent_messages: str,
) -> Dict[str, Any]:
    """Extract a current-state snapshot from recent messages."""
    user_prompt = CURRENT_STATE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        user_name=user_speaker,
        recent_messages=recent_messages,
    )
    raw = llm.chat(
        CURRENT_STATE_EXTRACTION_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.3,
        max_tokens=400,
    )
    return _parse_json(raw)


def run_state_axis_experiment(
    dataset_dir: str = "dataset",
    output_dir: str = "data/state_axis_eval",
    chat_filter: Optional[List[str]] = None,
    num_checkpoints: int = 5,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_files = sorted(dataset_dir.glob("Chat_*.json"))
    if chat_filter:
        chat_files = [f for f in chat_files if f.stem in chat_filter]

    print(f"[State Axis] Processing {len(chat_files)} chat files")
    llm = LLMClient()
    all_results: List[Dict[str, Any]] = []

    for chat_file in chat_files:
        print(f"\n[State Axis] Processing {chat_file.name}")
        chat = load_json(str(chat_file))
        user_speaker, _ = detect_speakers(chat)
        turns = flatten_messages(chat)
        user_turns = [t for t in turns if t["speaker"] == user_speaker]

        if len(user_turns) < num_checkpoints * 3:
            print(f"  [Skip] Too few user turns ({len(user_turns)})")
            continue

        # Select checkpoint indices evenly spaced through the conversation
        step = len(user_turns) // num_checkpoints
        checkpoint_indices = [step * i + step // 2 for i in range(num_checkpoints)]
        checkpoint_indices = [i for i in checkpoint_indices if i < len(user_turns)]

        states: List[Dict[str, Any]] = []
        for ci, idx in enumerate(checkpoint_indices):
            # Get a window of recent user messages around this checkpoint
            window_start = max(0, idx - 5)
            window_turns = user_turns[window_start:idx + 1]
            recent = "\n".join(f"{user_speaker}: {t['content']}" for t in window_turns)

            print(f"  Checkpoint {ci+1}/{num_checkpoints} (turn {idx}/{len(user_turns)})...")
            state = extract_current_state(llm, user_speaker, recent)
            state["checkpoint_index"] = ci
            state["turn_index"] = idx
            state["sample_messages"] = [t["content"][:80] for t in window_turns[-3:]]
            states.append(state)

            if "error" not in state:
                print(f"    Emotion: {state.get('emotional_state', '?')}, "
                      f"intensity: {state.get('emotional_intensity', '?')}, "
                      f"valence: {state.get('emotional_valence', '?')}")

        all_results.append({
            "chat_file": chat_file.name,
            "user_speaker": user_speaker,
            "num_checkpoints": len(states),
            "states": states,
            "timestamp": datetime.now().isoformat(),
        })

    results_path = output_dir / "state_axis_results.json"
    save_json(str(results_path), {"results": all_results})
    print(f"\n[State Axis] Results saved to {results_path}")

    summary = aggregate_state_axis(all_results)
    summary_path = output_dir / "state_axis_summary.json"
    save_json(str(summary_path), summary)
    print(f"[State Axis] Summary saved to {summary_path}")
    return summary


def aggregate_state_axis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze how much states vary within each conversation."""
    chat_variations = []
    valence_transitions = 0
    total_transitions = 0
    all_valences = {"positive": 0, "neutral": 0, "negative": 0}
    all_emotions: Dict[str, int] = {}

    for r in results:
        states = [s for s in r["states"] if "error" not in s]
        if len(states) < 2:
            continue

        # Count unique emotions and valences
        emotions = [s.get("emotional_state", "").lower() for s in states]
        valences = [s.get("emotional_valence", "neutral") for s in states]
        unique_emotions = len(set(emotions))
        unique_valences = len(set(valences))

        for v in valences:
            all_valences[v] = all_valences.get(v, 0) + 1
        for e in emotions:
            if e:
                all_emotions[e] = all_emotions.get(e, 0) + 1

        # Count transitions (changes between consecutive checkpoints)
        for i in range(1, len(valences)):
            total_transitions += 1
            if valences[i] != valences[i-1]:
                valence_transitions += 1

        chat_variations.append({
            "chat": r["chat_file"],
            "num_checkpoints": len(states),
            "unique_emotions": unique_emotions,
            "unique_valences": unique_valences,
            "emotions": emotions,
            "valences": valences,
        })

    transition_rate = round(valence_transitions / total_transitions, 3) if total_transitions > 0 else 0
    avg_unique_emotions = round(sum(c["unique_emotions"] for c in chat_variations) / len(chat_variations), 2) if chat_variations else 0

    return {
        "num_chats": len(chat_variations),
        "avg_unique_emotions_per_chat": avg_unique_emotions,
        "valence_transition_rate": transition_rate,
        "valence_transition_count": f"{valence_transitions}/{total_transitions}",
        "valence_distribution": all_valences,
        "top_emotions": sorted(all_emotions.items(), key=lambda x: -x[1])[:10],
        "per_chat": [{"chat": c["chat"], "unique_emotions": c["unique_emotions"],
                       "valences": c["valences"]} for c in chat_variations],
    }


# ---------------------------------------------------------------------------
# Experiment 1c: Context Axis Validation
# ---------------------------------------------------------------------------

def extract_context_profile(
    llm: LLMClient,
    user_speaker: str,
    corpus: str,
) -> Dict[str, Any]:
    """Extract a context-specific profile from a conversation segment."""
    user_prompt = CONTEXT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        user_name=user_speaker,
        corpus=corpus,
    )
    raw = llm.chat(
        CONTEXT_PROFILE_EXTRACTION_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.3,
        max_tokens=600,
    )
    return _parse_json(raw)


def run_context_axis_experiment(
    dataset_dir: str = "dataset",
    output_dir: str = "data/context_axis_eval",
    chat_filter: Optional[List[str]] = None,
    num_segments: int = 3,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_files = sorted(dataset_dir.glob("Chat_*.json"))
    if chat_filter:
        chat_files = [f for f in chat_files if f.stem in chat_filter]

    print(f"[Context Axis] Processing {len(chat_files)} chat files")
    llm = LLMClient()
    all_results: List[Dict[str, Any]] = []

    for chat_file in chat_files:
        print(f"\n[Context Axis] Processing {chat_file.name}")
        chat = load_json(str(chat_file))
        user_speaker, _ = detect_speakers(chat)
        sessions = session_keys(chat)

        # Split into segments
        n = len(sessions)
        segment_size = n // num_segments
        segments = []
        for i in range(num_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < num_segments - 1 else n
            seg_sessions = sessions[start:end]
            segments.append((f"segment_{i+1}", seg_sessions))

        segment_profiles: List[Dict[str, Any]] = []
        for seg_name, seg_sessions in segments:
            # Build corpus from user messages in this segment
            turns = []
            for s in seg_sessions:
                for msg in chat.get(s, []):
                    if str(msg.get("speaker", "")).strip() == user_speaker:
                        content = str(msg.get("clean_text", "")).strip()
                        if content:
                            turns.append(f"{user_speaker}: {content}")
            corpus = "\n".join(turns[-100:])

            if len(corpus) < 100:
                continue

            print(f"  Segment {seg_name} ({len(seg_sessions)} sessions)...")
            profile = extract_context_profile(llm, user_speaker, corpus)
            profile["segment"] = seg_name
            if "error" not in profile:
                print(f"    Context: {profile.get('dominant_context', '?')}, "
                      f"{len(profile.get('context_specific_traits', []))} traits")
            segment_profiles.append(profile)

        all_results.append({
            "chat_file": chat_file.name,
            "user_speaker": user_speaker,
            "segments": segment_profiles,
            "timestamp": datetime.now().isoformat(),
        })

    results_path = output_dir / "context_axis_results.json"
    save_json(str(results_path), {"results": all_results})
    print(f"\n[Context Axis] Results saved to {results_path}")

    summary = aggregate_context_axis(all_results)
    summary_path = output_dir / "context_axis_summary.json"
    save_json(str(summary_path), summary)
    print(f"[Context Axis] Summary saved to {summary_path}")
    return summary


def aggregate_context_axis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze context diversity across segments."""
    context_distribution: Dict[str, int] = {}
    context_per_chat: List[Dict[str, Any]] = []
    total_traits = 0
    unique_trait_names: Dict[str, int] = {}

    for r in results:
        segments = [s for s in r["segments"] if "error" not in s]
        if not segments:
            continue

        contexts = [s.get("dominant_context", "unknown") for s in segments]
        chat_contexts = set(contexts)

        for ctx in contexts:
            context_distribution[ctx] = context_distribution.get(ctx, 0) + 1

        # Count traits
        for seg in segments:
            traits = seg.get("context_specific_traits", [])
            total_traits += len(traits)
            for t in traits:
                if isinstance(t, dict):
                    name = t.get("trait", "").lower()
                    if name:
                        unique_trait_names[name] = unique_trait_names.get(name, 0) + 1

        context_per_chat.append({
            "chat": r["chat_file"],
            "contexts": contexts,
            "num_unique_contexts": len(chat_contexts),
        })

    avg_contexts = (
        round(sum(c["num_unique_contexts"] for c in context_per_chat) / len(context_per_chat), 2)
        if context_per_chat else 0
    )

    return {
        "num_chats": len(context_per_chat),
        "avg_unique_contexts_per_chat": avg_contexts,
        "context_distribution": context_distribution,
        "total_context_traits_extracted": total_traits,
        "unique_trait_types": len(unique_trait_names),
        "top_traits": sorted(unique_trait_names.items(), key=lambda x: -x[1])[:10],
        "per_chat": context_per_chat,
    }


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiments 1b & 1c: State Axis + Context Axis Validation"
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--state-output", default="data/state_axis_eval")
    parser.add_argument("--context-output", default="data/context_axis_eval")
    parser.add_argument("--chats", nargs="*", default=None)
    parser.add_argument("--checkpoints", type=int, default=5)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--experiment", choices=["both", "state", "context"], default="both")
    args = parser.parse_args()

    if args.experiment in ("both", "state"):
        print("\n" + "="*60)
        print("EXPERIMENT 1b: STATE AXIS VALIDATION")
        print("="*60)
        run_state_axis_experiment(
            args.dataset_dir, args.state_output, args.chats, args.checkpoints
        )

    if args.experiment in ("both", "context"):
        print("\n" + "="*60)
        print("EXPERIMENT 1c: CONTEXT AXIS VALIDATION")
        print("="*60)
        run_context_axis_experiment(
            args.dataset_dir, args.context_output, args.chats, args.segments
        )


if __name__ == "__main__":
    main()
