"""
Experiment 1a: Profile Time Evolution (Ablation)

This experiment validates the TIME AXIS of the multi-dimensional user profile
(Innovation 1). It demonstrates that:

1. User profiles EVOLVE over time — extracting from early sessions produces
   a different (thinner) profile than extracting from all sessions.
2. The stable personality structure is preserved across time windows, but
   the RICHNESS and ACCURACY of the profile increases with more conversation data.

Method:
- Split each conversation into 3 time windows: early (sessions 1-N), middle, late.
- Extract profiles independently from each window.
- Measure: (a) profile richness (tag count, evidence density),
            (b) cross-window consistency of stable traits,
            (c) how much the profile "grows" from early to late.
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
    PROFILE_EXTRACTION_SYSTEM_PROMPT,
    PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
    PROFILE_CONSISTENCY_SYSTEM_PROMPT,
    PROFILE_CONSISTENCY_USER_PROMPT_TEMPLATE,
)
from .persona_simulation import detect_speakers, flatten_messages, session_keys
from .cross_conversation_consistency import (
    extract_profile_from_chat,
    evaluate_profile_consistency,
    _parse_json,
)


# ---------------------------------------------------------------------------
# Time-window splitting
# ---------------------------------------------------------------------------

def split_sessions_into_windows(
    chat: Dict[str, Any],
    num_windows: int = 3,
) -> List[Tuple[str, List[str]]]:
    """
    Split sessions into N chronological windows.
    Returns list of (window_name, [session_keys]).
    """
    sessions = session_keys(chat)
    n = len(sessions)
    if n < num_windows:
        # Not enough sessions; just make each window a single session
        windows = [(f"window_{i+1}", [sessions[i]]) for i in range(n)]
    else:
        chunk = n // num_windows
        windows = []
        names = ["early", "middle", "late"][:num_windows]
        for i in range(num_windows):
            start = i * chunk
            end = (i + 1) * chunk if i < num_windows - 1 else n
            window_sessions = sessions[start:end]
            windows.append((names[i], window_sessions))
    return windows


def build_sub_chat(
    chat: Dict[str, Any],
    sessions: List[str],
) -> Dict[str, Any]:
    """Build a partial chat dict containing only the specified sessions."""
    sub_chat = {"name": chat.get("name", {})}
    for s in sessions:
        if s in chat:
            sub_chat[s] = chat[s]
    return sub_chat


# ---------------------------------------------------------------------------
# Profile richness analysis
# ---------------------------------------------------------------------------

def measure_profile_richness(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Measure how rich/detailed a profile is."""
    if "error" in profile:
        return {"error": True, "tag_count": 0, "avg_evidence_length": 0}

    tag_count = 0
    evidence_lengths = []
    filled_sections = 0

    for section, tags in profile.items():
        if isinstance(tags, dict):
            section_has_content = False
            for tag_name, tag_val in tags.items():
                if isinstance(tag_val, dict):
                    tag_count += 1
                    section_has_content = True
                    ev = str(tag_val.get("evidence", ""))
                    evidence_lengths.append(len(ev))
                elif isinstance(tag_val, str) and tag_val.strip():
                    tag_count += 1
                    section_has_content = True
            if section_has_content:
                filled_sections += 1

    return {
        "tag_count": tag_count,
        "avg_evidence_length": round(sum(evidence_lengths) / len(evidence_lengths), 1) if evidence_lengths else 0,
        "filled_sections": filled_sections,
        "total_evidence_chars": sum(evidence_lengths),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_time_evolution_experiment(
    dataset_dir: str = "dataset",
    output_dir: str = "data/time_evolution_eval",
    chat_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_files = sorted(dataset_dir.glob("Chat_*.json"))
    if chat_filter:
        chat_files = [f for f in chat_files if f.stem in chat_filter]

    print(f"[Time Evolution] Processing {len(chat_files)} chat files")
    llm = LLMClient()

    all_results: List[Dict[str, Any]] = []

    for chat_file in chat_files:
        print(f"\n[Time Evolution] Processing {chat_file.name}")
        chat = load_json(str(chat_file))
        user_speaker, agent_speaker = detect_speakers(chat)

        windows = split_sessions_into_windows(chat, num_windows=3)
        print(f"  User: {user_speaker}, {len(session_keys(chat))} sessions")
        for wname, wsessions in windows:
            print(f"    {wname}: sessions {wsessions[0]}..{wsessions[-1]} ({len(wsessions)} sessions)")

        window_profiles: Dict[str, Dict[str, Any]] = {}
        window_richness: Dict[str, Dict[str, Any]] = {}

        for wname, wsessions in windows:
            print(f"  Extracting profile from {wname} window...")
            sub_chat = build_sub_chat(chat, wsessions)
            profile = extract_profile_from_chat(llm, sub_chat, user_speaker)
            window_profiles[wname] = profile
            window_richness[wname] = measure_profile_richness(profile)
            r = window_richness[wname]
            print(f"    Richness: {r.get('tag_count', 0)} tags, "
                  f"{r.get('avg_evidence_length', 0)} avg evidence chars")

        # Compare early vs late
        if "early" in window_profiles and "late" in window_profiles:
            early = window_profiles["early"]
            late = window_profiles["late"]
            if "error" not in early and "error" not in late:
                print(f"  Comparing early vs late consistency...")
                consistency = evaluate_profile_consistency(
                    llm, early, late,
                    "early window", "late window",
                    user_speaker,
                )
            else:
                consistency = {"error": "profile parse failed"}
        else:
            consistency = {"error": "insufficient windows"}

        result = {
            "chat_file": chat_file.name,
            "user_speaker": user_speaker,
            "num_sessions": len(session_keys(chat)),
            "windows": {w: [str(s) for s in sessions] for w, sessions in windows},
            "profile_richness": window_richness,
            "early_vs_late_consistency": consistency,
            "early_profile": window_profiles.get("early", {}),
            "late_profile": window_profiles.get("late", {}),
            "middle_profile": window_profiles.get("middle", {}),
            "timestamp": datetime.now().isoformat(),
        }
        all_results.append(result)

    # Save full results
    results_path = output_dir / "time_evolution_results.json"
    save_json(str(results_path), {"results": all_results})
    print(f"\n[Time Evolution] Results saved to {results_path}")

    # Aggregate
    summary = aggregate_time_evolution(all_results)
    summary_path = output_dir / "time_evolution_summary.json"
    save_json(str(summary_path), summary)
    print(f"[Time Evolution] Summary saved to {summary_path}")

    return summary


def aggregate_time_evolution(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics across all chats."""
    richness_evolution: Dict[str, List[float]] = {"early": [], "middle": [], "late": []}
    consistency_scores = []

    for r in results:
        # Richness evolution
        for window in ["early", "middle", "late"]:
            rich = r.get("profile_richness", {}).get(window, {})
            tc = rich.get("tag_count", 0)
            if tc > 0:
                richness_evolution[window].append(tc)

        # Consistency
        cons = r.get("early_vs_late_consistency", {})
        if "error" not in cons:
            score = cons.get("overall_consistency", 0)
            if isinstance(score, (int, float)):
                consistency_scores.append(score)

    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0

    early_avg = avg(richness_evolution["early"])
    middle_avg = avg(richness_evolution["middle"])
    late_avg = avg(richness_evolution["late"])

    return {
        "num_chats": len(results),
        "richness_evolution": {
            "early_avg_tags": early_avg,
            "middle_avg_tags": middle_avg,
            "late_avg_tags": late_avg,
            "growth_from_early_to_late": round(late_avg - early_avg, 2),
            "growth_pct": round((late_avg - early_avg) / early_avg * 100, 1) if early_avg > 0 else 0,
            "raw": richness_evolution,
        },
        "early_vs_late_consistency": {
            "num_compared": len(consistency_scores),
            "avg_consistency": avg(consistency_scores),
            "scores": consistency_scores,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1a: Profile Time Evolution"
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/time_evolution_eval")
    parser.add_argument("--chats", nargs="*", default=None)
    args = parser.parse_args()

    run_time_evolution_experiment(args.dataset_dir, args.output_dir, args.chats)


if __name__ == "__main__":
    main()
