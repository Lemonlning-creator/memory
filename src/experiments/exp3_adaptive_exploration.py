"""
Experiment 3: RQ3 — Does adaptive exploration emerge naturally?

Objective: Validate whether exploration behavior naturally decreases as the
user profile matures, demonstrating the effect of dynamic epistemic weight ω(t).

Experimental Settings:
  1. No Exploration — omega(t) = 0 always (pure exploit)
  2. Fixed Exploration — omega(t) = constant (0.5)
  3. Always Exploration — omega(t) = 1 always (pure explore)
  4. Adaptive Exploration (Ours) — omega(t) decays naturally

Behavior Analysis:
  - Exploration Question Ratio: ratio of active exploration questions
  - Portrait Entropy: posterior entropy of user profile

Visualization:
  - Question Ratio ↓ over sessions
  - Portrait Entropy ↓ over sessions
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..llm_client import LLMClient
from ..utils import load_json, save_json
from ..epistemic_decay import EpistemicDecayTracker, compute_portrait_entropy, EXPLORATION_MODES
from ..metrics import detect_exploration_question, compute_exploration_ratio
from .experiment_utils import (
    load_chat_files,
    extract_explicit_profile,
    build_eval_points_at_sessions,
    save_experiment_results,
    robust_parse_json,
)
from .persona_simulation import (
    detect_speakers, flatten_messages, session_keys,
    format_conversation_history, condense_profile, condense_persona,
)
from ..prompts.templates_en import (
    EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
    EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE,
)


@dataclass
class Exp3Config:
    dataset_dir: str = "dataset"
    output_dir: str = "data/exp3_adaptive_exploration"
    min_context_sessions: int = 2
    max_eval_points_per_chat: int = 15
    chat_filter: Optional[List[str]] = None


def run_exp3(config: Exp3Config) -> Dict[str, Any]:
    """Run Experiment 3: Adaptive Exploration analysis."""
    chat_files = load_chat_files(config.dataset_dir, config.chat_filter)
    print(f"[Exp3] Processing {len(chat_files)} chat files")

    llm = LLMClient()
    all_results: List[Dict[str, Any]] = []

    exploration_modes = ["no_exploration", "fixed_exploration", "always_exploration", "adaptive"]

    for chat_file in chat_files:
        print(f"\n[Exp3] Processing {chat_file.name}")
        chat = load_json(str(chat_file))
        user_speaker, agent_speaker = detect_speakers(chat)

        # Load profile and persona
        profile_path = Path(config.dataset_dir) / "output" / "user" / f"{user_speaker.lower().replace(' ', '_')}_profile.json"
        persona_path = Path(config.dataset_dir) / "output" / "agent" / f"{agent_speaker.lower().replace(' ', '_')}_persona.json"

        profile = load_json(str(profile_path)) if profile_path.exists() else {}
        persona = load_json(str(persona_path)) if persona_path.exists() else {}

        static_profile = profile.get("state_axis", {}).get("static_profile", profile)

        eval_points = build_eval_points_at_sessions(
            chat, agent_speaker, user_speaker, config.min_context_sessions
        )
        if len(eval_points) > config.max_eval_points_per_chat:
            eval_points = eval_points[:config.max_eval_points_per_chat]
        print(f"  Evaluation points: {len(eval_points)}")

        # Initialize trackers for each mode
        trackers = {
            mode: EpistemicDecayTracker(mode=mode)
            for mode in exploration_modes
        }

        for ep in eval_points:
            print(f"  Evaluating {ep['eval_id']}...")
            target_msg = ep["target_message"]
            context_turns = ep["context_turns"]
            boundary_idx = ep["boundary_idx"]

            history_text = format_conversation_history(context_turns[-10:])
            profile_text = condense_profile(profile)
            persona_text = condense_persona(persona)

            mode_responses: Dict[str, Dict[str, Any]] = {}

            for mode in exploration_modes:
                tracker = trackers[mode]
                omega = tracker.compute(static_profile)

                # Run empathy alignment reasoning with this omega
                user_prompt = EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE.format(
                    recent_context=history_text[:2000],
                    user_message=target_msg,
                    user_profile=profile_text[:2000],
                    agent_persona=persona_text[:1000],
                    current_state=json.dumps({}, ensure_ascii=False),
                    epistemic_omega=omega,
                )

                try:
                    reasoning = robust_parse_json(llm.chat(
                        EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
                        user_prompt,
                        temperature=0.3,
                        max_tokens=1000,
                    ))
                except Exception as e:
                    reasoning = {"error": str(e)}

                # Extract exploration decision
                exploration = reasoning.get("exploration", {})
                empathy_state = reasoning.get("empathy_state", {})
                exploration_score = empathy_state.get("exploration", 0)

                # Determine if response contains exploration questions
                response_guidance = empathy_state.get("response_guidance", "")
                has_exploration = detect_exploration_question(response_guidance) or \
                                  exploration.get("decision") == "explore"

                mode_responses[mode] = {
                    "omega": omega,
                    "exploration_decision": exploration.get("decision", "unknown"),
                    "exploration_score": exploration_score,
                    "has_exploration_question": has_exploration,
                    "reasoning_summary": {
                        "omega_value": exploration.get("omega_value", omega),
                        "rationale": exploration.get("rationale", ""),
                        "focus": exploration.get("exploration_focus", None),
                    },
                }

                tracker.increment()

            # Compute portrait entropy at this session
            portrait_entropy = compute_portrait_entropy(static_profile)

            result = {
                "chat_file": chat_file.name,
                "eval_id": ep["eval_id"],
                "boundary_idx": boundary_idx,
                "target_session": ep["target_session"],
                "portrait_entropy": portrait_entropy,
                "mode_responses": mode_responses,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result)

            # Print summary
            for mode in exploration_modes:
                mr = mode_responses[mode]
                print(f"    {mode}: omega={mr['omega']:.3f}, "
                      f"decision={mr['exploration_decision']}, "
                      f"explore_q={mr['has_exploration_question']}")
            print(f"    portrait_entropy={portrait_entropy:.4f}")

    # Aggregate summary
    summary = _aggregate_exp3_results(all_results, exploration_modes)

    save_experiment_results(
        config.output_dir, "exp3_adaptive_exploration",
        all_results, summary, vars(config)
    )
    print(f"\n[Exp3] Summary saved.")
    return summary


def _aggregate_exp3_results(
    results: List[Dict[str, Any]],
    modes: List[str],
) -> Dict[str, Any]:
    """Aggregate exploration metrics across all evaluation points."""
    if not results:
        return {"error": "No results"}

    mode_stats: Dict[str, Dict[str, Any]] = {
        m: {"question_ratios": [], "omegas": [], "explore_counts": [], "total": 0}
        for m in modes
    }

    # Per-session data for visualization
    per_session: Dict[int, Dict[str, Any]] = {}

    for r in results:
        boundary_idx = r.get("boundary_idx", 0)
        if boundary_idx not in per_session:
            per_session[boundary_idx] = {
                "question_ratios": {m: [] for m in modes},
                "omegas": {m: [] for m in modes},
                "portrait_entropy": [],
            }

        for mode in modes:
            mr = r.get("mode_responses", {}).get(mode, {})
            omega = mr.get("omega", 0)
            has_explore = mr.get("has_exploration_question", False)

            mode_stats[mode]["omegas"].append(omega)
            mode_stats[mode]["explore_counts"].append(1 if has_explore else 0)
            mode_stats[mode]["total"] += 1

            per_session[boundary_idx]["question_ratios"][mode].append(1 if has_explore else 0)
            per_session[boundary_idx]["omegas"][mode].append(omega)

        per_session[boundary_idx]["portrait_entropy"].append(r.get("portrait_entropy", 1.0))

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    comparison = {}
    for mode in modes:
        stats = mode_stats[mode]
        total = max(stats["total"], 1)
        comparison[mode] = {
            "avg_omega": avg(stats["omegas"]),
            "exploration_question_ratio": round(sum(stats["explore_counts"]) / total, 4),
            "num_evaluations": stats["total"],
        }

    # Per-session trends for visualization
    session_trend = {}
    for session_idx in sorted(per_session.keys()):
        ps = per_session[session_idx]
        session_trend[str(session_idx)] = {
            "question_ratio": {m: avg(ps["question_ratios"][m]) for m in modes},
            "avg_omega": {m: avg(ps["omegas"][m]) for m in modes},
            "portrait_entropy": avg(ps["portrait_entropy"]),
        }

    return {
        "comparison": comparison,
        "per_session_trend": session_trend,
        "num_eval_points": len(results),
        "key_finding": {
            "adaptive_question_ratio": comparison.get("adaptive", {}).get("exploration_question_ratio", 0),
            "always_question_ratio": comparison.get("always_exploration", {}).get("exploration_question_ratio", 0),
            "no_question_ratio": comparison.get("no_exploration", {}).get("exploration_question_ratio", 0),
            "adaptive_reduces_exploration": (
                comparison.get("adaptive", {}).get("exploration_question_ratio", 1) <
                comparison.get("always_exploration", {}).get("exploration_question_ratio", 1)
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Adaptive Exploration")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/exp3_adaptive_exploration")
    parser.add_argument("--min-context-sessions", type=int, default=2)
    parser.add_argument("--max-eval-points", type=int, default=15)
    parser.add_argument("--chats", nargs="*", default=None)
    args = parser.parse_args()

    config = Exp3Config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        min_context_sessions=args.min_context_sessions,
        max_eval_points_per_chat=args.max_eval_points,
        chat_filter=args.chats,
    )
    run_exp3(config)


if __name__ == "__main__":
    main()
