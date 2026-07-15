"""
Experiment 4: RQ4 — Does Bayesian updating enable long-term personalization?

Objective: Validate whether Bayesian online updating continuously optimizes the
user profile and ultimately improves long-term companionship quality.

Experimental Settings:
  1. Static Profile — extracted once, never updated
  2. Periodic Rebuild — rebuilt from scratch every N sessions
  3. Bayesian Online Updating (Ours) — incremental Bayesian update

Evaluation Metrics:
  - Persona Consistency
  - Memory QA Accuracy
  - Future Emotion Accuracy

Visualization:
  - Posterior Entropy ↓
  - Prediction Accuracy ↑
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..llm_client import LLMClient
from ..utils import load_json, save_json, parse_json
from ..epistemic_decay import compute_portrait_entropy, compute_profile_completeness
from ..metrics import compute_style_similarity
from .experiment_utils import (
    load_chat_files,
    extract_explicit_profile,
    extract_emotion_sentiment,
    extract_topic,
    evaluate_profile_consistency,
    build_eval_points_at_sessions,
    save_experiment_results,
    robust_parse_json,
)
from .persona_simulation import (
    detect_speakers, flatten_messages, session_keys,
    format_conversation_history, condense_profile,
)
from ..prompts.prompt_loader import (
    PROFILE_EVOLUTION_SYSTEM_PROMPT,
    PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE,
    PERIODIC_REBUILD_SYSTEM_PROMPT,
    PERIODIC_REBUILD_USER_PROMPT_TEMPLATE,
)


@dataclass
class Exp4Config:
    dataset_dir: str = "dataset"
    output_dir: str = "data/exp4_bayesian_updating"
    min_context_sessions: int = 2
    max_eval_points_per_chat: int = 15
    periodic_rebuild_interval: int = 5
    chat_filter: Optional[List[str]] = None


def run_exp4(config: Exp4Config) -> Dict[str, Any]:
    """Run Experiment 4: Bayesian Updating vs alternatives."""
    chat_files = load_chat_files(config.dataset_dir, config.chat_filter)
    print(f"[Exp4] Processing {len(chat_files)} chat files")

    llm = LLMClient()
    all_results: List[Dict[str, Any]] = []

    update_modes = ["static", "periodic_rebuild", "bayesian_online"]

    for chat_file in chat_files:
        print(f"\n[Exp4] Processing {chat_file.name}")
        chat = load_json(str(chat_file))
        user_speaker, agent_speaker = detect_speakers(chat)

        sessions = session_keys(chat)
        turns = flatten_messages(chat)

        if len(sessions) < 3:
            print(f"  [Skip] Only {len(sessions)} sessions, need at least 3")
            continue

        # Initial profile extraction (same for all modes)
        print("  Extracting initial profile...")
        initial_profile = extract_explicit_profile(llm, chat, user_speaker)

        # Simulate profile evolution under each mode
        mode_profiles: Dict[str, Dict[str, Any]] = {
            "static": deepcopy(initial_profile),
            "periodic_rebuild": deepcopy(initial_profile),
            "bayesian_online": deepcopy(initial_profile),
        }

        # Process sessions incrementally
        eval_points = build_eval_points_at_sessions(
            chat, agent_speaker, user_speaker, config.min_context_sessions
        )
        if len(eval_points) > config.max_eval_points_per_chat:
            eval_points = eval_points[:config.max_eval_points_per_chat]

        print(f"  Evaluation points: {len(eval_points)}")

        for ep_idx, ep in enumerate(eval_points):
            target_msg = ep["target_message"]
            context_turns = ep["context_turns"]
            boundary_idx = ep["boundary_idx"]

            # Get conversation data up to this point for profile updates
            session_conversations = []
            for t in context_turns:
                session_conversations.append({
                    "speaker": t["speaker"],
                    "content": t["content"],
                })

            # --- Update profiles based on mode ---

            # Static: no update
            # (profile stays as initial)

            # Periodic Rebuild: rebuild every N sessions
            if (boundary_idx % config.periodic_rebuild_interval == 0
                    and boundary_idx > 0):
                print(f"    Periodic rebuild at session {boundary_idx}")
                conversation_text = "\n".join(
                    f"{t['speaker']}: {t['content']}" for t in session_conversations[-100:]
                )
                try:
                    rebuilt = robust_parse_json(llm.chat(
                        PERIODIC_REBUILD_SYSTEM_PROMPT,
                        PERIODIC_REBUILD_USER_PROMPT_TEMPLATE.format(
                            user_name=user_speaker,
                            full_conversation=conversation_text[:8000],
                        ),
                        temperature=0.3,
                        max_tokens=3000,
                    ))
                    if "error" not in rebuilt:
                        mode_profiles["periodic_rebuild"] = rebuilt
                except Exception as e:
                    print(f"    Periodic rebuild error: {e}")

            # Bayesian Online: update from recent conversations
            if boundary_idx > 0 and len(session_conversations) > 5:
                # Create a mock "long-term memory" from recent conversations
                recent_msgs = session_conversations[-10:]
                mock_ltm = {
                    "type": "behavior_evidence",
                    "content": "Recent conversation patterns: " + " | ".join(
                        m["content"][:100] for m in recent_msgs if m.get("content")
                    ),
                    "confidence": 0.6,
                }
                try:
                    state_axis = mode_profiles["bayesian_online"].get("state_axis", {})
                    current_static = state_axis.get("static_profile", mode_profiles["bayesian_online"])
                    updated = robust_parse_json(llm.chat(
                        PROFILE_EVOLUTION_SYSTEM_PROMPT,
                        PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE.format(
                            static_profile=json.dumps(current_static, ensure_ascii=False, indent=2)[:3000],
                            long_term_memories=json.dumps([mock_ltm], ensure_ascii=False, indent=2),
                        ),
                        temperature=0.3,
                        max_tokens=3000,
                    ))
                    if "error" not in updated:
                        new_static = updated.get("static_profile", updated)
                        if "state_axis" in mode_profiles["bayesian_online"]:
                            mode_profiles["bayesian_online"]["state_axis"]["static_profile"] = new_static
                        else:
                            mode_profiles["bayesian_online"] = new_static
                except Exception as e:
                    print(f"    Bayesian update error: {e}")

            # --- Evaluate each mode's profile ---
            gt_emotion = extract_emotion_sentiment(llm, target_msg)
            gt_topic = extract_topic(llm, target_msg)

            mode_evaluations: Dict[str, Dict[str, Any]] = {}
            for mode in update_modes:
                p = mode_profiles[mode]
                static_p = p.get("state_axis", {}).get("static_profile", p)

                # Portrait entropy
                entropy = compute_portrait_entropy(static_p)

                # Profile completeness
                completeness = compute_profile_completeness(static_p)

                # Emotion prediction accuracy using profile
                profile_text = json.dumps(
                    p.get("state_axis", {}).get("static_profile", p),
                    ensure_ascii=False, indent=2
                )[:2000]
                history_text = format_conversation_history(context_turns[-10:])

                pred_prompt = f"""Given this user profile, predict the user's emotional state for their next message.

USER PROFILE (updated via {mode}):
{profile_text}

CONVERSATION CONTEXT:
{history_text[:2000]}

USER'S NEXT MESSAGE: "{target_msg}"

Predict the user's emotion and sentiment. Output JSON:
{{
  "predicted_emotion": "emotion label",
  "predicted_sentiment": "positive/negative/neutral"
}}"""

                try:
                    pred = robust_parse_json(llm.chat(
                        "You are a user state predictor. Output only JSON.",
                        pred_prompt,
                        temperature=0.2,
                        max_tokens=300,
                    ))
                except Exception:
                    pred = {}

                emotion_acc = 1.0 if pred.get("predicted_emotion", "").lower().strip() == gt_emotion.get("emotion", "").lower().strip() else 0.0
                sentiment_acc = 1.0 if pred.get("predicted_sentiment", "").lower().strip() == gt_emotion.get("sentiment", "").lower().strip() else 0.0

                mode_evaluations[mode] = {
                    "portrait_entropy": entropy,
                    "profile_completeness": completeness,
                    "emotion_accuracy": emotion_acc,
                    "sentiment_accuracy": sentiment_acc,
                }

            # --- Cross-session profile consistency ---
            # Compare profiles at this session with initial profile
            consistency_scores = {}
            if ep_idx > 0:
                for mode in update_modes:
                    p = mode_profiles[mode]
                    try:
                        cons = evaluate_profile_consistency(
                            llm, initial_profile, p,
                            source_a="initial_extraction",
                            source_b=f"{mode}_at_session_{boundary_idx}",
                            speaker_name=user_speaker,
                        )
                        consistency_scores[mode] = cons.get("overall_consistency", 0)
                    except Exception:
                        consistency_scores[mode] = 0

            result = {
                "chat_file": chat_file.name,
                "eval_id": ep["eval_id"],
                "boundary_idx": boundary_idx,
                "target_session": ep["target_session"],
                "ground_truth_emotion": gt_emotion,
                "mode_evaluations": mode_evaluations,
                "consistency_scores": consistency_scores,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result)

            for mode in update_modes:
                me = mode_evaluations[mode]
                print(f"    {mode}: entropy={me['portrait_entropy']:.3f}, "
                      f"emotion_acc={me['emotion_accuracy']:.1f}")

    # Aggregate summary
    summary = _aggregate_exp4_results(all_results, update_modes)

    save_experiment_results(
        config.output_dir, "exp4_bayesian_updating",
        all_results, summary, vars(config)
    )
    print(f"\n[Exp4] Summary saved.")
    return summary


def _aggregate_exp4_results(
    results: List[Dict[str, Any]],
    modes: List[str],
) -> Dict[str, Any]:
    """Aggregate results across all evaluation points."""
    if not results:
        return {"error": "No results"}

    mode_stats: Dict[str, Dict[str, List[float]]] = {
        m: {"entropy": [], "emotion": [], "sentiment": [], "consistency": [], "completeness": []}
        for m in modes
    }

    per_session: Dict[int, Dict[str, Dict[str, List[float]]]] = {}

    for r in results:
        boundary_idx = r.get("boundary_idx", 0)
        if boundary_idx not in per_session:
            per_session[boundary_idx] = {m: {"entropy": [], "emotion": []} for m in modes}

        for mode in modes:
            me = r.get("mode_evaluations", {}).get(mode, {})
            if "portrait_entropy" in me:
                mode_stats[mode]["entropy"].append(me["portrait_entropy"])
                per_session[boundary_idx][mode]["entropy"].append(me["portrait_entropy"])
            if "emotion_accuracy" in me:
                mode_stats[mode]["emotion"].append(me["emotion_accuracy"])
                per_session[boundary_idx][mode]["emotion"].append(me["emotion_accuracy"])
            if "sentiment_accuracy" in me:
                mode_stats[mode]["sentiment"].append(me["sentiment_accuracy"])
            if "profile_completeness" in me:
                mode_stats[mode]["completeness"].append(me["profile_completeness"])

        cs = r.get("consistency_scores", {})
        for mode in modes:
            if mode in cs:
                mode_stats[mode]["consistency"].append(cs[mode])

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    comparison = {}
    for mode in modes:
        stats = mode_stats[mode]
        comparison[mode] = {
            "avg_portrait_entropy": avg(stats["entropy"]),
            "emotion_accuracy": avg(stats["emotion"]),
            "sentiment_accuracy": avg(stats["sentiment"]),
            "persona_consistency": avg(stats["consistency"]),
            "profile_completeness": avg(stats["completeness"]),
            "num_evaluations": len(stats["entropy"]),
        }

    # Per-session trends for visualization
    session_trend = {}
    for session_idx in sorted(per_session.keys()):
        ps = per_session[session_idx]
        session_trend[str(session_idx)] = {
            "portrait_entropy": {m: avg(ps[m]["entropy"]) for m in modes},
            "emotion_accuracy": {m: avg(ps[m]["emotion"]) for m in modes},
        }

    return {
        "comparison": comparison,
        "per_session_trend": session_trend,
        "num_eval_points": len(results),
        "improvement": {
            "bayesian_vs_static_entropy": round(
                comparison.get("static", {}).get("avg_portrait_entropy", 0) -
                comparison.get("bayesian_online", {}).get("avg_portrait_entropy", 0), 4
            ),
            "bayesian_vs_static_emotion": round(
                comparison.get("bayesian_online", {}).get("emotion_accuracy", 0) -
                comparison.get("static", {}).get("emotion_accuracy", 0), 4
            ),
            "bayesian_vs_periodic_emotion": round(
                comparison.get("bayesian_online", {}).get("emotion_accuracy", 0) -
                comparison.get("periodic_rebuild", {}).get("emotion_accuracy", 0), 4
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Bayesian Updating")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/exp4_bayesian_updating")
    parser.add_argument("--min-context-sessions", type=int, default=2)
    parser.add_argument("--max-eval-points", type=int, default=15)
    parser.add_argument("--periodic-rebuild-interval", type=int, default=5)
    parser.add_argument("--chats", nargs="*", default=None)
    args = parser.parse_args()

    config = Exp4Config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        min_context_sessions=args.min_context_sessions,
        max_eval_points_per_chat=args.max_eval_points,
        periodic_rebuild_interval=args.periodic_rebuild_interval,
        chat_filter=args.chats,
    )
    run_exp4(config)


if __name__ == "__main__":
    main()
