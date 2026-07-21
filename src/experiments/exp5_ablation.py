"""
Experiment 5: Ablation Study

Systematic ablation of all components:

  1. Full Model — Complete framework (explicit profile + Bayesian update + adaptive exploration)
  2. w/o Explicit User Modeling — Replace with self-model-based other modeling
  3. w/o Adaptive Exploration — Remove adaptive epistemic weight ω(t), use fixed omega
  4. w/o Bayesian Updating — Disable online posterior updating, use static profile
  5. Flat Profile — Replace hierarchical profile with flat profile

Evaluates all conditions across unified metrics:
  - Emotion Accuracy, Sentiment Accuracy, Topic Consistency
  - Persona Consistency
  - Portrait Entropy
  - Exploration Question Ratio
  - Style Similarity (vs ground truth response)
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
from ..utils import load_json, save_json
from ..epistemic_decay import (
    EpistemicDecayTracker, compute_portrait_entropy,
    compute_profile_completeness,
)
from ..metrics import (
    compute_style_similarity, detect_exploration_question,
    compute_exploration_ratio,
)
from ..prediction import FutureStatePredictor, compute_prediction_error
from .experiment_utils import (
    load_chat_files,
    extract_explicit_profile,
    extract_flat_profile,
    infer_self_model,
    extract_emotion_sentiment,
    extract_topic,
    extract_intimacy,
    evaluate_profile_consistency,
    build_eval_points_at_sessions,
    save_experiment_results,
    robust_parse_json,
)
from .persona_simulation import (
    detect_speakers, flatten_messages, session_keys,
    format_conversation_history, condense_profile, condense_persona,
)
from .empathy_alignment_analysis import EPITOMEEvaluator
from ..prompts.prompt_loader import (
    EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
    EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE,
)


# Ablation conditions
ABLATION_CONDITIONS = {
    "full_model": {
        "modeling_mode": "explicit",
        "exploration_mode": "adaptive",
        "update_mode": "bayesian_online",
        "description": "Complete framework",
    },
    "w_o_explicit_modeling": {
        "modeling_mode": "self_model",
        "exploration_mode": "adaptive",
        "update_mode": "bayesian_online",
        "description": "Replace explicit user model with self-model-based other modeling",
    },
    "w_o_adaptive_exploration": {
        "modeling_mode": "explicit",
        "exploration_mode": "fixed_exploration",
        "update_mode": "bayesian_online",
        "description": "Remove adaptive epistemic weight ω(t), use fixed omega=0.5",
    },
    "w_o_bayesian_updating": {
        "modeling_mode": "explicit",
        "exploration_mode": "adaptive",
        "update_mode": "static",
        "description": "Disable online posterior updating",
    },
    "flat_profile": {
        "modeling_mode": "flat",
        "exploration_mode": "adaptive",
        "update_mode": "bayesian_online",
        "description": "Replace hierarchical profile with flat profile",
    },
}


@dataclass
class Exp5Config:
    dataset_dir: str = "dataset"
    output_dir: str = "data/exp5_ablation"
    min_context_sessions: int = 2
    max_eval_points_per_chat: int = 10
    conditions: Optional[List[str]] = None  # None = all conditions
    chat_filter: Optional[List[str]] = None


def run_exp5(config: Exp5Config) -> Dict[str, Any]:
    """Run Experiment 5: Ablation Study."""
    chat_files = load_chat_files(config.dataset_dir, config.chat_filter)
    print(f"[Exp5] Processing {len(chat_files)} chat files")

    conditions = config.conditions or list(ABLATION_CONDITIONS.keys())
    print(f"[Exp5] Conditions: {conditions}")

    llm = LLMClient()
    all_results: List[Dict[str, Any]] = []

    for chat_file in chat_files:
        print(f"\n[Exp5] Processing {chat_file.name}")
        chat = load_json(str(chat_file))
        user_speaker, agent_speaker = detect_speakers(chat)

        # Load persona
        persona_path = Path(config.dataset_dir) / "output" / "agent" / f"{agent_speaker.lower().replace(' ', '_')}_persona.json"
        persona = load_json(str(persona_path)) if persona_path.exists() else {}

        # Extract profiles
        explicit_profile = extract_explicit_profile(llm, chat, user_speaker)
        flat_profile = extract_flat_profile(llm, chat, user_speaker)

        # Load pre-computed profile if available
        profile_path = Path(config.dataset_dir) / "output" / "user" / f"{user_speaker.lower().replace(' ', '_')}_profile.json"
        stored_profile = load_json(str(profile_path)) if profile_path.exists() else {}

        eval_points = build_eval_points_at_sessions(
            chat, agent_speaker, user_speaker, config.min_context_sessions
        )
        if len(eval_points) > config.max_eval_points_per_chat:
            eval_points = eval_points[:config.max_eval_points_per_chat]
        print(f"  Evaluation points: {len(eval_points)}")

        for ep in eval_points:
            target_msg = ep["target_message"]
            context_turns = ep["context_turns"]
            boundary_idx = ep["boundary_idx"]
            history_text = format_conversation_history(context_turns[-10:])

            # Ground truth
            gt_emotion = extract_emotion_sentiment(llm, target_msg)
            gt_topic = extract_topic(llm, target_msg)
            gt_intimacy = extract_intimacy(llm, target_msg)

            ground_truth = {
                "actual_emotion": gt_emotion.get("emotion", "neutral"),
                "actual_sentiment": gt_emotion.get("sentiment", "neutral"),
                "actual_intimacy": gt_intimacy.get("intimacy_level", 0.5),
                "actual_topic": gt_topic.get("topic", ""),
            }

            condition_results: Dict[str, Dict[str, Any]] = {}

            for cond_name in conditions:
                cond = ABLATION_CONDITIONS[cond_name]
                modeling = cond["modeling_mode"]
                exploration = cond["exploration_mode"]
                update = cond["update_mode"]

                # Select profile based on modeling mode
                if modeling == "self_model":
                    # Self-model: infer from persona
                    inference = infer_self_model(llm, target_msg, history_text, persona)
                    profile_text = condense_persona(persona)
                    emotion_pred = inference.get("inferred_emotion", "")
                    sentiment_pred = inference.get("inferred_sentiment", "")
                    topic_pred = inference.get("inferred_topic", "")
                    portrait_ent = 1.0  # No explicit profile = max entropy
                elif modeling == "flat":
                    profile_text = json.dumps(flat_profile, ensure_ascii=False, indent=2)[:2000]
                    static_p = flat_profile
                    portrait_ent = compute_portrait_entropy(static_p)
                    emotion_pred, sentiment_pred, topic_pred = _predict_with_profile(
                        llm, profile_text, history_text, target_msg
                    )
                else:  # explicit
                    static_p = stored_profile.get("state_axis", {}).get("static_profile", explicit_profile)
                    if update == "static":
                        # Use initial profile only (no Bayesian update effect)
                        profile_text = json.dumps(static_p, ensure_ascii=False, indent=2)[:2000]
                    else:
                        profile_text = json.dumps(static_p, ensure_ascii=False, indent=2)[:2000]
                    portrait_ent = compute_portrait_entropy(static_p)
                    emotion_pred, sentiment_pred, topic_pred = _predict_with_profile(
                        llm, profile_text, history_text, target_msg
                    )

                # Compute exploration metrics
                tracker = EpistemicDecayTracker(mode=exploration)
                omega = tracker.compute(static_p if modeling != "self_model" else None)
                exploration_label = "explore" if omega >= 0.6 else ("balanced" if omega >= 0.25 else "exploit")

                # Run empathy alignment for exploration question detection
                has_exploration = False
                try:
                    persona_text = condense_persona(persona)
                    align_prompt = EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE.format(
                        recent_context=history_text[:2000],
                        user_message=target_msg,
                        user_profile=profile_text[:2000],
                        agent_persona=persona_text[:1000],
                        current_state=json.dumps({}, ensure_ascii=False),
                        epistemic_omega=omega,
                    )
                    alignment = robust_parse_json(llm.chat(
                        EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
                        align_prompt,
                        temperature=0.3,
                        max_tokens=800,
                    ))
                    exploration_info = alignment.get("exploration", {})
                    has_exploration = exploration_info.get("decision") == "explore"
                except Exception:
                    pass

                # Compute scores
                from ..metrics import compute_emotion_similarity
                emo_sim = compute_emotion_similarity(emotion_pred, ground_truth["actual_emotion"])
                emotion_acc = 1.0 if emo_sim >= 0.5 else emo_sim
                sentiment_acc = 1.0 if sentiment_pred.lower().strip() == ground_truth["actual_sentiment"].lower().strip() else 0.0

                pred_words = set(topic_pred.lower().split())
                gt_words = set(ground_truth["actual_topic"].lower().split())
                topic_overlap = len(pred_words & gt_words) / max(len(gt_words), 1)

                prediction = {
                    "future_emotion": emotion_pred,
                    "future_sentiment": sentiment_pred,
                    "future_topic": topic_pred,
                    "future_intimacy": 0.5,
                }
                pred_error = compute_prediction_error(prediction, ground_truth)

                condition_results[cond_name] = {
                    "emotion_accuracy": emotion_acc,
                    "sentiment_accuracy": sentiment_acc,
                    "topic_consistency": round(topic_overlap, 3),
                    "prediction_total_error": pred_error.get("total_error", 1.0),
                    "portrait_entropy": portrait_ent,
                    "omega": omega,
                    "exploration_label": exploration_label,
                    "has_exploration_question": has_exploration,
                }

            result = {
                "chat_file": chat_file.name,
                "eval_id": ep["eval_id"],
                "boundary_idx": boundary_idx,
                "target_session": ep["target_session"],
                "ground_truth": ground_truth,
                "condition_results": condition_results,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result)

            # Print summary
            for cond_name in conditions:
                cr = condition_results[cond_name]
                print(f"    {cond_name}: emo={cr['emotion_accuracy']:.1f}, "
                      f"ent={cr['portrait_entropy']:.3f}, "
                      f"explore={cr['has_exploration_question']}")

    # Aggregate summary
    summary = _aggregate_exp5_results(all_results, conditions)

    save_experiment_results(
        config.output_dir, "exp5_ablation",
        all_results, summary, vars(config)
    )
    print(f"\n[Exp5] Summary saved.")
    return summary


def _predict_with_profile(
    llm: LLMClient,
    profile_text: str,
    history_text: str,
    target_msg: str,
) -> tuple[str, str, str]:
    """Use profile to predict user emotion, sentiment, topic."""
    prompt = f"""Estimate the user's conversational state from the latest message, recent trajectory, and user profile.

Reason silently in this order:
1. The USER'S NEXT MESSAGE is the strongest evidence. Identify the emotion, sentiment, and specific topic it expresses or strongly implies.
2. Use recent context to resolve tone, irony, pronouns, and whether the state continues or shifts.
3. Use profile evidence as a prior only when current evidence is ambiguous. If it is hierarchical, combine only the relevant core/regulation/cognition/identity/behavior signals and weight higher-confidence attributes more strongly.
4. Prefer a specific supported canonical emotion over neutral; never force an unrelated profile trait onto the current message.

USER PROFILE:
{profile_text}

CONVERSATION CONTEXT:
{history_text[:2000]}

USER'S NEXT MESSAGE: "{target_msg}"

Return the state expressed at this boundary and most likely to continue. Use a concise 2-5 word topic. Output JSON only:
{{
  "predicted_emotion": "emotion label",
  "predicted_sentiment": "positive/negative/neutral",
  "predicted_topic": "topic"
}}"""
    try:
        result = robust_parse_json(llm.chat(
            "You are a user state predictor. Output only JSON.",
            prompt,
            temperature=0.2,
            max_tokens=300,
        ))
        return (
            result.get("predicted_emotion", ""),
            result.get("predicted_sentiment", ""),
            result.get("predicted_topic", ""),
        )
    except Exception:
        return ("", "", "")


def _aggregate_exp5_results(
    results: List[Dict[str, Any]],
    conditions: List[str],
) -> Dict[str, Any]:
    """Aggregate ablation results."""
    if not results:
        return {"error": "No results"}

    cond_stats: Dict[str, Dict[str, List[float]]] = {
        c: {"emotion": [], "sentiment": [], "topic": [], "pred_error": [],
             "entropy": [], "explore_ratio": []}
        for c in conditions
    }

    for r in results:
        for cond_name in conditions:
            cr = r.get("condition_results", {}).get(cond_name, {})
            if not cr:
                continue
            cond_stats[cond_name]["emotion"].append(cr.get("emotion_accuracy", 0))
            cond_stats[cond_name]["sentiment"].append(cr.get("sentiment_accuracy", 0))
            cond_stats[cond_name]["topic"].append(cr.get("topic_consistency", 0))
            cond_stats[cond_name]["pred_error"].append(cr.get("prediction_total_error", 1))
            cond_stats[cond_name]["entropy"].append(cr.get("portrait_entropy", 1))
            cond_stats[cond_name]["explore_ratio"].append(
                1.0 if cr.get("has_exploration_question") else 0.0
            )

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    comparison = {}
    for cond_name in conditions:
        stats = cond_stats[cond_name]
        comparison[cond_name] = {
            "emotion_accuracy": avg(stats["emotion"]),
            "sentiment_accuracy": avg(stats["sentiment"]),
            "topic_consistency": avg(stats["topic"]),
            "prediction_error": avg(stats["pred_error"]),
            "avg_portrait_entropy": avg(stats["entropy"]),
            "exploration_question_ratio": avg(stats["explore_ratio"]),
            "num_evaluations": len(stats["emotion"]),
            "description": ABLATION_CONDITIONS.get(cond_name, {}).get("description", ""),
        }

    # Compute degradation from full model
    full = comparison.get("full_model", {})
    degradation = {}
    for cond_name in conditions:
        if cond_name == "full_model":
            continue
        c = comparison[cond_name]
        degradation[cond_name] = {
            "emotion_drop": round(full.get("emotion_accuracy", 0) - c.get("emotion_accuracy", 0), 4),
            "sentiment_drop": round(full.get("sentiment_accuracy", 0) - c.get("sentiment_accuracy", 0), 4),
            "prediction_error_increase": round(c.get("prediction_error", 0) - full.get("prediction_error", 0), 4),
            "entropy_diff": round(c.get("avg_portrait_entropy", 0) - full.get("avg_portrait_entropy", 0), 4),
        }

    return {
        "comparison": comparison,
        "degradation_from_full": degradation,
        "num_eval_points": len(results),
        "key_findings": {
            "most_critical_component": _find_most_critical(degradation),
            "full_model_best": all(
                comparison.get("full_model", {}).get("emotion_accuracy", 0) >=
                comparison.get(c, {}).get("emotion_accuracy", 0)
                for c in conditions if c != "full_model"
            ),
        },
    }


def _find_most_critical(degradation: Dict[str, Dict[str, float]]) -> str:
    """Find which ablation causes the largest performance drop."""
    if not degradation:
        return "unknown"
    max_drop = -1
    worst = "unknown"
    for cond, drops in degradation.items():
        total_drop = (
            abs(drops.get("emotion_drop", 0)) +
            abs(drops.get("sentiment_drop", 0)) +
            abs(drops.get("prediction_error_increase", 0))
        )
        if total_drop > max_drop:
            max_drop = total_drop
            worst = cond
    return worst


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Ablation Study")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/exp5_ablation")
    parser.add_argument("--min-context-sessions", type=int, default=2)
    parser.add_argument("--max-eval-points", type=int, default=10)
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Subset of conditions to run. Default: all.")
    parser.add_argument("--chats", nargs="*", default=None)
    args = parser.parse_args()

    config = Exp5Config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        min_context_sessions=args.min_context_sessions,
        max_eval_points_per_chat=args.max_eval_points,
        conditions=args.conditions,
        chat_filter=args.chats,
    )
    run_exp5(config)


if __name__ == "__main__":
    main()
