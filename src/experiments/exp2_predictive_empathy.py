"""
Experiment 2: RQ2 — Does better understanding improve predictive empathy?

Objective: Validate whether explicit user modeling improves future user state
prediction ability, and generates empathy responses more aligned with the
user's future state.

Experimental Settings:
  1. LLM Only — zero-shot, no user info
  2. Dialogue History — only conversation context
  3. User Profile — profile + history, no explicit prediction module
  4. Full Deep Empathy Framework (Ours) — complete framework

Input: Current Session
Predict: Future Emotion, Future Sentiment, Future Intimacy, Future Topic

Evaluation Metrics:
  Prediction: Future Emotion, Future Sentiment, Future Intimacy
  Generation: Style Similarity, EPITOME

Visualization: Prediction Error vs. Session
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
from ..prediction import FutureStatePredictor, compute_prediction_error
from ..metrics import compute_style_similarity
from .experiment_utils import (
    load_chat_files,
    extract_emotion_sentiment,
    extract_topic,
    extract_intimacy,
    build_eval_points_at_sessions,
    save_experiment_results,
    robust_parse_json,
)
from .persona_simulation import (
    detect_speakers, flatten_messages, session_keys,
    format_conversation_history, condense_profile, condense_persona,
)
from .empathy_alignment_analysis import (
    EmpathyAlignmentReasoner, EPITOMEEvaluator,
    AlignedResponseGenerator, DirectResponseGenerator,
)


@dataclass
class Exp2Config:
    dataset_dir: str = "dataset"
    output_dir: str = "data/exp2_predictive_empathy"
    min_context_sessions: int = 2
    max_eval_points_per_chat: int = 10
    chat_filter: Optional[List[str]] = None


def run_exp2(config: Exp2Config) -> Dict[str, Any]:
    """Run Experiment 2: Predictive Empathy evaluation."""
    chat_files = load_chat_files(config.dataset_dir, config.chat_filter)
    print(f"[Exp2] Processing {len(chat_files)} chat files")

    llm = LLMClient()
    all_results: List[Dict[str, Any]] = []

    for chat_file in chat_files:
        print(f"\n[Exp2] Processing {chat_file.name}")
        chat = load_json(str(chat_file))
        user_speaker, agent_speaker = detect_speakers(chat)

        # Load profile and persona
        profile_path = Path(config.dataset_dir) / "output" / "user" / f"{user_speaker.lower().replace(' ', '_')}_profile.json"
        persona_path = Path(config.dataset_dir) / "output" / "agent" / f"{agent_speaker.lower().replace(' ', '_')}_persona.json"

        profile = load_json(str(profile_path)) if profile_path.exists() else {}
        persona = load_json(str(persona_path)) if persona_path.exists() else {}

        # Initialize predictors for each mode
        predictors = {
            "llm_only": FutureStatePredictor(llm, mode="llm_only"),
            "dialogue_history": FutureStatePredictor(llm, mode="dialogue_history"),
            "user_profile": FutureStatePredictor(llm, mode="user_profile"),
            "full_framework": FutureStatePredictor(llm, mode="full_framework"),
        }

        # Initialize response generators and evaluators
        reasoner = EmpathyAlignmentReasoner(llm)
        epitome_eval = EPITOMEEvaluator(llm)

        eval_points = build_eval_points_at_sessions(
            chat, agent_speaker, user_speaker, config.min_context_sessions
        )
        if len(eval_points) > config.max_eval_points_per_chat:
            eval_points = eval_points[:config.max_eval_points_per_chat]
        print(f"  Evaluation points: {len(eval_points)}")

        for ep in eval_points:
            print(f"  Evaluating {ep['eval_id']}...")
            target_msg = ep["target_message"]
            context_turns = ep["context_turns"]

            # Ground truth annotation
            gt_emotion = extract_emotion_sentiment(llm, target_msg)
            gt_topic = extract_topic(llm, target_msg)
            gt_intimacy = extract_intimacy(llm, target_msg)

            ground_truth = {
                "actual_emotion": gt_emotion.get("emotion", "neutral"),
                "actual_sentiment": gt_emotion.get("sentiment", "neutral"),
                "actual_intimacy": gt_intimacy.get("intimacy_level", 0.5),
                "actual_topic": gt_topic.get("topic", ""),
            }

            # Run predictions for each mode
            predictions = {}
            for mode_name, predictor in predictors.items():
                pred = predictor.predict(
                    user_message=target_msg,
                    conversation_history=context_turns[-15:] if context_turns else None,
                    user_profile=profile if mode_name in ("user_profile", "full_framework") else None,
                    empathy_reasoning=None,  # Can add empathy reasoning for full_framework
                    current_state=None,
                )
                predictions[mode_name] = pred

            # Compute prediction errors
            pred_errors = {}
            for mode_name, pred in predictions.items():
                if "error" not in pred:
                    pred_errors[mode_name] = compute_prediction_error(pred, ground_truth)
                else:
                    pred_errors[mode_name] = {"error": pred["error"]}

            result = {
                "chat_file": chat_file.name,
                "eval_id": ep["eval_id"],
                "boundary_idx": ep["boundary_idx"],
                "target_session": ep["target_session"],
                "target_message": target_msg,
                "ground_truth": ground_truth,
                "predictions": predictions,
                "prediction_errors": pred_errors,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result)

            # Print summary
            for mode_name in predictions:
                err = pred_errors.get(mode_name, {})
                total_err = err.get("total_error", "N/A")
                print(f"    {mode_name}: total_error={total_err}")

    # Aggregate summary
    summary = _aggregate_exp2_results(all_results)

    save_experiment_results(
        config.output_dir, "exp2_predictive_empathy",
        all_results, summary, vars(config)
    )
    print(f"\n[Exp2] Summary saved.")
    return summary


def _aggregate_exp2_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate prediction errors across all evaluation points."""
    if not results:
        return {"error": "No results"}

    modes = ["llm_only", "dialogue_history", "user_profile", "full_framework"]
    mode_errors: Dict[str, Dict[str, List[float]]] = {
        m: {"emotion": [], "sentiment": [], "intimacy": [], "topic": [], "total": []}
        for m in modes
    }

    # Per-session errors for visualization
    per_session_errors: Dict[int, Dict[str, List[float]]] = {}

    for r in results:
        boundary_idx = r.get("boundary_idx", 0)
        if boundary_idx not in per_session_errors:
            per_session_errors[boundary_idx] = {m: [] for m in modes}

        for mode in modes:
            err = r.get("prediction_errors", {}).get(mode, {})
            if "error" in err:
                continue
            mode_errors[mode]["emotion"].append(err.get("emotion_match", 0))
            mode_errors[mode]["sentiment"].append(err.get("sentiment_match", 0))
            mode_errors[mode]["intimacy"].append(err.get("intimacy_error", 1.0))
            mode_errors[mode]["topic"].append(err.get("topic_overlap", 0))
            mode_errors[mode]["total"].append(err.get("total_error", 1.0))
            per_session_errors[boundary_idx][mode].append(err.get("total_error", 1.0))

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    comparison = {}
    for mode in modes:
        comparison[mode] = {
            "emotion_accuracy": avg(mode_errors[mode]["emotion"]),
            "sentiment_accuracy": avg(mode_errors[mode]["sentiment"]),
            "intimacy_error": avg(mode_errors[mode]["intimacy"]),
            "topic_consistency": avg(mode_errors[mode]["topic"]),
            "total_prediction_error": avg(mode_errors[mode]["total"]),
            "num_evaluations": len(mode_errors[mode]["total"]),
        }

    # Per-session prediction error (for visualization)
    session_trend = {}
    for session_idx in sorted(per_session_errors.keys()):
        session_trend[str(session_idx)] = {}
        for mode in modes:
            vals = per_session_errors[session_idx][mode]
            session_trend[str(session_idx)][mode] = avg(vals)

    return {
        "comparison": comparison,
        "prediction_error_per_session": session_trend,
        "num_eval_points": len(results),
        "improvement": {
            "full_vs_llm_only": round(
                comparison.get("llm_only", {}).get("total_prediction_error", 0) -
                comparison.get("full_framework", {}).get("total_prediction_error", 0), 4
            ),
            "full_vs_dialogue_history": round(
                comparison.get("dialogue_history", {}).get("total_prediction_error", 0) -
                comparison.get("full_framework", {}).get("total_prediction_error", 0), 4
            ),
            "full_vs_user_profile": round(
                comparison.get("user_profile", {}).get("total_prediction_error", 0) -
                comparison.get("full_framework", {}).get("total_prediction_error", 0), 4
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Predictive Empathy")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/exp2_predictive_empathy")
    parser.add_argument("--min-context-sessions", type=int, default=2)
    parser.add_argument("--max-eval-points", type=int, default=10)
    parser.add_argument("--chats", nargs="*", default=None)
    args = parser.parse_args()

    config = Exp2Config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        min_context_sessions=args.min_context_sessions,
        max_eval_points_per_chat=args.max_eval_points,
        chat_filter=args.chats,
    )
    run_exp2(config)


if __name__ == "__main__":
    main()
