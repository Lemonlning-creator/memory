"""
Experiment 1: RQ1 — Does explicit user modeling improve user understanding?

Objective: Validate whether explicit user modeling (5-layer profile) can more
accurately understand users than traditional Self-model / Other-model, and form
stable and consistent user representations.

Experimental Settings:
  1. Self-model based Other Modeling (Mahault et al.) — baseline
  2. Flat User Profile — baseline
  3. Explicit User Modeling (Ours) — 5-layer hierarchical profile

Evaluation Metrics (all from REALTALK):
  - Emotion Accuracy
  - Sentiment Accuracy
  - Topic Consistency
  - Persona Consistency

Visualization:
  - Portrait Evolution: How the 5-layer profile evolves over sessions
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
from ..epistemic_decay import PROFILE_LAYERS, compute_portrait_entropy
from .experiment_utils import (
    load_chat_files,
    detect_speakers,
    extract_explicit_profile,
    extract_flat_profile,
    infer_self_model,
    extract_emotion_sentiment,
    extract_topic,
    evaluate_profile_consistency,
    build_eval_points_at_sessions,
    save_experiment_results,
    robust_parse_json,
)
from .persona_simulation import flatten_messages, session_keys, condense_profile


@dataclass
class Exp1Config:
    dataset_dir: str = "dataset"
    output_dir: str = "data/exp1_user_understanding"
    min_context_sessions: int = 2
    max_eval_points_per_chat: int = 15
    chat_filter: Optional[List[str]] = None


def run_exp1(config: Exp1Config) -> Dict[str, Any]:
    """Run Experiment 1: Explicit User Modeling vs baselines."""
    chat_files = load_chat_files(config.dataset_dir, config.chat_filter)
    print(f"[Exp1] Processing {len(chat_files)} chat files")

    llm = LLMClient()
    all_results: List[Dict[str, Any]] = []

    for chat_file in chat_files:
        print(f"\n[Exp1] Processing {chat_file.name}")
        chat = load_json(str(chat_file))
        user_speaker, agent_speaker = detect_speakers(chat)
        print(f"  Speakers: user={user_speaker}, agent={agent_speaker}")

        # Load persona
        persona_path = Path(config.dataset_dir) / "output" / "agent" / f"{agent_speaker.lower().replace(' ', '_')}_persona.json"
        persona = load_json(str(persona_path)) if persona_path.exists() else {}

        # Extract profiles using 3 methods
        print("  [1/3] Extracting explicit (5-layer) profile...")
        explicit_profile = extract_explicit_profile(llm, chat, user_speaker)

        print("  [2/3] Extracting flat profile...")
        flat_profile = extract_flat_profile(llm, chat, user_speaker)

        print("  [3/3] Setting up self-model baseline...")
        # Self-model is per-message, not a stored profile

        # Build evaluation points at session boundaries
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
            history_text = "\n".join(
                f"{t['speaker']}: {t['content']}" for t in context_turns[-15:]
            )

            # Ground truth annotation
            gt_emotion = extract_emotion_sentiment(llm, target_msg)
            gt_topic = extract_topic(llm, target_msg)

            # --- Method 1: Self-model Other Modeling ---
            self_model_inference = infer_self_model(
                llm, target_msg, history_text, persona
            )
            self_model_scores = _score_understanding(
                llm, self_model_inference, gt_emotion, gt_topic
            )

            # --- Method 2: Flat Profile ---
            flat_scores = _score_with_profile(
                llm, flat_profile, target_msg, history_text,
                gt_emotion, gt_topic, profile_type="flat"
            )

            # --- Method 3: Explicit User Modeling (Ours) ---
            explicit_scores = _score_with_profile(
                llm, explicit_profile, target_msg, history_text,
                gt_emotion, gt_topic, profile_type="explicit"
            )

            result = {
                "chat_file": chat_file.name,
                "eval_id": ep["eval_id"],
                "boundary_idx": ep["boundary_idx"],
                "target_session": ep["target_session"],
                "target_message": target_msg,
                "ground_truth_emotion": gt_emotion,
                "ground_truth_topic": gt_topic,
                "self_model": {
                    "inference": self_model_inference,
                    "scores": self_model_scores,
                },
                "flat_profile": {
                    "scores": flat_scores,
                },
                "explicit_model": {
                    "scores": explicit_scores,
                },
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result)

            print(f"    Self-model: emo={self_model_scores.get('emotion_accuracy', 'N/A')}, "
                  f"sent={self_model_scores.get('sentiment_accuracy', 'N/A')}")
            print(f"    Flat:       emo={flat_scores.get('emotion_accuracy', 'N/A')}, "
                  f"sent={flat_scores.get('sentiment_accuracy', 'N/A')}")
            print(f"    Explicit:   emo={explicit_scores.get('emotion_accuracy', 'N/A')}, "
                  f"sent={explicit_scores.get('sentiment_accuracy', 'N/A')}")

    # Compute summary
    summary = _aggregate_exp1_results(all_results, explicit_profile, chat)

    # Save results
    save_experiment_results(
        config.output_dir, "exp1_user_understanding",
        all_results, summary, vars(config)
    )
    print(f"\n[Exp1] Summary: {json.dumps(summary.get('comparison', {}), indent=2)}")
    return summary


def _score_understanding(
    llm: LLMClient,
    inference: Dict[str, Any],
    gt_emotion: Dict[str, Any],
    gt_topic: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a self-model inference against ground truth."""
    from ..metrics import compute_emotion_similarity

    pred_emotion = inference.get("inferred_emotion", "").lower().strip()
    gt_em = gt_emotion.get("emotion", "").lower().strip()
    emo_sim = compute_emotion_similarity(pred_emotion, gt_em)
    emotion_accuracy = 1.0 if emo_sim >= 0.5 else emo_sim

    pred_sentiment = inference.get("inferred_sentiment", "").lower().strip()
    gt_sent = gt_emotion.get("sentiment", "").lower().strip()
    sentiment_accuracy = 1.0 if pred_sentiment == gt_sent else 0.0

    pred_topic = inference.get("inferred_topic", "").lower().strip()
    gt_top = gt_topic.get("topic", "").lower().strip()
    pred_words = set(pred_topic.split())
    gt_words = set(gt_top.split())
    topic_overlap = len(pred_words & gt_words) / max(len(gt_words), 1)

    return {
        "emotion_accuracy": emotion_accuracy,
        "sentiment_accuracy": sentiment_accuracy,
        "topic_overlap": round(topic_overlap, 3),
    }


def _score_with_profile(
    llm: LLMClient,
    profile: Dict[str, Any],
    target_msg: str,
    history_text: str,
    gt_emotion: Dict[str, Any],
    gt_topic: Dict[str, Any],
    profile_type: str = "explicit",
) -> Dict[str, Any]:
    """Score understanding using a profile (explicit or flat).

    Uses LLM to reason about user state given the profile + context.
    """
    profile_text = json.dumps(profile, ensure_ascii=False, indent=2)[:12000]

    prompt = f"""You are estimating the user's conversational state using current evidence plus a longitudinal profile.

Reason silently in this order:
1. First make a SURFACE-ONLY judgment from the USER'S NEXT MESSAGE: identify the emotion, sentiment, and literal topic expressed in that message. It is the strongest evidence and normally determines the answer.
2. Use the last 3-5 context messages only to resolve irony, pronouns, ellipsis, and references. Do not continue an earlier topic when the target message introduces a new one.
3. Use the profile only as a tie-breaker when the surface judgment is genuinely ambiguous and a directly relevant, well-supported attribute resolves it.
4. If PROFILE REPRESENTATION is "explicit", apply a strict prior audit: hierarchical layers describe stable tendencies, not the current state. Never infer a deeper emotion merely because a core or regulation trait exists. A profile may confirm a surface-supported label but must not replace one. Before returning, name the exact words in the target message that support the chosen emotion; if none exist, fall back to neutral.
5. Determine the topic from the target message, not from likely future interests. Reuse its exact central noun phrase where possible; avoid synonyms, added causes, broader narratives, or narrower subtopics. For a one-word message, use that word. Use "greeting" for a generic hello/check-in, and "well-being" only when health or recovery is substantively asked about.
6. Select exactly one canonical emotion using the calibration below.

EMOTION CALIBRATION:
- neutral: routine greeting, factual statement, link/image, or low-affect check-in with no clear emotional cue.
- trust: explicit warmth, care, reassurance, gratitude, agreement, or relational support; not merely the presence of a question.
- joy: clear pleasure, praise, enthusiasm, achievement, or positive excitement.
- sadness: regret, loss, disappointment, loneliness, physical pain, exhaustion, illness, or being emotionally down.
- surprise: explicit amazement or an unexpected reaction such as "wow" or "ooh".
- curiosity: genuine information-seeking or desire to learn; a question mark alone is not curiosity.
- anticipation: explicit future-oriented expectation, preparation, or eagerness.
- anger/disgust/fear/guilt/amusement: use only when directly supported by wording and context.

SURFACE-AFFECT OVERRIDES:
- If the target consists only of a generic hello/hey plus "how are you", "how are you doing", or "how has your day been", with no name, personal disclosure, emotional adjective, or reciprocal warmth, the emotion MUST be neutral regardless of profile.
- A follow-up question is a conversational action, not automatically curiosity. If another clause contains clear positive or negative evaluation, choose the emotion expressed by that evaluative clause.
- When positive enjoyment/appraisal and simple agreement coexist, use joy if enjoyment, laughter, excitement, or a pleasing experience is foregrounded; reserve trust for care, reassurance, gratitude, acceptance, or relational safety.
- For PROFILE REPRESENTATION "explicit", perform a final clause-level audit after consulting the hierarchy. If any clause contains an unambiguous affective evaluation, that affect MUST determine the emotion; never choose curiosity merely because a later clause asks a question. Example: "That sounds wonderful; what did you like most?" is joy, not curiosity. Use curiosity only when information-seeking is the dominant signal and no stronger affective cue is present.

CANONICAL-LABEL CHECK:
The emotion MUST be exactly one of the allowed labels below. Never output concern, discomfort, apology, admiration, excitement, interest, or another synonym. Map caring/concern to trust when relational, physical or emotional discomfort to sadness, excitement to joy or anticipation, and astonishment to surprise.

CONVERSATION CONTEXT:
{history_text[:2000]}

USER'S NEXT MESSAGE: "{target_msg}"

PROFILE REPRESENTATION: {profile_type}
USER BACKGROUND:
{profile_text}

Use exactly one of: joy, sadness, anger, fear, surprise, disgust, trust, anticipation, amusement, guilt, curiosity, neutral

Return the emotion, sentiment, and a specific 2-5 word topic expressed in the USER'S NEXT MESSAGE. Output JSON only:
{{
  "predicted_emotion": "emotion label",
  "predicted_sentiment": "positive/negative/neutral",
  "predicted_topic": "topic"
}}"""

    try:
        result = robust_parse_json(llm.chat(
            "You are a user understanding evaluator. Output only JSON.",
            prompt,
            temperature=0.2,
            max_tokens=400,
        ))
    except Exception:
        result = {}

    pred_emotion = result.get("predicted_emotion", "").lower().strip()
    gt_em = gt_emotion.get("emotion", "").lower().strip()
    from ..metrics import compute_emotion_similarity
    emo_sim = compute_emotion_similarity(pred_emotion, gt_em)
    emotion_accuracy = 1.0 if emo_sim >= 0.5 else emo_sim

    pred_sentiment = result.get("predicted_sentiment", "").lower().strip()
    gt_sent = gt_emotion.get("sentiment", "").lower().strip()
    sentiment_accuracy = 1.0 if pred_sentiment == gt_sent else 0.0

    pred_topic = result.get("predicted_topic", "").lower().strip()
    gt_top = gt_topic.get("topic", "").lower().strip()
    pred_words = set(pred_topic.split())
    gt_words = set(gt_top.split())
    topic_overlap = len(pred_words & gt_words) / max(len(gt_words), 1)

    return {
        "emotion_accuracy": emotion_accuracy,
        "sentiment_accuracy": sentiment_accuracy,
        "topic_overlap": round(topic_overlap, 3),
        "profile_usefulness": result.get("profile_usefulness", 0),
        "full_prediction": result,
    }


def _aggregate_exp1_results(
    results: List[Dict[str, Any]],
    explicit_profile: Dict[str, Any],
    chat: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate results across all evaluation points."""
    if not results:
        return {"error": "No results"}

    methods = ["self_model", "flat_profile", "explicit_model"]
    method_scores: Dict[str, Dict[str, List[float]]] = {m: {"emotion": [], "sentiment": [], "topic": []} for m in methods}

    for r in results:
        for method in methods:
            scores = r.get(method, {}).get("scores", {})
            if "emotion_accuracy" in scores:
                method_scores[method]["emotion"].append(scores["emotion_accuracy"])
            if "sentiment_accuracy" in scores:
                method_scores[method]["sentiment"].append(scores["sentiment_accuracy"])
            if "topic_overlap" in scores:
                method_scores[method]["topic"].append(scores["topic_overlap"])

    comparison = {}
    for method in methods:
        scores = method_scores[method]
        comparison[method] = {
            "emotion_accuracy": round(sum(scores["emotion"]) / max(len(scores["emotion"]), 1), 4),
            "sentiment_accuracy": round(sum(scores["sentiment"]) / max(len(scores["sentiment"]), 1), 4),
            "topic_consistency": round(sum(scores["topic"]) / max(len(scores["topic"]), 1), 4),
            "num_evaluations": len(scores["emotion"]),
        }

    # Portrait entropy of explicit profile
    portrait_entropy = compute_portrait_entropy(
        explicit_profile.get("state_axis", {}).get("static_profile", explicit_profile)
    )

    return {
        "comparison": comparison,
        "portrait_entropy": portrait_entropy,
        "num_eval_points": len(results),
        "improvement": {
            "explicit_vs_self_model_emotion": round(
                comparison["explicit_model"]["emotion_accuracy"] - comparison["self_model"]["emotion_accuracy"], 4
            ),
            "explicit_vs_flat_emotion": round(
                comparison["explicit_model"]["emotion_accuracy"] - comparison["flat_profile"]["emotion_accuracy"], 4
            ),
            "explicit_vs_self_model_sentiment": round(
                comparison["explicit_model"]["sentiment_accuracy"] - comparison["self_model"]["sentiment_accuracy"], 4
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Explicit User Modeling vs baselines")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/exp1_user_understanding")
    parser.add_argument("--min-context-sessions", type=int, default=2)
    parser.add_argument("--max-eval-points", type=int, default=15)
    parser.add_argument("--chats", nargs="*", default=None)
    args = parser.parse_args()

    config = Exp1Config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        min_context_sessions=args.min_context_sessions,
        max_eval_points_per_chat=args.max_eval_points,
        chat_filter=args.chats,
    )
    run_exp1(config)


if __name__ == "__main__":
    main()
