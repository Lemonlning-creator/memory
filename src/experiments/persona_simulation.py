from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..llm_client import LLMClient
from ..utils import load_json, save_json, parse_json
from ..prompts.prompt_loader import (
    EI_EVALUATION_SYSTEM_PROMPT,
    EI_EVALUATION_USER_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def session_keys(chat: Dict[str, Any]) -> List[str]:
    keys = [
        key for key, value in chat.items()
        if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
    ]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def detect_speakers(chat: Dict[str, Any]) -> tuple[str, str]:
    names = chat.get("name", {})
    speaker_1 = str(names.get("speaker_1", "speaker_1")).strip()
    speaker_2 = str(names.get("speaker_2", "speaker_2")).strip()
    return speaker_1, speaker_2


def flatten_messages(chat: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for key in session_keys(chat):
        for idx, msg in enumerate(chat[key]):
            content = str(msg.get("clean_text") or "").strip()
            if not content:
                continue
            turns.append({
                "session_id": key,
                "session_index": int(key.split("_")[1]),
                "message_index": idx,
                "speaker": str(msg.get("speaker") or "").strip(),
                "content": content,
                "dia_id": msg.get("dia_id", ""),
                "date_time": msg.get("date_time", ""),
            })
    return turns


def build_eval_points(
    chat: Dict[str, Any],
    agent_speaker: str,
    min_context_sessions: int = 2,
) -> List[Dict[str, Any]]:
    """
    Build evaluation points at session boundaries.
    Each eval point: agent has observed sessions 1..N,
    must generate agent_speaker's first message in session N+1.
    """
    turns = flatten_messages(chat)
    sessions = session_keys(chat)
    eval_points: List[Dict[str, Any]] = []

    for boundary_idx in range(1, len(sessions)):
        context_sessions = set(sessions[:boundary_idx])
        target_session = sessions[boundary_idx]

        if boundary_idx < min_context_sessions:
            continue

        # Find agent_speaker's first message in target session
        target_msg = None
        for turn in turns:
            if turn["session_id"] == target_session and turn["speaker"] == agent_speaker:
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

        eval_points.append({
            "eval_id": f"boundary_{boundary_idx}",
            "context_sessions": sorted(context_sessions),
            "target_session": target_session,
            "ground_truth": target_msg["content"],
            "ground_truth_dia_id": target_msg["dia_id"],
            "context_turns": context_turns,
        })

    return eval_points


def format_conversation_history(turns: List[Dict[str, Any]], max_turns: int = 30) -> str:
    """Format conversation history, keeping only the most recent turns."""
    if len(turns) > max_turns:
        turns = turns[-max_turns:]
    lines = []
    for t in turns:
        lines.append(f"{t['speaker']}: {t['content']}")
    return "\n".join(lines)


def get_last_user_message(turns: List[Dict[str, Any]], user_speaker: str) -> str:
    """Get the last message from the user speaker."""
    for turn in reversed(turns):
        if turn["speaker"] == user_speaker:
            return turn["content"]
    return ""


def format_context_for_eval(turns: List[Dict[str, Any]], last_n: int = 5) -> str:
    recent = turns[-last_n:] if len(turns) > last_n else turns
    return format_conversation_history(recent)


def condense_profile(profile: Dict[str, Any]) -> str:
    """
    Condense profile into a readable summary instead of raw JSON.
    Handles both flat profile format and nested state_axis format.
    """
    # Handle nested format (state_axis.static_profile)
    if "state_axis" in profile:
        profile = profile["state_axis"].get("static_profile", profile)
    
    lines = []
    for section, fields in profile.items():
        if not isinstance(fields, dict):
            continue
        lines.append(f"[{section}]")
        for key, val in fields.items():
            if isinstance(val, dict):
                if "value" in val:
                    v = val["value"]
                    if isinstance(v, list):
                        v = ", ".join(str(x) for x in v)
                    lines.append(f"  - {key}: {v}")
            else:
                lines.append(f"  - {key}: {val}")
    return "\n".join(lines)


def condense_persona(persona: Dict[str, Any]) -> str:
    """Condense persona into a readable summary."""
    lines = []
    
    # Handle nested format with meta_info
    if "meta_info" in persona:
        meta = persona["meta_info"]
        lines.append(f"Name: {meta.get('name', 'Unknown')}")
        lines.append(f"Core personality: {meta.get('core_personality', '')}")
        if "persona_principles" in meta:
            lines.append("Principles:")
            for p in meta["persona_principles"]:
                lines.append(f"  - {p}")
    
    # Handle flat format
    if "name" in persona and "meta_info" not in persona:
        lines.append(f"Name: {persona.get('name', 'Unknown')}")
        lines.append(f"Personality: {persona.get('personality', '')}")
        lines.append(f"Tone: {persona.get('tone', '')}")
        if "interaction_principles" in persona:
            lines.append("Interaction principles:")
            for p in persona["interaction_principles"]:
                lines.append(f"  - {p}")
        if "expression_patterns" in persona:
            lines.append("Expression patterns:")
            for p in persona["expression_patterns"]:
                lines.append(f"  - {p}")
    
    # Handle strategy_layer and expression_layer
    if "strategy_layer" in persona:
        strat = persona["strategy_layer"]
        lines.append("\n[Strategy]")
        for k, v in strat.items():
            lines.append(f"  - {k}: {v}")
    
    if "expression_layer" in persona:
        expr = persona["expression_layer"]
        lines.append("\n[Expression]")
        if "tone" in expr:
            tones = expr["tone"] if isinstance(expr["tone"], list) else [expr["tone"]]
            lines.append(f"  - tone: {', '.join(tones)}")
        if "expression_patterns" in expr:
            lines.append("  - patterns:")
            for p in expr["expression_patterns"]:
                lines.append(f"    * {p}")
    
    return "\n".join(lines) if lines else json.dumps(persona, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Method variants
# ---------------------------------------------------------------------------

class BaselineLLM:
    """Zero-shot LLM baseline: just conversation history, no profile/memory."""

    SYSTEM_PROMPT = """You are {agent_speaker}, continuing a long-term conversation with {user_speaker}.

CONVERSATION HISTORY:
{history}

{user_speaker}'s last message: "{last_msg}"

TASK: Generate your next message as {agent_speaker}.

IMPORTANT:
- Your message should be 1-3 sentences, casual text message style
- You can respond to what they said, share what you've been doing, or continue the conversation naturally
- Stay in character as {agent_speaker}
- Do NOT write long paragraphs

Generate ONLY your message text:"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(
        self,
        context_turns: List[Dict[str, Any]],
        agent_speaker: str,
        user_speaker: str,
    ) -> str:
        history = format_conversation_history(context_turns, max_turns=20)
        last_msg = get_last_user_message(context_turns, user_speaker)
        
        user_prompt = self.SYSTEM_PROMPT.format(
            agent_speaker=agent_speaker,
            user_speaker=user_speaker,
            history=history,
            last_msg=last_msg,
        )
        return self.llm.chat(
            "You are a helpful assistant generating natural conversation messages.",
            user_prompt,
            temperature=0.7,
            max_tokens=150,
        )


class ProfileOnlyAgent:
    """Agent with profile but no memory retrieval."""

    SYSTEM_PROMPT = """You are {agent_speaker}, in a long-term conversation with {user_speaker}.

YOUR PERSONA:
{persona_text}

ABOUT {user_speaker}:
{profile_text}

CONVERSATION HISTORY:
{history}

{user_speaker}'s last message: "{last_msg}"

TASK: Generate your next message as {agent_speaker}.

GUIDELINES:
- Use your knowledge of {user_speaker} to personalize your response
- Stay in character according to your persona
- Keep it natural and brief (1-3 sentences, casual text style)
- You can respond to their message, share what you've been doing, or continue naturally

Generate ONLY your message text:"""

    def __init__(
        self,
        llm: LLMClient,
        profile: Dict[str, Any],
        persona: Dict[str, Any],
    ):
        self.llm = llm
        self.profile = profile
        self.persona = persona

    def generate(
        self,
        context_turns: List[Dict[str, Any]],
        agent_speaker: str,
        user_speaker: str,
    ) -> str:
        history = format_conversation_history(context_turns, max_turns=20)
        last_msg = get_last_user_message(context_turns, user_speaker)
        profile_text = condense_profile(self.profile)
        persona_text = condense_persona(self.persona)

        user_prompt = self.SYSTEM_PROMPT.format(
            agent_speaker=agent_speaker,
            user_speaker=user_speaker,
            persona_text=persona_text,
            profile_text=profile_text,
            history=history,
            last_msg=last_msg,
        )
        return self.llm.chat(
            "You are a helpful assistant generating personalized conversation messages.",
            user_prompt,
            temperature=0.7,
            max_tokens=150,
        )


class FullAgent:
    """Full agent with profile + memory context + persona."""

    SYSTEM_PROMPT = """You are {agent_speaker}, in a long-term conversation with {user_speaker}.

YOUR PERSONA:
{persona_text}

ABOUT {user_speaker}:
{profile_text}

RELEVANT MEMORIES (things {user_speaker} has shared):
{memory_text}

CONVERSATION HISTORY:
{history}

{user_speaker}'s last message: "{last_msg}"

TASK: Generate your next message as {agent_speaker}.

GUIDELINES:
- Use relevant memories to show you remember what they've shared
- Stay in character according to your persona
- Keep it natural and brief (1-3 sentences, casual text style)
- Reference memories naturally if relevant, but don't force them

Generate ONLY your message text:"""

    def __init__(
        self,
        llm: LLMClient,
        profile: Dict[str, Any],
        persona: Dict[str, Any],
    ):
        self.llm = llm
        self.profile = profile
        self.persona = persona

    def extract_memories_from_context(self, context_turns: List[Dict[str, Any]], user_speaker: str) -> str:
        """Extract key topics/events from conversation history as 'memories'."""
        user_msgs = [t for t in context_turns if t["speaker"] == user_speaker]
        if not user_msgs:
            return "(No specific memories available)"
        
        # Take last 10 user messages and summarize key topics
        recent_user_msgs = user_msgs[-10:]
        memories = []
        for msg in recent_user_msgs:
            content = msg["content"]
            # Only include substantive messages (not just greetings)
            if len(content) > 30 and not content.lower().startswith(("hi", "hello", "hey", "good")):
                memories.append(f"- {content[:100]}...")
        
        return "\n".join(memories) if memories else "(No specific memories available)"

    def generate(
        self,
        context_turns: List[Dict[str, Any]],
        agent_speaker: str,
        user_speaker: str,
    ) -> str:
        history = format_conversation_history(context_turns, max_turns=20)
        last_msg = get_last_user_message(context_turns, user_speaker)
        profile_text = condense_profile(self.profile)
        persona_text = condense_persona(self.persona)
        memory_text = self.extract_memories_from_context(context_turns, user_speaker)

        user_prompt = self.SYSTEM_PROMPT.format(
            agent_speaker=agent_speaker,
            user_speaker=user_speaker,
            persona_text=persona_text,
            profile_text=profile_text,
            memory_text=memory_text,
            history=history,
            last_msg=last_msg,
        )
        return self.llm.chat(
            "You are a helpful assistant generating personalized, memory-aware conversation messages.",
            user_prompt,
            temperature=0.7,
            max_tokens=150,
        )


# ---------------------------------------------------------------------------
# EI Evaluation
# ---------------------------------------------------------------------------

class EIEvaluator:
    """Evaluate generated messages against ground truth on EI dimensions."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(
        self,
        context_turns: List[Dict[str, Any]],
        ground_truth: str,
        generated: str,
    ) -> Dict[str, Any]:
        context = format_context_for_eval(context_turns, last_n=5)
        user_prompt = EI_EVALUATION_USER_PROMPT_TEMPLATE.format(
            context=context,
            ground_truth=ground_truth,
            generated=generated,
        )
        try:
            result = parse_json(self.llm.chat(
                EI_EVALUATION_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.1,
                max_tokens=500,
            ))
            return result
        except Exception as e:
            print(f"[EI Eval Error] {e}")
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    dataset_dir: str = "dataset"
    output_dir: str = "data/persona_simulation_eval"
    min_context_sessions: int = 2
    max_eval_points_per_chat: int = 10
    methods: List[str] = field(default_factory=lambda: [
        "baseline_llm",
        "profile_only",
        "full_agent",
    ])


def run_persona_simulation_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    dataset_dir = Path(config.dataset_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_files = sorted(dataset_dir.glob("Chat_*.json"))
    if not chat_files:
        raise FileNotFoundError(f"No chat files found in {dataset_dir}")

    print(f"[Persona Simulation] Found {len(chat_files)} chat files")

    llm = LLMClient()
    ei_evaluator = EIEvaluator(llm)
    all_results: List[Dict[str, Any]] = []

    for chat_file in chat_files:
        print(f"\n[Persona Simulation] Processing {chat_file.name}")
        chat = load_json(str(chat_file))
        user_speaker, agent_speaker = detect_speakers(chat)
        print(f"  Speakers: user={user_speaker}, agent={agent_speaker}")

        eval_points = build_eval_points(chat, agent_speaker, config.min_context_sessions)
        if len(eval_points) > config.max_eval_points_per_chat:
            eval_points = eval_points[:config.max_eval_points_per_chat]
        print(f"  Evaluation points: {len(eval_points)}")

        # Load profile and persona from dataset/output/
        profile_path = dataset_dir / "output" / "user" / f"{user_speaker.lower()}_profile.json"
        persona_path = dataset_dir / "output" / "agent" / f"{agent_speaker.lower()}_persona.json"

        if not profile_path.exists():
            print(f"  [Warning] Profile not found: {profile_path}, skipping profile_only and full_agent")
            profile = {}
        else:
            profile = load_json(str(profile_path))
            print(f"  Loaded profile from {profile_path}")

        if not persona_path.exists():
            print(f"  [Warning] Persona not found: {persona_path}, skipping profile_only and full_agent")
            persona = {}
        else:
            persona = load_json(str(persona_path))
            print(f"  Loaded persona from {persona_path}")

        # Initialize method variants
        methods = {}
        if "baseline_llm" in config.methods:
            methods["baseline_llm"] = BaselineLLM(llm)
        if "profile_only" in config.methods and profile and persona:
            methods["profile_only"] = ProfileOnlyAgent(llm, profile, persona)
        if "full_agent" in config.methods and profile and persona:
            methods["full_agent"] = FullAgent(llm, profile, persona)

        # Run evaluation
        for eval_point in eval_points:
            print(f"  Evaluating {eval_point['eval_id']}...")
            context_turns = eval_point["context_turns"]
            ground_truth = eval_point["ground_truth"]

            for method_name, method in methods.items():
                try:
                    generated = method.generate(context_turns, agent_speaker, user_speaker)
                    ei_scores = ei_evaluator.evaluate(context_turns, ground_truth, generated)

                    result = {
                        "chat_file": chat_file.name,
                        "eval_id": eval_point["eval_id"],
                        "method": method_name,
                        "ground_truth": ground_truth,
                        "generated": generated,
                        "ei_scores": ei_scores,
                        "timestamp": datetime.now().isoformat(),
                    }
                    all_results.append(result)
                    print(f"    {method_name}: ref={ei_scores.get('reflectiveness', 'N/A')}, grd={ei_scores.get('grounding', 'N/A')}, emp={ei_scores.get('empathy_score', 'N/A')}")

                except Exception as e:
                    print(f"    [Error] {method_name}: {e}")
                    all_results.append({
                        "chat_file": chat_file.name,
                        "eval_id": eval_point["eval_id"],
                        "method": method_name,
                        "error": str(e),
                    })

    # Save results
    results_path = output_dir / "persona_simulation_results.json"
    save_json(str(results_path), {"results": all_results, "config": vars(config)})
    print(f"\n[Persona Simulation] Results saved to {results_path}")

    # Aggregate summary
    summary = aggregate_results(all_results)
    summary_path = output_dir / "persona_simulation_summary.json"
    save_json(str(summary_path), summary)
    print(f"[Persona Simulation] Summary saved to {summary_path}")

    return summary


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    method_scores: Dict[str, List[Dict[str, Any]]] = {}

    for result in results:
        if "error" in result:
            continue
        method = result["method"]
        if method not in method_scores:
            method_scores[method] = []
        method_scores[method].append(result["ei_scores"])

    summary: Dict[str, Any] = {}
    for method, scores_list in method_scores.items():
        if not scores_list:
            continue

        avg_scores: Dict[str, float] = {}
        for key in ["reflectiveness", "grounding", "sentiment_score", "emotion_score", "intimacy_score", "empathy_score"]:
            values = [s.get(key, 0) for s in scores_list if isinstance(s.get(key), (int, float))]
            avg_scores[key] = round(sum(values) / len(values), 3) if values else 0.0

        summary[method] = {
            "num_evaluations": len(scores_list),
            "average_scores": avg_scores,
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Persona Simulation Benchmark on REALTALK dataset.")
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing REALTALK chat JSON files.")
    parser.add_argument("--output-dir", default="data/persona_simulation_eval", help="Output directory for results.")
    parser.add_argument("--min-context-sessions", type=int, default=2, help="Minimum context sessions before evaluation.")
    parser.add_argument("--max-eval-points", type=int, default=10, help="Max evaluation points per chat file.")
    parser.add_argument("--methods", nargs="+", default=["baseline_llm", "profile_only", "full_agent"],
                        help="Methods to evaluate: baseline_llm, profile_only, full_agent")
    args = parser.parse_args()

    config = ExperimentConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        min_context_sessions=args.min_context_sessions,
        max_eval_points_per_chat=args.max_eval_points,
        methods=args.methods,
    )

    run_persona_simulation_experiment(config)


if __name__ == "__main__":
    main()
