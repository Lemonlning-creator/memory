"""
Experiment 2c: Cross-Conversation Profile & Persona Consistency

This experiment validates that the multi-dimensional user modeling (Innovation 1)
captures STABLE personality traits rather than conversation-specific noise.

In the REALTALK dataset, each speaker appears in exactly 2 conversations.
We independently extract profiles/personas from each conversation, then compare:

1. SAME-PERSON pairs: profiles from 2 conversations with the same speaker
2. DIFFERENT-PERSON pairs (baseline): profiles from 2 different speakers

If the modeling is valid, same-person profiles should score significantly
higher on consistency than different-person profiles.
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
    PERSONA_EXTRACTION_SYSTEM_PROMPT,
    PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE,
    PROFILE_CONSISTENCY_SYSTEM_PROMPT,
    PROFILE_CONSISTENCY_USER_PROMPT_TEMPLATE,
    PERSONA_CONSISTENCY_SYSTEM_PROMPT,
    PERSONA_CONSISTENCY_USER_PROMPT_TEMPLATE,
)
from .persona_simulation import detect_speakers, flatten_messages, session_keys


# ---------------------------------------------------------------------------
# Per-conversation profile/persona extraction
# ---------------------------------------------------------------------------

def build_corpus(chat: Dict[str, Any], speaker: str) -> str:
    """Build a text corpus of one speaker's messages from a single conversation."""
    turns = flatten_messages(chat)
    lines = []
    for t in turns:
        if t["speaker"] == speaker:
            lines.append(f"{speaker}: {t['content']}")
        else:
            lines.append(f"Partner: {t['content']}")
    return "\n".join(lines[-200:])  # cap at last 200 turns for token budget


def extract_profile_from_chat(
    llm: LLMClient,
    chat: Dict[str, Any],
    user_speaker: str,
) -> Dict[str, Any]:
    """Extract a user profile from a single conversation."""
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
    # robust JSON extraction
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
        except json.JSONDecodeError:
            return {"error": "parse failed", "raw": raw[:500]}


def extract_persona_from_chat(
    llm: LLMClient,
    chat: Dict[str, Any],
    agent_speaker: str,
) -> Dict[str, Any]:
    """Extract an agent persona from a single conversation."""
    corpus = build_corpus(chat, agent_speaker)
    user_prompt = PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        agent_name=agent_speaker, corpus=corpus
    )
    raw = llm.chat(
        PERSONA_EXTRACTION_SYSTEM_PROMPT.format(agent_name=agent_speaker),
        user_prompt,
        temperature=0.3,
        max_tokens=800,
    )
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
        except json.JSONDecodeError:
            return {"error": "parse failed", "raw": raw[:500]}


# ---------------------------------------------------------------------------
# Consistency evaluation
# ---------------------------------------------------------------------------

def evaluate_profile_consistency(
    llm: LLMClient,
    profile_a: Dict[str, Any],
    profile_b: Dict[str, Any],
    source_a: str,
    source_b: str,
    speaker_name: str,
) -> Dict[str, Any]:
    """Evaluate how consistent two profiles of the same person are."""
    user_prompt = PROFILE_CONSISTENCY_USER_PROMPT_TEMPLATE.format(
        profile_a_json=json.dumps(profile_a, ensure_ascii=False, indent=2),
        profile_b_json=json.dumps(profile_b, ensure_ascii=False, indent=2),
        source_a=source_a,
        source_b=source_b,
        speaker_name=speaker_name,
    )
    raw = llm.chat(
        PROFILE_CONSISTENCY_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.1,
        max_tokens=1000,
    )
    return _parse_json(raw)


def evaluate_persona_consistency(
    llm: LLMClient,
    persona_a: Dict[str, Any],
    persona_b: Dict[str, Any],
    source_a: str,
    source_b: str,
    agent_name: str,
) -> Dict[str, Any]:
    """Evaluate how consistent two personas of the same agent are."""
    user_prompt = PERSONA_CONSISTENCY_USER_PROMPT_TEMPLATE.format(
        persona_a_json=json.dumps(persona_a, ensure_ascii=False, indent=2),
        persona_b_json=json.dumps(persona_b, ensure_ascii=False, indent=2),
        source_a=source_a,
        source_b=source_b,
        agent_name=agent_name,
    )
    raw = llm.chat(
        PERSONA_CONSISTENCY_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.1,
        max_tokens=800,
    )
    return _parse_json(raw)


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
# Speaker mapping: find who appears in multiple conversations
# ---------------------------------------------------------------------------

def build_speaker_map(
    dataset_dir: Path,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Build a map of speaker -> list of (chat_file, role, chat_stem).
    role is "user" (speaker_1) or "agent" (speaker_2).
    """
    speaker_map: Dict[str, List[Tuple[str, str, str]]] = {}
    for f in sorted(dataset_dir.glob("Chat_*.json")):
        chat = load_json(str(f))
        s1, s2 = detect_speakers(chat)
        speaker_map.setdefault(s1, []).append((str(f), "user", f.stem))
        speaker_map.setdefault(s2, []).append((str(f), "agent", f.stem))
    return speaker_map


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_consistency_experiment(
    dataset_dir: str = "dataset",
    output_dir: str = "data/consistency_eval",
    max_chats: Optional[int] = None,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()
    speaker_map = build_speaker_map(dataset_dir)

    # Find speakers appearing in 2+ conversations with the same role
    multi_chat_speakers = {}
    for speaker, appearances in speaker_map.items():
        if len(appearances) >= 2:
            multi_chat_speakers[speaker] = appearances

    print(f"[Consistency] Found {len(multi_chat_speakers)} speakers in multiple conversations")

    # Phase 1: Extract per-conversation profiles/personas
    per_chat_profiles: Dict[str, Dict[str, Dict]] = {}  # speaker -> {chat_stem -> profile}
    per_chat_personas: Dict[str, Dict[str, Dict]] = {}

    for speaker, appearances in sorted(multi_chat_speakers.items()):
        print(f"\n[Consistency] Extracting per-chat models for '{speaker}'")
        per_chat_profiles[speaker] = {}
        per_chat_personas[speaker] = {}

        for chat_path, role, chat_stem in appearances:
            chat = load_json(chat_path)
            if role == "user":
                print(f"  [{chat_stem}] Extracting user profile...")
                profile = extract_profile_from_chat(llm, chat, speaker)
                per_chat_profiles[speaker][chat_stem] = profile
            else:
                print(f"  [{chat_stem}] Extracting agent persona...")
                persona = extract_persona_from_chat(llm, chat, speaker)
                per_chat_personas[speaker][chat_stem] = persona

    # Save extracted models
    save_json(str(output_dir / "per_chat_profiles.json"), per_chat_profiles)
    save_json(str(output_dir / "per_chat_personas.json"), per_chat_personas)
    print(f"\n[Consistency] Saved per-chat profiles and personas")

    # Phase 2: Same-person consistency evaluation
    same_person_results: List[Dict[str, Any]] = []
    for speaker, appearances in sorted(multi_chat_speakers.items()):
        if len(appearances) < 2:
            continue
        # Take the first two appearances
        (path_a, role_a, stem_a), (path_b, role_b, stem_b) = appearances[0], appearances[1]

        if role_a == "user" and role_b == "user":
            pa = per_chat_profiles[speaker].get(stem_a, {})
            pb = per_chat_profiles[speaker].get(stem_b, {})
            if "error" in pa or "error" in pb:
                print(f"  [Skip] Parse error for {speaker}")
                continue
            print(f"  Evaluating profile consistency: {stem_a} vs {stem_b}")
            result = evaluate_profile_consistency(llm, pa, pb, stem_a, stem_b, speaker)
            same_person_results.append({
                "speaker": speaker,
                "type": "user_profile",
                "source_a": stem_a,
                "source_b": stem_b,
                "consistency": result,
            })
        elif role_a == "agent" and role_b == "agent":
            pa = per_chat_personas[speaker].get(stem_a, {})
            pb = per_chat_personas[speaker].get(stem_b, {})
            if "error" in pa or "error" in pb:
                print(f"  [Skip] Parse error for {speaker}")
                continue
            print(f"  Evaluating persona consistency: {stem_a} vs {stem_b}")
            result = evaluate_persona_consistency(llm, pa, pb, stem_a, stem_b, speaker)
            same_person_results.append({
                "speaker": speaker,
                "type": "agent_persona",
                "source_a": stem_a,
                "source_b": stem_b,
                "consistency": result,
            })

    # Phase 3: Different-person baseline (random cross-pairing)
    diff_person_results: List[Dict[str, Any]] = []
    all_profiles = []
    all_personas = []
    for speaker in per_chat_profiles:
        for stem, prof in per_chat_profiles[speaker].items():
            if "error" not in prof:
                all_profiles.append((speaker, stem, prof))
    for speaker in per_chat_personas:
        for stem, pers in per_chat_personas[speaker].items():
            if "error" not in pers:
                all_personas.append((speaker, stem, pers))

    # Compare a few different-person profile pairs
    import random
    random.seed(42)
    if len(all_profiles) >= 4:
        for _ in range(min(4, len(all_profiles))):
            idx_a, idx_b = random.sample(range(len(all_profiles)), 2)
            sp_a, stem_a, pa = all_profiles[idx_a]
            sp_b, stem_b, pb = all_profiles[idx_b]
            if sp_a == sp_b:
                continue
            print(f"  Baseline profile: {sp_a}({stem_a}) vs {sp_b}({stem_b})")
            result = evaluate_profile_consistency(llm, pa, pb, stem_a, stem_b, f"{sp_a} vs {sp_b}")
            diff_person_results.append({
                "speaker_a": sp_a,
                "speaker_b": sp_b,
                "type": "user_profile_baseline",
                "source_a": stem_a,
                "source_b": stem_b,
                "consistency": result,
            })

    if len(all_personas) >= 4:
        for _ in range(min(4, len(all_personas))):
            idx_a, idx_b = random.sample(range(len(all_personas)), 2)
            sp_a, stem_a, pa = all_personas[idx_a]
            sp_b, stem_b, pb = all_personas[idx_b]
            if sp_a == sp_b:
                continue
            print(f"  Baseline persona: {sp_a}({stem_a}) vs {sp_b}({stem_b})")
            result = evaluate_persona_consistency(llm, pa, pb, stem_a, stem_b, f"{sp_a} vs {sp_b}")
            diff_person_results.append({
                "speaker_a": sp_a,
                "speaker_b": sp_b,
                "type": "agent_persona_baseline",
                "source_a": stem_a,
                "source_b": stem_b,
                "consistency": result,
            })

    # Phase 4: Aggregate and save
    all_results = {
        "same_person": same_person_results,
        "different_person_baseline": diff_person_results,
        "config": {
            "dataset_dir": str(dataset_dir),
            "num_multi_chat_speakers": len(multi_chat_speakers),
        },
        "timestamp": datetime.now().isoformat(),
    }
    results_path = output_dir / "consistency_results.json"
    save_json(str(results_path), all_results)
    print(f"\n[Consistency] Results saved to {results_path}")

    summary = aggregate_consistency_results(same_person_results, diff_person_results)
    summary_path = output_dir / "consistency_summary.json"
    save_json(str(summary_path), summary)
    print(f"[Consistency] Summary saved to {summary_path}")

    return summary


def aggregate_consistency_results(
    same_person: List[Dict[str, Any]],
    diff_person: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate statistics."""
    def extract_scores(results):
        scores = []
        for r in results:
            c = r.get("consistency", {})
            if "error" in c:
                continue
            s = c.get("overall_consistency", 0)
            if isinstance(s, (int, float)):
                scores.append(s)
        return scores

    same_scores = extract_scores(same_person)
    diff_scores = extract_scores(diff_person)

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    # Section-level breakdown for same-person profiles
    section_scores = {}
    for r in same_person:
        c = r.get("consistency", {})
        if "error" in c:
            continue
        for section, info in c.get("section_scores", {}).items():
            if isinstance(info, dict):
                s = info.get("score", 0)
                if isinstance(s, (int, float)):
                    section_scores.setdefault(section, []).append(s)

    return {
        "same_person": {
            "num_pairs": len(same_scores),
            "avg_consistency": avg(same_scores),
            "min": min(same_scores) if same_scores else 0,
            "max": max(same_scores) if same_scores else 0,
            "scores": same_scores,
        },
        "different_person_baseline": {
            "num_pairs": len(diff_scores),
            "avg_consistency": avg(diff_scores),
            "min": min(diff_scores) if diff_scores else 0,
            "max": max(diff_scores) if diff_scores else 0,
            "scores": diff_scores,
        },
        "section_breakdown": {
            s: {"avg": avg(v), "count": len(v)} for s, v in section_scores.items()
        },
        "separation_gap": round(avg(same_scores) - avg(diff_scores), 3) if same_scores and diff_scores else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2c: Cross-Conversation Profile Consistency"
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/consistency_eval")
    args = parser.parse_args()

    run_consistency_experiment(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
