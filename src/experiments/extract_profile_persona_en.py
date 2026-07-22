"""
Extract user profile and persona configuration from REALTALK dataset (English version).
Usage: python src/experiments/extract_profile_persona_en.py --dataset <path_to_dataset>
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.llm_client import LLMClient
from src.prompts.templates_en import (
    PROFILE_EXTRACTION_SYSTEM_PROMPT,
    PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
    PERSONA_EXTRACTION_SYSTEM_PROMPT,
    PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE,
)

CONFIG_PATH = "config.ini"

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_sessions(path: str) -> List[List[Dict[str, Any]]]:
    """Load conversation sessions from a REALTALK JSON file."""
    data = json.load(open(path, encoding="utf-8"))
    sessions = sorted(
        [k for k in data if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1]),
    )
    return [data[s] for s in sessions]


def format_session(messages: List[Dict[str, Any]]) -> str:
    """Format a single session as a readable string."""
    return "\n".join(
        f'{m["speaker"]}: {m["clean_text"]}'
        for m in messages if m.get("clean_text", "").strip()
    )


def format_all_sessions(sessions: List[List[Dict[str, Any]]]) -> str:
    """Format all sessions into a single readable string."""
    parts = []
    for i, s in enumerate(sessions, 1):
        parts.append(f"=== Session {i} ===\n{format_session(s)}")
    return "\n\n".join(parts)


def detect_speakers(chat_data: Dict[str, Any]) -> tuple[str, str]:
    """Detect user and agent speaker names from chat data."""
    names = chat_data.get("name", {})
    speaker_1 = str(names.get("speaker_1", "speaker_1")).strip()
    speaker_2 = str(names.get("speaker_2", "speaker_2")).strip()
    return speaker_1, speaker_2


# ── Core steps ────────────────────────────────────────────────────────────────

def extract_profile(llm: LLMClient, train_sessions: List[List[Dict[str, Any]]], user_name: str) -> Dict[str, Any]:
    """Extract user profile from training sessions."""
    print(f"[1/2] Extracting {user_name}'s profile...")
    corpus = format_all_sessions(train_sessions)
    
    system_prompt = PROFILE_EXTRACTION_SYSTEM_PROMPT.format(user_name=user_name)
    user_prompt = PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE.format(user_name=user_name, corpus=corpus)
    
    raw = llm.chat(system_prompt, user_prompt)
    profile = json.loads(raw.strip().strip("```json").strip("```").strip())
    print(f"  Profile extraction completed")
    return profile


def extract_persona(llm: LLMClient, train_sessions: List[List[Dict[str, Any]]], agent_name: str) -> Dict[str, Any]:
    """Extract agent persona from training sessions."""
    print(f"[2/2] Extracting {agent_name}'s persona...")
    corpus = format_all_sessions(train_sessions)
    
    system_prompt = PERSONA_EXTRACTION_SYSTEM_PROMPT.format(agent_name=agent_name)
    user_prompt = PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE.format(agent_name=agent_name, corpus=corpus)
    
    raw = llm.chat(system_prompt, user_prompt)
    persona = json.loads(raw.strip().strip("```json").strip("```").strip())
    print(f"  Persona extraction completed")
    return persona


def main():
    parser = argparse.ArgumentParser(description="Extract user profile and persona from REALTALK dataset (English version)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to REALTALK chat JSON file")
    parser.add_argument("--train-sessions", type=int, default=15, help="Number of sessions to use for extraction (default: 15)")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Path to config.ini file")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    # Load chat data and detect speakers
    with open(dataset_path, "r", encoding="utf-8") as f:
        chat_data = json.load(f)
    
    user_name, agent_name = detect_speakers(chat_data)
    print(f"Detected speakers: user={user_name}, agent={agent_name}")
    
    # Load sessions
    all_sessions = load_sessions(str(dataset_path))
    print(f"Total {len(all_sessions)} sessions, using first {args.train_sessions} for extraction\n")
    
    train = all_sessions[:args.train_sessions]
    
    # Initialize LLM client
    llm = LLMClient(args.config)
    
    # Extract profile and persona
    profile = extract_profile(llm, train, user_name)
    persona = extract_persona(llm, train, agent_name)
    
    # Create output directories
    output_dir = dataset_path.parent / "output"
    user_dir = output_dir / "user"
    agent_dir = output_dir / "agent"
    
    user_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)
    
    # Save profile and persona
    user_filename = user_name.lower().replace(" ", "_")
    agent_filename = agent_name.lower().replace(" ", "_")
    
    profile_path = user_dir / f"{user_filename}_profile.json"
    persona_path = agent_dir / f"{agent_filename}_persona.json"
    
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    
    with open(persona_path, "w", encoding="utf-8") as f:
        json.dump(persona, f, ensure_ascii=False, indent=2)
    
    print(f"\nExtraction completed. Files saved:")
    print(f"  - {profile_path}")
    print(f"  - {persona_path}")


if __name__ == "__main__":
    main()
