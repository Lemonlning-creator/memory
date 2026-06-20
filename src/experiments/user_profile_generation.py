from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..agent import StateDrivenCompanionAgent
from ..memory_os_local import MemoryOSLocal
from ..utils import save_json, load_json
from .agent_persona_generation import detect_agent_speaker

def detect_user_speaker(chat: Dict[str, Any]) -> str:
    names = chat.get("name", {})
    if isinstance(names, dict) and names.get("speaker_1"):
        return str(names["speaker_1"])
    return "default_user"

def session_keys(chat: Dict[str, Any]) -> List[str]:
    keys = [
        key for key, value in chat.items()
        if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
    ]
    return sorted(keys, key=lambda key: int(key.split("_")[1]))

def flatten_dialogue(chat: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for key in session_keys(chat):
        messages = chat[key]
        for message_index, message in enumerate(messages, start=1):
            content = str(message.get("clean_text") or "").strip()
            if not content:
                continue
            turns.append({
                "turn_index": len(turns) + 1,
                "session_id": key,
                "message_index": message_index,
                "session_message_count": len(messages),
                "dia_id": message.get("dia_id", ""),
                "speaker": str(message.get("speaker") or "").strip(),
                "content": content,
            })
    return turns

def checkpoint_path_for(memory_path: Path) -> Path:
    return memory_path / "pipeline_checkpoint.json"

def load_checkpoint(memory_path: Path) -> Dict[str, Any] | None:
    path = checkpoint_path_for(memory_path)
    if not path.exists():
        return None
    return load_json(path)

def save_checkpoint(memory_path: Path, payload: Dict[str, Any]) -> None:
    save_json(str(checkpoint_path_for(memory_path)), payload)

def run_profile_pipeline(
    realtalk_path: str | Path,
    profile_path: str | None = None,
    resume: bool = False,
    start_turn: int | None = None,
    replay_window: int = 20
) -> None:
    source_path = Path(realtalk_path)
    chat = load_json(source_path)
    user_name = detect_user_speaker(chat)
    agent_name = detect_agent_speaker(chat)
    persona_path = f"agent/{agent_name}_persona.json"
    profile_name = user_name + "_" + agent_name
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("[Profile Generation] start", f"source={source_path}", f"speaker={user_name}")
    agent = StateDrivenCompanionAgent(profile_path=profile_path, persona_path=persona_path, user_name=profile_name)

    memory_path = Path("data") / "realtalk_memory_runs" / f"{profile_name}_{run_id}"

    checkpoint = load_checkpoint(memory_path) if resume else None
    if start_turn is not None:
        replay_start_turn = max(1, start_turn)
        resume_source = "--start-turn"
    elif checkpoint:
        last_completed_turn = int(checkpoint.get("last_completed_turn", 0) or 0)
        replay_start_turn = max(1, last_completed_turn - replay_window + 1)
        resume_source = "checkpoint"
    else:
        replay_start_turn = 1
        resume_source = "fresh"

    agent.memory_manager = MemoryOSLocal(
        persist_path=str(memory_path)
    )
    print(f"[Pipeline] resume={resume} resume_source={resume_source} replay_start_turn={replay_start_turn}")

    target_turns = 0
    current_session = ""
    turns = flatten_dialogue(chat)
    for turn in turns:
        if turn["turn_index"] < replay_start_turn or turn["speaker"] != user_name:
            continue

        if turn["session_id"] != current_session:
            current_session = turn["session_id"]
            print(f"[Pipeline] session key={current_session}")

        agent.observe_dialogue_turn("user", turn["content"])

        target_turns += 1

        save_checkpoint(memory_path, {
            "source_path": str(source_path),
            "user_name": user_name,
            "profile_path": agent.profile_path,
            "memory_path": str(memory_path),
            "last_completed_turn": turn["turn_index"],
            "updated_at": datetime.now().isoformat(),
        })

    finalize_result = agent.finalize_session()
    save_json(agent.profile_path, agent.user_profile)
    print(f"[Pipeline] profile saved path={agent.profile_path}")

    result: Dict[str, Any] = {
        "source_path": str(source_path),
        "user_name": user_name,
        "profile_path": agent.profile_path,
        "memory_path": str(memory_path),
        "replay_start_turn": replay_start_turn,
        "target_turns": target_turns,
        "finalize_result": finalize_result,
    }

    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a user profile from a REALTALK chat file.")
    parser.add_argument("--realtalk", required=True, help="Path to a REALTALK chat JSON file.")
    parser.add_argument("--profile-path", default="user", help="Defaults to {user_name}_{agent_name}_profile.json.")
    parser.add_argument("--start-turn", type=int, default=None, help="Replay from this 1-based dialogue turn.")
    parser.add_argument("--replay-window", type=int, default=20, help="When resuming from checkpoint, replay this many previous turns.")
    args = parser.parse_args()

    run_profile_pipeline(
        realtalk_path=args.realtalk,
        profile_path=args.profile_path,
        start_turn=args.start_turn,
        replay_window=args.replay_window
    )

if __name__ == "__main__":
    main()