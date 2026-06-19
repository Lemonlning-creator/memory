from __future__ import annotations

import argparse
import re

from pathlib import Path
from typing import Any, Dict, List
from ..llm_client import LLMClient
from ..utils import parse_json, save_json, load_json
from ..prompts.templates import (
    PERSONA_SYSTEM_PROMPT,
    PERSONA_USER_PROMPT_TEMPLATE,
)

def detect_agent_speaker(chat: Dict[str, Any]) -> str:
    names = chat.get("name", {})
    if isinstance(names, dict) and names.get("speaker_2"):
        return str(names["speaker_2"])
    return "agent"

def collect_speaker_utterances(chat: Dict[str, Any], speaker_name: str) -> List[Dict[str, Any]]:
    utterances: List[Dict[str, Any]] = []
    for session_id in session_keys(chat):
        for message in chat[session_id]:
            speaker = str(message.get("speaker") or "").strip()
            content = str(message.get("clean_text") or "").strip()
            if speaker != speaker_name or not content:
                continue
            utterances.append({
                "session_id": session_id,
                "dia_id": message.get("dia_id", ""),
                "content": content,
            })
    return utterances

def session_keys(chat: Dict[str, Any]) -> List[str]:
    keys = [
        key for key, value in chat.items()
        if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
    ]
    return sorted(keys, key=lambda key: int(key.split("_")[1]))

def format_utterances(
    utterances: List[Dict[str, Any]],
    max_utterances: int,
    max_chars: int,
) -> str:
    if len(utterances) <= max_utterances:
        selected = utterances
    else:
        recent_count = max_utterances // 2
        history_count = max_utterances - recent_count
        history_pool = utterances[:-recent_count]
        recent = utterances[-recent_count:]
        history_step = max(1, len(history_pool) // history_count)
        history = history_pool[::history_step][:history_count]
        selected = history + recent
    lines: List[str] = []
    total_chars = 0

    for index, item in enumerate(selected, start=1):
        line = f"{index}. [{item['session_id']}:{item.get('dia_id', '')}] {item['content']}"
        if total_chars + len(line) > max_chars:
            break
        lines.append(line)
        total_chars += len(line)
    return "\n".join(lines)

def build_agent_persona(
    realtalk_path: str | Path,
    output_dir: str | Path = "agent",
    config_path: str = "config.ini",
    max_utterances: int = 180,
    max_chars: int = 24000,
) -> None:
    source_path = Path(realtalk_path)
    chat = load_json(source_path)
    target_speaker = detect_agent_speaker(chat)
    utterances = collect_speaker_utterances(chat, target_speaker)
    if not utterances:
        raise ValueError(f"No utterances found for speaker: {target_speaker}")

    print(
        "[Persona Generation] start",
        f"source={source_path}",
        f"speaker={target_speaker} utterances={len(utterances)}"
    )

    llm = LLMClient(config_path)
    utterance_text = format_utterances(
        utterances,
        max_utterances=max_utterances,
        max_chars=max_chars,
    )
    persona = parse_json(llm.chat(
        PERSONA_SYSTEM_PROMPT,
        PERSONA_USER_PROMPT_TEMPLATE.format(
            speaker_name=target_speaker,
            utterances=utterance_text
        ),
        temperature=0.2,
        max_tokens=1800,
    ))

    output_path = Path(output_dir) / f"{target_speaker}_persona.json"
    save_json(str(output_path), persona)
    print(f"[Persona Generation] saved={output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an agent persona from REALTALK speaker_2 utterances.")
    parser.add_argument("--realtalk", required=True, help="Path to a REALTALK chat JSON file.")
    parser.add_argument("--output-dir", default="agent", help="Defaults to agent/{speaker_name}_persona.json.")
    parser.add_argument("--config", default="config.ini", help="LLM config path.")
    parser.add_argument("--max-utterances", type=int, default=180, help="Max utterances sent to LLM.")
    parser.add_argument("--max-chars", type=int, default=24000, help="Max prompt chars for utterance list.")
    args = parser.parse_args()

    build_agent_persona(
        realtalk_path=args.realtalk,
        output_dir=args.output_dir,
        config_path=args.config,
        max_utterances=args.max_utterances,
        max_chars=args.max_chars,
    )

if __name__ == "__main__":
    main()
