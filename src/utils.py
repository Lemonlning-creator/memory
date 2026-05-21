from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def create_default_memory() -> Dict[str, Any]:
    now = datetime.now().isoformat()
    return {
        "memory_meta": {
            "created_at": now,
            "last_updated": now,
            "total_turns": 0,
            "last_mid_term_turn": 0,
            "last_long_term_summary_count": 0,
            "last_profile_evolution_memory_count": 0,
        },
        "short_term_memory": {"max_messages": 20, "messages": []},
        "mid_term_memory": {"summaries": []},
        "long_term_memory": {"memories": []},
    }
