from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from .utils import load_json, save_json, parse_json, create_default_memory
from .prompts import (
    MID_TERM_MEMORY_SYSTEM_PROMPT,
    MID_TERM_MEMORY_USER_PROMPT_TEMPLATE,
    LONG_TERM_MEMORY_SYSTEM_PROMPT,
    LONG_TERM_MEMORY_USER_PROMPT_TEMPLATE,
    PROFILE_EVOLUTION_SYSTEM_PROMPT,
    PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE,
)


class MemoryManager:
    def __init__(self, memory_path: str):
        self.memory_path = memory_path
        self.memory = self._load_or_create_memory()
        self._ensure_schema()

    def _load_or_create_memory(self) -> Dict[str, Any]:
        try:
            return load_json(self.memory_path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[Memory Load Warning] {e}; recreating {self.memory_path}")
            memory = create_default_memory()
            save_json(self.memory_path, memory)
            return memory

    def _ensure_schema(self) -> None:
        default = create_default_memory()
        self.memory.setdefault("memory_meta", default["memory_meta"])
        self.memory.setdefault("short_term_memory", default["short_term_memory"])
        self.memory.setdefault("mid_term_memory", default["mid_term_memory"])
        self.memory.setdefault("long_term_memory", default["long_term_memory"])
        meta = self.memory["memory_meta"]
        meta.setdefault("total_turns", 0)
        meta.setdefault("last_mid_term_turn", 0)
        meta.setdefault("last_long_term_summary_count", 0)
        meta.setdefault("last_profile_evolution_memory_count", 0)
        self.memory["short_term_memory"].setdefault("max_messages", 20)
        self.memory["short_term_memory"].setdefault("messages", [])
        self.memory["mid_term_memory"].setdefault("summaries", [])
        self.memory["long_term_memory"].setdefault("memories", [])

    def save(self) -> None:
        self.memory["memory_meta"]["last_updated"] = datetime.now().isoformat()
        save_json(self.memory_path, self.memory)

    def reset(self) -> None:
        self.memory = create_default_memory()
        save_json(self.memory_path, self.memory)

    def append_stm(self, role: str, content: str) -> None:
        stm = self.memory["short_term_memory"]
        stm["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        if len(stm["messages"]) > stm["max_messages"]:
            stm["messages"] = stm["messages"][-stm["max_messages"]:]
        if role == "assistant":
            self.memory["memory_meta"]["total_turns"] += 1

    def get_recent_messages(self, limit: int = 6) -> List[Dict[str, Any]]:
        return self.memory["short_term_memory"]["messages"][-limit:]

    def retrieve_relevant_memory(self, user_input: str) -> Dict[str, Any]:
        candidates = {
            "recent_messages": self.get_recent_messages(),
            "mid_term_summaries": self.memory["mid_term_memory"].get("summaries", [])[-5:],
            "long_term_memories": self.memory["long_term_memory"].get("memories", [])[-10:],
        }
        tokens = {
            t.lower()
            for t in user_input.replace(",", " ").replace("，", " ").split()
            if len(t.strip()) >= 2
        }
        if not tokens:
            return {
                "recent_messages": candidates["recent_messages"][-4:],
                "mid_term_summaries": candidates["mid_term_summaries"][-3:],
                "long_term_memories": candidates["long_term_memories"][-5:],
            }

        def score(item: Any) -> int:
            text = json.dumps(item, ensure_ascii=False).lower()
            return sum(1 for t in tokens if t in text)

        def top_relevant(items: List[Any], fallback: int, limit: int) -> List[Any]:
            ranked = sorted(
                ((score(item), i, item) for i, item in enumerate(items)),
                key=lambda r: (r[0], r[1]),
                reverse=True,
            )
            selected = [item for s, _, item in ranked if s > 0][:limit]
            return selected or items[-fallback:]

        return {
            "recent_messages": candidates["recent_messages"][-4:],
            "mid_term_summaries": top_relevant(candidates["mid_term_summaries"], 2, 3),
            "long_term_memories": top_relevant(candidates["long_term_memories"], 3, 5),
        }

    def build_mid_term_summary(self, llm) -> None:
        stm_messages = self.memory["short_term_memory"]["messages"]
        if len(stm_messages) < 6:
            return
        meta = self.memory.setdefault("memory_meta", {})
        if meta.get("total_turns", 0) - meta.get("last_mid_term_turn", 0) < 3:
            return
        conversation = "\n".join(f'{m["role"]}: {m["content"]}' for m in stm_messages)
        try:
            result = parse_json(llm.chat(
                MID_TERM_MEMORY_SYSTEM_PROMPT,
                MID_TERM_MEMORY_USER_PROMPT_TEMPLATE.format(conversation=conversation),
            ))
            result["id"] = f'mtm_{len(self.memory["mid_term_memory"]["summaries"]) + 1}'
            result["evidence_count"] = len(stm_messages)
            result["created_at"] = result["updated_at"] = datetime.now().isoformat()
            self.memory["mid_term_memory"]["summaries"].append(result)
            meta["last_mid_term_turn"] = meta.get("total_turns", 0)
        except Exception as e:
            print(f"[Mid-term Memory Error] {e}")

    def extract_long_term_memory(self, llm) -> None:
        summaries = self.memory["mid_term_memory"]["summaries"]
        if len(summaries) < 3:
            return
        meta = self.memory.setdefault("memory_meta", {})
        if len(summaries) - meta.get("last_long_term_summary_count", 0) < 3:
            return
        try:
            result = parse_json(llm.chat(
                LONG_TERM_MEMORY_SYSTEM_PROMPT,
                LONG_TERM_MEMORY_USER_PROMPT_TEMPLATE.format(
                    mid_term_summaries=json.dumps(summaries[-5:], ensure_ascii=False, indent=2),
                ),
            ))
            result["id"] = f'ltm_{len(self.memory["long_term_memory"]["memories"]) + 1}'
            result["source"] = [s["id"] for s in summaries[-3:]]
            result["created_at"] = result["updated_at"] = datetime.now().isoformat()
            self.memory["long_term_memory"]["memories"].append(result)
            meta["last_long_term_summary_count"] = len(summaries)
        except Exception as e:
            print(f"[Long-term Memory Error] {e}")

    def deduplicate_long_term_memory(self) -> None:
        memories = self.memory["long_term_memory"]["memories"]
        unique: Dict[str, Any] = {}
        for m in memories:
            key = m["content"]
            if key not in unique or m.get("confidence", 0) > unique[key].get("confidence", 0):
                unique[key] = m
        self.memory["long_term_memory"]["memories"] = list(unique.values())

    def evolve_profile(self, llm, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        memories = self.memory["long_term_memory"]["memories"]
        if len(memories) < 3:
            return user_profile
        meta = self.memory.setdefault("memory_meta", {})
        if len(memories) - meta.get("last_profile_evolution_memory_count", 0) < 3:
            return user_profile
        static_profile = user_profile["state_axis"]["static_profile"]
        try:
            updated = parse_json(llm.chat(
                PROFILE_EVOLUTION_SYSTEM_PROMPT,
                PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE.format(
                    static_profile=json.dumps(static_profile, ensure_ascii=False, indent=2),
                    long_term_memories=json.dumps(memories[-5:], ensure_ascii=False, indent=2),
                ),
            ))
            user_profile["state_axis"]["static_profile"] = updated
            meta["last_profile_evolution_memory_count"] = len(memories)
        except Exception as e:
            print(f"[Profile Evolution Error] {e}")
        return user_profile
