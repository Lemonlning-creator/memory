from __future__ import annotations

import configparser
import json
import os
import threading
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, Generator, List

from openai import OpenAI

from prompt import (
    UNIFIED_REASONING_SYSTEM_PROMPT,
    UNIFIED_REASONING_USER_PROMPT_TEMPLATE,
    RESPONSE_SYSTEM_PROMPT,
    RESPONSE_USER_PROMPT_TEMPLATE,
    MID_TERM_MEMORY_SYSTEM_PROMPT,
    MID_TERM_MEMORY_USER_PROMPT_TEMPLATE,
    LONG_TERM_MEMORY_SYSTEM_PROMPT,
    LONG_TERM_MEMORY_USER_PROMPT_TEMPLATE,
    PROFILE_EVOLUTION_SYSTEM_PROMPT,
    PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE,
)

DEFAULT_CONFIG_PATH = "config.ini"
DEFAULT_PROFILE_PATH = "user_profile.json"
DEFAULT_PERSONA_PATH = "agent_persona.json"
DEFAULT_MEMORY_PATH = "memory.json"

# =========================
# 1. LLM 客户端
# =========================
class LLMClient:
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")
        api_config = config["API"]
        self.model = api_config.get("model")
        self.enable_thinking = api_config.getboolean("enable_thinking", fallback=False)
        api_key = api_config.get("api_key")
        base_url = api_config.get("base_url")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.last_model_timing: Dict[str, float | None] = {
            "first_char_seconds": None,
        }
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

     # 非流式调用
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.6,
        max_tokens: int | None = None,
    ) -> str:
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "extra_body": {"enable_thinking": self.enable_thinking},
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**request)
        self._record_usage(response)
        return response.choices[0].message.content.strip()

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] += total_tokens
        self.token_usage["calls"] += 1

        print(
            f"[token usage] prompt={prompt_tokens}, "
            f"completion={completion_tokens}, total={total_tokens}"
        )
        
    def chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.6,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": True,
            "extra_body": {"enable_thinking": self.enable_thinking},
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        request_start = perf_counter()
        self.last_model_timing = {"first_char_seconds": None}
        completion = self.client.chat.completions.create(**request)

        for chunk in completion:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if content:
                if self.last_model_timing["first_char_seconds"] is None:
                    first_char_seconds = round(perf_counter() - request_start, 3)
                    self.last_model_timing["first_char_seconds"] = first_char_seconds
                    print(f"[model timing] input_to_first_char_seconds={first_char_seconds}")
                yield content

# =========================
# 2. 文件读写工具
# =========================
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
        "short_term_memory": {
            "max_messages": 20,
            "messages": [],
        },
        "mid_term_memory": {
            "summaries": [],
        },
        "long_term_memory": {
            "memories": [],
        },
    }


# =========================
# 3. 核心 Agent
# =========================
class StateDrivenCompanionAgent:
    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        profile_path: str = DEFAULT_PROFILE_PATH,
        persona_path: str = DEFAULT_PERSONA_PATH,
        memory_path: str = DEFAULT_MEMORY_PATH,
    ):
        self.llm = LLMClient(config_path)
        self.profile_path = profile_path
        self.persona_path = persona_path
        self.memory_path = memory_path
        self.user_profile = load_json(profile_path)
        self.persona_config = load_json(persona_path)
        self.memory_manager = MemoryManager(memory_path)
        self._background_memory_running = False
        self._background_memory_lock = threading.Lock()

    def _call_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.4) -> Dict[str, Any]:
        raw = self.llm.chat(system_prompt, user_prompt, temperature=temperature, max_tokens=700)
        return parse_json(raw)

    def _build_unified_reasoning_user_prompt(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
    ) -> str:
        static_profile = self.user_profile.get("static_profile", {})
        existing_current_state = self.user_profile.get("current_state", {})

        return UNIFIED_REASONING_USER_PROMPT_TEMPLATE.format(
            user_input=user_input,
            static_profile=json.dumps(self._compact_data(static_profile), ensure_ascii=False),
            existing_current_state=json.dumps(existing_current_state, ensure_ascii=False),
            persona_config=json.dumps(self._compact_data(self.persona_config), ensure_ascii=False),
            relevant_memory=json.dumps(self._compact_relevant_memory(relevant_memory), ensure_ascii=False),
        )

    def _build_fast_response_user_prompt(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
    ) -> str:
        static_profile = self.user_profile.get("static_profile", {})
        current_state = self.user_profile.get("current_state", {})
        activated_persona = {
            "empathy_level": "中",
            "teasing_level": "低",
            "warmth_level": "中",
            "guidance_level": "中",
            "activated_tone": "自然、简洁、友好，优先直接回应用户当前问题",
        }
        decision = {
            "reply_goal": "直接回应用户当前输入",
            "reply_strategy": "先回答问题，再给低成本建议；避免长篇分析",
            "content_focus": "用户当前最关心的内容",
            "avoid": ["过度铺垫", "输出分析过程", "提及内部状态推理"],
            "suggested_action": "给出一个可以马上尝试的小步骤",
        }

        return RESPONSE_USER_PROMPT_TEMPLATE.format(
            user_input=user_input,
            static_profile=json.dumps(self._compact_data(static_profile), ensure_ascii=False),
            current_state=json.dumps(current_state, ensure_ascii=False),
            relevant_memory=json.dumps(self._compact_relevant_memory(relevant_memory), ensure_ascii=False),
            persona_config=json.dumps(self._compact_data(self.persona_config), ensure_ascii=False),
            activated_persona=json.dumps(activated_persona, ensure_ascii=False),
            decision=json.dumps(decision, ensure_ascii=False),
        )

    def unified_reasoning(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._call_json(
            UNIFIED_REASONING_SYSTEM_PROMPT,
            self._build_unified_reasoning_user_prompt(user_input, relevant_memory),
            temperature=0.4,
        )

    def _compact_data(self, data: Any, max_chars: int = 1800) -> Any:
        text = json.dumps(data, ensure_ascii=False)
        if len(text) <= max_chars:
            return data
        return {"summary": text[:max_chars]}

    def _compact_relevant_memory(self, relevant_memory: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "recent_messages": relevant_memory.get("recent_messages", [])[-4:],
            "mid_term_summaries": relevant_memory.get("mid_term_summaries", [])[-2:],
            "long_term_memories": relevant_memory.get("long_term_memories", [])[-3:],
        }

    def _start_background_memory_pipeline(self, save_state: bool) -> None:
        with self._background_memory_lock:
            if self._background_memory_running:
                return
            self._background_memory_running = True

        thread = threading.Thread(
            target=self._run_background_memory_pipeline,
            args=(save_state,),
            daemon=True,
        )
        thread.start()

    def _start_background_post_chat_pipeline(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
        save_state: bool,
    ) -> None:
        with self._background_memory_lock:
            if self._background_memory_running:
                return
            self._background_memory_running = True

        thread = threading.Thread(
            target=self._run_background_post_chat_pipeline,
            args=(user_input, relevant_memory, save_state),
            daemon=True,
        )
        thread.start()

    def _update_state_from_reasoning(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
        save_state: bool,
    ) -> None:
        try:
            reasoning = self.unified_reasoning(user_input, relevant_memory)
            self.user_profile["current_state"] = reasoning.get(
                "current_state",
                self.user_profile.get("current_state", {}),
            )
            self.user_profile["projected_state"] = reasoning.get(
                "projected_state",
                self.user_profile.get("projected_state", {}),
            )
            if save_state:
                save_json(self.profile_path, self.user_profile)
        except Exception as e:
            print(f"[Background State Update Error] {e}")

    def _run_background_post_chat_pipeline(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
        save_state: bool,
    ) -> None:
        try:
            self._update_state_from_reasoning(user_input, relevant_memory, save_state)
            self.memory_manager.build_mid_term_summary(self.llm)
            self.memory_manager.extract_long_term_memory(self.llm)
            self.memory_manager.deduplicate_long_term_memory()
            self.user_profile = self.memory_manager.evolve_profile(self.llm, self.user_profile)
            self.memory_manager.save()
            if save_state:
                save_json(self.profile_path, self.user_profile)
        finally:
            with self._background_memory_lock:
                self._background_memory_running = False

    def _run_background_memory_pipeline(self, save_state: bool) -> None:
        try:
            threading.Event().wait(3)
            self.memory_manager.build_mid_term_summary(self.llm)
            self.memory_manager.extract_long_term_memory(self.llm)
            self.memory_manager.deduplicate_long_term_memory()
            self.user_profile = self.memory_manager.evolve_profile(self.llm, self.user_profile)
            self.memory_manager.save()
            if save_state:
                save_json(self.profile_path, self.user_profile)
        finally:
            with self._background_memory_lock:
                self._background_memory_running = False

    def _apply_reasoning_result(
        self,
        reasoning: Dict[str, Any],
        static_profile: Dict[str, Any],
        relevant_memory: Dict[str, Any],
        save_state: bool,
    ) -> Dict[str, Any]:
        current_state = reasoning.get("current_state", self.user_profile.get("current_state", {}))
        projected_state = reasoning.get("projected_state", self.user_profile.get("projected_state", {}))
        activated_persona = reasoning.get("activated_persona", {})
        decision = reasoning.get("decision", {})
        assistant_response = reasoning.get("response", "").strip()
        if not assistant_response:
            assistant_response = "I got stuck for a moment. Could you say that again?"

        self.memory_manager.append_stm("assistant", assistant_response)
        self.user_profile["current_state"] = current_state
        self.user_profile["projected_state"] = projected_state
        self.memory_manager.save()

        if save_state:
            save_json(self.profile_path, self.user_profile)

        self._start_background_memory_pipeline(save_state)

        return {
            "response": assistant_response,
            "current_state": self.user_profile["current_state"],
            "projected_state": self.user_profile["projected_state"],
            "activated_persona": activated_persona,
            "decision": decision,
            "static_profile": static_profile,
            "relevant_memory": relevant_memory,
            "background_memory_running": self._background_memory_running,
            "model_timing": self.llm.last_model_timing,
        }

    # =========================
    # 对话流程
    # =========================
    def chat(self, user_input: str, save_state: bool = True) -> Dict[str, Any]:
        self.memory_manager.append_stm("user", user_input)
        static_profile = self.user_profile.get("static_profile", {})
        relevant_memory = self.memory_manager.retrieve_relevant_memory(user_input)
        reasoning = self.unified_reasoning(user_input, relevant_memory)
        return self._apply_reasoning_result(reasoning, static_profile, relevant_memory, save_state)

    def chat_stream(
        self,
        user_input: str,
        save_state: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        self.memory_manager.append_stm("user", user_input)
        static_profile = self.user_profile.get("static_profile", {})
        relevant_memory = self.memory_manager.retrieve_relevant_memory(user_input)
        user_prompt = self._build_fast_response_user_prompt(user_input, relevant_memory)
        response_parts: List[str] = []

        for content in self.llm.chat_stream(
            RESPONSE_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.6,
            max_tokens=450,
        ):
            response_parts.append(content)
            yield {"type": "token", "content": content}

        assistant_response = "".join(response_parts).strip()
        if not assistant_response:
            assistant_response = "我刚才卡了一下，你再说一遍？"
            yield {"type": "token", "content": assistant_response}

        self.memory_manager.append_stm("assistant", assistant_response)
        self.memory_manager.save()

        if save_state:
            save_json(self.profile_path, self.user_profile)

        self._start_background_post_chat_pipeline(user_input, relevant_memory, save_state)

        yield {
            "type": "done",
            "response": assistant_response,
            "current_state": self.user_profile.get("current_state", {}),
            "projected_state": self.user_profile.get("projected_state", {}),
            "activated_persona": {},
            "decision": {},
            "static_profile": static_profile,
            "relevant_memory": relevant_memory,
            "background_memory_running": self._background_memory_running,
        }

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
        default_memory = create_default_memory()
        self.memory.setdefault("memory_meta", default_memory["memory_meta"])
        self.memory.setdefault("short_term_memory", default_memory["short_term_memory"])
        self.memory.setdefault("mid_term_memory", default_memory["mid_term_memory"])
        self.memory.setdefault("long_term_memory", default_memory["long_term_memory"])
        self.memory["memory_meta"].setdefault("total_turns", 0)
        self.memory["memory_meta"].setdefault("last_mid_term_turn", 0)
        self.memory["memory_meta"].setdefault("last_long_term_summary_count", 0)
        self.memory["memory_meta"].setdefault("last_profile_evolution_memory_count", 0)
        self.memory["short_term_memory"].setdefault("max_messages", 20)
        self.memory["short_term_memory"].setdefault("messages", [])
        self.memory["mid_term_memory"].setdefault("summaries", [])
        self.memory["long_term_memory"].setdefault("memories", [])

    def save(self):
        self.memory["memory_meta"]["last_updated"] = datetime.now().isoformat()
        save_json(self.memory_path, self.memory)
    
    def append_stm(self, role: str, content: str):
        stm = self.memory["short_term_memory"]
        stm["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        max_messages = stm["max_messages"]

        if len(stm["messages"]) > max_messages:
            stm["messages"] = stm["messages"][-max_messages:]
        if role == "assistant":
            self.memory["memory_meta"]["total_turns"] += 1
    
    def get_recent_messages(self, limit: int = 6) -> List[Dict[str, Any]]:
        messages = self.memory["short_term_memory"]["messages"]
        return messages[-limit:]

    def retrieve_relevant_memory(self, user_input: str) -> Dict[str, Any]:
        candidates = {
            "recent_messages": self.get_recent_messages(),
            "mid_term_summaries": self.memory.get(
                "mid_term_memory",
                {},
            ).get("summaries", [])[-5:],
            "long_term_memories": self.memory.get(
                "long_term_memory",
                {},
            ).get("memories", [])[-10:],
        }

        return self._retrieve_relevant_memory_locally(user_input, candidates)

    def _retrieve_relevant_memory_locally(
        self,
        user_input: str,
        candidates: Dict[str, Any],
    ) -> Dict[str, Any]:
        tokens = {
            token.lower()
            for token in user_input.replace(",", " ").replace("，", " ").split()
            if len(token.strip()) >= 2
        }
        if not tokens:
            return {
                "recent_messages": candidates["recent_messages"][-4:],
                "mid_term_summaries": candidates["mid_term_summaries"][-3:],
                "long_term_memories": candidates["long_term_memories"][-5:],
            }

        def score(item: Any) -> int:
            text = json.dumps(item, ensure_ascii=False).lower()
            return sum(1 for token in tokens if token in text)

        def top_relevant(items: List[Any], fallback_count: int, max_count: int) -> List[Any]:
            ranked = sorted(
                ((score(item), index, item) for index, item in enumerate(items)),
                key=lambda row: (row[0], row[1]),
                reverse=True,
            )
            selected = [item for item_score, _, item in ranked if item_score > 0][:max_count]
            return selected or items[-fallback_count:]

        return {
            "recent_messages": candidates["recent_messages"][-4:],
            "mid_term_summaries": top_relevant(candidates["mid_term_summaries"], 2, 3),
            "long_term_memories": top_relevant(candidates["long_term_memories"], 3, 5),
        }
    
    def build_mid_term_summary(self, llm):
        stm_messages = self.memory["short_term_memory"]["messages"]
        if len(stm_messages) < 6:
            return
        meta = self.memory.setdefault("memory_meta", {})
        total_turns = meta.get("total_turns", 0)
        last_mid_term_turn = meta.get("last_mid_term_turn", 0)
        if total_turns - last_mid_term_turn < 3:
            return
        conversation = "\n".join([
            f'{m["role"]}: {m["content"]}' for m in stm_messages
        ])
        try:
            result = parse_json(
                llm.chat(
                    MID_TERM_MEMORY_SYSTEM_PROMPT,
                    MID_TERM_MEMORY_USER_PROMPT_TEMPLATE.format(
                        conversation=conversation
                    )
                )
            )
            result["id"] = (
                f'mtm_{len(self.memory["mid_term_memory"]["summaries"]) + 1}'
            )
            result["evidence_count"] = len(stm_messages)
            result["created_at"] = datetime.now().isoformat()
            result["updated_at"] = datetime.now().isoformat()
            self.memory["mid_term_memory"]["summaries"].append(result)
            meta["last_mid_term_turn"] = total_turns
        except Exception as e:
            print(f"[Mid-term Memory Error] {e}")
        
    def extract_long_term_memory(self, llm):
        summaries = self.memory["mid_term_memory"]["summaries"]
        if len(summaries) < 3:
            return
        meta = self.memory.setdefault("memory_meta", {})
        last_summary_count = meta.get("last_long_term_summary_count", 0)
        if len(summaries) - last_summary_count < 3:
            return
        try:
            result = parse_json(
                llm.chat(
                    LONG_TERM_MEMORY_SYSTEM_PROMPT,
                    LONG_TERM_MEMORY_USER_PROMPT_TEMPLATE.format(
                        mid_term_summaries=json.dumps(
                            summaries[-5:],
                            ensure_ascii=False,
                            indent=2
                        )
                    )
                )
            )
            result["id"] = ( f'ltm_{len(self.memory["long_term_memory"]["memories"]) + 1}' )
            result["source"] = [
                s["id"] for s in summaries[-3:]
            ]
            result["created_at"] = datetime.now().isoformat()
            result["updated_at"] = datetime.now().isoformat()
            self.memory["long_term_memory"]["memories"].append(result)
            meta["last_long_term_summary_count"] = len(summaries)
        except Exception as e:
            print(f"[Long-term Memory Error] {e}")
    
    def deduplicate_long_term_memory(self):
        memories = self.memory["long_term_memory"]["memories"]
        unique = {}
        for memory in memories:
            key = memory["content"]
            if key not in unique:
                unique[key] = memory
            else:
                if (memory.get("confidence", 0) > unique[key].get("confidence", 0)):
                    unique[key] = memory

        self.memory["long_term_memory"]["memories"] = list(unique.values())
    
    def evolve_profile(self, llm, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        long_term_memories = (
            self.memory["long_term_memory"]["memories"]
        )
        if len(long_term_memories) < 3:
            return user_profile
        meta = self.memory.setdefault("memory_meta", {})
        last_memory_count = meta.get("last_profile_evolution_memory_count", 0)
        if len(long_term_memories) - last_memory_count < 3:
            return user_profile
        static_profile = user_profile.get("static_profile", {})

        try:
            updated_profile = parse_json(
                llm.chat(
                    PROFILE_EVOLUTION_SYSTEM_PROMPT,
                    PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE.format(
                        static_profile=json.dumps(
                            static_profile,
                            ensure_ascii=False,
                            indent=2
                        ),
                        long_term_memories=json.dumps(
                            long_term_memories[-5:],
                            ensure_ascii=False,
                            indent=2
                        )
                    )
                )
            )
            user_profile["static_profile"] = updated_profile
            meta["last_profile_evolution_memory_count"] = len(long_term_memories)
            return user_profile

        except Exception as e:
            print(f"[Profile Evolution Error] {e}")
            return user_profile