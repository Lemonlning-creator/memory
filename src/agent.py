from __future__ import annotations

import json
import threading
from typing import Any, Dict, Generator, List

from .llm_client import LLMClient
from .memory_manager import MemoryManager
from .profile_utils import state_axis, context_axis, flatten_static_profile, migrate_profile
from .utils import load_json, save_json, parse_json
from .prompts import (
    UNIFIED_REASONING_SYSTEM_PROMPT,
    UNIFIED_REASONING_USER_PROMPT_TEMPLATE,
    RESPONSE_SYSTEM_PROMPT,
    RESPONSE_USER_PROMPT_TEMPLATE,
)

DEFAULT_CONFIG_PATH = "config.ini"
DEFAULT_PROFILE_PATH = "user_profile.json"
DEFAULT_PERSONA_PATH = "agent_persona.json"
DEFAULT_MEMORY_PATH = "memory.json"


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
        raw_profile = load_json(profile_path)
        if "static_profile" in raw_profile:
            self.user_profile = migrate_profile(raw_profile)
            save_json(profile_path, self.user_profile)
        else:
            self.user_profile = raw_profile
        self.persona_config = load_json(persona_path)
        self.memory_manager = MemoryManager(memory_path)
        # 每次启动清空动态状态和记忆
        sa = state_axis(self.user_profile)
        sa["current_state"] = {}
        sa["projected_state"] = {}
        self.user_profile["context_axis"] = {"current_context": "其他", "context_detail": "", "inferred_at_turn": 0}
        save_json(profile_path, self.user_profile)
        self.memory_manager.reset()
        self._background_memory_running = False
        self._background_memory_lock = threading.Lock()

    # ---------- prompt builders ----------

    def _compact(self, data: Any, max_chars: int = 1800) -> Any:
        text = json.dumps(data, ensure_ascii=False)
        return data if len(text) <= max_chars else {"summary": text[:max_chars]}

    def _compact_memory(self, mem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "recent_messages": mem.get("recent_messages", [])[-4:],
            "mid_term_summaries": mem.get("mid_term_summaries", [])[-2:],
            "long_term_memories": mem.get("long_term_memories", [])[-3:],
        }

    def _reasoning_prompt(self, user_input: str, relevant_memory: Dict[str, Any]) -> str:
        sa = state_axis(self.user_profile)
        return UNIFIED_REASONING_USER_PROMPT_TEMPLATE.format(
            user_input=user_input,
            static_profile=json.dumps(self._compact(flatten_static_profile(sa.get("static_profile", {}))), ensure_ascii=False),
            existing_current_state=json.dumps(sa.get("current_state", {}), ensure_ascii=False),
            existing_context=json.dumps(context_axis(self.user_profile), ensure_ascii=False),
            persona_config=json.dumps(self._compact(self.persona_config), ensure_ascii=False),
            relevant_memory=json.dumps(self._compact_memory(relevant_memory), ensure_ascii=False),
        )

    def _response_prompt(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
        activated_persona: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> str:
        sa = state_axis(self.user_profile)
        return RESPONSE_USER_PROMPT_TEMPLATE.format(
            user_input=user_input,
            static_profile=json.dumps(self._compact(flatten_static_profile(sa.get("static_profile", {}))), ensure_ascii=False),
            current_state=json.dumps(sa.get("current_state", {}), ensure_ascii=False),
            current_context=context_axis(self.user_profile).get("current_context", "其他"),
            relevant_memory=json.dumps(self._compact_memory(relevant_memory), ensure_ascii=False),
            persona_config=json.dumps(self._compact(self.persona_config), ensure_ascii=False),
            activated_persona=json.dumps(activated_persona, ensure_ascii=False),
            decision=json.dumps(decision, ensure_ascii=False),
        )

    # ---------- state helpers ----------

    def _apply_reasoning(self, reasoning: Dict[str, Any]) -> None:
        sa = state_axis(self.user_profile)
        sa["current_state"] = reasoning.get("current_state", sa.get("current_state", {}))
        sa["projected_state"] = reasoning.get("projected_state", sa.get("projected_state", {}))
        if "context" in reasoning:
            ctx = context_axis(self.user_profile)
            ctx.update(reasoning["context"])
            ctx["inferred_at_turn"] = self.memory_manager.memory["memory_meta"].get("total_turns", 0)

    # ---------- background pipelines ----------

    def _start_background(self, target, args: tuple) -> None:
        with self._background_memory_lock:
            if self._background_memory_running:
                return
            self._background_memory_running = True
        threading.Thread(target=target, args=args, daemon=True).start()

    def _memory_pipeline(self, save_state: bool) -> None:
        try:
            threading.Event().wait(3)
            self._run_memory_steps(save_state)
        finally:
            with self._background_memory_lock:
                self._background_memory_running = False

    def _post_chat_pipeline(self, user_input: str, relevant_memory: Dict[str, Any], save_state: bool) -> None:
        try:
            reasoning = parse_json(self.llm.chat(
                UNIFIED_REASONING_SYSTEM_PROMPT,
                self._reasoning_prompt(user_input, relevant_memory),
                temperature=0.4,
            ))
            self._apply_reasoning(reasoning)
            if save_state:
                save_json(self.profile_path, self.user_profile)
            self._run_memory_steps(save_state)
        except Exception as e:
            print(f"[Background State Update Error] {e}")
        finally:
            with self._background_memory_lock:
                self._background_memory_running = False

    def _run_memory_steps(self, save_state: bool) -> None:
        self.memory_manager.build_mid_term_summary(self.llm)
        self.memory_manager.extract_long_term_memory(self.llm)
        self.memory_manager.deduplicate_long_term_memory()
        self.user_profile = self.memory_manager.evolve_profile(self.llm, self.user_profile)
        self.memory_manager.save()
        if save_state:
            save_json(self.profile_path, self.user_profile)

    # ---------- public API ----------

    def chat(self, user_input: str, save_state: bool = True) -> Dict[str, Any]:
        self.memory_manager.append_stm("user", user_input)
        relevant_memory = self.memory_manager.retrieve_relevant_memory(user_input)

        reasoning = parse_json(self.llm.chat(
            UNIFIED_REASONING_SYSTEM_PROMPT,
            self._reasoning_prompt(user_input, relevant_memory),
            temperature=0.4,
        ))
        self._apply_reasoning(reasoning)

        activated_persona = reasoning.get("activated_persona", {})
        decision = reasoning.get("decision", {})
        response = self.llm.chat(
            RESPONSE_SYSTEM_PROMPT,
            self._response_prompt(user_input, relevant_memory, activated_persona, decision),
            temperature=0.6,
            max_tokens=450,
        ).strip() or "我刚才卡了一下，你再说一遍？"

        self.memory_manager.append_stm("assistant", response)
        self.memory_manager.save()
        if save_state:
            save_json(self.profile_path, self.user_profile)

        self._start_background(self._memory_pipeline, (save_state,))

        sa = state_axis(self.user_profile)
        return {
            "response": response,
            "current_state": sa["current_state"],
            "projected_state": sa["projected_state"],
            "context": context_axis(self.user_profile),
            "activated_persona": activated_persona,
            "decision": decision,
            "background_memory_running": self._background_memory_running,
            "model_timing": self.llm.last_model_timing,
        }

    def chat_stream(self, user_input: str, save_state: bool = True) -> Generator[Dict[str, Any], None, None]:
        self.memory_manager.append_stm("user", user_input)
        relevant_memory = self.memory_manager.retrieve_relevant_memory(user_input)

        fast_prompt = self._response_prompt(
            user_input, relevant_memory,
            activated_persona={
                "empathy_level": "中", "teasing_level": "低", "warmth_level": "中",
                "guidance_level": "中", "activated_tone": "自然、简洁、友好，优先直接回应用户当前问题",
            },
            decision={
                "reply_goal": "直接回应用户当前输入",
                "reply_strategy": "先回答问题，再给低成本建议；避免长篇分析",
                "content_focus": "用户当前最关心的内容",
                "avoid": ["过度铺垫", "输出分析过程", "提及内部状态推理"],
                "suggested_action": "给出一个可以马上尝试的小步骤",
            },
        )

        parts: List[str] = []
        for content in self.llm.chat_stream(RESPONSE_SYSTEM_PROMPT, fast_prompt, temperature=0.6, max_tokens=450):
            parts.append(content)
            yield {"type": "token", "content": content}

        response = "".join(parts).strip() or "我刚才卡了一下，你再说一遍？"
        if not parts:
            yield {"type": "token", "content": response}

        self.memory_manager.append_stm("assistant", response)
        self.memory_manager.save()
        if save_state:
            save_json(self.profile_path, self.user_profile)

        self._start_background(self._post_chat_pipeline, (user_input, relevant_memory, save_state))

        sa = state_axis(self.user_profile)
        yield {
            "type": "done",
            "response": response,
            "current_state": sa.get("current_state", {}),
            "projected_state": sa.get("projected_state", {}),
            "context": context_axis(self.user_profile),
            "activated_persona": {},
            "decision": {},
            "background_memory_running": self._background_memory_running,
        }
