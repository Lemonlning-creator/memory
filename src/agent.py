from __future__ import annotations

import json
import threading
from typing import Any, Dict, Generator, Optional, List

from .llm_client import LLMClient
from .memory_os_local import MemoryOSLocal
from .profile_utils import state_axis, context_axis, migrate_profile
from .utils import load_json, save_json, parse_json
from .prompts.templates import (
    BACKGROUND_REASONING_USER_PROMPT_TEMPLATE,
    DIRECT_RESPONSE_SYSTEM_PROMPT,
    DIRECT_RESPONSE_USER_PROMPT_TEMPLATE,
    PROFILE_EVOLUTION_SYSTEM_PROMPT,
    PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE,
)

DEFAULT_CONFIG_PATH = "config.ini"
DEFAULT_PROFILE_PATH = "user_profile.json"
DEFAULT_PERSONA_PATH = "agent_persona.json"

FALLBACK_RESPONSE = "我刚才卡了一下，你再说一遍？"
MID_TERM_SOURCE_MESSAGES = 14


class StateDrivenCompanionAgent:
    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        profile_path: str = DEFAULT_PROFILE_PATH,
        persona_path: str = DEFAULT_PERSONA_PATH,
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
        self.memory_manager = MemoryOSLocal()

        self._background_memory_running = False
        self._background_memory_lock = threading.Lock()
        self._background_generation = 0
    # ---------- prompt builders ----------
    def _prompt_context(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
    ) -> Dict[str, str]:
        state = state_axis(self.user_profile)
        context = context_axis(self.user_profile)
        return {
            "user_input": user_input,
            "static_profile": json.dumps(state.get("static_profile", {}), ensure_ascii=False),
            "current_state": json.dumps(state.get("current_state", {}), ensure_ascii=False),
            "current_context": json.dumps(context, ensure_ascii=False),
            "persona_config": json.dumps(self.persona_config, ensure_ascii=False),
            "relevant_memory": json.dumps(relevant_memory, ensure_ascii=False),
        }

    def _response_prompt(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
    ) -> str:
        return DIRECT_RESPONSE_USER_PROMPT_TEMPLATE.format(**self._prompt_context(user_input, relevant_memory))

    def _background_reasoning_prompt(
        self,
        user_input: str,
        assistant_response: str,
        relevant_memory: Dict[str, Any],
    ) -> str:
        return BACKGROUND_REASONING_USER_PROMPT_TEMPLATE.format(
            **self._prompt_context(user_input, relevant_memory),
            assistant_response=assistant_response,
        )

    def _apply_background_reasoning(self, reasoning: Dict[str, Any]) -> None:
        state = state_axis(self.user_profile)
        if "current_state" in reasoning:
            state["current_state"] = reasoning["current_state"]
        if "projected_state" in reasoning:
            state["projected_state"] = reasoning["projected_state"]
        if "context" in reasoning:
            context = context_axis(self.user_profile)
            context.update(reasoning["context"])

    # ---------- memory background pipelines ----------
    def _start_background(self, target, args: tuple) -> None:
        with self._background_memory_lock:
            if self._background_memory_running:
                return
            self._background_memory_running = True
            generation = self._background_generation
        threading.Thread(target=target, args=args + (generation,), daemon=True).start()

    def _is_generation_stale(self, generation: int) -> bool:
        return generation != self._background_generation
    
    def _memory_pipeline(
        self,
        user_input: str,
        assistant_response: str,
        relevant_memory: Dict[str, Any],
        save_state: bool,
        generation: int,
    ) -> None:
        try:
            threading.Event().wait(3)
            if self._is_generation_stale(generation):
                return
            # try: # 获取推理过程中的状态更新和人设激活，当前版本不直接用这些信息，但保留接口
            #     reasoning = parse_json(self.llm.chat(
            #         BACKGROUND_REASONING_SYSTEM_PROMPT,
            #         self._background_reasoning_prompt(user_input, assistant_response, relevant_memory),
            #         temperature=0.4,
            #     ))
            #     self._apply_background_reasoning(reasoning)
            #     if save_state:
            #         save_json(self.profile_path, self.user_profile)
            # except Exception as e:
            #     print(f"[Background Reasoning Error] {e}")
            if self._is_generation_stale(generation):
                return
            self._run_memory_steps(save_state, generation)
        finally:
            with self._background_memory_lock:
                if not self._is_generation_stale(generation):
                    self._background_memory_running = False

    def _run_memory_steps(self, save_state: bool, generation: Optional[int] = None) -> None:
        if generation is not None and self._is_generation_stale(generation):
            return
        if len(self.memory_manager.short_term_memory) >= 20:
            self.memory_manager.build_mid_term_summary(self.llm, MID_TERM_SOURCE_MESSAGES)

        if generation is not None and self._is_generation_stale(generation):
            return
        long_term_memory_id = self.memory_manager.extract_long_term_memory(self.llm)

        if generation is not None and self._is_generation_stale(generation):
            return
        if long_term_memory_id:
            self._evolve_profile_from_long_term(long_term_memory_id)

        if generation is not None and self._is_generation_stale(generation):
            return
        if long_term_memory_id and save_state:
            save_json(self.profile_path, self.user_profile)

    def _evolve_profile_from_long_term(self, long_term_memory_id: str) -> None:
        long_term_memories = self.memory_manager.get_memories_by_ids([long_term_memory_id])
        if not long_term_memories:
            return
        state = state_axis(self.user_profile)
        try:
            updated_static_profile = parse_json(self.llm.chat(
                PROFILE_EVOLUTION_SYSTEM_PROMPT,
                PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE.format(
                    static_profile=json.dumps(state.get("static_profile", {}), ensure_ascii=False, indent=2),
                    long_term_memories=json.dumps(long_term_memories, ensure_ascii=False, indent=2),
                ),
                temperature=0.3,
            ))
            if isinstance(updated_static_profile, dict):
                state["static_profile"] = updated_static_profile
        except Exception as e:
            print(f"[Profile Evolution Error] {e}")

    def chat_stream(
        self,
        user_input: str,
        ablate_dimension: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        self.memory_manager.append_stm("user", user_input)
        relevant_memory = self.memory_manager.retrieve_relevant_memory(user_input)
        
        parts: List[str] = []
        try:
            for content in self.llm.chat_stream(
                DIRECT_RESPONSE_SYSTEM_PROMPT,
                self._response_prompt(user_input, relevant_memory),
                temperature=0.4,
                max_tokens=450,
            ):
                parts.append(content)
                yield {"type": "token", "content": content}
        except Exception as e:
            print(f"[Stream Response Error] {e}")
            if not parts:
                parts.append(FALLBACK_RESPONSE)
                yield {"type": "token", "content": FALLBACK_RESPONSE}
        finally:
            print("LLM response finished, starting background memory processing...")

        response = "".join(parts).strip() or FALLBACK_RESPONSE

        self.memory_manager.append_stm("assistant", response)
        self._start_background(self._memory_pipeline, (user_input, response, relevant_memory, True))
        
        yield {
            "type": "done",
            "response": response,
            "background_memory_running": self._background_memory_running,
            "model_timing": self.llm.last_model_timing,
        }
