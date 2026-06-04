from __future__ import annotations

import json
import threading
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional

from .llm_client import LLMClient
from .memory_os_local import MemoryOSLocal
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

ABLATE_DIMENSIONS = {"state", "context", "memory"}
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

        self.total_turns = 0
        self._initial_user_profile = deepcopy(self.user_profile)

        self._background_memory_running = False
        self._background_memory_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._background_generation = 0
        self._prompt_state_override: Optional[Dict[str, Any]] = None
        self._prompt_context_override: Optional[Dict[str, Any]] = None

    # ---------- 消融实验：重置到初始状态 ----------
    def _normalize_ablate_dimension(self, ablate_dimension: Optional[str]) -> Optional[str]:
        if ablate_dimension is None:
            return None
        normalized = ablate_dimension.strip().lower()
        return normalized if normalized in ABLATE_DIMENSIONS else None
    
    def reset_to_initial_state(self) -> Dict[str, Any]:
        with self._background_memory_lock:
            self._background_generation += 1
            self._background_memory_running = False
        self._clear_prompt_overrides()
        self.user_profile = deepcopy(self._initial_user_profile)
        self.memory_manager.clear_stm()
        self.total_turns = 0
        save_json(self.profile_path, self.user_profile)
        return deepcopy(self.user_profile)

    def _is_generation_stale(self, generation: int) -> bool:
        return generation != self._background_generation

    # ---------- prompt builders ----------

    def _compact_memory(self, mem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "recent_messages": mem.get("recent_messages", [])[-4:],
            "mid_term_summaries": mem.get("mid_term_summaries", [])[-2:],
            "long_term_memories": mem.get("long_term_memories", [])[-3:],
        }

    def _build_context_for_prompt(
        self,
        ablate_dimension: Optional[str],
        relevant_memory: Dict[str, Any],
        user_input: str,
    ) -> Dict[str, str]:
        sa = state_axis(self.user_profile)
        active_state = self._prompt_state_override if self._prompt_state_override is not None else sa.get("current_state", {})
        active_context = self._prompt_context_override if self._prompt_context_override is not None else context_axis(self.user_profile)

        if ablate_dimension == "state":
            static_profile_payload = "[ABLATED: STATE]"
            current_state_payload = "[ABLATED: STATE]"
        else:
            static_profile_payload = json.dumps(
                flatten_static_profile(sa.get("static_profile", {})),
                ensure_ascii=False,
            )
            current_state_payload = json.dumps(active_state, ensure_ascii=False)

        if ablate_dimension == "context":
            context_payload = "[ABLATED: CONTEXT]"
        else:
            context_payload = json.dumps(active_context, ensure_ascii=False)

        return {
            "user_input": user_input,
            "static_profile": static_profile_payload,
            "current_state": current_state_payload,
            "current_context": context_payload,
            "persona_config": json.dumps(self.persona_config, ensure_ascii=False),
            "relevant_memory": relevant_memory,
        }

    def _reasoning_prompt(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
        ablate_dimension: Optional[str] = None,
    ) -> str:
        prompt_context = self._build_context_for_prompt(ablate_dimension, relevant_memory, user_input)
        return UNIFIED_REASONING_USER_PROMPT_TEMPLATE.format(**prompt_context)

    def _response_prompt(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
        activated_persona: Dict[str, Any],
        decision: Dict[str, Any],
        ablate_dimension: Optional[str] = None,
    ) -> str:
        prompt_context = self._build_context_for_prompt(ablate_dimension, relevant_memory, user_input)
        return RESPONSE_USER_PROMPT_TEMPLATE.format(
            **prompt_context,
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
            ctx["inferred_at_turn"] = self.total_turns

    def _snapshot_response_state(
        self,
        reasoning: Dict[str, Any],
        ablate_dimension: Optional[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        current_state = state_axis(self.user_profile).get("current_state", {})
        current_context = context_axis(self.user_profile)
        prompt_state = current_state if ablate_dimension == "state" else reasoning.get("current_state", current_state)
        prompt_context = current_context if ablate_dimension == "context" else reasoning.get("context", current_context)
        return prompt_state, prompt_context

    def _set_prompt_overrides(self, state_override: Dict[str, Any], context_override: Dict[str, Any]) -> None:
        self._prompt_state_override = state_override
        self._prompt_context_override = context_override

    def _clear_prompt_overrides(self) -> None:
        self._prompt_state_override = None
        self._prompt_context_override = None

    def _updated_fields_from_reasoning(self, reasoning: Dict[str, Any]) -> List[str]:
        updated_fields: List[str] = []
        if "current_state" in reasoning:
            updated_fields.append("state_axis.current_state")
        if "projected_state" in reasoning:
            updated_fields.append("state_axis.projected_state")
        if "context" in reasoning:
            updated_fields.append("context_axis")
        return updated_fields

    def _append_process_record(
        self,
        user_input: str,
        response: str,
        current_state: Dict[str, Any],
        projected_state: Dict[str, Any],
        activated_persona: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> None:
        record = {
            "user_input": user_input,
            "current_state": deepcopy(current_state),
            "projected_state": deepcopy(projected_state),
            "empathy_state": deepcopy(activated_persona),
            "decision": deepcopy(decision),
            "final_response": response,
        }

        with self._process_lock:
            try:
                with open(self.process_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                process_data = json.loads(content) if content else {"records": []}
            except (FileNotFoundError, json.JSONDecodeError):
                process_data = {"records": []}

            if not isinstance(process_data, dict):
                process_data = {"records": []}

            records = process_data.get("records")
            if not isinstance(records, list):
                records = []
            process_data["records"] = records
            records.append(record)

            with open(self.process_path, "w", encoding="utf-8") as f:
                json.dump(process_data, f, ensure_ascii=False, indent=2)

    # ---------- background pipelines ----------

    def _start_background(self, target, args: tuple) -> None:
        with self._background_memory_lock:
            if self._background_memory_running:
                return
            self._background_memory_running = True
            generation = self._background_generation
        threading.Thread(target=target, args=args + (generation,), daemon=True).start()

    def _memory_pipeline(self, save_state: bool, generation: int) -> None:
        try:
            threading.Event().wait(3)
            if self._is_generation_stale(generation):
                return
            self._run_memory_steps(save_state, generation)
        finally:
            with self._background_memory_lock:
                if generation == self._background_generation:
                    self._background_memory_running = False

    def _post_chat_pipeline(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
        save_state: bool,
        ablate_dimension: Optional[str],
        generation: int,
    ) -> None:
        try:
            if self._is_generation_stale(generation):
                return
            reasoning = parse_json(
                self.llm.chat(
                    UNIFIED_REASONING_SYSTEM_PROMPT,
                    self._reasoning_prompt(user_input, relevant_memory, ablate_dimension),
                    temperature=0.4,
                )
            )
            if self._is_generation_stale(generation):
                return

            self._apply_reasoning(reasoning)
            if save_state:
                save_json(self.profile_path, self.user_profile)
            self._run_memory_steps(save_state, generation)
        except Exception as e:
            print(f"[Background State Update Error] {e}")
        finally:
            with self._background_memory_lock:
                if generation == self._background_generation:
                    self._background_memory_running = False

    def _run_memory_steps(self, save_state: bool, generation: Optional[int] = None) -> None:
        if generation is not None and self._is_generation_stale(generation):
            return
        if len(self.memory_manager.short_term_memory) >= self.memory_manager.summary_prune_messages:
            self.memory_manager.build_mid_term_summary(self.llm, MID_TERM_SOURCE_MESSAGES)

        if generation is not None and self._is_generation_stale(generation):
            return
        self.memory_manager.extract_long_term_memory(self.llm)

        if generation is not None and self._is_generation_stale(generation):
            return
        if save_state:
            save_json(self.profile_path, self.user_profile)

    # ---------- public API ----------

    def chat_stream(
        self,
        user_input: str,
        save_state: bool = True,
        ablate_dimension: Optional[str] = None, 
    ) -> Generator[Dict[str, Any], None, None]:
        ablate_dimension = self._normalize_ablate_dimension(ablate_dimension)

        self.memory_manager.append_stm("user", user_input)
        relevant_memory = {} if ablate_dimension == "memory" else self.memory_manager.retrieve_relevant_memory(user_input)

        reasoning = parse_json(
            self.llm.chat(
                UNIFIED_REASONING_SYSTEM_PROMPT,
                self._reasoning_prompt(user_input, relevant_memory, ablate_dimension),
                temperature=0.4,
            )
        )

        activated_persona = reasoning.get("activated_persona", {})
        decision = reasoning.get("decision", {})
        prompt_state, prompt_context = self._snapshot_response_state(reasoning, ablate_dimension)
        self._set_prompt_overrides(prompt_state, prompt_context)
        response_prompt = self._response_prompt(
            user_input,
            relevant_memory,
            activated_persona,
            decision,
            ablate_dimension,
        )

        parts: List[str] = []
        try:
            for content in self.llm.chat_stream(
                RESPONSE_SYSTEM_PROMPT,
                response_prompt,
                temperature=0.6,
                max_tokens=450,
            ):
                parts.append(content)
                yield {"type": "token", "content": content}
        finally:
            self._clear_prompt_overrides()

        response = "".join(parts).strip() or FALLBACK_RESPONSE

        self._apply_reasoning(reasoning)
        self.memory_manager.append_stm("assistant", response)
        self.total_turns += 1
        if save_state:
            save_json(self.profile_path, self.user_profile)

        self._start_background(
            self._post_chat_pipeline,
            (user_input, relevant_memory, save_state, ablate_dimension),
        )

        sa = state_axis(self.user_profile)
        current_state = sa.get("current_state", {})
        projected_state = sa.get("projected_state", {})
        context = context_axis(self.user_profile)
        self._append_process_record(
            user_input=user_input,
            response=response,
            current_state=current_state,
            projected_state=projected_state,
            activated_persona=activated_persona,
            decision=decision,
        )
        yield {
            "type": "done",
            "response": response,
            "current_state": current_state,
            "projected_state": projected_state,
            "context": context,
            "activated_persona": activated_persona,
            "decision": decision,
            "updated_fields": self._updated_fields_from_reasoning(reasoning),
            "background_memory_running": self._background_memory_running,
            "model_timing": self.llm.last_model_timing,
            "ablate_dimension": ablate_dimension,
        }
