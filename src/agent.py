from __future__ import annotations

import json
from pathlib import Path
import re
import threading
from time import perf_counter
from typing import Any, Dict, Generator, Optional, List

from .llm_client import LLMClient
from .memory_os_local import MemoryOSLocal
from .profile_utils import state_axis, context_axis, create_empty_profile, migrate_profile
from .utils import load_json, save_json, parse_json
from .prompts.prompt_loader import (
    DIRECT_RESPONSE_SYSTEM_PROMPT,
    DIRECT_RESPONSE_USER_PROMPT_TEMPLATE,
    PROFILE_EVOLUTION_SYSTEM_PROMPT,
    PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE,
    UNDERSTANDING_FEEDBACK_SYSTEM_PROMPT,
    UNDERSTANDING_FEEDBACK_USER_PROMPT_TEMPLATE,
    EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
    EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE,
)
from .epistemic_decay import EpistemicDecayTracker

DEFAULT_CONFIG_PATH = "config.ini"
DEFAULT_PERSONA_PATH = "agent_persona.json"
DEFAULT_USER_DIR = "user"
DEFAULT_USER_NAME = "default_user"

FALLBACK_RESPONSE = "我刚才卡了一下，你再说一遍？"
MID_TERM_SOURCE_MESSAGES = 14


class StateDrivenCompanionAgent:
    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        profile_path: Optional[str] = None,
        persona_path: Optional[str] = None,
        user_name: str = DEFAULT_USER_NAME,
    ):
        self.llm = LLMClient(config_path)
        self.user_name = user_name
        self.profile_path = profile_path or self._profile_path_for_user(user_name)
        self.user_profile = self._load_or_create_user_profile(self.profile_path)
        self.persona_path = persona_path or DEFAULT_PERSONA_PATH
        state_axis_obj = self.user_profile.setdefault("state_axis", {})
        state_axis_obj["current_state"] = {}
        state_axis_obj["projected_state"] = {}
        self.epistemic_tracker = EpistemicDecayTracker()
        self.last_empathy_state: Dict[str, Any] = {}
        self.last_prediction: Dict[str, Any] = {}
        self.last_agent_response: str = ""
        self.user_profile["context_axis"] = {}
        save_json(profile_path, self.user_profile)
        self.persona_config = load_json(self.persona_path)

        self.memory_manager = MemoryOSLocal()
        self._background_memory_running = False
        self._background_memory_lock = threading.Lock()
        self._background_generation = 0

    def _profile_path_for_user(self, user_name: str) -> str:
        name = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", user_name).strip("_")
        return str(Path(DEFAULT_USER_DIR) / f"{name}_profile.json")

    def _load_or_create_user_profile(self, profile_path: str) -> Dict[str, Any]:
        path = Path(profile_path)

        if path.exists():
            profile = load_json(str(path))
            if "static_profile" in profile:
                profile = migrate_profile(profile)
                save_json(str(path), profile)
            return profile

        profile = create_empty_profile()
        save_json(str(path), profile)
        return profile
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
        generation: int,
    ) -> None:
        try:
            threading.Event().wait(3)
            if self._is_generation_stale(generation):
                return
            self._run_memory_steps(generation)
        finally:
            with self._background_memory_lock:
                if not self._is_generation_stale(generation):
                    self._background_memory_running = False

    def _run_memory_steps(self, generation: Optional[int] = None) -> None:
        def _step(name: str, fn):
            if self._is_generation_stale(generation):
                return
            start = perf_counter()
            result = fn()
            print(f"[pipeline] {name} done in {perf_counter() - start:.3f}s")
            return result

        if len(self.memory_manager.short_term_memory) >= 20:
            _step("build_mid_term_summary",
                  lambda: self.memory_manager.build_mid_term_summary(self.llm, MID_TERM_SOURCE_MESSAGES))

        long_term_memory_id = _step("extract_long_term_memory",
                                    lambda: self.memory_manager.extract_long_term_memory(self.llm))

        if long_term_memory_id:
            profile_updated = _step("evolve_profile", lambda: self._evolve_profile_from_long_term(long_term_memory_id))
            if profile_updated:
                save_json(self.profile_path, self.user_profile)

    def _evolve_profile_from_long_term(self, long_term_memory_id: str) -> bool:
        long_term_memories = self.memory_manager.get_memories_by_ids([long_term_memory_id])
        if not long_term_memories:
            print(f"[Profile Evolution] missing long-term memory id={long_term_memory_id}")
            return False

        state = state_axis(self.user_profile)
        try:
            result = parse_json(self.llm.chat(
                PROFILE_EVOLUTION_SYSTEM_PROMPT,
                PROFILE_EVOLUTION_USER_PROMPT_TEMPLATE.format(
                    static_profile=json.dumps(state.get("static_profile", {}), ensure_ascii=False, indent=2),
                    long_term_memories=json.dumps(long_term_memories, ensure_ascii=False, indent=2),
                ),
                temperature=0.3,
            ))
            # Bayesian update output: {"reasoning": {...}, "static_profile": {...}}
            if isinstance(result, dict):
                updated_profile = result.get("static_profile", result)
                if isinstance(updated_profile, dict):
                    state["static_profile"] = updated_profile
                    reasoning = result.get("reasoning", {})
                    if reasoning:
                        print(f"[Bayesian Profile Update] {reasoning.get('evidence_summary', '')}")
                        if reasoning.get("new_attributes"):
                            print(f"  New attributes: {reasoning['new_attributes']}")
                        if reasoning.get("removed_attributes"):
                            print(f"  Removed (low confidence): {reasoning['removed_attributes']}")
                    print(f"[Profile Evolution] Bayesian update from memory_id={long_term_memory_id}")
                    return True
        except Exception as e:
            print(f"[Profile Evolution Error] {e}")
            return False

    def _run_empathy_alignment(
        self,
        user_input: str,
        relevant_memory: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Deep Empathy alignment reasoning with omega(t) modulation."""
        from .profile_utils import flatten_static_profile

        state = state_axis(self.user_profile)
        static_profile = state.get("static_profile", {})
        flattened = flatten_static_profile(static_profile)
        omega = self.epistemic_tracker.compute(static_profile)

        user_prompt = EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE.format(
            recent_context=json.dumps(relevant_memory, ensure_ascii=False)[:2000],
            user_message=user_input,
            user_profile=json.dumps(flattened, ensure_ascii=False)[:2000],
            agent_persona=json.dumps(self.persona_config, ensure_ascii=False)[:1000],
            current_state=json.dumps(state.get("current_state", {}), ensure_ascii=False),
            epistemic_omega=omega,
        )

        try:
            result = parse_json(self.llm.chat(
                EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.3,
            ))
            if isinstance(result, dict):
                self.last_empathy_state = result.get("empathy_state", {})
                self.last_prediction = result.get("prediction", {})
            return result
        except Exception as e:
            print(f"[Empathy Alignment Error] {e}")
            return {}

    def _understanding_feedback(self, user_input: str) -> None:
        """UPDATING step: assess how previous empathy was received and update understanding."""
        if not self.last_agent_response:
            return

        from .profile_utils import flatten_static_profile
        state = state_axis(self.user_profile)
        static_profile = state.get("static_profile", {})
        flattened = flatten_static_profile(static_profile)

        try:
            feedback = parse_json(self.llm.chat(
                UNDERSTANDING_FEEDBACK_SYSTEM_PROMPT,
                UNDERSTANDING_FEEDBACK_USER_PROMPT_TEMPLATE.format(
                    previous_empathy_state=json.dumps(self.last_empathy_state, ensure_ascii=False),
                    previous_prediction=json.dumps(self.last_prediction, ensure_ascii=False),
                    agent_response=self.last_agent_response,
                    user_message=user_input,
                    user_profile=json.dumps(flattened, ensure_ascii=False)[:2000],
                ),
                temperature=0.3,
            ))
            if isinstance(feedback, dict):
                learning = feedback.get("learning", {})
                if learning.get("new_insight"):
                    print(f"[Understanding Update] {learning['new_insight']}")
                calibration = feedback.get("understanding_update", {})
                if calibration.get("calibration_note"):
                    print(f"[Calibration] {calibration['calibration_note']}")
        except Exception as e:
            print(f"[Understanding Feedback Error] {e}")


    def finalize_session(self) -> Dict[str, Any]:
        with self._background_memory_lock:
            self._background_generation += 1
            self._background_memory_running = False

        flushed_mid_term_ids = self.memory_manager.flush_short_term_memory(self.llm)
        long_term_memory_id = self.memory_manager.extract_long_term_memory(self.llm)
        if long_term_memory_id:
            print(f"[Agent Profile] final evolve from long_term_memory_id={long_term_memory_id}")
            self._evolve_profile_from_long_term(long_term_memory_id)
            print(f"[Agent Profile] final save profile path={self.profile_path}")
            save_json(self.profile_path, self.user_profile)

        return {
            "flushed_mid_term_ids": flushed_mid_term_ids,
            "long_term_memory_id": long_term_memory_id
        }

    def observe_dialogue_turn(self, role: str, content: str) -> None:
        # UPDATING step runs in background so it never blocks the user's next interaction
        self.memory_manager.append_stm(role, content)
        self.epistemic_tracker.increment()
        if role == "user" and self.last_agent_response:
            threading.Thread(target=self._understanding_feedback, args=(content,), daemon=True).start()
        self._run_memory_steps()

    def chat_stream(
        self,
        user_input: str,
        ablate_dimension: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:

        self.memory_manager.append_stm("user", user_input)
        relevant_memory = self.memory_manager.retrieve_relevant_memory(user_input)

        # Empathy alignment reasoning runs in background; does NOT block streaming.
        # If a previous alignment result exists, use it; otherwise skip for this turn.
        empathy_state = self.last_empathy_state if self.last_empathy_state else {}
        threading.Thread(
            target=self._run_empathy_alignment,
            args=(user_input, relevant_memory),
            daemon=True,
        ).start()

        parts: List[str] = []
        first_token_logged = False
        t_start = perf_counter()
        try:
            for content in self.llm.chat_stream(
                DIRECT_RESPONSE_SYSTEM_PROMPT,
                self._response_prompt(user_input, relevant_memory),
                temperature=0.4,
                max_tokens=450,
            ):
                if not first_token_logged:
                    print(f"[chat] first_token in {perf_counter() - t_start:.3f}s from interaction start")
                    first_token_logged = True
                parts.append(content)
                yield {"type": "token", "content": content}
        except Exception as e:
            print(f"[Stream Response Error] {e}")
            if not parts:
                parts.append(FALLBACK_RESPONSE)
                yield {"type": "token", "content": FALLBACK_RESPONSE}

        response = "".join(parts).strip() or FALLBACK_RESPONSE

        self.memory_manager.append_stm("assistant", response)
        self.last_agent_response = response
        self.epistemic_tracker.increment()
        self._start_background(self._memory_pipeline, ())

        yield {
            "type": "done",
            "response": response,
            "background_memory_running": self._background_memory_running,
            "model_timing": self.llm.last_model_timing,
        }
