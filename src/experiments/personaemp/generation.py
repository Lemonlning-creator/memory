from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...epistemic_decay import PROFILE_LAYERS, compute_omega
from ...profile_utils import flatten_static_profile
from ...prompts import templates_en
from ...prompts.templates_en import (
    EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
    EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE,
    PROFILE_EXTRACTION_SYSTEM_PROMPT,
    PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
)
from .client import ChatBackend, ChatResult
from .dataset import PersonaEmpSample


DIRECT_RESPONSE_SYSTEM_PROMPT = getattr(
    templates_en,
    "DIRECT_RESPONSE_SYSTEM_PROMPT",
    templates_en.DDIRECT_RESPONSE_SYSTEM_PROMPT,
)
DIRECT_RESPONSE_USER_PROMPT_TEMPLATE = templates_en.DIRECT_RESPONSE_USER_PROMPT_TEMPLATE

BASE_QWEN3_SYSTEM_PROMPT = (
    "You are a helpful, warm, and empathetic AI assistant."
)
BASE_QWEN3_USER_PROMPT = """You will be provided with memories extracted from previous dialogue.
Generate a response to the user's current query.

Memories:
{memory}

User query:
{query}
"""


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_reasoning(text: str) -> str:
    value = text.strip()
    if "</think>" in value:
        value = value.split("</think>", 1)[1].strip()
    return value


def _parse_json_object(text: str) -> dict[str, Any]:
    value = _strip_reasoning(text)
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response does not contain a JSON object")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response JSON root must be an object")
    return parsed


def _memory_block(sample: PersonaEmpSample) -> str:
    return "\n".join(
        f"{index}. {value}" for index, value in enumerate(sample.memory_items, 1)
    )


def _profile_corpus(sample: PersonaEmpSample) -> str:
    parts = [
        "Extracted long-term memory evidence:",
        _memory_block(sample),
    ]
    if sample.conversation:
        parts.extend(
            [
                "",
                "Available dialogue excerpt:",
                *[
                    f"{turn.get('role', 'unknown')}: {turn.get('text', '')}"
                    for turn in sample.conversation
                ],
            ]
        )
    return "\n".join(parts)


def _profile_text(profile: dict[str, Any]) -> str:
    flattened = flatten_static_profile(profile)
    lines: list[str] = []
    for layer in PROFILE_LAYERS:
        fields = flattened.get(layer, {})
        if not isinstance(fields, dict):
            continue
        for key, value in fields.items():
            if value not in (None, "", [], {}):
                lines.append(f"[{layer}] {key}: {value}")
    return "\n".join(lines) or json.dumps(
        flattened,
        ensure_ascii=False,
        sort_keys=True,
    )


def _validate_profile(profile: dict[str, Any]) -> None:
    missing = [layer for layer in PROFILE_LAYERS if layer not in profile]
    if missing:
        raise ValueError(f"profile is missing layers: {', '.join(missing)}")
    invalid = [
        layer for layer in PROFILE_LAYERS if not isinstance(profile.get(layer), dict)
    ]
    if invalid:
        raise ValueError(f"profile layers must be objects: {', '.join(invalid)}")


@dataclass(frozen=True)
class StageUsage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    attempts: int
    logical_calls: int

    @classmethod
    def from_result(cls, result: ChatResult) -> "StageUsage":
        return cls(
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_seconds=result.latency_seconds,
            attempts=result.attempts,
            logical_calls=1,
        )

    @classmethod
    def combine(cls, results: list[ChatResult]) -> "StageUsage":
        if not results:
            raise ValueError("at least one result is required")
        return cls(
            model=results[-1].model,
            prompt_tokens=sum(result.prompt_tokens for result in results),
            completion_tokens=sum(
                result.completion_tokens for result in results
            ),
            latency_seconds=round(
                sum(result.latency_seconds for result in results),
                4,
            ),
            attempts=sum(result.attempts for result in results),
            logical_calls=len(results),
        )


@dataclass(frozen=True)
class GenerationOutput:
    response: str
    method: str
    profile_hash: str | None
    alignment_hash: str | None
    omega: float | None
    stages: dict[str, StageUsage]

    def to_record(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "method": self.method,
            "profile_hash": self.profile_hash,
            "alignment_hash": self.alignment_hash,
            "omega": self.omega,
            "stages": {
                name: asdict(usage) for name, usage in self.stages.items()
            },
        }


class ProfileCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, cache_key: str) -> Path:
        return self.directory / f"{cache_key}.json"

    def load(self, cache_key: str) -> dict[str, Any] | None:
        path = self._path(cache_key)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"invalid profile cache entry: {path}")
        _validate_profile(value)
        return value

    def save(self, cache_key: str, profile: dict[str, Any]) -> None:
        _validate_profile(profile)
        path = self._path(cache_key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)


class ProfileBuilder:
    def __init__(
        self,
        backend: ChatBackend,
        cache: ProfileCache,
        schema_attempts: int = 3,
    ) -> None:
        self.backend = backend
        self.cache = cache
        self.schema_attempts = schema_attempts

    def cache_key(self, sample: PersonaEmpSample) -> str:
        payload = {
            "session_id": sample.session_id,
            "memory": sample.memory_items,
            "conversation": sample.conversation,
            "model": self.backend.model,
            "system_prompt_hash": prompt_hash(PROFILE_EXTRACTION_SYSTEM_PROMPT),
            "user_prompt_hash": prompt_hash(PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def build(
        self,
        sample: PersonaEmpSample,
    ) -> tuple[dict[str, Any], StageUsage | None]:
        cache_key = self.cache_key(sample)
        cached = self.cache.load(cache_key)
        if cached is not None:
            return cached, None

        user_prompt = PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
            user_name="the user",
            corpus=_profile_corpus(sample),
        )
        last_error: Exception | None = None
        logical_results: list[ChatResult] = []
        for _ in range(self.schema_attempts):
            result = self.backend.chat(
                PROFILE_EXTRACTION_SYSTEM_PROMPT.format(user_name="the user"),
                user_prompt,
                temperature=0.2,
                max_tokens=3000,
            )
            logical_results.append(result)
            try:
                profile = _parse_json_object(result.content)
                _validate_profile(profile)
                self.cache.save(cache_key, profile)
                return profile, StageUsage.combine(logical_results)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc

        raise RuntimeError(
            f"profile extraction failed schema validation after "
            f"{self.schema_attempts} logical attempts: {last_error}"
        )


class BaseQwen3Generator:
    method = "base_qwen3"

    def __init__(self, backend: ChatBackend) -> None:
        self.backend = backend

    def generate(self, sample: PersonaEmpSample) -> GenerationOutput:
        result = self.backend.chat(
            BASE_QWEN3_SYSTEM_PROMPT,
            BASE_QWEN3_USER_PROMPT.format(
                memory=_memory_block(sample),
                query=sample.query,
            ),
            temperature=0.6,
            max_tokens=450,
        )
        return GenerationOutput(
            response=_strip_reasoning(result.content),
            method=self.method,
            profile_hash=None,
            alignment_hash=None,
            omega=None,
            stages={"response": StageUsage.from_result(result)},
        )


class DeepEmpathyGenerator:
    method = "ours"

    def __init__(
        self,
        backend: ChatBackend,
        profile_builder: ProfileBuilder,
        agent_persona: dict[str, Any] | None = None,
        schema_attempts: int = 3,
    ) -> None:
        self.backend = backend
        self.profile_builder = profile_builder
        self.agent_persona = agent_persona or {}
        self.schema_attempts = schema_attempts

    def _alignment(
        self,
        sample: PersonaEmpSample,
        profile: dict[str, Any],
        omega: float,
    ) -> tuple[dict[str, Any], StageUsage]:
        recent_context = json.dumps(
            list(sample.conversation[-20:]),
            ensure_ascii=False,
        )
        user_prompt = EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE.format(
            recent_context=recent_context,
            user_message=sample.query,
            user_profile=json.dumps(
                flatten_static_profile(profile),
                ensure_ascii=False,
            ),
            agent_persona=json.dumps(
                self.agent_persona,
                ensure_ascii=False,
            ),
            current_state="{}",
            epistemic_omega=omega,
        )

        last_error: Exception | None = None
        logical_results: list[ChatResult] = []
        for _ in range(self.schema_attempts):
            result = self.backend.chat(
                EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.2,
                max_tokens=1800,
            )
            logical_results.append(result)
            try:
                alignment = _parse_json_object(result.content)
                if not isinstance(alignment.get("empathy_state"), dict):
                    raise ValueError("alignment.empathy_state must be an object")
                if not isinstance(alignment.get("prediction"), dict):
                    raise ValueError("alignment.prediction must be an object")
                if not isinstance(alignment.get("exploration"), dict):
                    raise ValueError("alignment.exploration must be an object")
                return alignment, StageUsage.combine(logical_results)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc

        raise RuntimeError(
            f"empathy alignment failed schema validation after "
            f"{self.schema_attempts} logical attempts: {last_error}"
        )

    def generate(self, sample: PersonaEmpSample) -> GenerationOutput:
        profile, profile_usage = self.profile_builder.build(sample)
        interaction_count = max(len(sample.conversation), len(sample.memory_items))
        omega = compute_omega(interaction_count, profile)
        alignment, alignment_usage = self._alignment(sample, profile, omega)

        understanding = alignment.get("understanding", {})
        current_state = {
            "user_domain": understanding.get("user_domain", {}),
            "prediction": alignment.get("prediction", {}),
        }
        current_context = {
            "scenario": sample.scenario,
            "category": sample.category,
            "empathy_state": alignment.get("empathy_state", {}),
            "exploration": alignment.get("exploration", {}),
        }
        response_prompt = DIRECT_RESPONSE_USER_PROMPT_TEMPLATE.format(
            user_input=sample.query,
            static_profile=_profile_text(profile),
            current_state=json.dumps(current_state, ensure_ascii=False),
            current_context=json.dumps(current_context, ensure_ascii=False),
            persona_config=json.dumps(self.agent_persona, ensure_ascii=False),
            relevant_memory=json.dumps(
                list(sample.memory_items),
                ensure_ascii=False,
            ),
        )
        response_result = self.backend.chat(
            DIRECT_RESPONSE_SYSTEM_PROMPT,
            response_prompt,
            temperature=0.4,
            max_tokens=450,
        )

        stages = {
            "alignment": alignment_usage,
            "response": StageUsage.from_result(response_result),
        }
        if profile_usage is not None:
            stages["profile"] = profile_usage

        profile_hash = hashlib.sha256(
            json.dumps(
                profile,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        alignment_hash = hashlib.sha256(
            json.dumps(
                alignment,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return GenerationOutput(
            response=_strip_reasoning(response_result.content),
            method=self.method,
            profile_hash=profile_hash,
            alignment_hash=alignment_hash,
            omega=omega,
            stages=stages,
        )
