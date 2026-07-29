from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...epistemic_decay import PROFILE_LAYERS, compute_omega
from ...profile_utils import flatten_static_profile
from ...prompts.templates_en import (
    EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
    EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE,
    PROFILE_EXTRACTION_SYSTEM_PROMPT,
    PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
)
from .client import ChatBackend, ChatResult
from .dataset import PersonaEmpSample


PERSONAEMP_RESPONSE_SYSTEM_PROMPT = """You are a warm, empathetic conversation partner completing a single-turn personalized empathy benchmark.

Respond under all of these requirements:
1. Use the same language as the user's query.
2. Directly address the user's current need. When the user asks for advice, a decision, wording, or practical help, give at least one actionable suggestion or example phrase before any optional follow-up question.
3. Write exactly one paragraph containing 2 to 4 concise, natural sentences.
4. Validate the user's feelings when appropriate, without sounding clinical, formal, or patronizing.
5. Personalize only from the evidence provided. Do not mention memories, profiles, hidden context, or how the response was generated.
6. Do not invent user facts and do not describe yourself as an AI.
7. Ask at most one follow-up question.
8. Output only the final response."""

RESPONSE_MAX_TOKENS = 350
BASE_MODEL_USER_PROMPT = """You will be provided with memories extracted from previous dialogue.
Use them as background evidence and generate the final response.

User memory evidence:
{memory}

User query:
{query}
"""
OURS_USER_PROMPT = """Generate the final response using the following evidence and derived reasoning.

User memory evidence:
{memory}

Derived five-layer user profile:
{profile}

Derived deep-empathy state:
{alignment}

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
    return "\n".join(
        [
            "Extracted long-term memory evidence:",
            _memory_block(sample),
        ]
    )


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

    def load(
        self,
        cache_key: str,
    ) -> tuple[dict[str, Any], StageUsage | None] | None:
        path = self._path(cache_key)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"invalid profile cache entry: {path}")
        if value.get("format_version") == 2:
            profile = value.get("profile")
            usage_value = value.get("generation_usage")
            if not isinstance(profile, dict):
                raise ValueError(f"invalid profile cache entry: {path}")
            usage = (
                StageUsage(**usage_value)
                if isinstance(usage_value, dict)
                else None
            )
        else:
            profile = value
            usage = None
        _validate_profile(profile)
        return profile, usage

    def save(
        self,
        cache_key: str,
        profile: dict[str, Any],
        usage: StageUsage,
    ) -> None:
        _validate_profile(profile)
        path = self._path(cache_key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "profile": profile,
                    "generation_usage": asdict(usage),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
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
            profile, _generation_usage = cached
            return profile, None

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
                usage = StageUsage.combine(logical_results)
                self.cache.save(cache_key, profile, usage)
                return profile, usage
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc

        raise RuntimeError(
            f"profile extraction failed schema validation after "
            f"{self.schema_attempts} logical attempts: {last_error}"
        )


class BaseModelGenerator:
    method = "base_model"

    def __init__(self, backend: ChatBackend) -> None:
        self.backend = backend

    def generate(self, sample: PersonaEmpSample) -> GenerationOutput:
        result = self.backend.chat(
            PERSONAEMP_RESPONSE_SYSTEM_PROMPT,
            BASE_MODEL_USER_PROMPT.format(
                memory=_memory_block(sample),
                query=sample.query,
            ),
            temperature=0.6,
            max_tokens=RESPONSE_MAX_TOKENS,
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
        schema_attempts: int = 3,
    ) -> None:
        self.backend = backend
        self.profile_builder = profile_builder
        self.schema_attempts = schema_attempts

    def _alignment(
        self,
        sample: PersonaEmpSample,
        profile: dict[str, Any],
        omega: float,
    ) -> tuple[dict[str, Any], StageUsage]:
        user_prompt = EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE.format(
            recent_context=_memory_block(sample),
            user_message=sample.query,
            user_profile=json.dumps(
                flatten_static_profile(profile),
                ensure_ascii=False,
            ),
            agent_persona="{}",
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
        interaction_count = len(sample.memory_items)
        omega = compute_omega(interaction_count, profile)
        alignment, alignment_usage = self._alignment(sample, profile, omega)

        response_prompt = OURS_USER_PROMPT.format(
            memory=_memory_block(sample),
            profile=_profile_text(profile),
            alignment=json.dumps(alignment, ensure_ascii=False),
            query=sample.query,
        )
        response_result = self.backend.chat(
            PERSONAEMP_RESPONSE_SYSTEM_PROMPT,
            response_prompt,
            temperature=0.6,
            max_tokens=RESPONSE_MAX_TOKENS,
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
