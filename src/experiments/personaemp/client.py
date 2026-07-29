from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    attempts: int


class ChatBackend(Protocol):
    model: str

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult: ...


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return None


class OpenAICompatibleChatBackend:
    """Small retrying client isolated from the runtime agent configuration."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 180.0,
        max_attempts: int = 6,
        enable_thinking: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")

        self.model = model
        self.max_attempts = max_attempts
        self.enable_thinking = enable_thinking
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for real PersonaEmp model calls. "
                "Install the project dependencies before running generation."
            ) from exc
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    @classmethod
    def from_env(
        cls,
        prefix: str = "PERSONAEMP_GENERATOR",
    ) -> "OpenAICompatibleChatBackend":
        def env(name: str, fallback: str = "") -> str:
            return os.getenv(f"{prefix}_{name}", fallback).strip()

        return cls(
            api_key=env("API_KEY", os.getenv("API_KEY", "")),
            base_url=env("BASE_URL", os.getenv("BASE_URL", "")),
            model=env("MODEL", "qwen3-8b"),
            timeout_seconds=float(env("TIMEOUT_SECONDS", "180")),
            max_attempts=int(env("MAX_ATTEMPTS", "6")),
            enable_thinking=env("ENABLE_THINKING", "false").lower()
            in {"1", "true", "yes", "on"},
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if self.enable_thinking:
                    request["extra_body"] = {"enable_thinking": True}

                response = self.client.chat.completions.create(**request)
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise RuntimeError("model returned an empty response")

                usage = getattr(response, "usage", None)
                return ChatResult(
                    content=content,
                    model=self.model,
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(
                        getattr(usage, "completion_tokens", 0) or 0
                    ),
                    latency_seconds=round(time.perf_counter() - started, 4),
                    attempts=attempt,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                retry_after = _retry_after_seconds(exc)
                wait_seconds = (
                    retry_after
                    if retry_after is not None
                    else min(1.5 * (2 ** (attempt - 1)), 30.0) + random.random()
                )
                time.sleep(wait_seconds)

        assert last_error is not None
        raise RuntimeError(
            f"model call failed after {self.max_attempts} attempts: {last_error}"
        ) from last_error
