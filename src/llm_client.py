from __future__ import annotations

import configparser
from time import perf_counter
from typing import Any, Dict, Generator
from openai import OpenAI

class LLMClient:
    def __init__(self, config_path: str = "config.ini"):
        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")
        api_config = config["API"]
        self.model = api_config.get("model")
        self.enable_thinking = api_config.getboolean("enable_thinking", fallback=False)
        
        self.client = OpenAI(
            api_key = api_config.get("api_key"),
            base_url = api_config.get("base_url"),
        )
        self.last_model_timing: Dict[str, float | None] = {"first_char_seconds": None}
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
        }

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
                    elapsed = round(perf_counter() - request_start, 3)
                    self.last_model_timing["first_char_seconds"] = elapsed
                    print(f"[model timing] input_to_first_char_seconds={elapsed}")
                yield content

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["calls"] += 1
        print(f"[token usage] prompt={prompt_tokens}, completion={completion_tokens}")
