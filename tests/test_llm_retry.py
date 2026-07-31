import unittest
from types import SimpleNamespace

from src.llm_client import LLMClient


class StatusError(Exception):
    def __init__(self, status_code, retry_after=None):
        self.status_code = status_code
        self.response = type(
            "Response", (), {"headers": {"Retry-After": retry_after} if retry_after else {}}
        )()


class LLMRetryTests(unittest.TestCase):
    def test_retryable_status_codes(self):
        self.assertTrue(LLMClient._is_retryable(StatusError(429)))
        self.assertTrue(LLMClient._is_retryable(StatusError(503)))
        self.assertFalse(LLMClient._is_retryable(StatusError(401)))
        self.assertFalse(LLMClient._is_retryable(StatusError(400)))

    def test_retry_after_is_honored(self):
        self.assertEqual(LLMClient._retry_after_seconds(StatusError(429, "7")), 7.0)

    def test_kimi_temperature_is_normalized(self):
        client = object.__new__(LLMClient)
        client.model = "kimi-k2.6"
        client.base_url = "https://api.moonshot.cn/v1"
        self.assertEqual(client._normalized_temperature(0.0), 0.6)

    def test_transport_retry_returns_one_value(self):
        client = object.__new__(LLMClient)
        client.max_retries = 4
        client.retry_base_seconds = 0
        client.retry_max_seconds = 0
        client.last_request_attempts = 0
        calls = {"count": 0}

        def operation():
            calls["count"] += 1
            if calls["count"] < 3:
                raise StatusError(503)
            return "accepted"

        self.assertEqual(client._call_with_retry(operation, "test"), "accepted")
        self.assertEqual(calls["count"], 3)
        self.assertEqual(client.last_request_attempts, 3)

    def test_dashscope_qwen_schema_uses_required_tool_call(self):
        captured = {}

        def create(**request):
            captured.update(request)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(function=SimpleNamespace(
                        name="state_schema",
                        arguments='{"emotion":"joy"}',
                    ))],
                ))],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            )

        client = object.__new__(LLMClient)
        client.model = "qwen3-30b-a3b-instruct-2507"
        client.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        client.enable_thinking = False
        client.max_retries = 1
        client.retry_base_seconds = 0
        client.retry_max_seconds = 0
        client.last_request_attempts = 0
        client.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
        }
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        schema = {
            "name": "state_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"emotion": {"type": "string"}},
                "required": ["emotion"],
                "additionalProperties": False,
            },
        }

        result = client.chat(
            "system",
            "user",
            temperature=0.0,
            max_tokens=100,
            response_schema=schema,
        )

        self.assertEqual(result, '{"emotion":"joy"}')
        self.assertIn("tools", captured)
        self.assertIn("tool_choice", captured)
        self.assertNotIn("response_format", captured)
        self.assertNotIn("extra_body", captured)
