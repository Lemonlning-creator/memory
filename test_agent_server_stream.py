import unittest

import app as algorithm_app


class FakeAgent:
    user_profile = {}

    def chat_stream(self, user_input, ablate_dimension=None):
        yield {"type": "token", "content": "你"}
        yield {"type": "token", "content": "好"}
        yield {"type": "done", "response": "你好"}

    def finalize_session(self):
        return {"flushed_mid_term_ids": [], "long_term_memory_id": None}


class AgentServerStreamTest(unittest.TestCase):
    def setUp(self):
        algorithm_app.app.config["TESTING"] = True
        algorithm_app.ALGORITHM_API_KEY = "test-key"
        algorithm_app.agent = FakeAgent()
        algorithm_app.active_character_id = "emi"
        algorithm_app.conversation_history.clear()
        self.client = algorithm_app.app.test_client()

    def test_stream_requires_authorization(self):
        response = self.client.post("/stream", json={"user_text": "hello"})

        self.assertEqual(response.status_code, 401)

    def test_stream_requires_user_text(self):
        response = self.client.post(
            "/stream",
            headers={"Authorization": "Bearer test-key"},
            json={"user_text": ""},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("request_id", response.get_json())

    def test_stream_returns_agent_server_compatible_events(self):
        response = self.client.post(
            "/stream",
            headers={"Authorization": "Bearer test-key"},
            json={
                "device_id": "device-1",
                "session_id": "session-1",
                "user_text": "hello",
                "conversation_context": [],
                "session_state": "",
            },
        )

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('data: {"delta": "你"', body)
        self.assertIn('data: {"delta": "好"', body)
        self.assertIn("data: [DONE]", body)
        self.assertEqual(
            algorithm_app.conversation_history,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "你好"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
