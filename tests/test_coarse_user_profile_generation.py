from __future__ import annotations

import json
import unittest

from src.experiments.coarse_user_profile_generation import (
    PROFILE_LAYERS,
    build_compact_dialogue,
    collect_dialogue_messages,
    validate_coarse_profile,
)


class CoarseUserProfileGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chat = {
            "name": {"speaker_1": "Emi", "speaker_2": "elise", "unused": "drop"},
            "session_1": [
                {
                    "speaker": "Emi",
                    "clean_text": "I enjoy careful planning.",
                    "dia_id": "1",
                    "timestamp": "ignored",
                },
                {
                    "speaker": "elise",
                    "clean_text": "What do you usually plan?",
                    "dia_id": "2",
                },
                {"speaker": "Emi", "clean_text": "   ", "dia_id": "3"},
            ],
            "session_2": [
                {"speaker": "Emi", "clean_text": "Mostly research projects.", "score": 1},
            ],
            "session_2_date_time": "ignored",
        }

    def test_collect_dialogue_keeps_only_requested_message_fields(self) -> None:
        messages = collect_dialogue_messages(self.chat)
        self.assertEqual(len(messages), 3)
        self.assertEqual(set(messages[0]), {"speaker", "clean_text"})
        self.assertNotIn("dia_id", messages[0])

    def test_compact_dialogue_keeps_only_names_and_clean_messages(self) -> None:
        payload = json.loads(build_compact_dialogue(self.chat))
        self.assertEqual(
            payload["name"],
            {"speaker_1": "Emi", "speaker_2": "elise"},
        )
        self.assertEqual(set(payload), {"name", "messages"})
        self.assertTrue(all(set(message) == {"speaker", "clean_text"} for message in payload["messages"]))

    def test_compact_dialogue_obeys_limits_and_keeps_last_message(self) -> None:
        chat = {
            "name": {"speaker_1": "A", "speaker_2": "B"},
            "session_1": [
                {"speaker": "A" if index % 2 == 0 else "B", "clean_text": f"message-{index}-" + "x" * 40}
                for index in range(30)
            ],
        }
        compact = build_compact_dialogue(chat, max_utterances=10, max_chars=600)
        payload = json.loads(compact)
        self.assertLessEqual(len(compact), 600)
        self.assertLessEqual(len(payload["messages"]), 10)
        self.assertEqual(payload["messages"][-1]["clean_text"], "message-29-" + "x" * 40)

    def test_validate_profile_requires_exactly_five_string_layers(self) -> None:
        valid = {layer: f"{layer} summary" for layer in PROFILE_LAYERS}
        self.assertEqual(validate_coarse_profile(valid), valid)

        invalid = dict(valid)
        invalid["core"] = {"value": "nested output"}
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            validate_coarse_profile(invalid)

        invalid = dict(valid)
        invalid["extra"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "Invalid profile layers"):
            validate_coarse_profile(invalid)


if __name__ == "__main__":
    unittest.main()
