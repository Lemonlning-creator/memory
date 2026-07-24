import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.interactive_profile_simulation import (
    InteractiveProfileSimulation,
    SimulationConfig,
    contains_direct_self_description,
)
from src.profile_batch_updater import PROFILE_LAYERS, ProfileBatchUpdater


class RecordingAgentModel:
    def __init__(self):
        self.calls = []

    def complete(self, system_prompt, messages):
        self.calls.append((system_prompt, list(messages)))
        return f"agent-reply-{len(self.calls)}"


class RecordingUserModel:
    def __init__(self, replies=None):
        self.calls = []
        self.replies = list(replies or [])

    def complete(self, system_prompt, messages):
        self.calls.append((system_prompt, list(messages)))
        if self.replies:
            return self.replies.pop(0)
        latest_agent = messages[-1]["content"]
        return f"听到你说{latest_agent}，我会先把具体情况弄清楚再决定。"


class DeterministicExtractor:
    """Only replaces remote extraction; queue, worker, validation and merge stay real."""

    available = True

    def __init__(self):
        self.batches = []

    def extract(self, current_profile, turns):
        self.batches.append([turn["message_id"] for turn in turns])
        evidence_ids = [turn["message_id"] for turn in turns]
        return {
            "layers": {
                "core": {
                    "summary": {
                        "value": "重视把事情弄清楚后再作决定。",
                        "confidence": 0.82,
                        "evidence_message_ids": evidence_ids,
                    },
                    "attributes": {
                        "values": {
                            "value": "重视事实核对和稳妥选择。",
                            "confidence": 0.82,
                            "evidence_message_ids": evidence_ids,
                        }
                    },
                },
                **{layer: {"summary": None, "attributes": {}} for layer in PROFILE_LAYERS if layer != "core"},
            }
        }


class InteractiveProfileSimulationTests(unittest.TestCase):
    def _real_updater(self, profile_path, batch_size):
        extractor = DeterministicExtractor()
        updater = ProfileBatchUpdater(
            str(profile_path),
            extractor=extractor,
            min_user_messages=batch_size,
            max_wait_seconds=900,
        )
        return updater, extractor

    def test_dialogue_is_causal_and_profile_activates_through_real_background_worker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.json"
            agent = RecordingAgentModel()
            user = RecordingUserModel()
            updater, extractor = self._real_updater(profile_path, batch_size=2)
            persona = {"private_trait": "容易反复揣摩关系里的细节", "strength": "愿意照顾他人感受"}
            simulation = InteractiveProfileSimulation(
                agent_model=agent,
                user_model=user,
                updater=updater,
                hidden_persona=persona,
                profile_path=profile_path,
                config=SimulationConfig(turns=4, batch_size=2, topics=("关系", "消费")),
            )

            record = simulation.run()

            self.assertFalse(record["preset_user_messages"])
            self.assertEqual(record["mode"], "causal_interactive_dialogue")
            self.assertEqual([item["after_user_turn"] for item in record["activations"]], [2, 4])
            self.assertTrue(all(item["trigger"] == "automatic_message_threshold" for item in record["activations"]))
            self.assertEqual(len(extractor.batches), 2)
            self.assertTrue(all(len(batch) == 2 for batch in extractor.batches))
            self.assertEqual(len([turn for turn in record["transcript"] if turn["speaker"] == "user"]), 4)
            self.assertTrue(all("agent-reply-" in turn["content"] for turn in record["transcript"] if turn["speaker"] == "user"))
            self.assertTrue(all("private_trait" not in turn["content"] for turn in record["transcript"]))
            self.assertTrue(all("容易反复揣摩" not in turn["content"] for turn in record["transcript"]))

            for index, (_, messages) in enumerate(user.calls, start=1):
                self.assertEqual(messages[-1]["role"], "user")
                self.assertEqual(messages[-1]["content"], f"agent-reply-{index}")
            self.assertEqual(agent.calls[1][1][-1]["role"], "user")
            self.assertIn("agent-reply-1", agent.calls[1][1][-1]["content"])
            self.assertIn("private_trait", user.calls[0][0])
            self.assertNotIn("private_trait", agent.calls[0][0])
            self.assertEqual(record["final_profile"]["state_axis"]["static_profile"]["core"]["summary"]["value"], "重视把事情弄清楚后再作决定。")

    def test_direct_self_description_is_regenerated_as_concrete_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.json"
            user = RecordingUserModel([
                "我是一个很敏感又容易焦虑的人。",
                "如果朋友临时改约，我嘴上会说没事，但回家后大概还会想很久。",
            ])
            updater, _ = self._real_updater(profile_path, batch_size=1)
            simulation = InteractiveProfileSimulation(
                agent_model=RecordingAgentModel(),
                user_model=user,
                updater=updater,
                hidden_persona={"relationship": "在意关系安全感"},
                profile_path=profile_path,
                config=SimulationConfig(turns=1, batch_size=1, topics=("临时改约",)),
            )

            record = simulation.run()

            self.assertEqual(len(user.calls), 2)
            final_user_turn = record["transcript"][-1]["content"]
            self.assertFalse(contains_direct_self_description(final_user_turn))
            self.assertEqual(record["activations"][0]["trigger"], "automatic_message_threshold")

    def test_direct_self_description_detector_catches_test_and_trait_labels(self):
        self.assertTrue(contains_direct_self_description("我的缺点是总容易想太多。"))
        self.assertTrue(contains_direct_self_description("我很敏感，也比较容易焦虑。"))
        self.assertTrue(contains_direct_self_description("我其实是偏感性的人。"))
        self.assertTrue(contains_direct_self_description("作为一次画像测试，我会这样回答。"))
        self.assertFalse(contains_direct_self_description("朋友没回消息时，我会先等一晚，第二天再问。"))


if __name__ == "__main__":
    unittest.main()
