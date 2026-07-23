import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.profile_batch_updater import (
    KimiProfileExtractor,
    PROFILE_LAYERS,
    ProfileBatchUpdater,
    ProfileUpdateError,
    merge_patch,
    validate_patch,
)


def item(value="稳定偏好", confidence=0.8, evidence_ids=None):
    return {
        "value": value,
        "confidence": confidence,
        "evidence_message_ids": evidence_ids or ["m1"],
    }


def valid_patch():
    layers = {
        layer: {"summary": item(f"{layer}摘要"), "attributes": {}}
        for layer in PROFILE_LAYERS
    }
    layers["core"]["attributes"] = {"values": item("重视持续成长")}
    return {"layers": layers}


class FakeCompletions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages = []

    def create(self, **kwargs):
        self.messages.append(kwargs["messages"])
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(output, ensure_ascii=False)))]
        )


class FakeClient:
    def __init__(self, outputs):
        self.chat = SimpleNamespace(completions=FakeCompletions(outputs))


class StaticExtractor:
    available = True

    def __init__(self, patch=None, error=None):
        self.patch = patch or valid_patch()
        self.error = error

    def extract(self, current_profile, turns):
        if self.error:
            raise self.error
        patch = json.loads(json.dumps(self.patch))
        message_id = turns[0]["message_id"]
        for layer in patch["layers"].values():
            if layer["summary"]:
                layer["summary"]["evidence_message_ids"] = [message_id]
            for attribute in layer["attributes"].values():
                attribute["evidence_message_ids"] = [message_id]
        return patch


class ProfileBatchUpdaterTests(unittest.TestCase):
    def test_schema_rejects_unknown_layer_field_and_evidence(self):
        patch = valid_patch()
        patch["layers"]["core"]["attributes"]["invented"] = item()
        with self.assertRaises(ProfileUpdateError):
            validate_patch(patch, {"m1"})

        patch = valid_patch()
        patch["layers"]["core"]["attributes"]["values"] = item(evidence_ids=["outside"])
        with self.assertRaises(ProfileUpdateError):
            validate_patch(patch, {"m1"})

        patch = valid_patch()
        patch["layers"]["core"]["summary"] = None
        with self.assertRaises(ProfileUpdateError):
            validate_patch(patch, {"m1"})

    def test_merge_is_field_level_and_keeps_existing_values(self):
        profile = {
            "state_axis": {
                "static_profile": {
                    "core": {
                        "fears": {"value": "旧恐惧", "confidence": 0.6},
                        "values": {"value": "旧价值观", "confidence": 0.5},
                    }
                },
                "current_state": {"mood": "calm"},
                "projected_state": {},
            },
            "context_axis": {"current_context": "work"},
        }
        merged = merge_patch(profile, valid_patch(), [{"message_id": "m1", "user": "我重视持续成长"}])
        core = merged["state_axis"]["static_profile"]["core"]
        self.assertEqual(core["fears"]["value"], "旧恐惧")
        self.assertEqual(core["values"]["value"], "重视持续成长")
        self.assertEqual(core["summary"]["value"], "core摘要")
        self.assertEqual(merged["state_axis"]["current_state"], {"mood": "calm"})
        self.assertEqual(merged["context_axis"], {"current_context": "work"})

    def test_task_local_retry_includes_validation_error(self):
        invalid = valid_patch()
        del invalid["layers"]["behavior"]
        fake = FakeClient([invalid, valid_patch()])
        extractor = KimiProfileExtractor(client=fake, model="test", max_attempts=2)
        result = extractor.extract({}, [{"message_id": "m1", "user": "我重视成长", "assistant": "好的"}])
        self.assertEqual(set(result["layers"]), set(PROFILE_LAYERS))
        second_messages = fake.chat.completions.messages[1]
        self.assertTrue(any("校验失败" in message["content"] for message in second_messages))

    def test_queue_advances_only_after_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.json"
            profile_path.write_text(json.dumps({"state_axis": {"static_profile": {}}}), encoding="utf-8")
            failed = ProfileBatchUpdater(
                str(profile_path),
                extractor=StaticExtractor(error=RuntimeError("temporary")),
                min_user_messages=99,
                max_wait_seconds=9999,
            )
            failed.submit_turn("我重视成长")
            self.assertFalse(failed.process_pending())
            self.assertEqual(len(json.loads(failed.queue_path.read_text())["turns"]), 1)
            if failed._timer:
                failed._timer.cancel()

            succeeded = ProfileBatchUpdater(
                str(profile_path),
                extractor=StaticExtractor(),
                min_user_messages=99,
                max_wait_seconds=9999,
            )
            self.assertTrue(succeeded.process_pending())
            self.assertEqual(json.loads(succeeded.queue_path.read_text())["turns"], [])
            saved = json.loads(profile_path.read_text())
            self.assertEqual(saved["state_axis"]["static_profile"]["core"]["summary"]["value"], "core摘要")
            if succeeded._timer:
                succeeded._timer.cancel()


if __name__ == "__main__":
    unittest.main()
