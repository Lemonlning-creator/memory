import json
import tempfile
import unittest
from pathlib import Path

from src.profile_batch_updater import (
    KimiProfileExtractor,
    PROFILE_FIELDS,
    PROFILE_LAYERS,
    ProfileBatchUpdater,
    ProfileUpdateError,
    SYSTEM_PROMPT,
    build_profile_response_format,
    merge_patch,
    validate_patch,
)
from src.profile_schema import create_empty_static_profile


def empty_layers():
    return {
        layer: {"summary": None, **{field: None for field in PROFILE_FIELDS[layer]}}
        for layer in PROFILE_LAYERS
    }


def valid_patch():
    layers = empty_layers()
    layers["core"]["summary"] = "用户重视稳定成长，并希望长期发展。"
    layers["core"]["values"] = ["重视稳定和长期发展。"]
    return {"layers": layers}


class FakeCompletions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return type("Response", (), {
            "choices": [type("Choice", (), {"message": type("Message", (), {"content": output})()})()]
        })()


class FakeClient:
    def __init__(self, outputs):
        self.chat = type("Chat", (), {"completions": FakeCompletions(outputs)})()


class StaticExtractor:
    def __init__(self, patch=None, error=None):
        self.patch = patch if patch is not None else valid_patch()["layers"]
        self.error = error
        self.calls = []

    @property
    def available(self):
        return True

    def extract(self, current_profile, turns):
        self.calls.append((current_profile, list(turns)))
        if self.error:
            raise self.error
        return self.patch


class ProfileBatchUpdaterTests(unittest.TestCase):
    def test_strict_response_format_has_only_fixed_bare_fields(self):
        response_format = build_profile_response_format()
        self.assertEqual(response_format["type"], "json_schema")
        schema = response_format["json_schema"]["schema"]
        layers = schema["properties"]["layers"]
        self.assertEqual(layers["required"], list(PROFILE_LAYERS))
        core = layers["properties"]["core"]
        self.assertEqual(set(core["properties"]), {"summary", *PROFILE_FIELDS["core"]})
        self.assertNotIn("evidence_message_ids", json.dumps(response_format, ensure_ascii=False))
        self.assertNotIn("confidence", json.dumps(response_format, ensure_ascii=False))

    def test_validation_accepts_null_updates_and_rejects_unknown_or_legacy_fields(self):
        self.assertEqual(
            validate_patch({"layers": empty_layers()}),
            {layer: {} for layer in PROFILE_LAYERS},
        )
        bad = valid_patch()
        bad["layers"]["core"]["evidence"] = []
        with self.assertRaises(ProfileUpdateError):
            validate_patch(bad)
        bad = valid_patch()
        bad["layers"]["core"]["values"] = {"value": "旧叶子对象"}
        with self.assertRaises(ProfileUpdateError):
            validate_patch(bad)

    def test_merge_is_incremental_and_persists_only_bare_five_layers(self):
        original = create_empty_static_profile()
        original["core"]["values"] = ["重视稳定。"]
        original["behavior"]["habits"] = ["保持运动习惯。"]
        patch = validate_patch(valid_patch())
        merged = merge_patch(original, patch)
        self.assertEqual(set(merged), set(PROFILE_LAYERS))
        self.assertEqual(merged["core"]["values"], ["重视稳定和长期发展。"])
        self.assertEqual(merged["behavior"]["habits"], ["保持运动习惯。"])
        self.assertEqual(merged["identity"]["summary"], "")
        self.assertNotIn("state_axis", merged)
        self.assertNotIn("evidence_message_ids", json.dumps(merged, ensure_ascii=False))

    def test_extractor_injects_old_profile_and_entire_pending_batch(self):
        fake = FakeClient([json.dumps(valid_patch(), ensure_ascii=False)])
        extractor = KimiProfileExtractor(client=fake)
        current = create_empty_static_profile()
        current["core"]["values"] = ["重视稳定。"]
        turns = [
            {"message_id": "m1", "user": "我希望毕业后有稳定的发展。", "created_at": "2026-07-25T00:00:00+00:00"},
            {"message_id": "m2", "user": "我也愿意长期学习。", "created_at": "2026-07-25T00:01:00+00:00"},
        ]
        extractor.extract(current, turns)
        request = fake.chat.completions.requests[0]
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(payload["current_profile"]["core"]["values"], ["重视稳定。"])
        self.assertEqual(payload["raw_dialogue_batch"], turns)
        self.assertEqual(request["response_format"], build_profile_response_format())
        self.assertEqual(request["temperature"], 0.6)

    def test_system_prompt_directs_model_to_ignore_microphone_noise_without_short_message_rule(self):
        self.assertIn("语音误收音", SYSTEM_PROMPT)
        self.assertIn("不要因为消息短就忽略", SYSTEM_PROMPT)
        self.assertIn("只有能稳定归属", SYSTEM_PROMPT)

    def test_success_consumes_snapshot_and_preserves_newly_arrived_turn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "u_profile.json"
            profile_path.write_text(json.dumps(create_empty_static_profile()), encoding="utf-8")

            class ConcurrentExtractor(StaticExtractor):
                def __init__(self):
                    super().__init__()
                    self.updater = None

                def extract(self, current_profile, turns):
                    result = super().extract(current_profile, turns)
                    self.updater.submit_turn("处理中新增")
                    return result

            extractor = ConcurrentExtractor()
            updater = ProfileBatchUpdater(str(profile_path), extractor=extractor, min_user_messages=99, max_wait_seconds=9999)
            extractor.updater = updater
            updater.submit_turn("第一条")
            updater.submit_turn("第二条")
            self.assertTrue(updater.process_pending())
            saved = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertEqual(set(saved), set(PROFILE_LAYERS))
            remaining = updater._load_queue()["turns"]
            self.assertEqual([turn["user"] for turn in remaining], ["处理中新增"])
            self.assertEqual(len(extractor.calls[0][1]), 2)
            if updater._timer:
                updater._timer.cancel()

    def test_empty_update_is_success_and_consumes_noise_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "u_profile.json"
            original = create_empty_static_profile()
            profile_path.write_text(json.dumps(original), encoding="utf-8")
            empty_patch = {layer: {} for layer in PROFILE_LAYERS}
            updater = ProfileBatchUpdater(str(profile_path), extractor=StaticExtractor(patch=empty_patch), min_user_messages=99, max_wait_seconds=9999)
            updater.submit_turn("嗯嗯嗯嗯")
            self.assertTrue(updater.process_pending())
            self.assertEqual(updater._load_queue()["turns"], [])
            self.assertEqual(json.loads(profile_path.read_text(encoding="utf-8")), original)
            if updater._timer:
                updater._timer.cancel()

    def test_failure_keeps_pending(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "u_profile.json"
            profile_path.write_text(json.dumps(create_empty_static_profile()), encoding="utf-8")
            updater = ProfileBatchUpdater(str(profile_path), extractor=StaticExtractor(error=RuntimeError("down")), min_user_messages=99, max_wait_seconds=9999)
            updater.submit_turn("需要保留")
            self.assertFalse(updater.process_pending())
            self.assertEqual(len(updater._load_queue()["turns"]), 1)
            if updater._timer:
                updater._timer.cancel()


if __name__ == "__main__":
    unittest.main()
