import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.profile_batch_updater import (
    KimiProfileExtractor,
    PROFILE_LAYERS,
    compact_profile_for_prompt,
    ProfileBatchUpdater,
    ProfileUpdateError,
    merge_patch,
    normalize_patch_field_names,
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

    def test_compacts_old_profile_without_old_evidence_metadata(self):
        compact = compact_profile_for_prompt({
            "core": {
                "values": {
                    "value": "重视证据",
                    "confidence": 0.8,
                    "evidence": "旧消息",
                    "evidence_message_ids": ["old-id"],
                    "updated_at": "yesterday",
                }
            }
        })
        self.assertEqual(compact, {"core": {"values": {"value": "重视证据", "confidence": 0.8}}})

    def test_normalizes_known_snake_case_aliases_only(self):
        patch = valid_patch()
        patch["layers"]["core"]["attributes"] = {"sources_of_meaning": item("科研与成长")}
        patch["layers"]["regulation"]["attributes"] = {"people_pleasing": item("倾向照顾他人感受")}
        normalized = normalize_patch_field_names(patch)
        self.assertIn("sources of meaning", normalized["layers"]["core"]["attributes"])
        self.assertIn("people-pleasing", normalized["layers"]["regulation"]["attributes"])
        validate_patch(normalized, {"m1"})

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



    def test_schema_rejects_report_style_profile_values(self):
        patch = {
            "layers": {
                layer: {"summary": None, "attributes": {}}
                for layer in PROFILE_LAYERS
            }
        }
        patch["layers"]["core"] = {
            "summary": {
                "value": "最终画像呈现为敏感但有边界的人。",
                "confidence": 0.8,
                "evidence_message_ids": ["m1"],
            },
            "attributes": {},
        }
        with self.assertRaises(ProfileUpdateError):
            validate_patch(patch, {"m1"})

        patch["layers"]["core"]["summary"]["value"] = "深层价值收敛为长期信任与人格底色。"
        with self.assertRaises(ProfileUpdateError):
            validate_patch(patch, {"m1"})

        patch["layers"]["core"]["summary"]["value"] = "对方重视长期信任，也愿意持续成长。"
        with self.assertRaises(ProfileUpdateError):
            validate_patch(patch, {"m1"})


    def test_system_prompt_enforces_agent_viewpoint(self):
        from src.profile_batch_updater import SYSTEM_PROMPT

        for required in [
            "它正在陪伴的这个人是什么样的人",
            "最终文字要像已写入画像的直接判断",
            "优先把相邻属性自然合成一句",
            "只归纳同层、已有证据支持的属性",
            "五层 summary 风格严格参照 deployment 分支预设画像",
            "面对压力时容易产生焦虑，但通常不会停下行动",
            "思考方式偏务实，关注现实可行性",
            "不要直接搬运用户的自我描述或任务要求",
            "从自然话题中的选择、反应、偏好、取舍和反复出现的行为中归纳",
            "测试验收",
        ]:
            self.assertIn(required, SYSTEM_PROMPT)

        for banned_example in ["最终画像呈现为", "深层价值收敛为", "人格底色是", "用户表示自己是", "总结来说", "可见其", "说明对方"]:
            self.assertIn(banned_example, SYSTEM_PROMPT)

    def test_prompt_payload_uses_raw_dialogue_without_assistant_replies(self):
        class CapturingClient:
            def __init__(self):
                self.messages = None

            class Chat:
                def __init__(self, outer):
                    self.completions = self
                    self.outer = outer

                def create(self, **kwargs):
                    self.outer.messages = kwargs["messages"]
                    content = json.dumps({
                        "layers": {
                            layer: {"summary": None, "attributes": {}}
                            for layer in PROFILE_LAYERS
                        }
                    })
                    return type("Response", (), {
                        "choices": [type("Choice", (), {
                            "message": type("Message", (), {"content": content})()
                        })()]
                    })()

            @property
            def chat(self):
                return self.Chat(self)

        client = CapturingClient()
        extractor = KimiProfileExtractor(client=client)
        extractor.extract({}, [{
            "message_id": "m1",
            "user": "我在讨论一个产品方案，不是在要求你给我贴标签。",
            "created_at": "2026-07-24T00:00:00+00:00",
        }])
        payload = json.loads(client.messages[1]["content"])
        self.assertEqual(list(payload["raw_dialogue_batch"][0]), ["message_id", "user", "created_at"])
        self.assertNotIn("assistant", payload["raw_dialogue_batch"][0])
        self.assertIn("field_whitelist", payload)


if __name__ == "__main__":
    unittest.main()
