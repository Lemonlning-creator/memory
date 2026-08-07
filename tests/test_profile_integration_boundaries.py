import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.profile_schema import PROFILE_FIELDS, PROFILE_LAYERS, create_empty_static_profile
from src.profile_utils import (
    convert_to_flat_profile,
    count_profile_attributes,
    get_attribute_confidences,
    migrate_profile,
    serialize_static_profile_for_prompt,
)


class FakeLLM:
    def __init__(self, *args, **kwargs):
        self.last_model_timing = {"first_char_seconds": None}
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


class FakeMemory:
    def __init__(self, *args, **kwargs):
        self.short_term_memory = []

    def flush_short_term_memory(self, llm):
        return []

    def extract_long_term_memory(self, llm):
        return None

    def get_recent_messages(self, limit=100):
        return []


class ProfileBoundaryTests(unittest.TestCase):
    def test_prompt_serialization_preserves_grouping_and_all_fields(self):
        profile = create_empty_static_profile()
        profile["core"]["summary"] = "重视可靠交付。"
        profile["core"]["values"] = ["重视证据。", "尊重事实。"]

        serialized = serialize_static_profile_for_prompt(profile)
        expected_lines = []
        for layer in PROFILE_LAYERS:
            expected_lines.append(f"{layer}:")
            for field in ("summary", *PROFILE_FIELDS[layer]):
                value = profile[layer][field]
                text = "；".join(value) if isinstance(value, list) else value
                expected_lines.append(f"  - {field}: {text}")

        self.assertEqual(serialized, "\n".join(expected_lines))
        self.assertIn("core:\n  - summary: 重视可靠交付。", serialized)
        self.assertIn("  - values: 重视证据。；尊重事实。", serialized)
        self.assertIn("  - motivations: ", serialized)
        self.assertIn("regulation:\n  - summary: \n  - stress_response: ", serialized)
        self.assertEqual(len(serialized.splitlines()), 25)

    def test_bare_profile_metrics_and_flattening_ignore_summary(self):
        profile = create_empty_static_profile()
        profile["core"]["summary"] = "用户重视证据与长期发展。"
        profile["core"]["values"] = ["重视证据。"]
        self.assertEqual(count_profile_attributes(profile), 1)
        self.assertEqual(get_attribute_confidences(profile), {"core.values": 0.5})
        self.assertEqual(convert_to_flat_profile(profile), {
            "core_values": ["重视证据。"],
            "core_motivations": [], "core_long_term_goals": [],
            "regulation_stress_response": [], "regulation_emotion_regulation": [], "regulation_conflict_style": [],
            "cognition_thinking_style": [], "cognition_decision_style": [], "cognition_beliefs": [],
            "identity_self_identity": [], "identity_social_identity": [], "identity_life_context": [],
            "behavior_interaction_style": [], "behavior_habits": [], "behavior_preferences": [],
        })

    def test_legacy_wrapper_migrates_to_bare_contract(self):
        legacy = {
            "state_axis": {
                "static_profile": {
                    "core": {
                        "summary": {"value": "旧摘要", "confidence": 0.9},
                        "values": {"value": "重视稳定", "memory_ids": ["m1"]},
                    }
                },
                "current_state": {"mood": "x"},
            },
            "context_axis": {"current_context": "旧上下文"},
        }
        migrated = migrate_profile(legacy)
        self.assertEqual(set(migrated), set(PROFILE_LAYERS))
        self.assertEqual(migrated["core"]["summary"], "旧摘要")
        self.assertEqual(migrated["core"]["values"], ["重视稳定"])
        self.assertEqual(migrated["identity"]["life_context"], [])
        self.assertNotIn("state_axis", migrated)
        self.assertNotIn("memory_ids", json.dumps(migrated, ensure_ascii=False))

    def test_legacy_fields_are_mapped_only_to_clear_fixed_destinations(self):
        legacy = {
            "state_axis": {"static_profile": {
                "core": {
                    "desires": {"value": "希望创造长期价值。"},
                    "sources of meaning": {"value": "通过帮助他人成长获得意义。"},
                    "fears": {"value": "担心停滞。"},
                },
                "regulation": {
                    "avoidance": {"value": "面对困难时主动处理。"},
                    "aggression": {"value": "分歧时倾向理性沟通。"},
                },
                "cognition": {"decision style": {"value": "会比较多个方案。"}},
                "identity": {"occupation": {"value": "是一名工程师。"}},
                "behavior": {
                    "content preferences": {"value": "偏好技术内容。"},
                    "long-term behavior patterns": {"value": "会持续迭代成果。"},
                },
            }}
        }
        migrated = migrate_profile(legacy)
        self.assertEqual(migrated["core"]["long_term_goals"], ["希望创造长期价值。"])
        self.assertEqual(migrated["core"]["motivations"], ["通过帮助他人成长获得意义。"])
        self.assertEqual(migrated["regulation"]["stress_response"], ["面对困难时主动处理。"])
        self.assertEqual(migrated["regulation"]["conflict_style"], ["分歧时倾向理性沟通。"])
        self.assertEqual(migrated["cognition"]["decision_style"], ["会比较多个方案。"])
        self.assertEqual(migrated["identity"]["social_identity"], ["是一名工程师。"])
        self.assertEqual(migrated["behavior"]["preferences"], ["偏好技术内容。"])
        self.assertEqual(migrated["behavior"]["habits"], ["会持续迭代成果。"])
        self.assertNotIn("担心停滞。", json.dumps(migrated, ensure_ascii=False))

    def test_new_batch_updater_is_scoped_to_bayesian_online(self):
        from src.agent import StateDrivenCompanionAgent
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            persona = root / "persona.json"
            persona.write_text("{}", encoding="utf-8")
            with patch("src.agent.LLMClient", FakeLLM), patch("src.agent.MemoryOSLocal", FakeMemory):
                agents = {}
                for mode in ("bayesian_online", "static", "periodic_rebuild"):
                    profile = root / f"{mode}.json"
                    profile.write_text(json.dumps(create_empty_static_profile()), encoding="utf-8")
                    agents[mode] = StateDrivenCompanionAgent(
                        profile_path=str(profile), persona_path=str(persona), update_mode=mode,
                    )
            self.assertIsNotNone(agents["bayesian_online"].profile_batch_updater)
            self.assertIsNone(agents["static"].profile_batch_updater)
            self.assertIsNone(agents["periodic_rebuild"].profile_batch_updater)
            timer = agents["bayesian_online"].profile_batch_updater._timer
            if timer:
                timer.cancel()

    def test_agent_migrates_legacy_file_without_rewriting_wrapper(self):
        from src.agent import StateDrivenCompanionAgent
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profile = root / "legacy.json"
            persona = root / "persona.json"
            persona.write_text("{}", encoding="utf-8")
            profile.write_text(json.dumps({"state_axis": {"static_profile": {"core": {"values": {"value": "重视稳定"}}}}}), encoding="utf-8")
            with patch("src.agent.LLMClient", FakeLLM), patch("src.agent.MemoryOSLocal", FakeMemory):
                agent = StateDrivenCompanionAgent(profile_path=str(profile), persona_path=str(persona), update_mode="static")
            saved = json.loads(profile.read_text(encoding="utf-8"))
            self.assertEqual(set(saved), set(PROFILE_LAYERS))
            self.assertEqual(agent.user_profile["state_axis"]["static_profile"], saved)
            self.assertIn("current_state", agent.user_profile["state_axis"])

    def test_profile_api_returns_and_accepts_only_bare_contract(self):
        import app as app_module

        class ApiAgent:
            def __init__(self, path):
                self.profile_path = str(path)
                self.user_profile = {
                    "state_axis": {
                        "static_profile": create_empty_static_profile(),
                        "current_state": {"mood": "runtime-only"},
                        "projected_state": {},
                    },
                    "context_axis": {"current_context": "runtime-only"},
                }

            def _on_profile_updated(self, profile):
                self.user_profile["state_axis"]["static_profile"] = profile

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "api_profile.json"
            api_agent = ApiAgent(path)
            with patch.object(app_module, "agent", api_agent):
                client = app_module.app.test_client()
                response = client.get("/api/profile")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(set(response.get_json()), set(PROFILE_LAYERS))
                self.assertNotIn("state_axis", response.get_json())

                updated = create_empty_static_profile()
                updated["behavior"]["preferences"] = ["偏好安静环境。"]
                response = client.post("/api/profile", json=updated)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), updated)
                self.assertEqual(api_agent.user_profile["state_axis"]["current_state"], {"mood": "runtime-only"})

                response = client.post("/api/profile", json={"core": {}})
                self.assertEqual(response.status_code, 400)

    def test_chat_accepts_frontend_character_id_and_rejects_identity_mismatch(self):
        import app as app_module

        class ChatAgent:
            user_profile = create_empty_static_profile()

            def chat_stream(self, message, ablate_dimension=None):
                yield {"type": "done", "response": f"echo:{message}"}

        with patch.object(app_module, "agent", ChatAgent()), patch.object(
            app_module, "active_profile_id", "user"
        ), patch.object(app_module, "active_persona_id", "persona"), patch.object(
            app_module, "active_chat_count", 0
        ):
            client = app_module.app.test_client()
            response = client.post("/api/chat", json={
                "message": "hello",
                "character_id": "user",
                "persona_id": "persona",
            })
            self.assertEqual(response.status_code, 200)
            self.assertIn('"message": "echo:hello"', response.get_data(as_text=True))

            response = client.post("/api/chat", json={
                "message": "hello",
                "profile_id": "other",
                "persona_id": "persona",
            })
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.get_json()["error"], "character selection mismatch")

    def test_new_user_profile_is_created_as_bare_contract(self):
        import app as app_module

        with tempfile.TemporaryDirectory() as temp_dir:
            working = Path(temp_dir)
            with patch.object(app_module, "WORKING_PROFILE_DIR", working), patch.object(
                app_module, "USER_PROFILES", {}
            ), patch.object(app_module, "ALLOW_CREATE_USER_PROFILES", True):
                profile_info = app_module.ensure_user_profile("new_user")

            profile_path = Path(profile_info["source_path"])
            saved = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertEqual(profile_path, working / "new_user_profile.json")
            self.assertEqual(saved, create_empty_static_profile())
            self.assertEqual(set(saved), set(PROFILE_LAYERS))
            self.assertNotIn("state_axis", saved)

    def test_character_reselection_reuses_bare_working_profile(self):
        import app as app_module

        class FakeAgent:
            def __init__(self, profile_path, persona_path, user_name):
                self.profile_path = profile_path
                self.persona_path = persona_path
                self.user_name = user_name
                self.user_profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.json"
            persona = root / "persona.json"
            source.write_text(json.dumps(create_empty_static_profile()), encoding="utf-8")
            persona.write_text("{}", encoding="utf-8")
            working = root / "working"
            with patch.object(app_module, "USER_PROFILES", {"user": {"source_path": str(source), "display_name": "User"}}), patch.object(
                app_module, "AGENT_PERSONAS", {"persona": {"source_path": str(persona), "display_name": "Persona"}}
            ), patch.object(app_module, "WORKING_PROFILE_DIR", working), patch.object(app_module, "StateDrivenCompanionAgent", FakeAgent):
                first = app_module.build_agent_for_character("user", "persona")
                saved = json.loads(Path(first.profile_path).read_text(encoding="utf-8"))
                saved["core"]["summary"] = "已增长"
                Path(first.profile_path).write_text(json.dumps(saved), encoding="utf-8")
                second = app_module.build_agent_for_character("user", "persona")
                self.assertEqual(second.user_profile["core"]["summary"], "已增长")
                self.assertEqual(set(second.user_profile), set(PROFILE_LAYERS))


if __name__ == "__main__":
    unittest.main()
