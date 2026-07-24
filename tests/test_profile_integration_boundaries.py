import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.profile_utils import (
    convert_to_flat_profile,
    count_profile_attributes,
    get_attribute_confidences,
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
    def test_summary_does_not_change_experiment_attribute_metrics(self):
        static = {
            "core": {
                "summary": {"value": "概括", "confidence": 0.9},
                "values": {"value": "重视证据", "confidence": 0.8},
            }
        }
        self.assertEqual(count_profile_attributes(static), 1)
        self.assertEqual(list(get_attribute_confidences(static)), ["core.values"])
        self.assertEqual(list(convert_to_flat_profile(static)), ["core_values"])

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
                    profile.write_text(json.dumps({
                        "state_axis": {"static_profile": {}, "current_state": {}, "projected_state": {}},
                        "context_axis": {},
                    }), encoding="utf-8")
                    agents[mode] = StateDrivenCompanionAgent(
                        profile_path=str(profile),
                        persona_path=str(persona),
                        update_mode=mode,
                    )
            self.assertIsNotNone(agents["bayesian_online"].profile_batch_updater)
            self.assertIsNone(agents["static"].profile_batch_updater)
            self.assertIsNone(agents["periodic_rebuild"].profile_batch_updater)
            timer = agents["bayesian_online"].profile_batch_updater._timer
            if timer:
                timer.cancel()

    def test_character_reselection_reuses_working_profile(self):
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
            source.write_text(json.dumps({"core": {}}), encoding="utf-8")
            persona.write_text("{}", encoding="utf-8")
            working = root / "working"
            with patch.object(app_module, "USER_PROFILES", {
                "user": {"source_path": str(source), "display_name": "User"}
            }), patch.object(app_module, "AGENT_PERSONAS", {
                "persona": {"source_path": str(persona), "display_name": "Persona"}
            }), patch.object(app_module, "WORKING_PROFILE_DIR", working), patch.object(
                app_module, "StateDrivenCompanionAgent", FakeAgent
            ):
                first = app_module.build_agent_for_character("user", "persona")
                saved = json.loads(Path(first.profile_path).read_text(encoding="utf-8"))
                saved["state_axis"]["static_profile"]["core"]["summary"] = {"value": "已增长"}
                Path(first.profile_path).write_text(json.dumps(saved), encoding="utf-8")
                second = app_module.build_agent_for_character("user", "persona")
                self.assertEqual(
                    second.user_profile["state_axis"]["static_profile"]["core"]["summary"]["value"],
                    "已增长",
                )


if __name__ == "__main__":
    unittest.main()
