from __future__ import annotations

import unittest
from pathlib import Path

from src.agent import StateDrivenCompanionAgent
from src.experiments.exp2_controlled_state_ablation import (
    _audit_condition_inputs,
    _build_response_prompt,
    _condition_manifest,
    _parse_conditions,
    _reconstruct_preceding_states,
    _selected_state,
)
from src.prompts.exp2_versions import get_exp2_prompt_bundle


class ControlledStateAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.full_state = {
            "emotional_reaction": 1,
            "interpretation": 2,
            "exploration": 0,
            "activated_tone": "warm but casual",
            "response_guidance": "acknowledge the disappointment briefly",
        }

    def test_state_conditions_change_only_intended_payload(self) -> None:
        self.assertEqual(_selected_state(self.full_state, "full_state"), self.full_state)
        self.assertEqual(
            _selected_state(self.full_state, "scores_only"),
            {
                "emotional_reaction": 1,
                "interpretation": 2,
                "exploration": 0,
            },
        )
        self.assertEqual(_selected_state(self.full_state, "no_state"), {})
        scores_plus = _selected_state(
            self.full_state,
            "scores_plus_tone",
        )
        self.assertNotIn("response_guidance", scores_plus)
        self.assertEqual(
            scores_plus,
            {
                "emotional_reaction": 1,
                "interpretation": 2,
                "exploration": 0,
                "activated_tone": "warm but casual",
            },
        )
        self.assertIn("response_guidance", self.full_state)

    def test_scores_only_rejects_missing_numeric_contract(self) -> None:
        with self.assertRaises(ValueError):
            _selected_state({"emotional_reaction": 1}, "scores_only")

    def test_scores_plus_rejects_missing_tone(self) -> None:
        incomplete = dict(self.full_state)
        incomplete.pop("activated_tone")
        with self.assertRaises(ValueError):
            _selected_state(incomplete, "scores_plus_tone")

    def test_first_state_is_empty_and_later_state_is_source_predecessor(self) -> None:
        predictions = [{"example_id": "a"}, {"example_id": "b"}, {"example_id": "c"}]
        states = [
            {"example_id": "a", "core_current_state": {"mood": "a"}},
            {"example_id": "b", "core_current_state": {"mood": "b"}},
            {"example_id": "c", "core_current_state": {"mood": "c"}},
        ]
        preceding = _reconstruct_preceding_states(predictions, states)
        self.assertEqual(preceding["a"], {})
        self.assertEqual(preceding["b"], {"mood": "a"})
        self.assertEqual(preceding["c"], {"mood": "b"})

    def test_prompt_replay_keeps_frozen_inputs_equal(self) -> None:
        bundle = get_exp2_prompt_bundle("v18_reflective_grounding_joint_gate")
        prediction = {
            "user_message": "I finally finished it.",
            "generation_input_audit": {
                "previous_empathy_state": self.full_state,
                "relevant_memory": {"short_term": ["same history"]},
            },
        }
        common = {
            "bundle": bundle,
            "prediction": prediction,
            "profile_text": "- values: ['family']",
            "persona": {"core_layer": {"summary": "casual"}},
            "current_state": {"emotional_state": "relieved"},
            "current_context": {},
        }
        full_prompt, full_frozen, full_payload = _build_response_prompt(
            **common, condition="full_state"
        )
        score_prompt, score_frozen, score_payload = _build_response_prompt(
            **common, condition="scores_only"
        )
        empty_prompt, empty_frozen, empty_payload = _build_response_prompt(
            **common, condition="no_state"
        )
        scores_plus_prompt, scores_plus_frozen, scores_plus_payload = (
            _build_response_prompt(
                **common,
                condition="scores_plus_tone",
            )
        )
        self.assertEqual(full_frozen, score_frozen)
        self.assertEqual(score_frozen, empty_frozen)
        self.assertEqual(empty_frozen, scores_plus_frozen)
        self.assertNotEqual(full_prompt, score_prompt)
        self.assertNotEqual(score_prompt, empty_prompt)
        self.assertNotEqual(full_prompt, scores_plus_prompt)
        self.assertEqual(full_payload, self.full_state)
        self.assertEqual(set(score_payload), {
            "emotional_reaction", "interpretation", "exploration"
        })
        self.assertEqual(empty_payload, {})
        self.assertNotIn("response_guidance", scores_plus_payload)
        self.assertEqual(
            scores_plus_payload["activated_tone"],
            self.full_state["activated_tone"],
        )

    def test_condition_parser_is_strict(self) -> None:
        self.assertEqual(
            _parse_conditions("full_state,no_state"),
            ["full_state", "no_state"],
        )
        with self.assertRaises(ValueError):
            _parse_conditions("full_state,unknown")

    def test_input_audit_rejects_missing_condition_outputs(self) -> None:
        with self.assertRaises(RuntimeError):
            _audit_condition_inputs(
                Path("does-not-exist"),
                ["full_state", "no_state"],
                [],
            )

    def test_v26_is_standalone_role_clear_and_scores_only(self) -> None:
        v18 = get_exp2_prompt_bundle("v18_reflective_grounding_joint_gate")
        v26 = get_exp2_prompt_bundle("v26_local_evidence_single_act")
        self.assertEqual(v26.response_state_policy, "scores_only")
        self.assertFalse(v26.response_system.startswith(v18.response_system))
        self.assertIn("no comparable recent target-speaker reply", v26.response_system)
        self.assertIn("TARGET SPEAKER PERSONA", v26.response_user)
        self.assertIn("CURRENT USER PROFILE", v26.response_user)
        self.assertIn("THREE EMPATHY SCORES", v26.response_user)

    def test_v27_changes_only_v18_grounding_policy_and_uses_scores_only(self) -> None:
        v7 = get_exp2_prompt_bundle("v7_recent_style_imitation")
        v18 = get_exp2_prompt_bundle("v18_reflective_grounding_scores_only")
        v27 = get_exp2_prompt_bundle("v27_grounding_three_mode_gate")

        self.assertEqual(v27.response_state_policy, "scores_only")
        self.assertEqual(v27.response_user, v18.response_user)
        self.assertEqual(v27.alignment_system, v18.alignment_system)
        self.assertEqual(v27.alignment_user, v18.alignment_user)
        self.assertTrue(v18.response_system.startswith(v7.response_system))
        self.assertTrue(v27.response_system.startswith(v7.response_system))
        self.assertNotEqual(v27.response_system, v18.response_system)

        reflective_boundary = (
            "A. REFLECTIVE. Use one natural self-observation only when the active "
            "topic is the target speaker's own feeling, motive, realization, "
            "decision, change, or recurring pattern."
        )
        self.assertIn(reflective_boundary, v18.response_system)
        self.assertIn(reflective_boundary, v27.response_system)
        ordinary_boundary = (
            "C. ORDINARY. When neither A nor B is clearly supported, use the "
            "direct answer, acknowledgement, opinion, reaction, or supported "
            "self-disclosure this speaker would normally give."
        )
        self.assertIn(ordinary_boundary, v18.response_system)
        self.assertIn(ordinary_boundary, v27.response_system)
        self.assertIn("1. DIRECT_RESPONSE:", v27.response_system)
        self.assertIn("2. REPAIR_GROUNDING:", v27.response_system)
        self.assertIn("3. ELABORATION_GROUNDING:", v27.response_system)
        self.assertIn(
            "Choose B only for mode 2 or 3. Otherwise choose C.",
            v27.response_system,
        )

    def test_agent_scores_only_policy_is_strict(self) -> None:
        agent = object.__new__(StateDrivenCompanionAgent)
        agent.prompt_bundle = get_exp2_prompt_bundle("v26_local_evidence_single_act")
        result = {
            "empathy_state": {
                **self.full_state,
                "extra_field": "must not reach the response",
            }
        }
        self.assertEqual(
            agent._response_state_from_alignment(result),
            {
                "emotional_reaction": 1,
                "interpretation": 2,
                "exploration": 0,
            },
        )
        invalid = {"empathy_state": {**self.full_state, "exploration": "0"}}
        with self.assertRaises(ValueError):
            agent._response_state_from_alignment(invalid)

    def test_controlled_manifest_separates_source_and_response_prompts(self) -> None:
        manifest = _condition_manifest(
            condition="scores_only",
            source_root=Path("source"),
            cases=[],
            source_files={},
            source_prompt_version="v18_reflective_grounding_joint_gate",
            response_prompt_version="v26_local_evidence_single_act",
            model="qwen-plus",
            temperature=0.0,
            max_tokens=450,
        )
        self.assertEqual(
            manifest["source_prompt_version"],
            "v18_reflective_grounding_joint_gate",
        )
        self.assertEqual(
            manifest["response_prompt_version"],
            "v26_local_evidence_single_act",
        )
        self.assertIn("response_prompt_sha256", manifest)

        legacy = _condition_manifest(
            condition="scores_only",
            source_root=Path("source"),
            cases=[],
            source_files={},
            source_prompt_version="v18_reflective_grounding_joint_gate",
            response_prompt_version="v18_reflective_grounding_joint_gate",
            model="qwen-plus",
            temperature=0.0,
            max_tokens=450,
        )
        self.assertNotIn("response_prompt_version", legacy)
        self.assertNotIn("response_prompt_sha256", legacy)

if __name__ == "__main__":
    unittest.main()
