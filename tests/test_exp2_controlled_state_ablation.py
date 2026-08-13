from __future__ import annotations

import unittest
from pathlib import Path

from src.experiments.exp2_controlled_state_ablation import (
    _audit_condition_inputs,
    _build_response_prompt,
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

if __name__ == "__main__":
    unittest.main()
