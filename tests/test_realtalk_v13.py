from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.realtalk_v13 import (
    _actor_contract,
    _select_gate6,
    _select_gate18,
    _validate_decision,
    conditional_statistics,
)
from src.experiments.realtalk_v13_schemas import (
    TURN_TRIGGERS,
    normalize_v13_decision,
)


def _decision() -> dict:
    return {
        "situation": {
            "topic": "plans",
            "partner_move": "asks about plans",
            "turn_obligation": "answer",
            "open_obligation": "answer-current-question",
            "obligation_source_turn_id": "session_1:turn_1",
            "explicit_affect": "",
            "support_request": False,
            "uncertainty": "low",
        },
        "relevant_user_domain": [],
        "alignment": {
            "orientation": "balanced",
            "lambda_trace": 0.5,
            "decision_basis": "A direct question creates a response obligation.",
        },
        "behavior_policy": {
            "primary_move": "answer",
            "companion_move": "ask",
            "reflection_depth": "surface",
            "question_plan": "reciprocal",
            "question_target": "the partner's plans",
            "relational_register": "casual-neutral",
            "message_shape": "typical",
            "content_direction": "answer and return the same plans slot",
            "tone": "casual",
        },
    }


class RealTalkV13Tests(unittest.TestCase):
    def test_direct_question_requires_adaptive_alignment(self):
        value = normalize_v13_decision(_decision())
        history = [{"turn_id": "session_1:turn_1", "content": "What are your plans?"}]
        self.assertIs(_validate_decision(value, history, "What are your plans?"), value)
        value["alignment"].update({"orientation": "self-led", "lambda_trace": 0.2})
        with self.assertRaisesRegex(ValueError, "direct partner question"):
            _validate_decision(value, history, "What are your plans?")

    def test_obligation_source_must_be_visible(self):
        value = normalize_v13_decision(_decision())
        value["situation"]["open_obligation"] = "answer-earlier-unanswered-question"
        with self.assertRaisesRegex(ValueError, "exact visible history turn"):
            _validate_decision(
                value,
                [{"turn_id": "session_1:turn_2", "content": "A statement"}],
                "A statement",
            )

    def test_question_obligation_source_must_really_be_a_question(self):
        value = normalize_v13_decision(_decision())
        history = [{"turn_id": "session_1:turn_1", "content": "A statement"}]
        with self.assertRaisesRegex(ValueError, "visible question mark"):
            _validate_decision(value, history, "A statement")

    def test_none_obligation_discards_redundant_source(self):
        value = _decision()
        value["situation"]["open_obligation"] = "none"
        normalized = normalize_v13_decision(value)
        self.assertEqual(normalized["situation"]["obligation_source_turn_id"], "")

    def test_orientation_and_lambda_ranges_are_coupled(self):
        value = _decision()
        value["alignment"]["lambda_trace"] = 0.2
        with self.assertRaisesRegex(ValueError, "does not match balanced"):
            normalize_v13_decision(value)

    def test_question_plan_requires_exactly_one_ask_move(self):
        value = _decision()
        value["behavior_policy"]["companion_move"] = "none"
        with self.assertRaisesRegex(ValueError, "exactly one ask move"):
            normalize_v13_decision(value)

        value = _decision()
        value["behavior_policy"]["primary_move"] = "ask"
        with self.assertRaisesRegex(ValueError, "exactly one ask move"):
            normalize_v13_decision(value)

    def test_actor_contract_has_exact_question_and_reflection_permissions(self):
        policy = _decision()["behavior_policy"]
        contract = _actor_contract(policy)
        self.assertIn("Ask exactly one reciprocal question", contract)
        self.assertIn("Do not explain motives", contract)
        policy.update({
            "question_plan": "none",
            "question_target": "",
            "companion_move": "none",
            "reflection_depth": "brief-reflective",
        })
        contract = _actor_contract(policy)
        self.assertIn("Do not ask any question", contract)
        self.assertIn("at most one brief reason", contract)

    def test_conditional_statistics_cover_all_triggers(self):
        turns = [
            {"session_id": "session_1", "speaker": "A", "content": "Hello", "message_indices": [0]},
            {"session_id": "session_1", "speaker": "B", "content": "How are you?", "message_indices": [1]},
            {"session_id": "session_1", "speaker": "A", "content": "I'm good. You?", "message_indices": [2]},
            {"session_id": "session_1", "speaker": "B", "content": "I went out today", "message_indices": [3]},
            {"session_id": "session_1", "speaker": "A", "content": "Nice", "message_indices": [4]},
        ]
        stats = conditional_statistics(turns, "A")
        self.assertEqual(set(stats), set(TURN_TRIGGERS))
        self.assertEqual(stats["session-opening"]["observations"], 1)
        self.assertEqual(stats["after-direct-question"]["question_rate"], 1.0)
        self.assertEqual(stats["after-partner-disclosure"]["observations"], 1)

    def test_progressive_gate_helpers_are_nested(self):
        speakers = ["Emi", "Nicolas", "Kevin", "Akib", "Muhhamed", "Nebraas"]
        sessions = ("session_1", "session_2", "session_3")
        items = []
        for index in range(30):
            category = ("reflect", "ground", "intimacy", "clean")[index % 4]
            classes = {key: key == category for key in ("reflect", "ground", "intimacy", "clean")}
            items.append({
                "result_id": f"id-{index:02d}",
                "speaker": speakers[index % len(speakers)],
                "session": sessions[index % len(sessions)],
                "message_level_index": index,
                "classes": classes,
                "error_count": int(category != "clean"),
            })
        gate6 = _select_gate6(items)
        gate18 = _select_gate18(items, gate6)
        self.assertEqual(len(gate6), 6)
        self.assertEqual(len(gate18), 18)
        self.assertTrue(set(gate6).issubset(gate18))


if __name__ == "__main__":
    unittest.main()
