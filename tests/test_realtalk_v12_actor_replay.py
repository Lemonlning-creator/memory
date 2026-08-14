from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.personaemp.client import ChatResult
from src.experiments.realtalk_v12_actor_replay import (
    GENERATION_USER_PROMPT,
    _action_contract,
    _compact_actor_action,
    _largest_remainder_allocation,
    _question_contract,
    _self_revelation_contract,
    _validate_actor_message,
    prepare_sample_manifests,
    run,
)


def action(primary="answer", revelation="state-only", continuation="none"):
    return {
        "communicative_intent": "unused free prose",
        "primary_move": primary,
        "content_direction": "current conversational slot",
        "self_expression": "unused delivery prose",
        "partner_adaptation": "unused partner prose",
        "tone": "casual",
        "message_scale": "short",
        "self_revelation_mode": revelation,
        "question_mode": "follow-up" if primary == "follow-up" else "none",
        "continuation_move": continuation,
        "missing_information": "their preference" if continuation != "none" else "",
    }


class FakeBackend:
    model = "deepseek-v4-flash"
    token_usage = {}

    def chat(self, *args, **kwargs):
        return ChatResult("A natural direct reply.", self.model, 1, 1, 0.01, 1, "")


class RealTalkV12ActorReplayTests(unittest.TestCase):
    def test_compact_action_omits_free_prose_and_private_alignment(self):
        compact = _compact_actor_action(action())
        self.assertEqual(set(compact), {
            "primary_move", "content_direction", "tone", "message_scale",
            "self_revelation_mode", "question_mode", "continuation_move",
            "missing_information",
        })
        self.assertNotIn("communicative_intent", compact)
        self.assertNotIn("partner_adaptation", compact)
        self.assertNotIn("self_expression", compact)

    def test_all_primary_moves_have_single_action_contracts(self):
        for primary in ("answer", "acknowledge", "self-disclose", "follow-up", "open", "topic-shift"):
            contract = _action_contract(_compact_actor_action(action(primary=primary)))
            self.assertTrue(contract)

    def test_revelation_contracts_are_structured_not_keyword_inferred(self):
        self.assertIn("Do not explain", _self_revelation_contract(_compact_actor_action(action(revelation="none"))))
        self.assertIn("do not explain why", _self_revelation_contract(_compact_actor_action(action(revelation="state-only"))))
        self.assertIn("At most one", _self_revelation_contract(_compact_actor_action(action(revelation="brief-reason-or-feeling"))))

    def test_early_v9_schema_gets_deterministic_projection(self):
        old = action()
        old.pop("self_revelation_mode")
        old.pop("missing_information")
        compact = _compact_actor_action(old, {"missing_information": "partner wellbeing"})
        self.assertEqual(compact["self_revelation_mode"], "state-only")
        self.assertEqual(compact["missing_information"], "partner wellbeing")
        old["primary_move"] = "acknowledge"
        self.assertEqual(
            _compact_actor_action(old, {})["self_revelation_mode"], "none"
        )

    def test_question_contract_and_runtime_validation(self):
        plain = _compact_actor_action(action())
        self.assertIn("Do not ask", _question_contract(plain))
        self.assertEqual(_validate_actor_message("Fine.", plain), "Fine.")
        with self.assertRaisesRegex(ValueError, "forbids"):
            _validate_actor_message("Fine?", plain)

        follow = _compact_actor_action(action(primary="follow-up"))
        self.assertEqual(_validate_actor_message("Which one?", follow), "Which one?")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _validate_actor_message("Which one", follow)

        reciprocal = _compact_actor_action(action(continuation="reciprocal-question"))
        self.assertIn("their preference", _question_contract(reciprocal))

    def test_prompt_excludes_metrics_lambda_and_full_user_domain(self):
        lowered = GENERATION_USER_PROMPT.casefold()
        for forbidden in ("grounding accuracy", "reflectiveness accuracy", "lambda_trace", "current user domain", "ground truth"):
            self.assertNotIn(forbidden, lowered)

    def test_selection_is_disjoint_and_balanced(self):
        rows = []
        speakers = ("A", "B")
        for speaker in speakers:
            for session in ("session_1", "session_2", "session_3"):
                for index in range(8):
                    rows.append({
                        "result_id": f"{speaker}:{session}:{index}",
                        "speaker": speaker,
                        "target_session": session,
                        "message_level_index": index,
                    })
        excluded = {f"{speaker}:{session}:0" for speaker in speakers for session in ("session_1", "session_2", "session_3")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = prepare_sample_manifests(
                rows, excluded, root, speakers=speakers, holdout_per_speaker=8
            )
            dev = set(json.loads((root / "v12_dev30_sample_ids.json").read_text()))
            holdout = set(json.loads((root / "v12_holdout80_sample_ids.json").read_text()))
            self.assertEqual(len(dev), 6)
            self.assertEqual(len(holdout), 16)
            self.assertFalse(dev & holdout)
            self.assertFalse(dev & excluded)
            self.assertFalse(holdout & excluded)
            self.assertEqual(result["holdout_records"], 16)

    def test_largest_remainder_respects_capacity(self):
        allocation = _largest_remainder_allocation(
            {"session_1": 10, "session_2": 2, "session_3": 0}, 8
        )
        self.assertEqual(sum(allocation.values()), 8)
        self.assertLessEqual(allocation["session_2"], 2)
        self.assertEqual(allocation["session_3"], 0)

    def test_replay_preserves_frozen_v9_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"; dataset = root / "dataset"
            source.mkdir(); dataset.mkdir()
            row = {
                "result_id": "a:message_1:session_1:turn_1",
                "speaker": "A", "test_chat": "chat.json",
                "target_session": "session_1", "message_level_index": 1,
                "context_turn_ids": ["session_1:turn_0"],
                "context_hash": "frozen", "ground_truth": "truth",
                "generated_message": "v9 old", "situation": {},
                "next_action": action(),
            }
            (source / "predictions.jsonl").write_text(json.dumps(row) + "\n")
            (source / "self_domains.json").write_text(json.dumps({"A": {
                "communication_signature": {}, "interaction_policy_prior": {},
                "affective_social_signature": {}, "observable_statistics": {},
            }}))
            (source / "run_manifest.json").write_text(json.dumps({"ours_model": "deepseek-v4-flash"}))
            sample_ids = root / "ids.json"
            sample_ids.write_text(json.dumps([row["result_id"]]))
            (dataset / "chat.json").write_text(json.dumps({
                "name": {"speaker_1": "A", "speaker_2": "B"},
                "session_1": [
                    {"speaker": "B", "clean_text": "context", "dia_id": "x"},
                    {"speaker": "A", "clean_text": "truth", "dia_id": "y"},
                ],
            }))
            with patch(
                "src.experiments.realtalk_v12_actor_replay._backend_from_env",
                return_value=FakeBackend(),
            ):
                manifest = run(source, dataset, sample_ids, root / "output")
            output = json.loads((root / "output" / "predictions.jsonl").read_text())
            self.assertEqual(manifest["records_complete"], 1)
            self.assertEqual(output["v9_generated_message"], "v9 old")
            self.assertEqual(output["next_action"], row["next_action"])
            self.assertEqual(output["context_hash"], "frozen")


if __name__ == "__main__":
    unittest.main()
