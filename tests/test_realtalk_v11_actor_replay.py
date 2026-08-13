from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.personaemp.client import ChatResult
from src.experiments.realtalk_v11_actor_replay import (
    _soft_action_contract,
    run,
    select_fixed_rows,
)


class FakeBackend:
    model = "deepseek-v4-flash"
    token_usage = {}

    def chat(self, *args, **kwargs):
        return ChatResult("A natural reply because it feels right.", self.model, 1, 1, 0.01, 1, "")


class RealTalkV11ActorReplayTests(unittest.TestCase):
    def test_fixed_selection_uses_first_and_last_per_session(self):
        rows = [
            {
                "result_id": f"a:{session}:{index}", "speaker": "A",
                "target_session": session, "message_level_index": index,
            }
            for session in ("session_1", "session_2", "session_3")
            for index in range(4)
        ]
        selected = select_fixed_rows(rows, ("A",))
        self.assertEqual(
            [row["result_id"] for row in selected],
            [
                "a:session_1:0", "a:session_1:3",
                "a:session_2:0", "a:session_2:3",
                "a:session_3:0", "a:session_3:3",
            ],
        )

    def test_soft_contract_allows_same_topic_addition_but_not_unlicensed_question(self):
        contract = _soft_action_contract("answer", "none")
        self.assertIn("Answer", contract)
        self.assertIn("Do not append", contract)
        reciprocal = _soft_action_contract("answer", "reciprocal-question")
        self.assertIn("exactly one short reciprocal question", reciprocal)

    def test_replay_keeps_v9_message_and_frozen_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            dataset = root / "dataset"
            source.mkdir(); dataset.mkdir()
            rows = []
            sessions = {}
            for session_number in range(1, 4):
                session = f"session_{session_number}"
                sessions[session] = [
                    {"speaker": "A", "clean_text": "truth", "dia_id": f"{session}-a"},
                    {"speaker": "B", "clean_text": "context", "dia_id": f"{session}-b"},
                    {"speaker": "A", "clean_text": "truth 2", "dia_id": f"{session}-c"},
                ]
                for index, turn in enumerate((0, 2)):
                    rows.append({
                        "result_id": f"a:message_{index}:{session}:turn_{turn}",
                        "speaker": "A", "test_chat": "chat.json",
                        "target_session": session, "message_level_index": session_number * 10 + index,
                        "context_turn_ids": [] if turn == 0 else [f"{session}:turn_0", f"{session}:turn_1"],
                        "ground_truth": "truth", "generated_message": "v9 old",
                        "situation": {},
                        "next_action": {"primary_move": "answer", "continuation_move": "none"},
                    })
            (dataset / "chat.json").write_text(json.dumps({
                "name": {"speaker_1": "A", "speaker_2": "B"}, **sessions,
            }), encoding="utf-8")
            (source / "predictions.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            (source / "self_domains.json").write_text(json.dumps({"A": {
                "communication_signature": {}, "interaction_policy_prior": {},
                "affective_social_signature": {}, "observable_statistics": {},
            }}), encoding="utf-8")
            (source / "run_manifest.json").write_text(
                json.dumps({"ours_model": "deepseek-v4-flash"}), encoding="utf-8"
            )
            with patch(
                "src.experiments.realtalk_v11_actor_replay._backend_from_env",
                return_value=FakeBackend(),
            ):
                manifest = run(source, dataset, root / "output", speakers=("A",))
            output = [json.loads(line) for line in (root / "output" / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["records_complete"], 6)
            self.assertTrue(all(row["v9_generated_message"] == "v9 old" for row in output))
            self.assertTrue(all(row["next_action"] == {"primary_move": "answer", "continuation_move": "none"} for row in output))


if __name__ == "__main__":
    unittest.main()
