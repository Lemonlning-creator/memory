from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.realtalk_paired_judge import run


class RealTalkPairedJudgeTests(unittest.TestCase):
    def test_reference_is_judged_once_for_two_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v9 = root / "v9.jsonl"; v11 = root / "v11.jsonl"
            base = {"result_id": "a:id", "speaker": "A", "ground_truth": "reference"}
            v9.write_text(json.dumps({**base, "generated_message": "old"}) + "\n")
            v11.write_text(json.dumps({**base, "generated_message": "new"}) + "\n")
            calls = []

            def fake_chat(*args):
                prompt = args[-1]
                calls.append(prompt)
                if "Return only JSON" in prompt:
                    return '{"emotional_reaction":0,"interpretation":0,"exploration":0}', {}
                return "False", {}

            with patch.dict("os.environ", {
                "REALTALK_JUDGE_API_KEY": "test", "REALTALK_JUDGE_BASE_URL": "https://example.test/v1",
            }), patch("src.experiments.realtalk_paired_judge._contexts", return_value={"a:id": "history"}), patch(
                "src.experiments.realtalk_paired_judge._chat", side_effect=fake_chat
            ):
                summary = run(v9, v11, root, root / "judge", "gpt-4o-mini")
            self.assertEqual(len(calls), 9)
            self.assertEqual(summary["shared_reference_judgments"], 3)
            self.assertEqual(summary["status"], "complete")

    def test_rejects_misaligned_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v9 = root / "v9.jsonl"; v11 = root / "v11.jsonl"
            v9.write_text(json.dumps({"result_id": "a", "ground_truth": "x"}) + "\n")
            v11.write_text(json.dumps({"result_id": "b", "ground_truth": "x"}) + "\n")
            with patch.dict("os.environ", {
                "REALTALK_JUDGE_API_KEY": "test", "REALTALK_JUDGE_BASE_URL": "https://example.test/v1",
            }):
                with self.assertRaisesRegex(ValueError, "identical result IDs"):
                    run(v9, v11, root, root / "judge", "gpt-4o-mini")

    def test_v12_candidate_is_named_in_protocol_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v9 = root / "v9.jsonl"; v12 = root / "v12.jsonl"
            base = {"result_id": "a:id", "speaker": "A", "ground_truth": "reference"}
            v9.write_text(json.dumps({**base, "generated_message": "old"}) + "\n")
            v12.write_text(json.dumps({**base, "generated_message": "new"}) + "\n")

            def fake_chat(*args):
                prompt = args[-1]
                if "Return only JSON" in prompt:
                    return '{"emotional_reaction":0,"interpretation":0,"exploration":0}', {}
                return "False", {}

            with patch.dict("os.environ", {
                "REALTALK_JUDGE_API_KEY": "test", "REALTALK_JUDGE_BASE_URL": "https://example.test/v1",
            }), patch("src.experiments.realtalk_paired_judge._contexts", return_value={"a:id": "history"}), patch(
                "src.experiments.realtalk_paired_judge._chat", side_effect=fake_chat
            ):
                summary = run(v9, v12, root, root / "judge", "gpt-4o-mini", "v12")
            self.assertIn("v12", summary["methods"])
            self.assertIn("delta_v12_minus_v9", summary)
            self.assertEqual(summary["protocol"], "realtalk_appendix_c_paired_v9_v12_v1")

    def test_v13_candidate_is_named_in_protocol_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v9 = root / "v9.jsonl"; v13 = root / "v13.jsonl"
            base = {"result_id": "a:id", "speaker": "A", "ground_truth": "reference"}
            v9.write_text(json.dumps({**base, "generated_message": "old"}) + "\n")
            v13.write_text(json.dumps({**base, "generated_message": "new"}) + "\n")

            def fake_chat(*args):
                prompt = args[-1]
                if "Return only JSON" in prompt:
                    return '{"emotional_reaction":0,"interpretation":0,"exploration":0}', {}
                return "False", {}

            with patch.dict("os.environ", {
                "REALTALK_JUDGE_API_KEY": "test", "REALTALK_JUDGE_BASE_URL": "https://example.test/v1",
            }), patch("src.experiments.realtalk_paired_judge._contexts", return_value={"a:id": "history"}), patch(
                "src.experiments.realtalk_paired_judge._chat", side_effect=fake_chat
            ):
                summary = run(v9, v13, root, root / "judge", "gpt-4o-mini", "v13")
            self.assertIn("v13", summary["methods"])
            self.assertIn("delta_v13_minus_v9", summary)
            self.assertEqual(summary["protocol"], "realtalk_appendix_c_paired_v9_v13_v1")


if __name__ == "__main__":
    unittest.main()
