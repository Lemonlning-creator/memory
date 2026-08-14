from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.realtalk_v9_judge_reuse import run


class RealTalkV9JudgeReuseTests(unittest.TestCase):
    def test_exact_rows_are_imported_and_first_source_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full.jsonl"
            row = {"result_id": "a:1", "generated_message": "same", "ground_truth": "truth"}
            full.write_text(json.dumps(row) + "\n")
            sources = []
            for index, grounding in enumerate((False, True)):
                predictions = root / f"pred{index}.jsonl"
                predictions.write_text(json.dumps(row) + "\n")
                judge = root / f"judge{index}"; judge.mkdir()
                judgments = {}
                for side in ("reference", "candidate"):
                    for metric in ("reflectiveness", "grounding", "empathy"):
                        value = {"emotional_reaction": 0, "interpretation": 0, "exploration": 0} if metric == "empathy" else (grounding if metric == "grounding" else False)
                        judgments[f"a:1:{side}:{metric}"] = {"value": value, "audit": {}}
                (judge / "checkpoint.json").write_text(json.dumps({"judgments": judgments, "errors": {}}))
                (judge / "summary.json").write_text(json.dumps({
                    "status": "complete",
                    "judge_protocol": "realtalk_appendix_c_full_prompt_within_session_v3",
                    "model_requested": "gpt-4o-mini",
                }))
                sources.append((f"source{index}", predictions, judge, "candidate"))
            result = run(full, root / "out", sources)
            checkpoint = json.loads((root / "out" / "checkpoint.json").read_text())
            self.assertEqual(result["imported_unique_ids"], 1)
            self.assertEqual(result["remaining_ids"], 0)
            self.assertEqual(result["duplicate_label_conflicts_ignored"], 2)
            self.assertFalse(checkpoint["judgments"]["a:1:candidate:grounding"]["value"])


if __name__ == "__main__":
    unittest.main()
