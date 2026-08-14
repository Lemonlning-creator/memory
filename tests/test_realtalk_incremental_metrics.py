from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.realtalk_incremental_metrics import run


class RealTalkIncrementalMetricsTests(unittest.TestCase):
    def test_reuses_frozen_rows_and_computes_only_new_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = self._row("a", "A", 1.0)
            added = self._row("b", "B", None)
            expanded = root / "expanded.jsonl"
            prior = root / "prior.jsonl"
            expanded.write_text("\n".join(json.dumps(x) for x in (previous, added)) + "\n")
            prior.write_text(json.dumps(previous) + "\n")

            def fake_metrics(predictions, output_dir):
                row = json.loads(predictions.read_text())
                scored = self._row(row["result_id"], row["speaker"], 0.5)
                output_dir.mkdir(parents=True)
                (output_dir / "results_with_local_metrics.jsonl").write_text(json.dumps(scored) + "\n")

            with patch("src.experiments.realtalk_incremental_metrics.run_local_metrics", side_effect=fake_metrics):
                result = run(expanded, prior, root / "out")
            self.assertEqual(result["manifest"]["records_reused"], 1)
            self.assertEqual(result["manifest"]["records_computed"], 1)
            self.assertEqual(result["summary"]["speaker_count"], 2)

    @staticmethod
    def _row(result_id, speaker, value):
        row = {
            "result_id": result_id, "speaker": speaker,
            "ground_truth": "truth", "generated_message": "prediction",
        }
        if value is not None:
            row["local_metrics"] = {
                "rouge_l": value,
                "sentiment_accuracy": value,
                "emotion_accuracy": value,
                "intimacy_absolute_difference": value,
                "bertscore_f1": value,
            }
        return row


if __name__ == "__main__":
    unittest.main()
