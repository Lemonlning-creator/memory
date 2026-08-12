import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.realtalk_local_metrics import run


class FakeLabels:
    def annotate(self, text):
        return {
            "emotion": "joy" if "same" in text else "sadness",
            "sentiment": "positive" if "same" in text else "negative",
            "intimacy": 0.7 if "same" in text else 0.2,
        }

    def metadata(self):
        return {"provider": "fake"}


class RealTalkLocalMetricsTest(unittest.TestCase):
    def test_offline_metrics_preserve_predictions_and_write_macro_summary(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            predictions = root / "predictions.jsonl"
            rows = [
                {
                    "result_id": "a:1", "speaker": "A",
                    "ground_truth": "same reference",
                    "generated_message": "same candidate",
                },
                {
                    "result_id": "b:1", "speaker": "B",
                    "ground_truth": "other reference",
                    "generated_message": "same candidate",
                },
            ]
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            before = predictions.read_bytes()
            result = run(
                predictions,
                root / "metrics",
                evaluator=FakeLabels(),
                bertscore_fn=lambda references, candidates: [0.8, 0.6],
            )
            self.assertEqual(predictions.read_bytes(), before)
            self.assertEqual(result["manifest"]["records"], 2)
            summary = result["summary"]
            self.assertEqual(summary["speaker_count"], 2)
            self.assertEqual(summary["message_count"], 2)
            self.assertEqual(summary["speaker_macro"]["bertscore_f1"]["mean"], 0.7)


if __name__ == "__main__":
    unittest.main()
