from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.realtalk_v12_report import run


def local_summary(values):
    return {"speaker_macro": {
        key: {"mean": value, "std_population": 0.01}
        for key, value in values.items()
    }}


class RealTalkV12ReportTests(unittest.TestCase):
    def test_strict_raw_gates_do_not_accept_display_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v9 = root / "v9.json"; v12 = root / "v12.json"; judge = root / "judge.json"
            metrics = {
                "rouge_l": 0.18, "bertscore_f1": 0.86,
                "sentiment_accuracy": 0.68, "emotion_accuracy": 0.57,
                "intimacy_absolute_difference": 0.0601,
            }
            v9.write_text(json.dumps(local_summary(metrics)))
            v12.write_text(json.dumps(local_summary(metrics)))
            judge.write_text(json.dumps({
                "status": "complete",
                "methods": {
                    "v9": {"speaker_macro": {
                        "reflectiveness_accuracy": {"mean": 0.8, "std_population": 0.1},
                        "grounding_accuracy": {"mean": 0.7, "std_population": 0.1},
                        "empathy_absolute_difference": {"mean": 1.0, "std_population": 0.1},
                    }},
                    "v12": {"speaker_macro": {
                        "reflectiveness_accuracy": {"mean": 0.8, "std_population": 0.1},
                        "grounding_accuracy": {"mean": 0.7, "std_population": 0.1},
                        "empathy_absolute_difference": {"mean": 1.24, "std_population": 0.1},
                    }},
                },
            }))
            result = run(v9, v12, judge, root / "report")
            self.assertFalse(result["paper_best_strict_gates"]["intimacy"])
            self.assertFalse(result["paper_best_strict_gates"]["empathy"])
            self.assertFalse(result["all_paper_best_strict_gates_passed"])


if __name__ == "__main__":
    unittest.main()
