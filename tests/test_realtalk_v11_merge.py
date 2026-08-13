from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.exp1_protocol import REALTALK_PERSONA_SPLITS
from src.experiments.realtalk_v11_merge import run


class RealTalkV11MergeTests(unittest.TestCase):
    def test_merges_two_frozen_cohorts_without_changing_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"; second = root / "second"
            self._write_cohort(first, REALTALK_PERSONA_SPLITS[:5])
            self._write_cohort(second, REALTALK_PERSONA_SPLITS[5:])
            first_before = (first / "predictions.jsonl").read_bytes()
            manifest = run(first, second, root / "all")
            output = (root / "all" / "predictions.jsonl").read_bytes()
            self.assertEqual(manifest["records"], 60)
            self.assertTrue(output.startswith(first_before))

    @staticmethod
    def _write_cohort(path, splits):
        path.mkdir()
        rows = []
        for split in splits:
            for index in range(6):
                rows.append({
                    "result_id": f"{split['speaker']}:{index}",
                    "speaker": split["speaker"],
                    "generated_message": "new",
                })
        text = "\n".join(json.dumps(row) for row in rows) + "\n"
        (path / "predictions.jsonl").write_text(text)
        (path / "v9_baseline_predictions.jsonl").write_text(text)
        (path / "manifest.json").write_text(json.dumps({
            "status": "complete", "unresolved_errors": 0,
        }))


if __name__ == "__main__":
    unittest.main()
