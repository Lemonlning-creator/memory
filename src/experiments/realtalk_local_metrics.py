"""Offline REALTALK Table 2 metrics for frozen persona predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .exp1_protocol import stable_hash
from .exp2_generation import compute_bertscore_f1
from .operation_checkpoint import OperationCheckpoint
from .realtalk_evaluator import RealTalkLabelEvaluator
from .realtalk_ours import RealTalkOursConfig, _run_local_metrics


def run(
    predictions: Path,
    output_dir: Path,
    *,
    compute_bertscore: bool = True,
    evaluator: RealTalkLabelEvaluator | None = None,
    bertscore_fn=None,
) -> dict:
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("predictions file is empty")
    result_ids = [row["result_id"] for row in rows]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("predictions contain duplicate result IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = hashlib.sha256(predictions.read_bytes()).hexdigest()
    signature = stable_hash({
        "protocol": "realtalk_table2_offline_local_metrics_v1",
        "predictions_sha256": source_sha256,
        "compute_bertscore": compute_bertscore,
    })
    checkpoint = OperationCheckpoint(output_dir / "checkpoint.json", signature)
    config = RealTalkOursConfig(
        output_dir=str(output_dir),
        compute_local_metrics=True,
        compute_bertscore=compute_bertscore,
    )
    summary = _run_local_metrics(
        output_dir,
        rows,
        checkpoint,
        evaluator or RealTalkLabelEvaluator(),
        config,
        bertscore_fn or compute_bertscore_f1,
    )
    manifest = {
        "status": "complete",
        "protocol": "realtalk_table2_offline_local_metrics_v1",
        "predictions": str(predictions.resolve()),
        "predictions_sha256": source_sha256,
        "records": len(rows),
        "compute_bertscore": compute_bertscore,
        "aggregation_for_table2": "speaker_macro_mean_and_population_std",
        "run_signature": signature,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"manifest": manifest, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-bertscore", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        args.predictions,
        args.output_dir,
        compute_bertscore=not args.skip_bertscore,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
