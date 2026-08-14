"""Incrementally extend frozen REALTALK local metrics to a larger prediction set."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .realtalk_local_metrics import run as run_local_metrics
from .realtalk_ours import _aggregate_local_metrics


def run(
    expanded_predictions: Path,
    previous_results: Path,
    output_dir: Path,
) -> dict[str, Any]:
    predictions = _read_jsonl(expanded_predictions)
    previous = _read_jsonl(previous_results)
    prediction_by_id = {row["result_id"]: row for row in predictions}
    previous_by_id = {row["result_id"]: row for row in previous}
    if len(prediction_by_id) != len(predictions) or len(previous_by_id) != len(previous):
        raise ValueError("duplicate result IDs in incremental metric inputs")
    unknown = set(previous_by_id) - set(prediction_by_id)
    if unknown:
        raise ValueError(f"previous metrics contain IDs absent from expansion: {sorted(unknown)[:3]}")
    for result_id, row in previous_by_id.items():
        prediction = prediction_by_id[result_id]
        for field in ("speaker", "ground_truth", "generated_message"):
            if row[field] != prediction[field]:
                raise ValueError(f"previous metric row {result_id} changed field {field}")

    missing = [row for row in predictions if row["result_id"] not in previous_by_id]
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_path = output_dir / "new_predictions.jsonl"
    _write_jsonl(missing_path, missing)
    if missing:
        run_local_metrics(missing_path, output_dir / "new_metrics")
        new_scored = _read_jsonl(output_dir / "new_metrics" / "results_with_local_metrics.jsonl")
    else:
        new_scored = []
    scored_by_id = {**previous_by_id, **{row["result_id"]: row for row in new_scored}}
    combined = [scored_by_id[row["result_id"]] for row in predictions]
    if len(combined) != len(predictions):
        raise ValueError("incremental metrics did not cover every expanded prediction")
    summary = _aggregate_local_metrics(combined)
    _write_jsonl(output_dir / "results_with_local_metrics.jsonl", combined)
    (output_dir / "local_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "protocol": "realtalk_incremental_local_metrics_v1",
        "expanded_predictions_sha256": _sha256(expanded_predictions),
        "previous_results_sha256": _sha256(previous_results),
        "records_reused": len(previous),
        "records_computed": len(new_scored),
        "records_total": len(combined),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"manifest": manifest, "summary": summary}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-predictions", type=Path, required=True)
    parser.add_argument("--previous-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.expanded_predictions, args.previous_results, args.output_dir
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
