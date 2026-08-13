"""Merge frozen first-five and second-five V11 replay cohorts."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .exp1_protocol import REALTALK_PERSONA_SPLITS, stable_hash


def run(first_dir: Path, second_dir: Path, output_dir: Path) -> dict[str, Any]:
    first = _validated_cohort(first_dir, expected_speakers=5)
    second = _validated_cohort(second_dir, expected_speakers=5)
    overlap = set(first["ids"]) & set(second["ids"])
    if overlap:
        raise ValueError(f"cohorts overlap on result IDs: {sorted(overlap)[:3]}")
    rows = first["rows"] + second["rows"]
    baseline = first["baseline"] + second["baseline"]
    expected_speakers = [item["speaker"] for item in REALTALK_PERSONA_SPLITS]
    actual_speakers = list(dict.fromkeys(row["speaker"] for row in rows))
    if [name.casefold() for name in actual_speakers] != [name.casefold() for name in expected_speakers]:
        raise ValueError(f"merged speaker order differs: {actual_speakers}")
    if len(rows) != 60 or len(baseline) != 60:
        raise ValueError(f"all10 merge requires 60 records, found {len(rows)}")
    if [row["result_id"] for row in rows] != [row["result_id"] for row in baseline]:
        raise ValueError("merged V9/V11 IDs are not aligned")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "predictions.jsonl", rows)
    _write_jsonl(output_dir / "v9_baseline_predictions.jsonl", baseline)
    ids = [row["result_id"] for row in rows]
    (output_dir / "sample_ids.json").write_text(
        json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "protocol": "realtalk_task1_ours_v11_soft_actor_cohort_merge_v1",
        "first_dir": str(first_dir.resolve()),
        "second_dir": str(second_dir.resolve()),
        "first_predictions_sha256": first["sha256"],
        "second_predictions_sha256": second["sha256"],
        "records": len(rows),
        "speakers": actual_speakers,
        "sample_ids_sha256": stable_hash(ids),
        "output_predictions_sha256": _sha256(output_dir / "predictions.jsonl"),
        "first_cohort_preserved_byte_for_byte_by_record": True,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _validated_cohort(path: Path, *, expected_speakers: int) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["unresolved_errors"] != 0:
        raise ValueError(f"cohort is not complete: {path}")
    rows = _read_jsonl(path / "predictions.jsonl")
    baseline = _read_jsonl(path / "v9_baseline_predictions.jsonl")
    if len(rows) != 30 or len(baseline) != 30:
        raise ValueError(f"cohort must contain 30 records: {path}")
    if len({row["speaker"] for row in rows}) != expected_speakers:
        raise ValueError(f"cohort must contain {expected_speakers} speakers: {path}")
    ids = [row["result_id"] for row in rows]
    if ids != [row["result_id"] for row in baseline]:
        raise ValueError(f"cohort V9/V11 IDs differ: {path}")
    return {
        "rows": rows, "baseline": baseline, "ids": ids,
        "sha256": _sha256(path / "predictions.jsonl"),
    }


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
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--second-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.first_dir, args.second_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
