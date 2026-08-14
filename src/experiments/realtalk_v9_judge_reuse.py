"""Build an auditable V9 Judge checkpoint by reusing exact prior judgments."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


METRICS = ("reflectiveness", "grounding", "empathy")
EXPECTED_PROTOCOL = "realtalk_appendix_c_full_prompt_within_session_v3"
EXPECTED_MODEL = "gpt-4o-mini"


def run(
    full_predictions: Path,
    output_dir: Path,
    sources: list[tuple[str, Path, Path, str]],
) -> dict[str, Any]:
    full_rows = _read_rows(full_predictions)
    full = {row["result_id"]: row for row in full_rows}
    if len(full) != len(full_rows):
        raise ValueError("full V9 predictions contain duplicate result IDs")
    if output_dir.exists():
        raise FileExistsError(f"reuse output already exists: {output_dir}")

    judgments: dict[str, Any] = {}
    imported_ids: set[str] = set()
    provenance = []
    conflicts = []
    for name, predictions, judge_dir, candidate_side in sources:
        if candidate_side not in {"candidate", "v9"}:
            raise ValueError(f"invalid candidate side for {name}: {candidate_side}")
        summary = json.loads((judge_dir / "summary.json").read_text(encoding="utf-8"))
        protocol = summary.get("judge_protocol") or summary.get("judge_prompt_protocol")
        if (
            summary.get("status") != "complete"
            or protocol != EXPECTED_PROTOCOL
            or summary.get("model_requested") != EXPECTED_MODEL
        ):
            raise ValueError(f"Judge identity mismatch for reuse source {name}")
        checkpoint_path = judge_dir / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        rows = _read_rows(predictions)
        matched = imported = duplicate = 0
        for row in rows:
            result_id = row["result_id"]
            target = full.get(result_id)
            if target is None or any(
                row.get(field) != target.get(field)
                for field in ("generated_message", "ground_truth")
            ):
                continue
            matched += 1
            keys = [
                f"{result_id}:{side}:{metric}"
                for side in ("reference", candidate_side)
                for metric in METRICS
            ]
            if not all(key in checkpoint["judgments"] for key in keys):
                continue
            if result_id in imported_ids:
                duplicate += 1
                _record_conflicts(
                    conflicts, judgments, checkpoint["judgments"],
                    result_id, candidate_side, name,
                )
                continue
            for metric in METRICS:
                judgments[f"{result_id}:reference:{metric}"] = checkpoint["judgments"][
                    f"{result_id}:reference:{metric}"
                ]
                judgments[f"{result_id}:candidate:{metric}"] = checkpoint["judgments"][
                    f"{result_id}:{candidate_side}:{metric}"
                ]
            imported_ids.add(result_id)
            imported += 1
        provenance.append({
            "name": name,
            "candidate_side": candidate_side,
            "prediction_sha256": _sha256(predictions),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "matched_exact_v9_rows": matched,
            "new_ids_imported": imported,
            "duplicate_ids_skipped": duplicate,
        })

    output_dir.mkdir(parents=True)
    (output_dir / "checkpoint.json").write_text(
        json.dumps({"judgments": judgments, "errors": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "protocol": "realtalk_v9_full519_judge_resume_v1",
        "judge_protocol": EXPECTED_PROTOCOL,
        "model": EXPECTED_MODEL,
        "full_predictions_sha256": _sha256(full_predictions),
        "full_records": len(full),
        "source_priority": provenance,
        "imported_unique_ids": len(imported_ids),
        "imported_judgments": len(judgments),
        "remaining_ids": len(full) - len(imported_ids),
        "remaining_judgments": (len(full) - len(imported_ids)) * 6,
        "duplicate_label_conflicts_ignored": len(conflicts),
        "conflict_ids": len({item["result_id"] for item in conflicts}),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (output_dir / "reuse_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "reuse_conflicts.json").write_text(
        json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _record_conflicts(
    conflicts: list[dict[str, str]],
    kept: dict[str, Any],
    incoming: dict[str, Any],
    result_id: str,
    candidate_side: str,
    source_name: str,
) -> None:
    for logical_side, actual_side in (("reference", "reference"), ("candidate", candidate_side)):
        for metric in METRICS:
            kept_value = kept[f"{result_id}:{logical_side}:{metric}"]["value"]
            incoming_value = incoming[f"{result_id}:{actual_side}:{metric}"]["value"]
            if kept_value != incoming_value:
                conflicts.append({
                    "result_id": result_id,
                    "side": logical_side,
                    "metric": metric,
                    "kept_source": "earlier_priority",
                    "ignored_source": source_name,
                })


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(value: str) -> tuple[str, Path, Path, str]:
    parts = value.split("|", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("source must be NAME|PREDICTIONS|JUDGE_DIR|SIDE")
    return parts[0], Path(parts[1]), Path(parts[2]), parts[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=_source, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(
        run(args.full_predictions, args.output_dir, args.source),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
