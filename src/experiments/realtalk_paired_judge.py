"""Paired Appendix-C judging for aligned V9 and V11 REALTALK predictions."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .realtalk_gpt_judge import (
    EMPATHY_PROMPT,
    GROUNDING_PROMPT,
    REFLECTIVENESS_PROMPT,
    _chat,
    _contexts,
    _parse_bool,
    _parse_empathy,
)


METRICS = ("reflectiveness", "grounding", "empathy")


def run(
    v9_predictions: Path,
    v11_predictions: Path,
    dataset_dir: Path,
    output_dir: Path,
    model: str,
    candidate_name: str = "v11",
) -> dict[str, Any]:
    if candidate_name not in {"v11", "v12"}:
        raise ValueError("candidate_name must be v11 or v12")
    methods = ("v9", candidate_name)
    api_key = os.environ["REALTALK_JUDGE_API_KEY"]
    base_url = os.environ["REALTALK_JUDGE_BASE_URL"]
    rows = {
        "v9": _read_rows(v9_predictions),
        candidate_name: _read_rows(v11_predictions),
    }
    ids = [row["result_id"] for row in rows["v9"]]
    if ids != [row["result_id"] for row in rows[candidate_name]]:
        raise ValueError(
            f"V9 and {candidate_name.upper()} predictions are not ordered on identical result IDs"
        )
    if any(
        left["ground_truth"] != right["ground_truth"]
        for left, right in zip(rows["v9"], rows[candidate_name])
    ):
        raise ValueError(f"V9 and {candidate_name.upper()} ground truths differ")

    contexts = _contexts(dataset_dir, rows["v9"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists() else {"judgments": {}, "errors": {}}
    )
    raw_path = output_dir / "raw_responses.jsonl"
    metric_specs = (
        ("reflectiveness", REFLECTIVENESS_PROMPT, _parse_bool),
        ("grounding", GROUNDING_PROMPT, _parse_bool),
        ("empathy", EMPATHY_PROMPT, _parse_empathy),
    )

    for index, result_id in enumerate(ids):
        history = contexts[result_id]
        turns = {
            "reference": rows["v9"][index]["ground_truth"],
            "v9": rows["v9"][index]["generated_message"],
            candidate_name: rows[candidate_name][index]["generated_message"],
        }
        for side, turn in turns.items():
            for metric, template, parser in metric_specs:
                key = f"{result_id}:{side}:{metric}"
                if key in checkpoint["judgments"]:
                    continue
                try:
                    content, audit = _chat(
                        base_url, api_key, model,
                        template.format(history=history or "(none)", turn=turn),
                    )
                    checkpoint["judgments"][key] = {
                        "value": parser(content), "audit": audit,
                    }
                    checkpoint["errors"].pop(key, None)
                    with raw_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "key": key, "raw": content, "audit": audit,
                            "recorded_at_utc": datetime.now(UTC).isoformat(),
                        }, ensure_ascii=False) + "\n")
                except Exception as exc:
                    checkpoint["errors"][key] = {
                        "type": type(exc).__name__, "error": str(exc),
                        "recorded_at_utc": datetime.now(UTC).isoformat(),
                    }
                checkpoint_path.write_text(
                    json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
                )

    scored = {method: [] for method in methods}
    for index, result_id in enumerate(ids):
        reference = _side_values(checkpoint, result_id, "reference")
        if reference is None:
            continue
        for method in methods:
            candidate = _side_values(checkpoint, result_id, method)
            if candidate is None:
                continue
            scored[method].append({
                "result_id": result_id,
                "speaker": rows[method][index]["speaker"],
                "reference": reference,
                "candidate": candidate,
                "metrics": {
                    "reflectiveness_accuracy": float(
                        reference["reflectiveness"] == candidate["reflectiveness"]
                    ),
                    "grounding_accuracy": float(
                        reference["grounding"] == candidate["grounding"]
                    ),
                    "empathy_absolute_difference": abs(
                        sum(reference["empathy"].values()) - sum(candidate["empathy"].values())
                    ),
                },
            })

    summaries = {method: _aggregate(items) for method, items in scored.items()}
    complete = all(len(scored[method]) == len(ids) for method in methods)
    summary = {
        "status": "complete" if complete and not checkpoint["errors"] else "incomplete",
        "protocol": f"realtalk_appendix_c_paired_v9_{candidate_name}_v1",
        "judge_prompt_protocol": "realtalk_appendix_c_full_prompt_within_session_v3",
        "model_requested": model,
        "messages": len(ids),
        "shared_reference_judgments": len(ids) * len(METRICS),
        "candidate_judgments_per_method": len(ids) * len(METRICS),
        "judgments_expected": len(ids) * len(METRICS) * 3,
        "judgments_complete": len(checkpoint["judgments"]),
        "unresolved_errors": len(checkpoint["errors"]),
        "methods": summaries,
        f"delta_{candidate_name}_minus_v9": _delta(summaries, candidate_name),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    for method, items in scored.items():
        with (output_dir / f"{method}_scored.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _side_values(checkpoint: dict[str, Any], result_id: str, side: str):
    values = {}
    for metric in METRICS:
        item = checkpoint["judgments"].get(f"{result_id}:{side}:{metric}")
        if item is None:
            return None
        values[metric] = item["value"]
    return values


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "reflectiveness_accuracy", "grounding_accuracy", "empathy_absolute_difference"
    )
    speakers = list(dict.fromkeys(item["speaker"] for item in items))
    by_speaker = {
        speaker: {
            metric: round(statistics.mean(
                item["metrics"][metric] for item in items if item["speaker"] == speaker
            ), 6)
            for metric in metric_names
        }
        for speaker in speakers
    }
    macro = {
        metric: {
            "mean": round(statistics.mean(row[metric] for row in by_speaker.values()), 6),
            "std_population": round(statistics.pstdev(row[metric] for row in by_speaker.values()), 6),
        }
        for metric in metric_names
    } if by_speaker else {}
    return {
        "messages_scored": len(items),
        "speaker_count": len(by_speaker),
        "by_speaker": by_speaker,
        "speaker_macro": macro,
    }


def _delta(summaries: dict[str, Any], candidate_name: str = "v11") -> dict[str, float]:
    result = {}
    for metric in (
        "reflectiveness_accuracy", "grounding_accuracy", "empathy_absolute_difference"
    ):
        if not summaries["v9"]["speaker_macro"] or not summaries[candidate_name]["speaker_macro"]:
            continue
        result[metric] = round(
            summaries[candidate_name]["speaker_macro"][metric]["mean"]
            - summaries["v9"]["speaker_macro"][metric]["mean"], 6
        )
    return result


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9-predictions", type=Path, required=True)
    parser.add_argument("--v11-predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--candidate-name", choices=("v11", "v12"), default="v11")
    args = parser.parse_args()
    print(json.dumps(run(
        args.v9_predictions, args.v11_predictions, args.dataset_dir,
        args.output_dir, args.model, args.candidate_name,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
