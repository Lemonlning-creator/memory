"""Four-row REALTALK Table 2 diagnostic report for paired V9/V11 subsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .realtalk_ours import PAPER_TABLE2_ROWS


KEYS = (
    "lexical", "semantic", "reflective", "grounding",
    "sentiment", "emotion", "intimacy", "empathy",
)


def run(
    v9_local: Path, v11_local: Path, paired_judge: Path, output_dir: Path
) -> dict[str, Any]:
    local = {
        "V9": json.loads(v9_local.read_text(encoding="utf-8")),
        "V11": json.loads(v11_local.read_text(encoding="utf-8")),
    }
    judge = json.loads(paired_judge.read_text(encoding="utf-8"))
    rows = dict(PAPER_TABLE2_ROWS)
    for label, method in (("V9", "v9"), ("V11", "v11")):
        macro = local[label]["speaker_macro"]
        gpt = judge["methods"][method]["speaker_macro"]
        rows[label] = {
            "lexical": _display(macro, "rouge_l"),
            "semantic": _display(macro, "bertscore_f1"),
            "reflective": _display(gpt, "reflectiveness_accuracy"),
            "grounding": _display(gpt, "grounding_accuracy"),
            "sentiment": _display(macro, "sentiment_accuracy"),
            "emotion": _display(macro, "emotion_accuracy"),
            "intimacy": _display(macro, "intimacy_absolute_difference"),
            "empathy": _display(gpt, "empathy_absolute_difference"),
        }
    payload = {
        "scope": "matched_subset_diagnostic_not_full_table2_result",
        "aggregation": "speaker_macro_mean_and_population_std",
        "rows": rows,
        "judge_status": judge["status"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table2_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# REALTALK Table 2 + V9/V11 matched diagnostic",
        "",
        "| Method | Lexical | Semantic | Reflective | Grounding | Sentiment | Emotion | Intimacy | Empathy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in rows.items():
        lines.append("| " + method + " | " + " | ".join(row[key] for key in KEYS) + " |")
    lines.extend([
        "",
        "The V9/V11 rows use the same fixed subset and shared reference judgments.",
        "They are diagnostics; the paper rows use the complete protocol.",
    ])
    (output_dir / "table2_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _display(summary: dict[str, Any], key: str) -> str:
    value = summary[key]
    if isinstance(value, dict):
        return f"{value['mean']:.2f} +/- {value['std_population']:.2f}"
    raise ValueError(f"metric {key} does not contain mean/std_population")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9-local", type=Path, required=True)
    parser.add_argument("--v11-local", type=Path, required=True)
    parser.add_argument("--paired-judge", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.v9_local, args.v11_local, args.paired_judge, args.output_dir
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
