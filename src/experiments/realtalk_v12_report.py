"""Three-row REALTALK Table 2 diagnostic report for paired V9/V12 subsets."""
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
PAPER_BEST = {
    "lexical": (0.14, "higher"),
    "semantic": (0.78, "higher"),
    "reflective": (0.77, "higher"),
    "grounding": (0.62, "higher"),
    "sentiment": (0.59, "higher"),
    "emotion": (0.46, "higher"),
    "intimacy": (0.06, "lower"),
    "empathy": (1.24, "lower"),
}


def run(v9_local: Path, v12_local: Path, paired_judge: Path, output_dir: Path) -> dict[str, Any]:
    local = {
        "V9": json.loads(v9_local.read_text(encoding="utf-8")),
        "V12": json.loads(v12_local.read_text(encoding="utf-8")),
    }
    judge = json.loads(paired_judge.read_text(encoding="utf-8"))
    if judge["status"] != "complete" or "v12" not in judge["methods"]:
        raise ValueError("V12 paired judge is incomplete or mislabeled")
    rows = dict(PAPER_TABLE2_ROWS)
    exact: dict[str, dict[str, float]] = {}
    for label, method in (("V9", "v9"), ("V12", "v12")):
        macro = local[label]["speaker_macro"]
        gpt = judge["methods"][method]["speaker_macro"]
        exact[label] = {
            "lexical": macro["rouge_l"]["mean"],
            "semantic": macro["bertscore_f1"]["mean"],
            "reflective": gpt["reflectiveness_accuracy"]["mean"],
            "grounding": gpt["grounding_accuracy"]["mean"],
            "sentiment": macro["sentiment_accuracy"]["mean"],
            "emotion": macro["emotion_accuracy"]["mean"],
            "intimacy": macro["intimacy_absolute_difference"]["mean"],
            "empathy": gpt["empathy_absolute_difference"]["mean"],
        }
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
    gates = {
        key: (
            exact["V12"][key] > threshold
            if direction == "higher" else exact["V12"][key] < threshold
        )
        for key, (threshold, direction) in PAPER_BEST.items()
    }
    payload = {
        "scope": "matched_subset_diagnostic_not_full_table2_result",
        "aggregation": "speaker_macro_mean_and_population_std",
        "rows": rows,
        "exact": exact,
        "paper_best_strict_gates": gates,
        "all_paper_best_strict_gates_passed": all(gates.values()),
        "judge_status": judge["status"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table2_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# REALTALK Table 2 + V9/V12 matched diagnostic",
        "",
        "| Method | Lexical | Semantic | Reflective | Grounding | Sentiment | Emotion | Intimacy | Empathy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in rows.items():
        lines.append("| " + method + " | " + " | ".join(row[key] for key in KEYS) + " |")
    lines.extend([
        "",
        f"Strict paper-best gates passed: {sum(gates.values())}/8.",
        "The V9/V12 rows use identical samples and shared reference judgments.",
        "They are diagnostics; only a frozen full-519 run is a final result.",
    ])
    (output_dir / "table2_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _display(summary: dict[str, Any], key: str) -> str:
    value = summary[key]
    return f"{value['mean']:.2f} +/- {value['std_population']:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9-local", type=Path, required=True)
    parser.add_argument("--v12-local", type=Path, required=True)
    parser.add_argument("--paired-judge", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.v9_local, args.v12_local, args.paired_judge, args.output_dir
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
