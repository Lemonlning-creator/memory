from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .official_eval import DIMENSIONS
from .visualize import build_visualization, load_metric_series


METHOD_ORDER = ("base_model", "memory", "rag", "ours")
METHOD_LABELS = {
    "base_model": "Base",
    "memory": "Memory",
    "rag": "RAG",
    "ours": "Ours",
}
OFFICIAL_REFERENCE = {
    "Qwen3 Judge / Random": {
        "Base": (3.70, 3.74, 3.76, 3.73),
        "Memory": (3.66, 3.72, 3.71, 3.69),
        "RAG": (3.53, 3.64, 3.73, 3.63),
    },
    "Qwen3 Judge / OOD": {
        "Base": (3.61, 3.63, 3.67, 3.64),
        "Memory": (3.50, 3.65, 3.59, 3.58),
        "RAG": (3.45, 3.51, 3.60, 3.52),
    },
    "DeepSeek Judge / Random": {
        "Base": (2.62, 2.95, 3.02, 2.86),
        "Memory": (2.47, 2.87, 2.86, 2.73),
        "RAG": (2.52, 2.90, 2.99, 2.80),
    },
    "DeepSeek Judge / OOD": {
        "Base": (2.60, 2.91, 2.96, 2.82),
        "Memory": (2.42, 2.83, 2.89, 2.71),
        "RAG": (2.50, 2.93, 2.92, 2.78),
    },
}


def _parse_result(value: str) -> tuple[str, str, str, Path]:
    identity, separator, path = value.partition("=")
    parts = identity.split(":")
    if not separator or len(parts) != 3:
        raise ValueError("--result must use SPLIT:JUDGE:METHOD=PATH")
    split, judge, method = (part.strip() for part in parts)
    if method not in METHOD_ORDER:
        raise ValueError(f"unknown method: {method}")
    return split, judge, method, Path(path).resolve()


def _load_records(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"result root must be a list: {path}")
    records: dict[tuple[str, str], dict[str, float]] = {}
    for row in data:
        key = (str(row.get("session_id") or ""), str(row.get("query_id") or ""))
        if not all(key) or key in records:
            raise ValueError(f"invalid or duplicate result identity in {path}: {key}")
        scores: dict[str, float] = {}
        for dimension in DIMENSIONS:
            score = (row.get(dimension) or {}).get("score")
            if not isinstance(score, (int, float)) or not 1 <= float(score) <= 5:
                raise ValueError(f"invalid {dimension} score for {key} in {path}")
            scores[dimension] = float(score)
        scores["average"] = mean(scores.values())
        records[key] = scores
    return records


def _user_metric(
    records: dict[tuple[str, str], dict[str, float]],
    metric: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for (user_id, _query_id), scores in records.items():
        grouped[user_id].append(scores[metric])
    return {user: mean(values) for user, values in grouped.items()}


def paired_user_bootstrap(
    baseline: dict[tuple[str, str], dict[str, float]],
    ours: dict[tuple[str, str], dict[str, float]],
    metric: str,
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    if set(baseline) != set(ours):
        raise ValueError("paired comparison requires identical result identities")
    baseline_users = _user_metric(baseline, metric)
    ours_users = _user_metric(ours, metric)
    users = sorted(baseline_users)
    if not users:
        raise ValueError("paired comparison has no users")
    differences = [ours_users[user] - baseline_users[user] for user in users]
    rng = random.Random(seed)
    samples = sorted(
        mean(rng.choices(differences, k=len(differences)))
        for _ in range(iterations)
    )
    return {
        "delta": mean(differences),
        "ci95_low": samples[int(iterations * 0.025)],
        "ci95_high": samples[min(int(iterations * 0.975), iterations - 1)],
        "users": len(users),
    }


def _write_official_reference(output_dir: Path) -> None:
    csv_path = output_dir / "official_table1_training_free_reference.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            ["Setting", "Method", "Resonation", "Expression", "Reception", "Average"]
        )
        for setting, methods in OFFICIAL_REFERENCE.items():
            for method, values in methods.items():
                writer.writerow([setting, method, *values])


def _write_controlled_csv(
    output_dir: Path,
    summary_rows: list[list[Any]],
    user_rows: list[list[Any]],
) -> None:
    with (output_dir / "controlled_reproduction_summary.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as destination:
        writer = csv.writer(destination)
        writer.writerow(
            [
                "Split",
                "Judge",
                "Method",
                "N",
                "Resonation",
                "Expression",
                "Reception",
                "Average",
            ]
        )
        writer.writerows(summary_rows)
    with (output_dir / "controlled_reproduction_user_metrics.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as destination:
        writer = csv.writer(destination)
        writer.writerow(
            [
                "Split",
                "Judge",
                "Method",
                "User",
                "Queries",
                "Resonation",
                "Expression",
                "Reception",
                "Average",
            ]
        )
        writer.writerows(user_rows)


def build_report(
    result_specs: list[tuple[str, str, str, Path]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for split, judge, method, path in result_specs:
        if method in grouped[(split, judge)]:
            raise ValueError(f"duplicate result for {split}:{judge}:{method}")
        grouped[(split, judge)][method] = path

    report: dict[str, Any] = {
        "protocol": "personaemp_public_controlled_reproduction_v1",
        "official_reference_is_not_directly_comparable": True,
        "settings": {},
    }
    markdown = [
        "# PersonaEmp 公开数据受控复现实验",
        "",
        "> 官方 Table 1 仅作外部参考，以下实测结果不得与其绝对分数拼接。",
        "",
    ]
    summary_rows: list[list[Any]] = []
    user_rows: list[list[Any]] = []
    for (split, judge), methods in sorted(grouped.items()):
        missing = [method for method in METHOD_ORDER if method not in methods]
        if missing:
            raise ValueError(
                f"{split}:{judge} is missing methods: {', '.join(missing)}"
            )
        records = {method: _load_records(path) for method, path in methods.items()}
        identities = {method: set(rows) for method, rows in records.items()}
        if len({frozenset(value) for value in identities.values()}) != 1:
            raise ValueError(f"{split}:{judge} methods are not sample-aligned")
        setting_key = f"{split}:{judge}"
        setting: dict[str, Any] = {"methods": {}, "ours_deltas": {}}
        metric_series = []
        markdown.extend([f"## {split} / {judge}", ""])
        markdown.append("| 方法 | N | Res | Exp | Rec | Avg |")
        markdown.append("|---|---:|---:|---:|---:|---:|")
        for method in METHOD_ORDER:
            series = load_metric_series(METHOD_LABELS[method], methods[method])
            metric_series.append(series)
            values = series.values
            setting["methods"][method] = {
                "records": series.records,
                **values,
            }
            summary_rows.append(
                [
                    split,
                    judge,
                    METHOD_LABELS[method],
                    series.records,
                    values["resonation"],
                    values["expression"],
                    values["reception"],
                    values["average_raw_1_to_5"],
                ]
            )
            users = sorted({user for user, _query in records[method]})
            for user in users:
                user_records = {
                    key: scores
                    for key, scores in records[method].items()
                    if key[0] == user
                }
                user_rows.append(
                    [
                        split,
                        judge,
                        METHOD_LABELS[method],
                        user,
                        len(user_records),
                        *[
                            _user_metric(user_records, metric)[user]
                            for metric in (*DIMENSIONS, "average")
                        ],
                    ]
                )
            markdown.append(
                f"| {METHOD_LABELS[method]} | {series.records} | "
                f"{values['resonation']:.3f} | {values['expression']:.3f} | "
                f"{values['reception']:.3f} | "
                f"{values['average_raw_1_to_5']:.3f} |"
            )
        for baseline in METHOD_ORDER[:-1]:
            setting["ours_deltas"][baseline] = {
                metric: paired_user_bootstrap(
                    records[baseline],
                    records["ours"],
                    metric,
                )
                for metric in (*DIMENSIONS, "average")
            }
        chart_dir = output_dir / "charts" / split / judge
        setting["visualization"] = build_visualization(
            metric_series,
            chart_dir,
            judge_label=judge,
            split_label=split,
            status_note=(
                "Public-data controlled reproduction; not a direct Table 1 extension."
            ),
        )
        report["settings"][setting_key] = setting
        markdown.append("")

    _write_controlled_csv(output_dir, summary_rows, user_rows)
    _write_official_reference(output_dir)
    report["artifacts"] = {
        "summary_csv": str(
            output_dir / "controlled_reproduction_summary.csv"
        ),
        "user_metrics_csv": str(
            output_dir / "controlled_reproduction_user_metrics.csv"
        ),
        "official_reference_csv": str(
            output_dir / "official_table1_training_free_reference.csv"
        ),
    }
    (output_dir / "controlled_reproduction_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "controlled_reproduction_report_zh.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the four-method PersonaEmp reproduction report."
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="SPLIT:JUDGE:METHOD=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_report(
        [_parse_result(value) for value in args.result],
        args.output_dir.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
