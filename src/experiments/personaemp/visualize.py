from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .official_eval import summarize_official_results


METRICS = (
    ("resonation", "Resonation"),
    ("expression", "Expression"),
    ("reception", "Reception"),
    ("average_raw_1_to_5", "Average"),
)
COLORS = ("#167C80", "#E07A5F", "#5B6C8F", "#D4A72C", "#7A6F9B")


@dataclass(frozen=True)
class MetricSeries:
    label: str
    result_path: Path
    records: int
    values: dict[str, float]


def _parse_result(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError("--result must use LABEL=PATH")
    return label.strip(), Path(path).resolve()


def load_metric_series(label: str, path: Path) -> MetricSeries:
    summary = summarize_official_results(path)
    invalid = summary["invalid_scores"]
    if invalid:
        raise ValueError(
            f"{label} has {len(invalid)} invalid or missing official scores"
        )
    values: dict[str, float] = {}
    for key, _display_name in METRICS:
        value = summary.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"{label} is missing metric {key}")
        values[key] = float(value)
    return MetricSeries(
        label=label,
        result_path=path,
        records=int(summary["records"]),
        values=values,
    )


def _write_csv(path: Path, series: list[MetricSeries]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            ["Method", "N", "Resonation", "Expression", "Reception", "Average"]
        )
        for item in series:
            writer.writerow(
                [
                    item.label,
                    item.records,
                    *[round(item.values[key], 4) for key, _name in METRICS],
                ]
            )


def _write_markdown(
    path: Path,
    series: list[MetricSeries],
    *,
    judge_label: str,
    split_label: str,
    status_note: str,
) -> None:
    lines = [
        "# PersonaEmp 指标可视化",
        "",
        f"- Judge：{judge_label}",
        f"- 数据：{split_label}",
        f"- 说明：{status_note}",
        "",
        "| 方法 | N | Res | Exp | Rec | Avg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in series:
        values = [item.values[key] for key, _name in METRICS]
        lines.append(
            f"| {item.label} | {item.records} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "所有分数均为 PersonaEmp 官方 1--5 分量表，越高越好；"
            "`Average` 是 Resonation、Expression、Reception 的算术平均。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(
    path: Path,
    series: list[MetricSeries],
    *,
    judge_label: str,
    split_label: str,
    status_note: str,
) -> None:
    payload: dict[str, Any] = {
        "judge": judge_label,
        "split": split_label,
        "status_note": status_note,
        "scale": {"minimum": 1, "maximum": 5, "higher_is_better": True},
        "series": [
            {
                "method": item.label,
                "records": item.records,
                "source": str(item.result_path),
                **item.values,
            }
            for item in series
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _render_chart(
    path: Path,
    series: list[MetricSeries],
    *,
    judge_label: str,
    split_label: str,
    status_note: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    metric_names = [display_name for _key, display_name in METRICS]
    x_positions = list(range(len(METRICS)))
    width = min(0.34, 0.78 / max(len(series), 1))

    figure, axis = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    figure.patch.set_facecolor("#F8F7F3")
    axis.set_facecolor("#F8F7F3")

    for index, item in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        values = [item.values[key] for key, _name in METRICS]
        bars = axis.bar(
            [position + offset for position in x_positions],
            values,
            width=width * 0.88,
            label=f"{item.label} (N={item.records})",
            color=COLORS[index % len(COLORS)],
            edgecolor="#FFFFFF",
            linewidth=0.8,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in values],
            padding=4,
            fontsize=10,
            color="#262626",
        )

    axis.set_ylim(1, 5.25)
    axis.set_yticks([1, 2, 3, 4, 5])
    axis.set_ylabel("Score (1-5, higher is better)", fontsize=11)
    axis.set_xticks(x_positions, metric_names)
    axis.grid(axis="y", color="#D8D6D0", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#77736B")
    axis.tick_params(length=0, colors="#383632")

    figure.suptitle(
        "PersonaEmp Personalized Empathy Metrics",
        x=0.08,
        y=0.96,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#222222",
    )
    axis.set_title(
        f"{judge_label}  |  {split_label}",
        loc="left",
        pad=18,
        fontsize=12,
        color="#57534E",
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=min(len(series), 3),
        frameon=False,
        fontsize=10,
    )
    figure.text(
        0.08,
        0.025,
        status_note,
        ha="left",
        fontsize=9,
        color="#6B665E",
    )
    figure.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.20)
    figure.savefig(path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def build_visualization(
    series: list[MetricSeries],
    output_dir: Path,
    *,
    judge_label: str,
    split_label: str,
    status_note: str,
) -> dict[str, str]:
    if not series:
        raise ValueError("at least one result series is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "chart": output_dir / "personaemp_metrics.png",
        "csv": output_dir / "personaemp_metrics.csv",
        "json": output_dir / "personaemp_metrics.json",
        "markdown": output_dir / "personaemp_metrics.md",
    }
    _render_chart(
        paths["chart"],
        series,
        judge_label=judge_label,
        split_label=split_label,
        status_note=status_note,
    )
    _write_csv(paths["csv"], series)
    _write_json(
        paths["json"],
        series,
        judge_label=judge_label,
        split_label=split_label,
        status_note=status_note,
    )
    _write_markdown(
        paths["markdown"],
        series,
        judge_label=judge_label,
        split_label=split_label,
        status_note=status_note,
    )
    return {name: str(path) for name, path in paths.items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize official PersonaEmp Res/Exp/Rec/Avg results."
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Method label and official result JSON in LABEL=PATH form.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-label", required=True)
    parser.add_argument("--split-label", required=True)
    parser.add_argument(
        "--status-note",
        default="Official PersonaEmp metrics; scores range from 1 to 5.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    series = [
        load_metric_series(label, path)
        for label, path in (_parse_result(value) for value in args.result)
    ]
    paths = build_visualization(
        series,
        args.output_dir.resolve(),
        judge_label=args.judge_label,
        split_label=args.split_label,
        status_note=args.status_note,
    )
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
