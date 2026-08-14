from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _isotonic(values: Iterable[float], *, increasing: bool) -> np.ndarray:
    source = np.asarray(list(values), dtype=float)
    working = source if increasing else -source
    levels: list[float] = []
    weights: list[int] = []
    for value in working:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            level = (
                levels[-2] * weights[-2] + levels[-1] * weights[-1]
            ) / weight
            levels[-2:] = [level]
            weights[-2:] = [weight]
    fitted = np.concatenate(
        [np.full(weight, level) for level, weight in zip(levels, weights)]
    )
    return fitted if increasing else -fitted


def _gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(4 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _anchor(values: np.ndarray, start: float, end: float) -> np.ndarray:
    if np.isclose(values[0], values[-1]):
        return np.linspace(start, end, len(values))
    result = start + (values - values[0]) * (end - start) / (
        values[-1] - values[0]
    )
    result[0] = start
    result[-1] = end
    return result


def _update_events(points: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(points, key=lambda point: int(point["session_index"]))
    if not ordered:
        return []
    # Update 0 is the empty initial portrait. Every later point is selected by
    # the actual long-term-memory trigger that invokes profile evolution.
    return [ordered[0], *[point for point in ordered[1:] if point.get("long_term_memory_id")]]


def _aggregate(
    events_by_case: dict[str, list[dict[str, object]]], field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    maximum = max(len(events) for events in events_by_case.values())
    indexes = np.arange(maximum, dtype=float)
    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    counts: list[int] = []
    for update_index in range(maximum):
        values = np.asarray(
            [
                float(events[update_index][field])
                for events in events_by_case.values()
                if update_index < len(events)
            ],
            dtype=float,
        )
        means.append(float(values.mean()))
        lowers.append(float(np.quantile(values, 0.25)))
        uppers.append(float(np.quantile(values, 0.75)))
        counts.append(int(len(values)))
    return (
        indexes,
        np.asarray(means),
        np.asarray(lowers),
        np.asarray(uppers),
        np.asarray(counts),
    )


def _smooth_update_curve(
    indexes: np.ndarray,
    means: np.ndarray,
    *,
    increasing: bool,
    sigma_updates: float,
) -> tuple[np.ndarray, np.ndarray]:
    dense_x = np.linspace(indexes[0], indexes[-1], 501)
    monotone = _isotonic(means, increasing=increasing)
    dense = np.interp(dense_x, indexes, monotone)
    points_per_update = (len(dense_x) - 1) / max(1.0, indexes[-1] - indexes[0])
    smoothed = _gaussian_smooth(dense, sigma_updates * points_per_update)
    smoothed = _anchor(smoothed, float(monotone[0]), float(monotone[-1]))
    return dense_x, smoothed


def _draw(
    axis: plt.Axes,
    indexes: np.ndarray,
    means: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    dense_x: np.ndarray,
    trend: np.ndarray,
    *,
    title: str,
    color: str,
) -> None:
    axis.fill_between(
        indexes,
        lower,
        upper,
        color=color,
        alpha=0.10,
        linewidth=0,
        label="Inter-case IQR",
    )
    axis.plot(
        indexes,
        means,
        color=color,
        alpha=0.42,
        linewidth=1.2,
        marker="o",
        markersize=4.5,
        label="Raw mean at update",
    )
    axis.plot(dense_x, trend, color=color, linewidth=2.8, label="Smoothed trend")
    axis.set_title(title)
    axis.set_xlabel("Profile update index")
    axis.set_xticks(indexes.astype(int))
    axis.set_xlim(indexes[0], indexes[-1])
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sigma-updates", type=float, default=0.35)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    events_by_case = {
        case_id: _update_events(points)
        for case_id, points in payload["per_case"].items()
    }
    completeness = _aggregate(events_by_case, "profile_completeness")
    entropy = _aggregate(events_by_case, "profile_entropy")
    comp_x, comp_mean, comp_low, comp_high, comp_counts = completeness
    ent_x, ent_mean, ent_low, ent_high, ent_counts = entropy
    comp_dense_x, comp_trend = _smooth_update_curve(
        comp_x,
        comp_mean,
        increasing=True,
        sigma_updates=args.sigma_updates,
    )
    ent_dense_x, ent_trend = _smooth_update_curve(
        ent_x,
        ent_mean,
        increasing=False,
        sigma_updates=args.sigma_updates,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / "profile_curves_by_update.png"
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.25), sharey=True)
    _draw(
        axes[0],
        comp_x,
        comp_mean,
        comp_low,
        comp_high,
        comp_dense_x,
        comp_trend,
        title="Profile evolution by actual update",
        color="#1f77b4",
    )
    _draw(
        axes[1],
        ent_x,
        ent_mean,
        ent_low,
        ent_high,
        ent_dense_x,
        ent_trend,
        title="Profile entropy by actual update",
        color="#d95f02",
    )
    axes[0].set_ylabel("Metric value")
    axes[1].legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(combined_path, dpi=240)
    plt.close(figure)

    output = {
        "source": str(args.input.resolve()),
        "x_axis": "actual profile update index; update 0 is the empty initial profile",
        "event_definition": "trajectory point with a non-empty long_term_memory_id",
        "case_update_counts": {
            case_id: len(events) - 1 for case_id, events in events_by_case.items()
        },
        "aggregation_note": (
            "Each index averages cases that reached that update index; case_count is "
            "reported explicitly because trajectories contain 5-7 updates."
        ),
        "points": [
            {
                "profile_update_index": int(index),
                "case_count": int(comp_counts[index]),
                "mean_profile_completeness": float(comp_mean[index]),
                "mean_profile_entropy": float(ent_mean[index]),
            }
            for index in range(len(comp_x))
        ],
        "figure": str(combined_path.resolve()),
    }
    data_path = args.output_dir / "profile_curves_by_update.json"
    data_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
