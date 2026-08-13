from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _isotonic(values: Iterable[float], *, increasing: bool) -> np.ndarray:
    """Least-squares isotonic regression via the pool-adjacent-violators algorithm."""
    source = np.asarray(list(values), dtype=float)
    working = source if increasing else -source
    levels: list[float] = []
    weights: list[int] = []
    for value in working:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_weight = weights[-2] + weights[-1]
            pooled = (
                levels[-2] * weights[-2] + levels[-1] * weights[-1]
            ) / total_weight
            levels[-2:] = [pooled]
            weights[-2:] = [total_weight]
    fitted = np.concatenate(
        [np.full(weight, level, dtype=float) for level, weight in zip(levels, weights)]
    )
    return fitted if increasing else -fitted


def _gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(4 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _anchor_endpoints(values: np.ndarray, start: float, end: float) -> np.ndarray:
    if np.isclose(values[0], values[-1]):
        return np.linspace(start, end, len(values))
    scaled = start + (values - values[0]) * (end - start) / (values[-1] - values[0])
    scaled[0] = start
    scaled[-1] = end
    return scaled


def _normalized_case_matrix(
    per_case: dict[str, list[dict[str, object]]],
    field: str,
    grid: np.ndarray,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for points in per_case.values():
        ordered = sorted(points, key=lambda point: int(point["session_index"]))
        sessions = np.asarray([float(point["session_index"]) for point in ordered])
        values = np.asarray([float(point[field]) for point in ordered])
        maximum = float(sessions[-1])
        progress = sessions / maximum if maximum else np.zeros_like(sessions)
        rows.append(np.interp(grid, progress, values))
    if not rows:
        raise ValueError("profile_curves.json contains no per-case trajectories")
    return np.vstack(rows)


def _plot_curve(
    path: Path,
    x: np.ndarray,
    raw: np.ndarray,
    trend: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    ylabel: str,
    color: str,
) -> None:
    plt.figure(figsize=(7.2, 4.4))
    plt.fill_between(
        x,
        lower,
        upper,
        color=color,
        alpha=0.10,
        linewidth=0,
        label="Inter-case IQR",
    )
    plt.plot(x, raw, color=color, linewidth=1.0, alpha=0.28, label="Raw normalized mean")
    plt.plot(x, trend, color=color, linewidth=2.8, label="Monotone Gaussian trend")
    plt.xlabel("Normalized training progress (%)")
    plt.ylabel(ylabel)
    plt.xlim(0, 100)
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.20)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="Gaussian smoothing width in percentage points on the 0-100 grid.",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    per_case = payload["per_case"]
    progress = np.linspace(0.0, 1.0, 101)
    x = progress * 100.0

    completeness_cases = _normalized_case_matrix(
        per_case, "profile_completeness", progress
    )
    entropy_cases = _normalized_case_matrix(per_case, "profile_entropy", progress)
    completeness_raw = completeness_cases.mean(axis=0)
    entropy_raw = entropy_cases.mean(axis=0)

    completeness_iso = _isotonic(completeness_raw, increasing=True)
    entropy_iso = _isotonic(entropy_raw, increasing=False)
    completeness_trend = _anchor_endpoints(
        _gaussian_smooth(completeness_iso, args.sigma),
        float(completeness_raw[0]),
        float(completeness_iso[-1]),
    )
    entropy_trend = _anchor_endpoints(
        _gaussian_smooth(entropy_iso, args.sigma),
        float(entropy_raw[0]),
        float(entropy_iso[-1]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evolution_path = args.output_dir / "profile_evolution_curve_smooth.png"
    entropy_path = args.output_dir / "profile_entropy_curve_smooth.png"
    combined_path = args.output_dir / "profile_curves_smooth_combined.png"
    _plot_curve(
        evolution_path,
        x,
        completeness_raw,
        completeness_trend,
        np.quantile(completeness_cases, 0.25, axis=0),
        np.quantile(completeness_cases, 0.75, axis=0),
        ylabel="Fixed-field profile coverage",
        color="#1f77b4",
    )
    _plot_curve(
        entropy_path,
        x,
        entropy_raw,
        entropy_trend,
        np.quantile(entropy_cases, 0.25, axis=0),
        np.quantile(entropy_cases, 0.75, axis=0),
        ylabel="Fixed-field profile entropy",
        color="#d95f02",
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.2), sharex=True, sharey=True)
    panels = (
        (
            axes[0],
            completeness_raw,
            completeness_trend,
            completeness_cases,
            "Profile evolution",
            "#1f77b4",
        ),
        (
            axes[1],
            entropy_raw,
            entropy_trend,
            entropy_cases,
            "Profile entropy",
            "#d95f02",
        ),
    )
    for axis, raw, trend, cases, title, color in panels:
        axis.fill_between(
            x,
            np.quantile(cases, 0.25, axis=0),
            np.quantile(cases, 0.75, axis=0),
            color=color,
            alpha=0.10,
            linewidth=0,
            label="Inter-case IQR",
        )
        axis.plot(x, raw, color=color, linewidth=1.0, alpha=0.28, label="Raw mean")
        axis.plot(x, trend, color=color, linewidth=2.8, label="Smoothed trend")
        axis.set_title(title)
        axis.set_xlabel("Normalized training progress (%)")
        axis.set_xlim(0, 100)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.20)
    axes[0].set_ylabel("Metric value")
    axes[1].legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(combined_path, dpi=240)
    plt.close(figure)

    rows = []
    for index in range(len(progress)):
        rows.append(
            {
                "progress_percent": float(x[index]),
                "case_count": int(completeness_cases.shape[0]),
                "raw_mean_profile_completeness": float(completeness_raw[index]),
                "smoothed_profile_completeness": float(completeness_trend[index]),
                "raw_mean_profile_entropy": float(entropy_raw[index]),
                "smoothed_profile_entropy": float(entropy_trend[index]),
            }
        )
    result = {
        "source": str(args.input.resolve()),
        "case_count": int(completeness_cases.shape[0]),
        "aggregation": "linear interpolation per case on normalized 0-100% training progress",
        "trend": {
            "method": "isotonic regression followed by Gaussian smoothing",
            "sigma_percentage_points": args.sigma,
            "raw_curve_preserved": True,
        },
        "points": rows,
        "figures": {
            "profile_evolution": str(evolution_path.resolve()),
            "profile_entropy": str(entropy_path.resolve()),
            "combined": str(combined_path.resolve()),
        },
    }
    (args.output_dir / "profile_curves_smooth.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
