from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, Sequence

from ..persona_schema import PERSONA_SCHEMA_VERSION, persona_schema_manifest, validate_persona
from ..prompts.exp2_versions import EXP2_PROMPT_SWEEP_SPECS
from ..utils import load_json, save_json
from .exp2_user_modeling import (
    PROFILE_ALGORITHM,
    TABLE2_BASELINES,
    TABLE2_METRICS,
    CasePaths,
    ExperimentCase,
    _agent_runtime_profile,
    _validate_extracted_profile,
    aggregate_table2_scores,
    build_cases,
)


SWEEP_VERSIONS = tuple(EXP2_PROMPT_SWEEP_SPECS)
HIGHER_IS_BETTER = {
    "lexical",
    "semantic",
    "reflective",
    "grounding",
    "sentiment",
    "emotion",
}


def _paper_baseline_results() -> list[Dict[str, Any]]:
    """Render the two published REALTALK Table 2 rows in sweep reports."""
    rows: list[Dict[str, Any]] = []
    for baseline in TABLE2_BASELINES:
        rows.append({
            "directory": None,
            "prompt_version": f"Paper: {baseline['method']}",
            "description": "Published REALTALK Table 2 result over all speakers.",
            "example_count": None,
            "speaker_count": None,
            "result_kind": "paper_baseline",
            "metrics": {
                metric: {
                    "mean": float(baseline[metric][0]),
                    "std": float(baseline[metric][1]),
                }
                for metric in TABLE2_METRICS
            },
        })
    return rows


def _select_cases(
    cases: Sequence[ExperimentCase],
    selectors: Sequence[str],
) -> list[ExperimentCase]:
    if not selectors:
        return list(cases)
    wanted = {selector.lower() for selector in selectors}
    selected = [
        case
        for case in cases
        if case.case_id.lower() in wanted
        or Path(case.dataset_path).name.lower() in wanted
    ]
    found = {
        value
        for case in selected
        for value in (case.case_id.lower(), Path(case.dataset_path).name.lower())
    }
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"unknown case selectors: {missing}")
    return selected


def _asset_protocol(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Return path-independent fields that must be identical when assets are reused."""
    case = manifest.get("case") if isinstance(manifest.get("case"), dict) else {}
    persona_schema = (
        manifest.get("persona_schema")
        if isinstance(manifest.get("persona_schema"), dict)
        else {}
    )
    return {
        "dataset_sha256": case.get("dataset_sha256"),
        "user_speaker": case.get("user_speaker"),
        "agent_speaker": case.get("agent_speaker"),
        "profile_algorithm": manifest.get("profile_algorithm"),
        "train_sessions": manifest.get("train_sessions"),
        "persona_schema_version": manifest.get("persona_schema_version"),
        "persona_prompt_sha256": persona_schema.get("extraction_prompt_sha256"),
    }


def _validate_source_assets(case: ExperimentCase, paths: CasePaths) -> None:
    required = (paths.persona, paths.profile, paths.asset_manifest)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete reusable assets for {case.case_id}: {', '.join(missing)}"
        )

    persona = load_json(str(paths.persona))
    profile = load_json(str(paths.profile))
    manifest = load_json(str(paths.asset_manifest))
    validate_persona(persona)
    _validate_extracted_profile(profile)

    current_persona = persona_schema_manifest()
    expected = {
        "dataset_sha256": case.dataset_sha256,
        "user_speaker": case.user_speaker,
        "agent_speaker": case.agent_speaker,
        "profile_algorithm": PROFILE_ALGORITHM,
        "train_sessions": list(case.train_sessions),
        "persona_schema_version": PERSONA_SCHEMA_VERSION,
        "persona_prompt_sha256": current_persona["extraction_prompt_sha256"],
    }
    actual = _asset_protocol(manifest)
    if actual != expected:
        raise RuntimeError(
            f"asset protocol mismatch for {case.case_id}: "
            f"expected={expected}, actual={actual}"
        )


def _validate_existing_target(source: CasePaths, target: CasePaths) -> None:
    required = (
        target.persona,
        target.profile,
        target.runtime_profile,
        target.asset_manifest,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "target contains partial reusable assets; do not overwrite it: "
            + ", ".join(missing)
        )
    if load_json(str(source.persona)) != load_json(str(target.persona)):
        raise RuntimeError(f"target persona differs from source: {target.persona}")
    if load_json(str(source.profile)) != load_json(str(target.profile)):
        raise RuntimeError(f"target user profile differs from source: {target.profile}")
    source_manifest = load_json(str(source.asset_manifest))
    target_manifest = load_json(str(target.asset_manifest))
    if _asset_protocol(source_manifest) != _asset_protocol(target_manifest):
        raise RuntimeError(f"target asset manifest differs from source: {target.asset_manifest}")


def _copy_reference_annotations(source: CasePaths, target: CasePaths) -> int:
    """Copy only content-bound reference labels; never copy generated labels or scores."""
    if not source.table2_annotations.is_file():
        return 0
    target.table2_annotations.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if target.table2_annotations.is_file():
        with target.table2_annotations.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if "annotation_id" in row:
                        existing.add(str(row["annotation_id"]))

    copied = 0
    with source.table2_annotations.open("r", encoding="utf-8") as source_handle, (
        target.table2_annotations.open("a", encoding="utf-8")
    ) as target_handle:
        for line in source_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            annotation_id = str(row.get("annotation_id") or "")
            if (
                row.get("variant") != "reference"
                or not annotation_id
                or not isinstance(row.get("labels"), dict)
                or not row.get("evaluator_fingerprint")
                or not row.get("candidate_sha256")
                or not row.get("context_sha256")
            ):
                continue
            if annotation_id in existing:
                continue
            target_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing.add(annotation_id)
            copied += 1
    return copied


def clone_assets(
    *,
    source_dir: str | Path,
    target_dir: str | Path,
    dataset_dir: str | Path,
    train_ratio: float,
    case_selectors: Sequence[str],
    reuse_reference_cache: bool,
) -> Dict[str, Any]:
    source_root = Path(source_dir).resolve()
    target_root = Path(target_dir).resolve()
    if source_root == target_root:
        raise ValueError("asset source and sweep target must be different directories")
    cases = _select_cases(build_cases(dataset_dir, train_ratio), case_selectors)
    target_root.mkdir(parents=True, exist_ok=True)

    copied_assets = 0
    reused_assets = 0
    copied_reference_annotations = 0
    for case in cases:
        source = CasePaths.for_case(source_root, case)
        target = CasePaths.for_case(target_root, case)
        _validate_source_assets(case, source)

        target_asset_files = (
            target.persona,
            target.profile,
            target.runtime_profile,
            target.asset_manifest,
        )
        if any(path.exists() for path in target_asset_files):
            _validate_existing_target(source, target)
            reused_assets += 1
        else:
            target.persona.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.persona, target.persona)
            shutil.copy2(source.profile, target.profile)
            profile = load_json(str(source.profile))
            save_json(str(target.runtime_profile), _agent_runtime_profile(profile))

            manifest = deepcopy(load_json(str(source.asset_manifest)))
            manifest["case"] = asdict(case)
            manifest["profile_path"] = str(target.profile)
            manifest["runtime_profile_path"] = str(target.runtime_profile)
            manifest["persona_path"] = str(target.persona)
            save_json(str(target.asset_manifest), manifest)
            copied_assets += 1

        if reuse_reference_cache:
            copied_reference_annotations += _copy_reference_annotations(source, target)

    result = {
        "source_dir": str(source_root),
        "target_dir": str(target_root),
        "cases": [case.case_id for case in cases],
        "copied_assets": copied_assets,
        "reused_assets": reused_assets,
        "copied_reference_annotations": copied_reference_annotations,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _load_result(root: Path) -> Dict[str, Any] | None:
    path = root / "evaluation" / "table2_main_results.json"
    if not path.is_file():
        return None
    data = load_json(str(path))
    prompt = data.get("protocol", {}).get("generation_prompts", {})
    return {
        "directory": str(root.resolve()),
        "prompt_version": prompt.get("version"),
        "prompt_sha256": prompt.get("sha256"),
        "description": prompt.get("description"),
        "example_count": data.get("example_count"),
        "speaker_count": data.get("speaker_count"),
        "metrics": data.get("ours", {}),
    }


def _load_baseline_result(
    root: Path,
    cases: Sequence[ExperimentCase],
) -> Dict[str, Any] | None:
    """Re-aggregate the baseline over exactly the sweep cases."""
    main_path = root / "evaluation" / "table2_main_results.json"
    if not main_path.is_file():
        return None
    main = load_json(str(main_path))
    all_scores: list[Dict[str, Any]] = []
    for case in cases:
        score_path = CasePaths.for_case(root, case).table2_scores
        if not score_path.is_file():
            return None
        score_payload = load_json(str(score_path))
        scores = score_payload.get("scores")
        if not isinstance(scores, list) or not scores:
            return None
        all_scores.extend(scores)
    aggregate = aggregate_table2_scores(all_scores)
    prompt = main.get("protocol", {}).get("generation_prompts", {})
    return {
        "directory": str(root.resolve()),
        "prompt_version": prompt.get("version"),
        "prompt_sha256": prompt.get("sha256"),
        "description": (
            str(prompt.get("description") or "")
            + " (re-aggregated over the sweep cases)"
        ).strip(),
        "example_count": aggregate["example_count"],
        "speaker_count": aggregate["speaker_count"],
        "metrics": aggregate["ours"],
    }


def _winner_versions(results: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    completed = [row for row in results if isinstance(row.get("metrics"), dict)]
    winners: Dict[str, str] = {}
    for metric in TABLE2_METRICS:
        eligible = [
            row for row in completed
            if isinstance(row["metrics"].get(metric), dict)
            and isinstance(row["metrics"][metric].get("mean"), (int, float))
        ]
        if not eligible:
            continue
        selector = max if metric in HIGHER_IS_BETTER else min
        winner = selector(eligible, key=lambda row: row["metrics"][metric]["mean"])
        winners[metric] = str(winner["prompt_version"])
    return winners


def _markdown_summary(payload: Dict[str, Any]) -> str:
    results = payload["results"]
    prompt_results = payload.get(
        "prompt_results",
        [row for row in results if row.get("result_kind") != "paper_baseline"],
    )
    winners = payload["winners"]
    lines = [
        "# Experiment 2 prompt sweep",
        "",
        "## Controlled prompt variants",
        "",
        "| Version | Axis | Strength | Primary metrics | Hypothesis |",
        "|---|---|---|---|---|",
    ]
    for version, spec in payload.get("design", {}).items():
        lines.append(
            f"| {version} | {spec['axis']} | {spec['strength']} | "
            f"{spec['primary_metrics']} | {spec['hypothesis']} |"
        )

    lines.extend((
        "",
        "## Table 2 comparison",
        "",
        "Higher is better for Lexical through Emotion; lower is better for "
        "Intimacy and Empathy.",
        "",
        "The two `Paper:` rows are the published REALTALK Table 2 results over "
        "all speakers. For a case-subset sweep, treat them as published reference "
        "values rather than a same-sample statistical comparison.",
        "",
        "| Method / Version | N | Lexical | Semantic | Reflective | Grounding | Sentiment | Emotion | Intimacy | Empathy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for row in results:
        if not row.get("metrics"):
            lines.append(f"| {row['prompt_version']} | pending | - | - | - | - | - | - | - | - |")
            continue
        values = []
        for metric in TABLE2_METRICS:
            stat = row["metrics"][metric]
            formatted = f"{stat['mean']:.4f} ± {stat['std']:.4f}"
            if winners.get(metric) == row["prompt_version"]:
                formatted = f"**{formatted}**"
            values.append(formatted)
        example_count = (
            "published"
            if row.get("result_kind") == "paper_baseline"
            else str(row.get("example_count") or "-")
        )
        lines.append(
            "| " + " | ".join(
                [str(row["prompt_version"]), example_count] + values
            ) + " |"
        )

    improvements = [
        row for row in prompt_results
        if row.get("oriented_improvement_vs_baseline") is not None
    ]
    if improvements:
        baseline_version = prompt_results[0]["prompt_version"]
        lines.extend((
            "",
            f"## Directional change versus {baseline_version}",
            "",
            "Positive values always mean improvement. For Intimacy and Empathy the "
            "raw metric direction is reversed before computing the change.",
            "",
            "| Version | ΔLexical | ΔSemantic | ΔReflective | ΔGrounding | ΔSentiment | ΔEmotion | ΔIntimacy | ΔEmpathy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ))
        for row in improvements:
            changes = row["oriented_improvement_vs_baseline"]
            lines.append(
                "| " + " | ".join(
                    [str(row["prompt_version"])]
                    + [f"{changes[metric]:+.4f}" for metric in TABLE2_METRICS]
                ) + " |"
            )

    lines.extend(("", "## Best mean by metric", ""))
    for metric in TABLE2_METRICS:
        direction = "higher" if metric in HIGHER_IS_BETTER else "lower"
        lines.append(f"- {metric} ({direction}): {winners.get(metric, 'pending')}")
    lines.append("")
    return "\n".join(lines)


def summarize(
    *,
    sweep_root: str | Path,
    baseline_dir: str | Path | None,
    best_dir: str | Path | None,
    versions: Sequence[str],
    dataset_dir: str | Path,
    train_ratio: float,
    case_selectors: Sequence[str],
) -> Dict[str, Any]:
    root = Path(sweep_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    paper_baselines = _paper_baseline_results()
    results: list[Dict[str, Any]] = []
    cases = _select_cases(build_cases(dataset_dir, train_ratio), case_selectors)

    if baseline_dir:
        baseline = _load_baseline_result(Path(baseline_dir), cases)
        if baseline is not None:
            baseline["comparison_role"] = "v7_baseline"
            results.append(baseline)

    if best_dir:
        best_path = Path(best_dir).resolve()
        baseline_path = Path(baseline_dir).resolve() if baseline_dir else None
        if best_path != baseline_path:
            best = _load_result(best_path)
            if best is not None:
                best["comparison_role"] = "best_full_after_v7"
                results.append(best)

    for version in versions:
        result = _load_result(root / version)
        if result is None:
            result = {
                "directory": str((root / version).resolve()),
                "prompt_version": version,
                "description": EXP2_PROMPT_SWEEP_SPECS.get(version, {}).get(
                    "hypothesis"
                ),
                "example_count": None,
                "speaker_count": None,
                "metrics": None,
            }
        result["comparison_role"] = "current"
        if any(
            row.get("directory") == result.get("directory")
            for row in results
        ):
            continue
        results.append(result)

    baseline_metrics = (
        results[0].get("metrics")
        if baseline_dir and results and results[0].get("directory") == str(Path(baseline_dir).resolve())
        else None
    )
    for row in results:
        metrics = row.get("metrics")
        if not metrics or not baseline_metrics:
            row["oriented_improvement_vs_baseline"] = None
            continue
        improvement: Dict[str, float] = {}
        for metric in TABLE2_METRICS:
            current = float(metrics[metric]["mean"])
            baseline = float(baseline_metrics[metric]["mean"])
            improvement[metric] = (
                current - baseline
                if metric in HIGHER_IS_BETTER
                else baseline - current
            )
        row["oriented_improvement_vs_baseline"] = improvement

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sweep_root": str(root),
        "baseline_dir": str(Path(baseline_dir).resolve()) if baseline_dir else None,
        "best_dir": str(Path(best_dir).resolve()) if best_dir else None,
        "cases": [case.case_id for case in cases],
        "paper_baselines": paper_baselines,
        "design": {
            version: EXP2_PROMPT_SWEEP_SPECS[version]
            for version in versions
            if version in EXP2_PROMPT_SWEEP_SPECS
        },
        "results": [*paper_baselines, *results],
        "prompt_results": results,
        "winners": _winner_versions([*paper_baselines, *results]),
    }
    save_json(str(root / "prompt_sweep_summary.json"), payload)
    (root / "prompt_sweep_summary.md").write_text(
        _markdown_summary(payload), encoding="utf-8"
    )
    print(f"summary: {root / 'prompt_sweep_summary.md'}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reuse fixed Experiment 2 assets and summarize a response-prompt sweep."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clone = subparsers.add_parser("clone-assets")
    clone.add_argument("--source-dir", required=True)
    clone.add_argument("--target-dir", required=True)
    clone.add_argument("--dataset-dir", default="dataset")
    clone.add_argument("--train-ratio", type=float, default=0.9)
    clone.add_argument("--case", action="append", default=[])
    clone.add_argument("--reuse-reference-cache", action="store_true")

    report = subparsers.add_parser("summarize")
    report.add_argument("--sweep-root", required=True)
    report.add_argument("--baseline-dir")
    report.add_argument(
        "--best-dir",
        help="Completed full-run result selected as the best version after V7 and before the current version.",
    )
    report.add_argument("--version", action="append", default=[])
    report.add_argument("--dataset-dir", default="dataset")
    report.add_argument("--train-ratio", type=float, default=0.9)
    report.add_argument("--case", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "clone-assets":
        clone_assets(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            dataset_dir=args.dataset_dir,
            train_ratio=args.train_ratio,
            case_selectors=args.case,
            reuse_reference_cache=args.reuse_reference_cache,
        )
        return
    summarize(
        sweep_root=args.sweep_root,
        baseline_dir=args.baseline_dir,
        best_dir=args.best_dir,
        versions=args.version or SWEEP_VERSIONS,
        dataset_dir=args.dataset_dir,
        train_ratio=args.train_ratio,
        case_selectors=args.case,
    )


if __name__ == "__main__":
    main()
