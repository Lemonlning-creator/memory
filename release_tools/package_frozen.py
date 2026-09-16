"""Collect existing results only. No inference, training, or Judge calls."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .common import CANONICAL, MODEL, PREDICTIONS_SHA, ROOT, read, rows, sha, write
from .data import verify_dataset, validate_predictions
from .report import combine, table

SOURCE_NAMES = {
    "generation": "realtalk-ours-v9-full519-evidencefix-flash-5927bbf",
    "local": "realtalk-ours-v9-full519-local-metrics-v1",
    "judge": "realtalk-ours-v9-full519-judge-resume-v1",
}
INCLUDE = {
    "generation": ["predictions.jsonl", "self_domains.json", "run_manifest.json", "dataset_manifest.json", "unresolved_errors.json", "GENERATION_COMPLETE"],
    "local": ["results_with_local_metrics.jsonl", "local_metrics_summary.json", "manifest.json"],
    "judge": ["scored.jsonl", "summary.json", "checkpoint.json", "reuse_manifest.json", "reuse_conflicts.json"],
}


def verify_package(output: Path) -> dict:
    manifest = read(output / "PACKAGE_MANIFEST.json")
    actual = {str(f.relative_to(output)).replace("\\", "/") for f in output.rglob("*")
              if f.is_file() and f.name != "PACKAGE_MANIFEST.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Package inventory differs")
    for name, expected in manifest["files"].items():
        if sha(output / name) != expected["sha256"]:
            raise ValueError(f"Package hash mismatch: {name}")
    if sha(output / "generation/predictions.jsonl") != PREDICTIONS_SHA:
        raise ValueError("Not the canonical V9 predictions")
    return {"verified_files": len(actual), "canonical_predictions_sha256": PREDICTIONS_SHA}


def package(runs: Path, dataset: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Package destination must not exist; old packages are immutable")
    sources = {key: runs / name for key, name in SOURCE_NAMES.items()}
    prediction_file = sources["generation"] / "predictions.jsonl"
    if sha(prediction_file) != PREDICTIONS_SHA:
        raise ValueError("Wrong source predictions")
    manifest = read(sources["generation"] / "run_manifest.json")
    if manifest["implementation_repository_commit"] != CANONICAL or manifest["ours_model"] != MODEL:
        raise ValueError("Wrong source commit/model")
    if any(manifest["stage_thinking"].values()) or manifest["training_or_finetuning"]:
        raise ValueError("Wrong thinking/training configuration")
    if read(sources["generation"] / "unresolved_errors.json"):
        raise ValueError("Unresolved generation errors")
    data_manifest, _ = verify_dataset(dataset)
    validate_predictions(rows(prediction_file), dataset, full=True)
    summary = combine(prediction_file, sources["local"], sources["judge"])
    if len(read(sources["generation"] / "self_domains.json")) != 10:
        raise ValueError("Expected 10 Self Domains")
    judge_checkpoint = read(sources["judge"] / "checkpoint.json")
    expected_judgments = {f"{r['result_id']}:{side}:{metric}" for r in rows(prediction_file)
                          for side in ("reference", "candidate") for metric in ("reflectiveness", "grounding", "empathy")}
    if set(judge_checkpoint["judgments"]) != expected_judgments or judge_checkpoint["errors"]:
        raise ValueError("Judge checkpoint must contain exactly 3114 complete judgments")
    # The inference manifest predates offline evaluation. Preserve it unchanged;
    # a new package-level completion record describes the verified joined result.
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise ValueError("Existing partial package; inspect it before choosing a new destination")
    staging.mkdir(parents=True)
    origin_hashes = {}
    for group, names in INCLUDE.items():
        (staging / group).mkdir()
        for name in names:
            source = sources[group] / name
            target = staging / group / name
            shutil.copyfile(source, target)
            origin_hashes[str(target.relative_to(staging))] = {"original_path": str(source), "sha256": sha(source)}
    (staging / "dataset").mkdir()
    for name in data_manifest["source_files_sha256"]:
        shutil.copyfile(dataset / name, staging / "dataset" / name)
    shutil.copytree(ROOT / "contracts", staging / "contracts")
    shutil.copytree(ROOT / "provenance", staging / "provenance")
    write(staging / "summary.json", summary)
    write(staging / "ORIGINS.json", origin_hashes)
    write(staging / "PIPELINE_COMPLETE", {"verified_records": 519, "verified_judgments": 3114, "verified_self_domains": 10,
                                         "unresolved_errors": 0, "metrics": 8, "meaning": "archival integrity, not superiority over the paper"})
    (staging / "TABLE2.md").write_text(table(summary), encoding="utf-8", newline="\n")
    readme = (ROOT / "docs/RESULT_PACKAGE_README.md").read_text(encoding="utf-8")
    (staging / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    files = {str(f.relative_to(staging)).replace("\\", "/"): {"sha256": sha(f), "bytes": f.stat().st_size}
             for f in sorted(staging.rglob("*")) if f.is_file()}
    write(staging / "PACKAGE_MANIFEST.json", {"canonical_commit": CANONICAL, "model": MODEL,
                                             "canonical_predictions_sha256": PREDICTIONS_SHA, "files": files})
    verify_package(staging)
    staging.rename(output)
    return verify_package(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(verify_package(args.output))
    elif args.runs is None:
        parser.error("--runs is required for packaging")
    else:
        print(package(args.runs, args.dataset, args.output))
