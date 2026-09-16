from __future__ import annotations

import argparse
import hashlib
import http.client
import subprocess
import time
import urllib.request
from pathlib import Path

from .common import ROOT, read, write, sha
from src.experiments.exp1_protocol import select_realtalk_splits
from src.experiments.realtalk_ours import RealTalkOursConfig, _prepare_dataset, _speaker_id


def verify_dataset(dataset: Path) -> tuple[dict, list[dict]]:
    frozen = read(ROOT / "provenance/dataset_manifest.json")
    for name, expected in frozen["source_files_sha256"].items():
        if not (dataset / name).is_file() or sha(dataset / name) != expected:
            raise ValueError(f"Dataset SHA256 mismatch or missing file: {name}")
    manifest, prepared = _prepare_dataset(
        RealTalkOursConfig(dataset_dir=str(dataset)), select_realtalk_splits(dataset))
    for key in ("table8_speaker_specific_splits", "targets_by_speaker", "total_targets", "source_files_sha256"):
        if manifest[key] != frozen[key]:
            raise ValueError(f"Dataset protocol changed: {key}")
    return manifest, prepared


def sample_manifest(prepared: list[dict]) -> list[dict]:
    return [{
        "result_id": f"{_speaker_id(person['speaker'])}:{point['sample_id']}",
        "speaker": person["speaker"],
        "target_session": point["target_session"],
        "context_hash": point["history_hash"],
        "ground_truth_sha256": hashlib.sha256(point["target_message"].encode()).hexdigest(),
    } for person in prepared for point in person["points"]]


def validate_predictions(predictions: list[dict], dataset: Path, *, full: bool = False) -> None:
    _, prepared = verify_dataset(dataset)
    expected = {row["result_id"]: row for row in sample_manifest(prepared)}
    seen = set()
    for row in predictions:
        rid = row["result_id"]
        if rid in seen or rid not in expected:
            raise ValueError(f"Duplicate or unknown result ID: {rid}")
        seen.add(rid)
        reference = expected[rid]
        if row["speaker"] != reference["speaker"] or row["target_session"] != reference["target_session"]:
            raise ValueError(f"Speaker/session mismatch: {rid}")
        if row["context_hash"] != reference["context_hash"] or row["context_truncated"]:
            raise ValueError(f"History mismatch: {rid}")
        if hashlib.sha256(row["ground_truth"].encode()).hexdigest() != reference["ground_truth_sha256"]:
            raise ValueError(f"Ground truth mismatch: {rid}")
        if not row["generated_message"].strip():
            raise ValueError(f"Empty prediction: {rid}")
    if full and seen != set(expected):
        raise ValueError(f"Expected all 519 IDs; found {len(seen)}")


def prepare(dataset: Path, source: Path | None = None, download: bool = False, source_repo: Path | None = None) -> dict:
    frozen = read(ROOT / "provenance/dataset_manifest.json")
    dataset.mkdir(parents=True, exist_ok=True)
    for name, expected in frozen["source_files_sha256"].items():
        target = dataset / name
        if target.exists():
            if sha(target) != expected:
                raise ValueError(f"Refusing to overwrite different data: {target}")
            continue
        if source_repo is not None:
            raw = subprocess.check_output(["git", "-C", str(source_repo), "show",
                                           f"{frozen['official_repository_commit']}:data/{name}"])
        elif source is not None:
            raw = (source / name).read_bytes()
        elif download:
            url = f"https://raw.githubusercontent.com/danny911kr/REALTALK/{frozen['official_repository_commit']}/data/{name}"
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        raw = response.read()
                    break
                except (OSError, EOFError, http.client.IncompleteRead):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        else:
            raise ValueError("Supply --source or --download for missing files")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Source bytes differ from frozen V9: {name}")
        target.write_bytes(raw)
    manifest, prepared = verify_dataset(dataset)
    write(dataset / "prepared_manifest.json", manifest)
    write(dataset / "sample_manifest.json", sample_manifest(prepared))
    return {"files": 10, "targets": manifest["total_targets"], "by_speaker": manifest["targets_by_speaker"]}


def main():
    parser = argparse.ArgumentParser(description="Prepare exact frozen V9 input data; no model calls")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--source", type=Path)
    group.add_argument("--source-repo", type=Path)
    group.add_argument("--download", action="store_true")
    args = parser.parse_args()
    print(prepare(args.dataset_dir, args.source, args.download, args.source_repo))


if __name__ == "__main__":
    main()
