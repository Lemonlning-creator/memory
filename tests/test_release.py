import argparse
import hashlib
import json
from pathlib import Path

import pytest

from release_tools.common import ROOT, bind_output, read, sha, verify_source, write
from release_tools.cli import generation_config
from release_tools.data import verify_dataset, sample_manifest, validate_predictions
from release_tools.export_contracts import export
from release_tools.report import combine
from src.experiments import realtalk_ours as ours


def test_original_source_is_byte_identical():
    assert verify_source()["byte_identical_files"] == 28


def test_prompt_and_schema_hashes_match_original_run():
    frozen = read(ROOT / "provenance/frozen_run_manifest.json")
    assert frozen["ours_model"] == "deepseek-v4-flash"
    assert not any(frozen["stage_thinking"].values())
    assert ours._prompt_hashes() == frozen["prompt_hashes"]
    for key, schema in (("self_domain", ours.SELF_DOMAIN_SCHEMA), ("user_domain", ours.USER_DOMAIN_SCHEMA), ("decision", ours.ALIGNMENT_SCHEMA)):
        assert ours.stable_hash(schema) == frozen["schema_hashes"][key]


def test_exports_are_exact_and_hash_stable(tmp_path):
    export(tmp_path)
    for name, value in read(tmp_path / "index.json")["files_sha256"].items():
        assert sha(tmp_path / name) == value
    assert (tmp_path / "generation_system_template.txt").read_bytes() == ours.GENERATION_SYSTEM_TEMPLATE.encode()


def test_full_519_protocol_and_mutation_rejection():
    manifest, prepared = verify_dataset(Path("dataset"))
    assert manifest["total_targets"] == 519
    assert manifest["targets_by_speaker"] == ours.EXPECTED_SPEAKER_TARGETS
    samples = sample_manifest(prepared)
    assert len({s["result_id"] for s in samples}) == 519
    person, point = prepared[0], prepared[0]["points"][0]
    row = {**samples[0], "context_truncated": False, "ground_truth": point["target_message"], "generated_message": "test"}
    validate_predictions([row], Path("dataset"))
    row["ground_truth"] += " altered"
    with pytest.raises(ValueError, match="Ground truth"):
        validate_predictions([row], Path("dataset"))


def test_fixed_model_configuration_and_overwrite_guard(tmp_path):
    args = argparse.Namespace(scope="full", speaker=None, output=tmp_path / "run", resume=False, dataset_dir=Path("dataset"))
    cfg = generation_config(args)
    assert cfg.model == "deepseek-v4-flash"
    assert cfg.decision_thinking is False and cfg.eval_points_per_session == 0
    assert cfg.profile_sessions == cfg.test_sessions == 3
    args.speaker = ["Emi"]
    with pytest.raises(ValueError, match="all ten"):
        generation_config(args)
    args.speaker = None
    write(args.output / "existing.json", {})
    with pytest.raises(ValueError, match="empty"):
        generation_config(args)


def test_evaluation_binding_prevents_wrong_cache_reuse(tmp_path):
    output = tmp_path / "judge"
    bind_output(output, {"sha": "one"})
    bind_output(output, {"sha": "one"})
    with pytest.raises(ValueError, match="different input"):
        bind_output(output, {"sha": "two"})
    unbound = tmp_path / "unbound"
    write(unbound / "checkpoint.json", {})
    with pytest.raises(ValueError, match="populated"):
        bind_output(unbound, {"sha": "one"})


def test_report_rejects_partial_or_duplicate_metrics(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    row = {"result_id": "r", "speaker": "Emi"}
    predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
    local = tmp_path / "local"
    judge = tmp_path / "judge"
    local.mkdir()
    judge.mkdir()
    (local / "results_with_local_metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (judge / "scored.jsonl").write_text(json.dumps({"result_id": "different"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="IDs must match"):
        combine(predictions, local, judge)


def test_package_rejects_modified_or_extra_file(tmp_path):
    from release_tools.package_frozen import verify_package
    write(tmp_path / "x.json", {"test": True})
    write(tmp_path / "PACKAGE_MANIFEST.json", {"files": {"x.json": {"sha256": sha(tmp_path / "x.json")}}})
    write(tmp_path / "extra.json", {})
    with pytest.raises(ValueError, match="inventory"):
        verify_package(tmp_path)
    (tmp_path / "extra.json").unlink()
    write(tmp_path / "x.json", {"test": False})
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_package(tmp_path)
