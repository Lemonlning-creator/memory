from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .dataset import PersonaEmpDataset, PersonaEmpSample
from .generation import (
    BASE_QWEN3_SYSTEM_PROMPT,
    BASE_QWEN3_USER_PROMPT,
    DIRECT_RESPONSE_SYSTEM_PROMPT,
    DIRECT_RESPONSE_USER_PROMPT_TEMPLATE,
    BaseQwen3Generator,
    DeepEmpathyGenerator,
    prompt_hash,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_value(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sample_id(
    dataset_fingerprint: str,
    sample: PersonaEmpSample,
    method: str,
) -> str:
    value = (
        f"{dataset_fingerprint}\0{sample.session_id}\0"
        f"{sample.query_id}\0{method}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _generation_prompt_hashes() -> dict[str, str]:
    return {
        "ours_system": prompt_hash(DIRECT_RESPONSE_SYSTEM_PROMPT),
        "ours_user_template": prompt_hash(
            DIRECT_RESPONSE_USER_PROMPT_TEMPLATE
        ),
        "base_system": prompt_hash(BASE_QWEN3_SYSTEM_PROMPT),
        "base_user_template": prompt_hash(BASE_QWEN3_USER_PROMPT),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid checkpoint JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"checkpoint record must be an object at {path}:{line_number}"
                )
            records.append(value)
    return records


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(serialized + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def _selected_dataset(
    dataset: PersonaEmpDataset,
    samples: Iterable[PersonaEmpSample],
) -> list[dict[str, Any]]:
    selected = {(sample.session_index, sample.query_index) for sample in samples}
    output: list[dict[str, Any]] = []
    for session_index, raw_session in enumerate(dataset.raw_sessions):
        queries = [
            query
            for query_index, query in enumerate(raw_session.get("queries", []))
            if (session_index, query_index) in selected
        ]
        if not queries:
            continue
        session_copy = dict(raw_session)
        session_copy["queries"] = queries
        output.append(session_copy)
    return output


def _prediction_rows(
    selected_dataset: list[dict[str, Any]],
    records: list[dict[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    response_by_query = {
        str(record["query_id"]): str(record["response"])
        for record in records
        if record.get("status") == "success" and record.get("method") == method
    }
    output: list[dict[str, Any]] = []
    for session in selected_dataset:
        responses: list[dict[str, str]] = []
        for query in session.get("queries", []):
            query_id = str(query.get("query_id") or "")
            if query_id not in response_by_query:
                raise RuntimeError(
                    f"cannot export {method}: missing successful response for {query_id}"
                )
            responses.append(
                {
                    "query_id": query_id,
                    "response": response_by_query[query_id],
                }
            )
        output.append(
            {
                "session_id": str(
                    session.get("session_id") or session.get("original_sid") or ""
                ),
                "responses": responses,
            }
        )
    return output


@dataclass(frozen=True)
class RunConfiguration:
    methods: tuple[str, ...]
    limit: int | None
    dataset_provenance: str
    expected_table1_dataset_sha256: str | None
    agent_persona_sha256: str | None
    generator_model: str

    def identity(
        self,
        dataset_fingerprint: str,
        prompt_hashes: dict[str, str],
    ) -> str:
        value = {
            "dataset_fingerprint": dataset_fingerprint,
            "methods": self.methods,
            "limit": self.limit,
            "dataset_provenance": self.dataset_provenance,
            "expected_table1_dataset_sha256": self.expected_table1_dataset_sha256,
            "agent_persona_sha256": self.agent_persona_sha256,
            "generator_model": self.generator_model,
            "prompt_hashes": prompt_hashes,
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True).encode("utf-8")
        ).hexdigest()


class PersonaEmpRunner:
    OFFICIAL_REPOSITORY = "https://github.com/ZhengWwwq/PersonalizedEmpathy"
    OFFICIAL_COMMIT = "b555447f267b8057039aab39a4be44725718ea7f"

    def __init__(
        self,
        *,
        repository_root: Path,
        dataset: PersonaEmpDataset,
        output_dir: Path,
        config: RunConfiguration,
        generators: dict[str, BaseQwen3Generator | DeepEmpathyGenerator],
    ) -> None:
        self.repository_root = repository_root
        self.dataset = dataset
        self.output_dir = output_dir.resolve()
        self.config = config
        self.generators = generators
        self.results_path = self.output_dir / "results.jsonl"
        self.errors_path = self.output_dir / "errors.jsonl"
        self.manifest_path = self.output_dir / "run_manifest.json"

    def _manifest(self, selected_count: int) -> dict[str, Any]:
        expected_hash = self.config.expected_table1_dataset_sha256
        table1_compatible = bool(
            expected_hash and expected_hash == self.dataset.fingerprint
        )
        generation_prompt_hashes = _generation_prompt_hashes()
        return {
            "experiment": "exp1_personaemp_deep_empathy",
            "created_at": _utc_now(),
            "run_identity": self.config.identity(
                self.dataset.fingerprint,
                generation_prompt_hashes,
            ),
            "repository_commit": _git_value(
                self.repository_root,
                "rev-parse",
                "HEAD",
            ),
            "repository_branch": _git_value(
                self.repository_root,
                "branch",
                "--show-current",
            ),
            "dataset": {
                "path": str(self.dataset.path),
                "sha256": self.dataset.fingerprint,
                "provenance": self.config.dataset_provenance,
                "total_sessions": len(self.dataset.raw_sessions),
                "total_queries": len(self.dataset.samples),
                "selected_queries": selected_count,
                "table1_direct_comparison_allowed": table1_compatible,
                "expected_table1_sha256": expected_hash,
            },
            "official_reference": {
                "repository": self.OFFICIAL_REPOSITORY,
                "commit": self.OFFICIAL_COMMIT,
                "paper": "arXiv:2606.00728v1",
            },
            "generation": {
                "model": self.config.generator_model,
                "methods": list(self.config.methods),
                "agent_persona_sha256": self.config.agent_persona_sha256,
                "model_inputs": {
                    "shared_evidence": ["extracted_memory", "query"],
                    "ours_transformation": (
                        "five_layer_profile_and_deep_empathy_alignment"
                    ),
                    "dataset_persona_visible_to_generators": False,
                    "dataset_persona_visible_to_official_judges": True,
                },
                "core_prompt_policy": "unchanged_from_upstream_experiment",
                "prompt_hashes": generation_prompt_hashes,
            },
        }

    def _prepare_manifest(self, selected_count: int) -> None:
        manifest = self._manifest(selected_count)
        if self.manifest_path.is_file():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("run_identity") != manifest["run_identity"]:
                raise RuntimeError(
                    "output directory belongs to a different run configuration; "
                    "choose a new --output-dir"
                )
            return
        _atomic_json(self.manifest_path, manifest)

    def run(self) -> dict[str, Any]:
        samples = list(self.dataset.iter_samples(self.config.limit))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_manifest(len(samples))

        prior_records = _load_jsonl(self.results_path)
        completed_ids = {
            str(record.get("sample_id"))
            for record in prior_records
            if record.get("status") == "success"
        }

        for sample in samples:
            for method in self.config.methods:
                sample_id = _sample_id(
                    self.dataset.fingerprint,
                    sample,
                    method,
                )
                if sample_id in completed_ids:
                    continue

                generator = self.generators[method]
                try:
                    output = generator.generate(sample)
                    record = {
                        "sample_id": sample_id,
                        "status": "success",
                        "completed_at": _utc_now(),
                        "session_id": sample.session_id,
                        "query_id": sample.query_id,
                        "query_sha256": hashlib.sha256(
                            sample.query.encode("utf-8")
                        ).hexdigest(),
                        **output.to_record(),
                    }
                    _append_jsonl(self.results_path, record)
                    completed_ids.add(sample_id)
                except Exception as exc:
                    error_record = {
                        "sample_id": sample_id,
                        "status": "error",
                        "failed_at": _utc_now(),
                        "session_id": sample.session_id,
                        "query_id": sample.query_id,
                        "method": method,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    _append_jsonl(self.errors_path, error_record)
                    raise

        all_records = _load_jsonl(self.results_path)
        selected_dataset = _selected_dataset(self.dataset, samples)
        evaluation_dataset_path = self.output_dir / "evaluation_dataset.json"
        _atomic_json(evaluation_dataset_path, selected_dataset)

        prediction_paths: dict[str, str] = {}
        for method in self.config.methods:
            predictions = _prediction_rows(selected_dataset, all_records, method)
            prediction_path = self.output_dir / "predictions" / f"{method}.json"
            _atomic_json(prediction_path, predictions)
            prediction_paths[method] = str(prediction_path)

        summary = {
            "selected_queries": len(samples),
            "methods": list(self.config.methods),
            "successful_results": sum(
                1
                for record in all_records
                if record.get("status") == "success"
                and record.get("method") in self.config.methods
            ),
            "evaluation_dataset": str(evaluation_dataset_path),
            "predictions": prediction_paths,
        }
        _atomic_json(self.output_dir / "summary.json", summary)
        return summary
