from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.experiments.personaemp.client import OpenAICompatibleChatBackend
from src.experiments.personaemp.official_eval import (
    DIMENSIONS,
    summarize_official_results,
    validate_prediction_alignment,
    verify_official_checkout,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_prediction(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError("--prediction must use LABEL=PATH")
    return label.strip(), Path(path).resolve()


def _load_official_modules(repository: Path) -> tuple[Any, Any, Any]:
    evaluation_dir = str((repository / "evaluation").resolve())
    sys.path.insert(0, evaluation_dir)
    try:
        prepare_criteria = importlib.import_module("prepare_criteria")
        evaluation = importlib.import_module("eval")
        utils = importlib.import_module("utils")
    finally:
        sys.path.pop(0)
    return prepare_criteria, evaluation, utils


def _chat_with_parse_retry(
    backend: OpenAICompatibleChatBackend,
    prompt: str,
    *,
    parser: Any | None = None,
    attempts: int = 3,
) -> tuple[str, Any | None]:
    last_response = ""
    for _attempt in range(attempts):
        result = backend.chat(
            "",
            prompt,
            temperature=0.2,
            max_tokens=2048,
        )
        last_response = result.content
        parsed = parser(last_response) if parser is not None else last_response
        if parsed not in (None, ""):
            return last_response, parsed
    raise RuntimeError(
        "pilot judge returned an unparsable response after "
        f"{attempts} logical attempts: {last_response[:200]!r}"
    )


def _prepare_criteria(
    *,
    backend: OpenAICompatibleChatBackend,
    dataset: list[dict[str, Any]],
    output_path: Path,
    prepare_module: Any,
) -> list[dict[str, Any]]:
    existing = _load_json(output_path) if output_path.is_file() else None
    criteria_data = prepare_module.init_output_data(dataset, existing)
    jobs = prepare_module.build_generation_jobs(
        data=dataset,
        output_data=criteria_data,
        dimensions=list(DIMENSIONS),
        overwrite=False,
    )
    for job in jobs:
        response, _parsed = _chat_with_parse_retry(
            backend,
            job["prompt"][0]["content"],
        )
        criteria_data[job["session_index"]]["criterias"][
            job["query_index"]
        ][job["dimension"]] = response
        _atomic_json(output_path, criteria_data)
    return criteria_data


def _judge_prediction(
    *,
    backend: OpenAICompatibleChatBackend,
    dataset: list[dict[str, Any]],
    prediction: list[dict[str, Any]],
    criteria_data: list[dict[str, Any]],
    evaluation_module: Any,
    utils_module: Any,
) -> list[dict[str, Any]]:
    processed = evaluation_module.preprocess_data(dataset, prediction)
    processed = evaluation_module.attach_criteria_to_processed_data(
        processed,
        dataset,
        criteria_data,
    )
    results: list[dict[str, Any]] = []
    for item in processed:
        record: dict[str, Any] = {
            "session_id": item["session_id"],
            "query_id": item["query_id"],
        }
        for dimension in DIMENSIONS:
            prompt = evaluation_module.prepare_prompt(
                dimension,
                item["memory"],
                item["query"],
                item["response"],
                item["persona"],
                item["scenario"],
                item["criteria"][dimension],
            )
            response, score = _chat_with_parse_retry(
                backend,
                prompt[0]["content"],
                parser=utils_module.extract_boxed_score,
            )
            record[dimension] = {
                "score": score,
                "full_judge": response,
            }
        results.append(record)
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a non-paper PersonaEmp pilot judge with the official prompts."
        )
    )
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Method label and prediction JSON in LABEL=PATH form.",
    )
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-prefix", default="PERSONAEMP_PILOT_JUDGE")
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository = args.official_repo.resolve()
    verify_official_checkout(repository)
    prepare_module, evaluation_module, utils_module = _load_official_modules(
        repository
    )
    backend = OpenAICompatibleChatBackend.from_env(args.env_prefix)
    dataset_path = args.dataset.resolve()
    dataset = _load_json(dataset_path)
    if not isinstance(dataset, list):
        raise ValueError("dataset root must be a list")

    criteria_path = args.criteria.resolve()
    criteria_data = _prepare_criteria(
        backend=backend,
        dataset=dataset,
        output_path=criteria_path,
        prepare_module=prepare_module,
    )

    outputs: dict[str, dict[str, Any]] = {}
    for label, prediction_path in (
        _parse_prediction(value) for value in args.prediction
    ):
        validate_prediction_alignment(dataset_path, prediction_path)
        prediction = _load_json(prediction_path)
        results = _judge_prediction(
            backend=backend,
            dataset=dataset,
            prediction=prediction,
            criteria_data=criteria_data,
            evaluation_module=evaluation_module,
            utils_module=utils_module,
        )
        result_path = args.output_dir.resolve() / f"{label}.kimi-pilot.json"
        _atomic_json(result_path, results)
        summary = summarize_official_results(result_path)
        summary.update(
            {
                "judge_name": "kimi_pilot",
                "judge_model": backend.model,
                "formal_personaemp_result": False,
            }
        )
        summary_path = result_path.with_suffix(".summary.json")
        _atomic_json(summary_path, summary)
        outputs[label] = {
            "result": str(result_path),
            "summary": str(summary_path),
        }

    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
