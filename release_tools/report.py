from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

from .common import read, rows, sha, write

METRICS = (
    ("rouge_l", "ROUGE-L", "up", .14, .14),
    ("bertscore_f1", "BERTScore", "up", .76, .78),
    ("reflectiveness_accuracy", "Reflectiveness", "up", .62, .77),
    ("grounding_accuracy", "Grounding", "up", .40, .62),
    ("sentiment_accuracy", "Sentiment", "up", .53, .59),
    ("emotion_accuracy", "Emotion", "up", .43, .46),
    ("intimacy_absolute_difference", "Intimacy AD", "down", .06, .07),
    ("empathy_absolute_difference", "Empathy AD", "down", 1.80, 1.24),
)


def combine(predictions: Path, local: Path, judge: Path) -> dict:
    predictions_rows = rows(predictions)
    local_rows = {r["result_id"]: r for r in rows(local / "results_with_local_metrics.jsonl")}
    judge_rows = {r["result_id"]: r for r in rows(judge / "scored.jsonl")}
    expected = {r["result_id"] for r in predictions_rows}
    if set(local_rows) != expected or set(judge_rows) != expected:
        raise ValueError("Generation/local/Judge IDs must match exactly; partial results cannot be finalized")
    jm = read(judge / "summary.json")
    lm = read(local / "manifest.json")
    if (judge / "release_binding.json").exists():
        binding = read(judge / "release_binding.json")
        if binding["predictions_sha256"] != sha(predictions) or binding["model"] != "gpt-4o-mini":
            raise ValueError("Judge binding does not match predictions/model")
    if jm["status"] != "complete" or jm["unresolved_errors"] or jm["model_requested"] != "gpt-4o-mini":
        raise ValueError("Judge must be complete, unresolved-free, and gpt-4o-mini")
    if lm["predictions_sha256"] != sha(predictions) or not lm["compute_bertscore"]:
        raise ValueError("Local metrics must include BERTScore and match prediction bytes")
    scores = []
    for row in predictions_rows:
        rid = row["result_id"]
        l, j = local_rows[rid], judge_rows[rid]
        for key in ("speaker", "ground_truth", "generated_message", "context_hash"):
            if l[key] != row[key]:
                raise ValueError(f"Local metric source mismatch: {rid}:{key}")
        if j["speaker"] != row["speaker"]:
            raise ValueError(f"Judge speaker mismatch: {rid}")
        merged = {**l["local_metrics"], **j["metrics"]}
        for key, *_ in METRICS:
            if not isinstance(merged.get(key), (float, int)) or not math.isfinite(merged[key]):
                raise ValueError(f"Missing/nonfinite metric: {rid}:{key}")
        scores.append({"result_id": rid, "speaker": row["speaker"],
                       "target_session": row["target_session"], "metrics": merged})
    speakers = list(dict.fromkeys(r["speaker"] for r in predictions_rows))
    by_speaker = {speaker: {key: statistics.mean(r["metrics"][key] for r in scores if r["speaker"] == speaker)
                           for key, *_ in METRICS} for speaker in speakers}
    macro = {key: {"mean": statistics.mean(s[key] for s in by_speaker.values()),
                   "std_population": statistics.pstdev(s[key] for s in by_speaker.values())}
             for key, *_ in METRICS}
    return {"records": len(scores), "speakers": speakers, "by_speaker": by_speaker,
            "speaker_macro": macro, "per_sample": scores, "predictions_sha256": sha(predictions),
            "comparison_status": "protocol_aligned_exploratory_not_runtime_identical"}


def table(summary: dict, label: str = "Ours V9") -> str:
    names = [f"{name} {'↑' if direction == 'up' else '↓'}" for _, name, direction, *_ in METRICS]
    lines = ["| Method | " + " | ".join(names) + " |", "|---|" + "---:|" * 8]
    for name, idx in (("Paper w/o FT", 3), ("Paper w/ FT", 4)):
        lines.append("| " + name + " | " + " | ".join(f"{m[idx]:.2f}" for m in METRICS) + " |")
    lines.append("| " + label + " | " + " | ".join(f"{summary['speaker_macro'][m[0]]['mean']:.6f}" for m in METRICS) + " |")
    return "\n".join(lines) + "\n"


def finalize(generation: Path, local: Path, judge: Path, output: Path, dataset: Path) -> dict:
    from .data import validate_predictions
    predictions = generation / "predictions.jsonl"
    validate_predictions(rows(predictions), dataset, full=True)
    if not (generation / "GENERATION_COMPLETE").exists() or read(generation / "unresolved_errors.json"):
        raise ValueError("Generation incomplete")
    summary = combine(predictions, local, judge)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Report output must be empty")
    write(output / "summary.json", summary)
    (output / "TABLE2.md").write_text(table(summary), encoding="utf-8", newline="\n")
    write(output / "PIPELINE_COMPLETE", {"records": 519, "metrics": 8,
                                         "predictions_sha256": sha(predictions), "meaning": "complete, not a performance pass"})
    return {"records": summary["records"], "report": str(output)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Finalize only a complete 519-message, eight-metric run")
    for name in ("generation", "local", "judge", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=Path("dataset"))
    print(finalize(**vars(p.parse_args())))
