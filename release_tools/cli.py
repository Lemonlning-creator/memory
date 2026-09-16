from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .common import MODEL, JUDGE, ROOT, bind_output, read, rows, sha, verify_source
from .data import verify_dataset, validate_predictions


def generation_config(args):
    from src.experiments.realtalk_ours import RealTalkOursConfig
    if args.scope == "full" and args.speaker:
        raise ValueError("Full means all ten speakers; omit --speaker")
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        raise ValueError("New run requires an empty output directory; use --resume or a new directory")
    return RealTalkOursConfig(
        dataset_dir=str(args.dataset_dir), output_dir=str(args.output),
        model=MODEL, decision_thinking=False, compute_local_metrics=False,
        eval_points_per_session=1 if args.scope == "smoke" else 0,
        speaker_filter=tuple(args.speaker or ()),
        preflight_only=args.scope == "preflight",
        fresh=not args.resume, resume=args.resume,
    )


def main():
    parser = argparse.ArgumentParser(description="Frozen V9 entry point: DeepSeek Flash, no training, CPU metrics")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-code")
    generate = sub.add_parser("generate")
    generate.add_argument("--scope", choices=("preflight", "smoke", "full"), required=True)
    generate.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--speaker", action="append")
    modes = generate.add_mutually_exclusive_group(required=True)
    modes.add_argument("--new", action="store_true")
    modes.add_argument("--resume", action="store_true")
    for command in ("local-metrics", "judge"):
        p = sub.add_parser(command)
        p.add_argument("--predictions", type=Path, required=True)
        p.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_source()
    if args.command == "verify-code":
        print(json.dumps(verify_source(), indent=2))
        return
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "generate":
        verify_dataset(args.dataset_dir)
        from src.experiments.realtalk_ours import run_realtalk_ours
        result = run_realtalk_ours(generation_config(args))
        if args.scope != "preflight":
            errors = args.output / "unresolved_errors.json"
            if not (args.output / "GENERATION_COMPLETE").exists() or (errors.exists() and read(errors)):
                raise RuntimeError("Generation incomplete; inspect checkpoint/unresolved_errors; do not score as a full result")
    else:
        predictions = rows(args.predictions)
        validate_predictions(predictions, args.dataset_dir)
        identity = {"predictions_sha256": sha(args.predictions),
                    "dataset_manifest_sha256": sha(ROOT / "provenance/dataset_manifest.json"),
                    "source_manifest_sha256": sha(ROOT / "provenance/canonical_source.json"),
                    "stage": args.command, "model": JUDGE if args.command == "judge" else "frozen_local_metrics"}
        bind_output(args.output, identity)
        if args.command == "judge":
            from src.experiments.realtalk_gpt_judge import run
            result = run(args.predictions, args.dataset_dir, args.output, JUDGE)
            if result["status"] != "complete":
                raise RuntimeError("Judge incomplete; rerun the same command to resume")
        else:
            from src.experiments.realtalk_local_metrics import run
            result = run(args.predictions, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
