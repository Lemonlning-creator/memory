#!/usr/bin/env python3
"""Actor-only replay: Rule A vs control on a fixed subset of frozen V9 positions.

Reuses the frozen Self Domain and the frozen Decision (situation + next_action), so
only the Actor (generation) call is re-executed. Nothing in the frozen package is
modified; outputs go to a separate directory.

Arms
----
control : byte-identical assembly of the frozen Actor prompt
ruleA   : same prompt + a conditional clause that forbids first-person self-statements
          when the partner's most recent message does not ask a question

Generation parameters mirror the frozen run: temperature 0.6, top_p 0.9,
max_tokens 300, thinking disabled.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RULE_A_CLAUSE = """
ADDITIONAL CONSTRAINT FOR THIS MESSAGE:
The partner's most recent message does not ask you a question. Therefore do not include
any first-person statement about your own feelings, preferences, intentions, motivations,
or self-observation (for example "I think I'll ...", "I should probably ...",
"I'm the opposite ...", "for me ..."). Continue the partner's topic without describing
your own internal state, and do not add a self-focused update.
"""

RULE_AB_CLAUSE = """
ADDITIONAL CONSTRAINTS FOR THIS MESSAGE:
The partner's most recent message does not ask you a question. Therefore:
(1) do not include any first-person statement about your own feelings, preferences,
    intentions, motivations, or self-observation (for example "I think I'll ...",
    "I should probably ...", "I'm the opposite ...", "for me ...");
(2) do not add new information, a new topic, advice, or an elaboration of your own,
    and do not confirm or restate what the partner already said. Give only a brief
    reaction that stays inside what the partner already said.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path,
                    default=Path("/amax/xidian_ty/Ly/personaemp-exp2/releases/exp2-realtalk-v9-code"))
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--arm", choices=("control", "ruleA", "ruleAB"), required=True)
    ap.add_argument("--ids-file", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo))
    ours = importlib.import_module("src.experiments.realtalk_ours")
    proto = importlib.import_module("src.experiments.exp1_protocol")
    utils = importlib.import_module("src.utils")

    frozen = args.frozen
    ds = frozen / "dataset"

    records: dict[str, dict] = {}
    for line in (frozen / "local" / "results_with_local_metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["result_id"]] = r
    self_domains = json.loads((frozen / "generation" / "self_domains.json").read_text(encoding="utf-8"))

    speakers = sorted({r["speaker"] for r in records.values()}, key=ours._split_order)
    points: dict[str, tuple[dict, str]] = {}
    for sp in speakers:
        for split in proto.select_realtalk_splits(ds, speaker_filter=[sp]):
            chat = utils.load_json(str(ds / split["test_chat"]))
            for p in proto.build_message_level_points(
                chat, split["speaker"], test_sessions=3, merge_adjacent_bubbles=True
            ):
                points[f"{ours._speaker_id(split['speaker'])}:{p['sample_id']}"] = (p, split["speaker"])

    ids = json.loads(args.ids_file.read_text(encoding="utf-8"))
    if args.limit:
        ids = ids[: args.limit]
    missing = [i for i in ids if i not in records or i not in points]
    if missing:
        raise ValueError(f"{len(missing)} ids not reconstructable, e.g. {missing[:3]}")

    backend = ours._backend_from_env("deepseek-v4-flash")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "generated.jsonl"

    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                done[o["result_id"]] = o
    todo = [i for i in ids if i not in done]

    print(f"[{args.arm}] ids={len(ids)} todo={len(todo)} workers={args.workers}", flush=True)
    lock = threading.Lock()
    fh = out_path.open("a", encoding="utf-8")
    fh_lock = threading.Lock()
    counter = {"n": 0}
    started = time.time()

    def work(rid: str):
        rec = records[rid]
        point, spk = points[rid]
        ctx = point["context_turns"]
        last = ctx[-1]["content"] if ctx else ""
        asked = "?" in last
        user = ours.GENERATION_USER_TEMPLATE.format(
            speaker=spk,
            history=ours._turns_with_session_boundaries(ctx),
            current_session=point["target_session"],
            target_spoke_in_current_session=ours._target_spoke_in_session(
                ctx, spk, point["target_session"]),
            behavioral_self_domain=ours._json(ours._behavioral_self_domain(self_domains[spk])),
            situation=ours._json(rec["situation"]),
            next_action=ours._json(rec["next_action"]),
            action_contract=ours._action_contract(
                rec["next_action"]["primary_move"], rec["next_action"]["continuation_move"]),
        )
        applied = args.arm in ("ruleA", "ruleAB") and not asked
        if applied:
            user = user + (RULE_A_CLAUSE if args.arm == "ruleA" else RULE_AB_CLAUSE)
        try:
            res = backend.chat(
                ours.GENERATION_SYSTEM_TEMPLATE.format(speaker=spk),
                user,
                temperature=0.6, top_p=0.9, max_tokens=300, enable_thinking=False,
            )
            msg = ours._normalize_generated_message(res.content, spk)
            return {"result_id": rid, "speaker": spk, "arm": args.arm,
                    "partner_asked": bool(asked), "clause_applied": bool(applied),
                    "generated_message": msg, "raw": res.content,
                    "model": res.model, "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "latency_seconds": res.latency_seconds, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"result_id": rid, "speaker": spk, "arm": args.arm,
                    "partner_asked": bool(asked), "clause_applied": bool(applied),
                    "generated_message": None, "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for fut in as_completed([pool.submit(work, i) for i in todo]):
            item = fut.result()
            with lock:
                counter["n"] += 1
                with fh_lock:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fh.flush()
                if counter["n"] % 20 == 0 or counter["n"] == len(todo):
                    rate = counter["n"] / max(1e-6, time.time() - started)
                    print(f"[{args.arm}] {counter['n']}/{len(todo)}  {rate:.2f}/s  "
                          f"eta {(len(todo)-counter['n'])/rate:.0f}s", flush=True)
    fh.close()

    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    errs = sum(1 for r in rows if r["error"])
    applied = sum(1 for r in rows if r["clause_applied"])
    print(f"[{args.arm}] DONE rows={len(rows)} errors={errs} clause_applied={applied} "
          f"elapsed={time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
