#!/usr/bin/env python3
"""Judge ONE arm's candidate side under the final config, then report the paper-convention macro.

Final config: paper-align prompt + pinned snapshot gpt-4o-mini-2024-07-18.
Reference-side labels are reused from the frozen final-config run (text never changes),
so only candidate cells are requested. Prints macro +/- sample std across 10 speakers.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics as st
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

METRICS = ("reflectiveness", "grounding", "empathy")
KEYMAP = {"reflectiveness": "reflectiveness_accuracy", "grounding": "grounding_accuracy",
          "empathy": "empathy_absolute_difference"}


def rd(p: Path):
    return {json.loads(l)["result_id"]: json.loads(l)
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--prompt-dir",
                    default="/amax/xidian_ty/Ly/personaemp-exp2/v9-judge-align-work/tools/prompts/paper-align")
    args = ap.parse_args()

    W = Path("/amax/xidian_ty/Ly/personaemp-exp2/v9-judge-align-work")
    REPO = Path("/amax/xidian_ty/Ly/personaemp-exp2/releases/exp2-realtalk-v9-code")
    sys.path.insert(0, str(REPO))
    judge = importlib.import_module("src.experiments.realtalk_gpt_judge")

    frozen = rd(W / "base-v9-frozen/local/results_with_local_metrics.jsonl")
    gen = rd(args.gen)
    ref = rd(W / "out/full-paperalign-pinned/scored.jsonl")
    ids = [i for i in frozen if i in gen and gen[i].get("generated_message") and i in ref]

    ctx = judge._contexts(W / "base-v9-frozen/dataset",
                          [{"result_id": i, "speaker": frozen[i]["speaker"]} for i in ids])
    tmpl = {m: (Path(args.prompt_dir) / f"judge_{m}_prompt.txt").read_text(encoding="utf-8") for m in METRICS}
    parsers = {"reflectiveness": judge._parse_bool, "grounding": judge._parse_bool, "empathy": judge._parse_empathy}
    key = os.environ["REALTALK_JUDGE_API_KEY"]
    url = os.environ["REALTALK_JUDGE_BASE_URL"]

    args.out.mkdir(parents=True, exist_ok=True)
    cp_path = args.out / "checkpoint.json"
    ckpt = json.loads(cp_path.read_text()) if cp_path.exists() else {}

    tasks = []
    for i in ids:
        h = ctx[i] or "(none)"
        for m in METRICS:
            k = f"{i}:{m}"
            if k not in ckpt:
                tasks.append((k, m, tmpl[m].format(dialogue_history_within_session=h, history=h,
                                                   turn=gen[i]["generated_message"])))
    print(f"[{args.label}] msgs={len(ids)} pending={len(tasks)} done={len(ckpt)} model={args.model}", flush=True)
    lock = threading.Lock()
    n = {"v": 0}
    errs = {"n": 0}
    t0 = time.time()

    def flush():
        t = cp_path.with_suffix(".json.tmp")
        t.write_text(json.dumps(ckpt, ensure_ascii=False))
        t.replace(cp_path)

    def work(it):
        k, m, pr = it
        try:
            c, _a = judge._chat(url, key, args.model, pr)
            return k, parsers[m](c), None
        except Exception as exc:  # noqa: BLE001
            return k, None, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(work, t) for t in tasks]):
            k, v, e = fut.result()
            with lock:
                n["v"] += 1
                if e is None:
                    ckpt[k] = v
                else:
                    errs["n"] += 1
                if n["v"] % 60 == 0 or n["v"] == len(tasks):
                    flush()
                    print(f"[{args.label}] {n['v']}/{len(tasks)} err={errs['n']} "
                          f"{n['v']/max(1e-6, time.time()-t0):.1f}/s", flush=True)
    flush()

    per = defaultdict(lambda: defaultdict(list))
    for i in ids:
        s = frozen[i]["speaker"]
        rr = ref[i]
        per[s]["reflectiveness"].append(float(bool(ckpt[f"{i}:reflectiveness"]) == bool(rr["reference"]["reflectiveness"])))
        per[s]["grounding"].append(float(bool(ckpt[f"{i}:grounding"]) == bool(rr["reference"]["grounding"])))
        per[s]["empathy"].append(abs(sum(ckpt[f"{i}:empathy"].values()) - sum(rr["reference"]["empathy"].values())))

    summary = {}
    print(f"\n=== {args.label} (n={len(ids)}, {len(per)} speakers) 论文口径 2 位小数 ===")
    line = f"{args.label:22s}"
    for m in METRICS:
        vals = [st.mean(per[s][m]) for s in per]
        summary[KEYMAP[m]] = {"mean": st.mean(vals), "std_n1": st.stdev(vals)}
        line += f"{st.mean(vals):.2f}±{st.stdev(vals):.2f}".rjust(14)
    print(line)
    for m in METRICS:
        v = summary[KEYMAP[m]]
        print(f"   {KEYMAP[m]:32s} {v['mean']:.4f}  std={v['std_n1']:.4f}")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
