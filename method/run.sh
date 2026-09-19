#!/bin/bash
R=/amax/xidian_ty/Ly/personaemp-exp2
W=$R/v9-judge-align-work
PY=$R/releases/exp2-realtalk-v9-code/.venv/bin/python
cd $W
export OMP_NUM_THREADS=8
set -a; . $R/secrets/realtalk_ours.env; . $R/secrets/realtalk_judge.env; set +a
for r in r1 r2; do
  $PY tools/actor_replay.py --frozen base-v9-frozen --output-dir out/actor-ruleAB-$r \
     --arm ruleAB --ids-file probe/all519_ids.json --workers 12 > logs/gen-ruleAB-$r.log 2>&1 &
done
wait
for r in r1 r2; do
  $PY tools/judge_arm.py --gen out/actor-ruleAB-$r/generated.jsonl --out out/judge-ruleAB-$r \
     --label ruleAB-$r --workers 16 > logs/judge-ruleAB-$r.log 2>&1 &
done
wait
echo RAB_ALL_DONE
