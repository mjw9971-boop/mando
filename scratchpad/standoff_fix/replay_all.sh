#!/bin/bash
# 사용: replay_all.sh <출력디렉터리>   — routes/ 가 있는 230426 3건만 폐루프 판단 재생 가능
set -e
OUT="$1"; mkdir -p "$OUT"
D=logs/batch/20260904_230426
for n in 정적회피집중_01_좌회전2 정적회피집중_02_직진3 정적회피집중_03_우회전5; do
  python3 run_agent.py --replay "$D/$n.jsonl" --route "$D/routes/route_$n.pkl" \
      --log "$OUT/$n.jsonl" > "$OUT/$n.stdout" 2>&1
done
