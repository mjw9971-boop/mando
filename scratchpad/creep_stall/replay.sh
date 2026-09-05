#!/bin/bash
# 사용: replay.sh <배치> <출력디렉터리>
set -e
B="$1"; OUT="$2"; mkdir -p "$OUT"
for n in 정적회피집중_01_좌회전2 정적회피집중_02_직진3 정적회피집중_03_우회전5; do
  python3 run_agent.py --replay "logs/batch/$B/$n.jsonl" --route "logs/batch/$B/routes/route_$n.pkl" \
      --log "$OUT/$n.jsonl" > "$OUT/$n.stdout" 2>&1
done
