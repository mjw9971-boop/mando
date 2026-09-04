#!/bin/bash
# $1=log  $2=route  — 신/구 트리에서 각각 리플레이하고 비교
set -u
ROOT=/home/cjw/mando
WT=/tmp/claude-1000/-home-cjw-mando/1487e01a-8cea-4925-a43a-9abaf7de1eb3/scratchpad/wt2
OUT=$ROOT/scratchpad/lane_target/replay
LOG="$1"; ROUTE="$2"
B=$(basename "$LOG" .jsonl)
timeout 900 python3 "$ROOT/run_agent.py" --replay "$LOG" --route "$ROUTE" \
   --graph "$ROOT/data/lane_graph.pkl" --config "$ROOT/config/params.yaml" \
   --log "$OUT/new/$B.jsonl" >/dev/null 2>&1
RC1=$?
cd "$WT" && timeout 900 python3 "$WT/run_agent.py" --replay "$LOG" --route "$ROUTE" \
   --graph "$ROOT/data/lane_graph.pkl" --config "$ROOT/config/params.yaml" \
   --log "$OUT/old/$B.jsonl" >/dev/null 2>&1
RC2=$?
echo "$B rc=$RC1/$RC2"
