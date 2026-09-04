#!/bin/bash
set -u
ROOT=/home/cjw/mando
WT=/tmp/claude-1000/-home-cjw-mando/1487e01a-8cea-4925-a43a-9abaf7de1eb3/scratchpad/wt2
OUT=$ROOT/scratchpad/lane_target/replay
LOG="$ROOT/$1"; ROUTE="$1"; ROUTE="$ROOT/$2"
B=$(basename "$LOG" .jsonl)
cd "$WT" || exit 9
timeout 900 python3 "$WT/run_agent.py" --replay "$LOG" --route "$ROUTE" \
   --graph "$ROOT/data/lane_graph.pkl" --config "$ROOT/config/params.yaml" \
   --log "$OUT/old/$B.jsonl" >/dev/null 2>&1
echo "$B old_rc=$?"
