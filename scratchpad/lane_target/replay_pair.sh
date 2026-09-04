#!/bin/bash
# $1=log $2=route $3=tag(고유 이름) $4=tree(new|old)
set -u
ROOT=/home/cjw/mando
WT=/tmp/claude-1000/-home-cjw-mando/1487e01a-8cea-4925-a43a-9abaf7de1eb3/scratchpad/wt2
OUT=$ROOT/scratchpad/lane_target/replay2
mkdir -p "$OUT/$4"
if [ "$4" = "new" ]; then D=$ROOT; else D=$WT; fi
cd "$D" || exit 9
timeout 900 python3 "$D/run_agent.py" --replay "$ROOT/$1" --route "$ROOT/$2" \
   --graph "$ROOT/data/lane_graph.pkl" --config "$ROOT/config/params.yaml" \
   --log "$OUT/$4/$3.jsonl" >/dev/null 2>&1
echo "$3 $4 rc=$?"
