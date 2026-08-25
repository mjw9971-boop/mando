#!/usr/bin/env bash
# 대회날에는 이것만 실행한다.
set -euo pipefail
cd "$(dirname "$0")"

HOST="${VTD_HOST:-192.168.10.1}"
PORT="${VTD_PORT:-9910}"
GRAPH="${GRAPH:-data/lane_graph.pkl}"

# 대회 배포 waypoints.csv 가 있으면 그걸로 route 를 빌드해 쓰고,
# 아니면 ROUTE(기본 data/route.pkl)를 그대로 쓴다.
if [[ -n "${CSV:-}" ]]; then
    exec python3 run_agent.py --host "$HOST" --port "$PORT" --graph "$GRAPH" --csv "$CSV"
else
    ROUTE="${ROUTE:-data/route.pkl}"
    exec python3 run_agent.py --host "$HOST" --port "$PORT" --graph "$GRAPH" --route "$ROUTE"
fi
