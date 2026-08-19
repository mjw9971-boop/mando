#!/usr/bin/env bash
# 대회날에는 이것만 실행한다.
set -euo pipefail
cd "$(dirname "$0")"

source /opt/ros/jazzy/setup.bash
source install/setup.bash

HOST="${VTD_HOST:-127.0.0.1}"
PORT="${VTD_PORT:-9910}"
GRAPH="${GRAPH:-data/lane_graph.pkl}"
ROUTE="${ROUTE:-data/route.pkl}"
RECORD="${RECORD:-true}"          # 기본 녹화 — 끝나고 원인 분석에 쓴다

exec ros2 launch hlfma drive.launch.py \
    host:="$HOST" port:="$PORT" \
    graph:="$GRAPH" route:="$ROUTE" \
    record:="$RECORD" bag_path:="bags/drive_$(date +%Y%m%d_%H%M%S)"
