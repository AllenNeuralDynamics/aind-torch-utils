#!/usr/bin/env bash
set -Eeuo pipefail

metrics_file="${GPU_METRICS_FILE:-/results/gpu_metrics.csv}"
metrics_interval="${GPU_METRICS_INTERVAL:-1}"
monitor_pid=""

stop_monitor() {
    if [[ -n "${monitor_pid}" ]]; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
        monitor_pid=""
    fi
}

trap stop_monitor EXIT INT TERM

mkdir -p /results

gpu_query="timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu"

nvidia-smi -L
nvidia-smi --query-gpu="${gpu_query}" --format=csv >/dev/null

nvidia-smi \
    --query-gpu="${gpu_query}" \
    --format=csv \
    -l "${metrics_interval}" \
    -f "${metrics_file}" &
monitor_pid=$!

set +e
python -m aind_torch_utils.distributed.ray_launcher "$@"
launcher_status=$?
set -e

stop_monitor
trap - EXIT INT TERM
exit "${launcher_status}"
