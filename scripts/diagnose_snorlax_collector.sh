#!/usr/bin/env bash
set -Eeuo pipefail

PERSIST_ROOT="${PERSIST_ROOT:-/workspace/snorlax-v1-v2}"
REPO_DIR="${REPO_DIR:-/workspace/testing4}"
LOG_DIR="${PERSIST_ROOT}/logs"
OUTPUT="${LOG_DIR}/collector-diagnostic.txt"
mkdir -p "${LOG_DIR}"

exec > >(tee "${OUTPUT}") 2>&1

echo "Snorlax collector diagnostic: $(date)"
echo "Repository commit: $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo
echo "=== Selected launcher profile ==="
grep -hE "Host:|Collector preset|Collector resource guard" \
  "${LOG_DIR}/vast-bootstrap-and-training.log" 2>/dev/null | tail -n 6 || true

echo
echo "=== Live topology ==="
echo "Collector Python processes: $(pgrep -fc 'metamon.rl.ataraxos_online.*--mode collect' || true)"
echo "Simulator Node processes: $(pgrep -fc 'battle_host_engine.js' || true)"
pgrep -af 'metamon.rl.ataraxos_online.*--mode collect' || true

echo
echo "=== CPU allocation and initial pressure ==="
echo -n "CPU quota: "
cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unavailable
cat /sys/fs/cgroup/cpu.pressure 2>/dev/null || cat /proc/pressure/cpu 2>/dev/null || true
usage_before="$(awk '$1 == "usage_usec" {print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)"
throttled_before="$(awk '$1 == "throttled_usec" {print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)"

echo
echo "=== RAM ==="
if [[ -r /sys/fs/cgroup/memory.current && -r /sys/fs/cgroup/memory.max ]]; then
  memory_current="$(cat /sys/fs/cgroup/memory.current)"
  memory_max="$(cat /sys/fs/cgroup/memory.max)"
  if [[ "${memory_max}" =~ ^[0-9]+$ ]]; then
    awk -v used="${memory_current}" -v limit="${memory_max}" \
      'BEGIN {printf "Used: %.1f GiB / %.1f GiB (%.1f%%)\n", used/1073741824, limit/1073741824, used*100/limit}'
  fi
fi
cat /sys/fs/cgroup/memory.pressure 2>/dev/null || cat /proc/pressure/memory 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true

echo
echo "=== Fifteen one-second GPU samples ==="
echo "timestamp, gpu_util, memory_util, memory_used, memory_total, power_draw"
for _sample in $(seq 1 15); do
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits 2>&1 || true
  sleep 1
done

usage_after="$(awk '$1 == "usage_usec" {print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)"
throttled_after="$(awk '$1 == "throttled_usec" {print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)"
echo
echo "=== CPU over GPU sample window ==="
awk -v before="${usage_before}" -v after="${usage_after}" \
  'BEGIN {printf "Average utilized CPU cores: %.2f\n", (after-before)/15000000}'
awk -v before="${throttled_before}" -v after="${throttled_after}" \
  'BEGIN {printf "Additional throttled CPU seconds: %.3f\n", (after-before)/1000000}'
cat /sys/fs/cgroup/cpu.pressure 2>/dev/null || cat /proc/pressure/cpu 2>/dev/null || true

echo
echo "=== Highest CPU users ==="
ps -eo pid,ppid,pcpu,pmem,rss,stat,comm,args --sort=-pcpu | head -n 35 || true

echo
echo "=== Latest measured collector boundary ==="
grep -hE '\[CollectorBoundary\]|\[VectorBoundary\]|profile over|opponent_act:|pump_settle:|advance_lanes:|build_obs:' \
  "${LOG_DIR}"/collector-*.log 2>/dev/null | tail -n 80 || true

echo
echo "Diagnostic saved to ${OUTPUT}"
