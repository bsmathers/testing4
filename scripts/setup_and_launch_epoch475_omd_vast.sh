#!/usr/bin/env bash
set -Eeuo pipefail

# Compatibility-named bootstrap for the SnorlaxV1 experiment on Vast.ai.
#
# Recommended instance:
#   - 1x RTX 5090 (32 GiB)
#   - 16+ dedicated CPU cores and 64+ GiB RAM
#   - 500+ GiB persistent NVMe mounted at /workspace
#   - a CUDA 12.8+ PyTorch development image (nvcc included)
#
# Strong-model data is downloaded from the pinned public Hugging Face dataset
# when GOOD_REPLAYS_DIR/gen1ou is not already populated.
#
# This script deliberately runs in the foreground. Re-run it after an
# interruption; the training launcher resumes from complete epoch-boundary
# checkpoints under PERSIST_ROOT.

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }
log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

REPO_URL="${REPO_URL:-https://github.com/bsmathers/testing4.git}"
REPO_REF="${REPO_REF:-main}"
REPO_DIR="${REPO_DIR:-/workspace/testing4}"
PERSIST_ROOT="${PERSIST_ROOT:-/workspace/snorlax-v1-v2}"
GOOD_REPLAYS_DIR="${GOOD_REPLAYS_DIR:-/workspace/good_replays}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UPDATE_REPO="${UPDATE_REPO:-1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-30000}"
ALLOW_LOW_VRAM="${ALLOW_LOW_VRAM:-0}"

[[ "${UPDATE_REPO}" =~ ^[01]$ ]] || die "UPDATE_REPO must be 0 or 1"
[[ "${INSTALL_FLASH_ATTN}" =~ ^[01]$ ]] || die "INSTALL_FLASH_ATTN must be 0 or 1"
[[ "${ALLOW_LOW_VRAM}" =~ ^[01]$ ]] || die "ALLOW_LOW_VRAM must be 0 or 1"
[[ "${MIN_VRAM_MIB}" =~ ^[1-9][0-9]*$ ]] || die "MIN_VRAM_MIB must be a positive integer"
[[ -d /workspace && -w /workspace ]] || die "/workspace must exist and be writable"

missing_bootstrap=()
for command_name in git curl; do
  command -v "${command_name}" >/dev/null || missing_bootstrap+=("${command_name}")
done
if (( ${#missing_bootstrap[@]} > 0 )); then
  [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null \
    || die "Missing bootstrap commands: ${missing_bootstrap[*]}"
  log "Installing missing bootstrap tools: ${missing_bootstrap[*]}"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y git curl ca-certificates
fi

for command_name in git curl "${PYTHON_BIN}" nvidia-smi; do
  command -v "${command_name}" >/dev/null || die "Missing required command: ${command_name}"
done

gpu_line="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -n 1)"
gpu_name="$(cut -d, -f1 <<<"${gpu_line}" | xargs)"
vram_mib="$(cut -d, -f2 <<<"${gpu_line}" | xargs)"
driver_version="$(cut -d, -f3 <<<"${gpu_line}" | xargs)"
[[ "${vram_mib}" =~ ^[0-9]+$ ]] || die "Could not read GPU memory from nvidia-smi: ${gpu_line}"
log "GPU: ${gpu_name}; VRAM: ${vram_mib} MiB; driver: ${driver_version}"
if (( vram_mib < MIN_VRAM_MIB )); then
  if [[ "${ALLOW_LOW_VRAM}" == "1" ]]; then
    warn "GPU has less than ${MIN_VRAM_MIB} MiB; batch size 14 may run out of memory"
  else
    die "GPU has less than ${MIN_VRAM_MIB} MiB. Use a 32 GiB GPU, or set ALLOW_LOW_VRAM=1 for an explicit experiment."
  fi
fi

affinity_cpu_count="$("${PYTHON_BIN}" - <<'PY'
import os
try:
    print(len(os.sched_getaffinity(0)))
except AttributeError:
    print(os.cpu_count() or 0)
PY
)"
cpu_quota_count=0
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r cpu_quota cpu_period < /sys/fs/cgroup/cpu.max
  if [[ "${cpu_quota}" =~ ^[0-9]+$ && "${cpu_period}" =~ ^[0-9]+$ ]] \
      && (( cpu_period > 0 )); then
    cpu_quota_count=$(( cpu_quota / cpu_period ))
    (( cpu_quota_count > 0 )) || cpu_quota_count=1
  fi
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us \
     && -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
  cpu_quota="$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
  cpu_period="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
  if [[ "${cpu_quota}" =~ ^[0-9]+$ && "${cpu_period}" =~ ^[0-9]+$ ]] \
      && (( cpu_period > 0 )); then
    cpu_quota_count=$(( cpu_quota / cpu_period ))
    (( cpu_quota_count > 0 )) || cpu_quota_count=1
  fi
fi
cpu_count="${affinity_cpu_count}"
if (( cpu_quota_count > 0 && cpu_quota_count < cpu_count )); then
  cpu_count="${cpu_quota_count}"
fi
host_ram_mib="$(awk '/MemTotal:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 0)"
cpu_model="$(awk -F: '/model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null || true)"
cgroup_ram_mib=0
if [[ -r /sys/fs/cgroup/memory.max ]]; then
  cgroup_limit="$(cat /sys/fs/cgroup/memory.max)"
  if [[ "${cgroup_limit}" =~ ^[0-9]+$ ]]; then
    cgroup_ram_mib=$(( cgroup_limit / 1024 / 1024 ))
  fi
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
  cgroup_limit="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
  if [[ "${cgroup_limit}" =~ ^[0-9]+$ ]]; then
    cgroup_ram_mib=$(( cgroup_limit / 1024 / 1024 ))
  fi
fi
ram_mib="${host_ram_mib}"
if (( cgroup_ram_mib > 0 && (ram_mib == 0 || cgroup_ram_mib < ram_mib) )); then
  ram_mib="${cgroup_ram_mib}"
fi
disk_free_gib="$(df -Pk /workspace | awk 'NR==2 {print int($4 / 1024 / 1024)}')"
log "Host: ${cpu_model:-unknown}; ${cpu_count} usable CPUs (affinity=${affinity_cpu_count}, quota=${cpu_quota_count:-0}); effective RAM ${ram_mib} MiB (host=${host_ram_mib}, cgroup=${cgroup_ram_mib:-0}); ${disk_free_gib} GiB free in /workspace"
(( cpu_count >= 16 )) || warn "Fewer than 16 CPU cores may bottleneck the 96-lane collector"
(( ram_mib >= 60000 )) || warn "Less than 64 GiB effective RAM may constrain replay loading and compilation"
(( disk_free_gib >= 250 )) || warn "Less than 250 GiB free; verify replay and retained-checkpoint growth during training"

if [[ ! -e "${REPO_DIR}" ]]; then
  log "Cloning ${REPO_URL} at ${REPO_REF}"
  git clone --branch "${REPO_REF}" --single-branch "${REPO_URL}" "${REPO_DIR}"
elif [[ -d "${REPO_DIR}/.git" ]]; then
  if [[ "${UPDATE_REPO}" == "1" ]]; then
    log "Synchronizing the disposable training checkout to origin/${REPO_REF}"
    git -C "${REPO_DIR}" fetch origin "${REPO_REF}"
    git -C "${REPO_DIR}" reset --hard "origin/${REPO_REF}"
  fi
else
  die "REPO_DIR exists but is not a Git checkout: ${REPO_DIR}"
fi

[[ -f "${REPO_DIR}/scripts/train_snorlax_v1.sh" ]] \
  || die "Training launcher is missing from ${REPO_DIR}"

if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  die "WANDB_API_KEY is required when WANDB_MODE=online; set WANDB_MODE=offline to run without it"
fi

mkdir -p "${PERSIST_ROOT}/logs"
commit="$(git -C "${REPO_DIR}" rev-parse HEAD)"
run_log="${PERSIST_ROOT}/logs/vast-bootstrap-and-training.log"
log "Launching commit ${commit}; progress will also be written to ${run_log}"

export REPO_DIR PERSIST_ROOT GOOD_REPLAYS_DIR PYTHON_BIN

# Collector topology is selected here and is authoritative. Inherited shell
# values cannot silently override the machine-specific preset.
if [[ "${cpu_model}" == *"EPYC 7642"* && "${gpu_name}" == *"5090"* ]] \
    && (( cpu_count >= 44 && ram_mib >= 60000 )); then
  # Vast EPYC 7642 profile with 48 allocated CPUs and about 64 GiB RAM.
  # Vast currently encodes the advertised 48-vCPU allocation as a 46.08-CPU
  # CFS quota (4608000/100000), so allow the small provider-side quota haircut.
  # Each collector has one Python coordinator plus seven single-threaded Node
  # simulator workers: 6 * (1 + 7) = 48 CPU-bound processes. Six actor-only
  # BF16 policies and 384 lanes fit comfortably on a 32 GiB RTX 5090, while
  # retaining fewer copies than the proven 64-core 8x64x7 profile.
  COLLECTOR_PROCESSES=6
  LANES=64
  COLLECTOR_WORKERS=7
  METAMON_PARTIAL_HIDDEN_CHUNK=64
  METAMON_FULL_BATCH_MAX_INACTIVE=8
  DLOADER_WORKERS=12
  SNORLAX_RESOURCE_GUARD=1
  SNORLAX_MIN_COLLECTOR_PROCESSES=4
  SNORLAX_CPU_PSI_LIMIT=25
  SNORLAX_MEMORY_USAGE_LIMIT=0.90
  collector_preset="epyc-7642-48cpu-64gb-5090-6x64x7"
elif [[ "${cpu_model}" == *"EPYC 7742"* && "${gpu_name}" == *"5090"* ]]; then
  # Exact production host requested for this run. Do not trust virtualized
  # MemTotal/cgroup reporting to choose a different topology.
  COLLECTOR_PROCESSES=8
  LANES=64
  COLLECTOR_WORKERS=7
  METAMON_PARTIAL_HIDDEN_CHUNK=64
  METAMON_FULL_BATCH_MAX_INACTIVE=8
  DLOADER_WORKERS=12
  collector_preset="epyc-7742-5090-8x64x7"
elif (( vram_mib >= 30000 && cpu_count >= 56 )); then
  if (( ram_mib < 56000 )); then
    # EPYC 7742 / 40-48 GiB / RTX 5090 class host:
    # 512 battles, 56 Node workers total. This is the previously optimized
    # Snorlax collection topology for a 64-core EPYC host.
    COLLECTOR_PROCESSES=8
    LANES=64
    COLLECTOR_WORKERS=7
    METAMON_PARTIAL_HIDDEN_CHUNK=64
    METAMON_FULL_BATCH_MAX_INACTIVE=8
    DLOADER_WORKERS=12
    collector_preset="epyc-40gb-8x64x7"
  else
    # Large-memory 64+ CPU host:
    # 512 battles, 56 Node workers total.
    COLLECTOR_PROCESSES=8
    LANES=64
    COLLECTOR_WORKERS=7
    METAMON_PARTIAL_HIDDEN_CHUNK=64
    METAMON_FULL_BATCH_MAX_INACTIVE=8
    DLOADER_WORKERS=32
    collector_preset="high-ram-8x64x7"
  fi
elif (( vram_mib >= 30000 && cpu_count >= 32 )); then
  COLLECTOR_PROCESSES=4
  LANES=64
  COLLECTOR_WORKERS=4
  METAMON_PARTIAL_HIDDEN_CHUNK=32
  METAMON_FULL_BATCH_MAX_INACTIVE=8
  DLOADER_WORKERS=8
  collector_preset="mid-cpu-4x64x4"
else
  COLLECTOR_PROCESSES=1
  LANES=96
  COLLECTOR_WORKERS=8
  METAMON_PARTIAL_HIDDEN_CHUNK=8
  METAMON_FULL_BATCH_MAX_INACTIVE=8
  DLOADER_WORKERS=8
  collector_preset="fallback-1x96x8"
fi
COLLECTOR_DEVICE=gpu
COLLECTOR_DTYPE=bf16
SNORLAX_RESOURCE_GUARD="${SNORLAX_RESOURCE_GUARD:-0}"
SNORLAX_MIN_COLLECTOR_PROCESSES="${SNORLAX_MIN_COLLECTOR_PROCESSES:-${COLLECTOR_PROCESSES}}"
SNORLAX_CPU_PSI_LIMIT="${SNORLAX_CPU_PSI_LIMIT:-25}"
SNORLAX_MEMORY_USAGE_LIMIT="${SNORLAX_MEMORY_USAGE_LIMIT:-0.90}"

export COLLECTOR_PROCESSES COLLECTOR_DEVICE COLLECTOR_DTYPE LANES
export COLLECTOR_WORKERS METAMON_PARTIAL_HIDDEN_CHUNK METAMON_FULL_BATCH_MAX_INACTIVE DLOADER_WORKERS
export SNORLAX_RESOURCE_GUARD SNORLAX_MIN_COLLECTOR_PROCESSES
export SNORLAX_CPU_PSI_LIMIT SNORLAX_MEMORY_USAGE_LIMIT
log "Collector preset ${collector_preset}: ${COLLECTOR_PROCESSES} processes x ${LANES} lanes x ${COLLECTOR_WORKERS} Showdown workers; DLOADER_WORKERS=${DLOADER_WORKERS}"
if [[ "${SNORLAX_RESOURCE_GUARD}" == "1" ]]; then
  log "Collector resource guard: CPU PSI >= ${SNORLAX_CPU_PSI_LIMIT}% or sustained RAM pressure >= ${SNORLAX_MEMORY_USAGE_LIMIT}; one-process downshifts, floor=${SNORLAX_MIN_COLLECTOR_PROCESSES}"
fi

MIXED_PRECISION=bf16
METAMON_LOG_INTERVAL=50
BATCH_SIZE_PER_GPU=14
GAMES_PER_LEARNER_EPOCH=2000
TOTAL_LEARNER_EPOCHS=1000000000
ONLINE_FIFO_SIZE=5000
export MIXED_PRECISION METAMON_LOG_INTERVAL BATCH_SIZE_PER_GPU
export GAMES_PER_LEARNER_EPOCH TOTAL_LEARNER_EPOCHS ONLINE_FIFO_SIZE

cd "${REPO_DIR}"
bash scripts/train_snorlax_v1.sh 2>&1 | tee -a "${run_log}"
