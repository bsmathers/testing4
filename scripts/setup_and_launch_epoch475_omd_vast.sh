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

cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)"
ram_mib="$(awk '/MemTotal:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 0)"
disk_free_gib="$(df -Pk /workspace | awk 'NR==2 {print int($4 / 1024 / 1024)}')"
log "Host: ${cpu_count} logical CPUs; ${ram_mib} MiB RAM; ${disk_free_gib} GiB free in /workspace"
(( cpu_count >= 16 )) || warn "Fewer than 16 CPU cores may bottleneck the 96-lane collector"
(( ram_mib >= 60000 )) || warn "Less than 64 GiB RAM may constrain replay loading and compilation"
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
if (( cpu_count >= 64 && vram_mib >= 30000 )); then
  # The vectorized simulator shards each collector across COLLECTOR_WORKERS
  # independent Node/Showdown processes.  On a 64-core EPYC, 8 collectors x
  # 8 workers creates 64 Node processes before counting the 8 Python
  # coordinators, IPC relay threads, and the resident learner.  That topology
  # is fast only when the host also has enough RAM and CPU headroom.
  if (( ram_mib >= 60000 )); then
    # High-RAM 64+ core host.
    default_collector_processes=8
    default_lanes=64
    default_collector_workers=8
    default_partial_hidden_chunk=64
    default_dloader_workers=32
    collector_preset="high-ram-8x64x8"
  elif (( ram_mib >= 36000 )); then
    # 40-48 GiB class hosts (for example EPYC 7742 + RTX 5090). Preserve the
    # full 512 concurrent battles and eight Python batching coordinators, but
    # halve simulator sharding: 32 Node workers total, 16 lanes per worker.
    # This leaves physical-core headroom for Python observation construction,
    # opponent batching, IPC, and the resident learner while greatly reducing
    # duplicated Node/V8 state.
    default_collector_processes=8
    default_lanes=64
    default_collector_workers=4
    default_partial_hidden_chunk=64
    default_dloader_workers=12
    collector_preset="low-ram-epyc-8x64x4"
  else
    # Very memory-constrained large-CPU host.
    default_collector_processes=4
    default_lanes=64
    default_collector_workers=4
    default_partial_hidden_chunk=32
    default_dloader_workers=8
    collector_preset="very-low-ram-4x64x4"
  fi
else
  default_collector_processes=1
  default_lanes=96
  default_collector_workers=8
  default_partial_hidden_chunk=8
  default_dloader_workers=8
  collector_preset="fallback-1x96x8"
fi
export COLLECTOR_PROCESSES="${COLLECTOR_PROCESSES:-${default_collector_processes}}"
export COLLECTOR_DEVICE="${COLLECTOR_DEVICE:-gpu}"
if [[ -z "${COLLECTOR_DTYPE:-}" ]]; then
  if [[ "${COLLECTOR_DEVICE}" == "cpu" ]]; then
    COLLECTOR_DTYPE=fp32
  else
    COLLECTOR_DTYPE=bf16
  fi
fi
export COLLECTOR_DTYPE
export LANES="${LANES:-${default_lanes}}"
export COLLECTOR_WORKERS="${COLLECTOR_WORKERS:-${default_collector_workers}}"
export METAMON_PARTIAL_HIDDEN_CHUNK="${METAMON_PARTIAL_HIDDEN_CHUNK:-${default_partial_hidden_chunk}}"
export DLOADER_WORKERS="${DLOADER_WORKERS:-${default_dloader_workers}}"
log "Collector preset ${collector_preset}: ${COLLECTOR_PROCESSES} processes x ${LANES} lanes x ${COLLECTOR_WORKERS} Showdown workers; DLOADER_WORKERS=${DLOADER_WORKERS}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export METAMON_LOG_INTERVAL="${METAMON_LOG_INTERVAL:-50}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-14}"
export GAMES_PER_LEARNER_EPOCH="${GAMES_PER_LEARNER_EPOCH:-2000}"
export TOTAL_LEARNER_EPOCHS="${TOTAL_LEARNER_EPOCHS:-1000000000}"
export ONLINE_FIFO_SIZE=5000

cd "${REPO_DIR}"
bash scripts/train_snorlax_v1.sh 2>&1 | tee -a "${run_log}"
