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
PERSIST_ROOT="${PERSIST_ROOT:-/workspace/snorlax-v1}"
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

if [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null; then
  log "Installing bootstrap tools"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git curl ca-certificates build-essential xz-utils
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
  # `pip install -e .` refreshes the repository's tracked egg-info files.
  # Those generated metadata changes are harmless and expected on a rerun;
  # continue to reject changes anywhere else so source edits are not lost.
  unexpected_changes="$(
    git -C "${REPO_DIR}" status --porcelain -- . \
      ':(exclude)metamon.egg-info/**'
  )"
  if [[ -n "${unexpected_changes}" ]]; then
    die "Existing checkout has local changes: ${REPO_DIR}. Commit/stash them or choose another REPO_DIR."
  fi
  if [[ -n "$(git -C "${REPO_DIR}" status --porcelain -- metamon.egg-info)" ]]; then
    warn "Ignoring generated metamon.egg-info changes from the earlier installation"
  fi
  current_branch="$(git -C "${REPO_DIR}" branch --show-current)"
  [[ "${current_branch}" == "${REPO_REF}" ]] \
    || die "Existing checkout is on ${current_branch:-detached HEAD}, expected ${REPO_REF}"
  if [[ "${UPDATE_REPO}" == "1" ]]; then
    log "Fast-forwarding the existing checkout"
    git -C "${REPO_DIR}" pull --ff-only origin "${REPO_REF}"
  fi
else
  die "REPO_DIR exists but is not a Git checkout: ${REPO_DIR}"
fi

[[ -f "${REPO_DIR}/scripts/train_snorlax_v1.sh" ]] \
  || die "Training launcher is missing from ${REPO_DIR}"
export GOOD_REPLAYS_DIR PYTHON_BIN
bash "${REPO_DIR}/scripts/prepare_snorlax_strong_replays.sh"
[[ -d "${GOOD_REPLAYS_DIR}/gen1ou" ]] \
  || die "Strong-model replay preparation did not create ${GOOD_REPLAYS_DIR}/gen1ou"

log "Checking the CUDA PyTorch runtime"
"${PYTHON_BIN}" - <<'PY'
import sys
import torch

print(f"Python: {sys.version.split()[0]}")
print(f"Torch: {torch.__version__}; compiled CUDA: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

if ! "${PYTHON_BIN}" -c 'import flash_attn' >/dev/null 2>&1; then
  [[ "${INSTALL_FLASH_ATTN}" == "1" ]] \
    || die "flash_attn is missing and INSTALL_FLASH_ATTN=0"
  command -v nvcc >/dev/null \
    || die "flash_attn is missing and nvcc is unavailable. Use a CUDA development image."
  log "Installing ${FLASH_ATTN_SPEC}; this may compile native CUDA kernels"
  "${PYTHON_BIN}" -m pip install -U pip setuptools wheel packaging ninja
  MAX_JOBS="${MAX_JOBS:-4}" "${PYTHON_BIN}" -m pip install --no-build-isolation "${FLASH_ATTN_SPEC}"
fi

if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  die "WANDB_API_KEY is required when WANDB_MODE=online; set WANDB_MODE=offline to run without it"
fi

mkdir -p "${PERSIST_ROOT}/logs"
commit="$(git -C "${REPO_DIR}" rev-parse HEAD)"
run_log="${PERSIST_ROOT}/logs/vast-bootstrap-and-training.log"
log "Launching commit ${commit}; progress will also be written to ${run_log}"

export REPO_DIR PERSIST_ROOT GOOD_REPLAYS_DIR PYTHON_BIN
export COLLECTOR_PROCESSES="${COLLECTOR_PROCESSES:-1}"
export COLLECTOR_DEVICE="${COLLECTOR_DEVICE:-gpu}"
if [[ -z "${COLLECTOR_DTYPE:-}" ]]; then
  if [[ "${COLLECTOR_DEVICE}" == "cpu" ]]; then
    COLLECTOR_DTYPE=fp32
  else
    COLLECTOR_DTYPE=bf16
  fi
fi
export COLLECTOR_DTYPE
export LANES="${LANES:-96}"
export COLLECTOR_WORKERS="${COLLECTOR_WORKERS:-8}"
export DLOADER_WORKERS=0
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-14}"
export GAMES_PER_LEARNER_EPOCH="${GAMES_PER_LEARNER_EPOCH:-2000}"
export TOTAL_LEARNER_EPOCHS="${TOTAL_LEARNER_EPOCHS:-1000000000}"
export ONLINE_FIFO_SIZE=5000

cd "${REPO_DIR}"
bash scripts/train_snorlax_v1.sh 2>&1 | tee -a "${run_log}"
