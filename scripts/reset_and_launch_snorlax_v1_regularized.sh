#!/usr/bin/env bash
set -Eeuo pipefail

# Destructively discard all generated SnorlaxV1 training state, then provision
# and launch the regularized Epoch475 continuation in the foreground. Downloaded
# offline datasets are preserved, but generated compiler caches are discarded.

die() { echo "ERROR: $*" >&2; exit 1; }
log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

[[ $# -eq 2 ]] || die "Usage: $0 WANDB_API_KEY HF_API_KEY"
[[ -n "$1" ]] || die "WANDB_API_KEY must not be empty"
[[ -n "$2" ]] || die "HF_API_KEY must not be empty"

REPO_DIR="${REPO_DIR:-/workspace/testing4}"
PERSIST_ROOT="${PERSIST_ROOT:-/workspace/snorlax-v1-v2}"

[[ "${REPO_DIR}" == /* ]] || die "REPO_DIR must be absolute"
[[ "${PERSIST_ROOT}" == /workspace/* ]] \
  || die "PERSIST_ROOT must be a child of /workspace"
# Normalize dot segments before comparing or deleting. Without this,
# /workspace/run/.. would pass the textual prefix check but name /workspace.
command -v python3 >/dev/null || die "python3 is required"
REPO_DIR="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${REPO_DIR}")"
PERSIST_ROOT="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${PERSIST_ROOT}")"
case "${PERSIST_ROOT}" in
  /|/workspace|/workspace/|"${REPO_DIR}")
    die "Refusing unsafe PERSIST_ROOT: ${PERSIST_ROOT}"
    ;;
esac
[[ -f "${REPO_DIR}/scripts/setup_and_launch_epoch475_omd_vast.sh" ]] \
  || die "Snorlax launcher not found in ${REPO_DIR}"

export WANDB_API_KEY="$1"
export HF_TOKEN="$2"
export HUGGING_FACE_HUB_TOKEN="$2"
export REPO_DIR PERSIST_ROOT
run_timestamp="$(date -u '+%Y%m%dT%H%M%SZ')"
export WANDB_RUN_ID="${WANDB_RUN_ID:-snorlax-v1-regularized-${run_timestamp}}"
export WANDB_NAME="${WANDB_NAME:-SnorlaxV1-regularized-v1}"
# This entry point always deletes the local training state, so allowing it to
# resume an older remote W&B run would splice two independent runs together.
export WANDB_RESUME=never

log "RESET: deleting FIFO buffers, generated model weights, logs, and provision state from ${PERSIST_ROOT}"
rm -rf -- \
  "${PERSIST_ROOT}/online_buffer" \
  "${PERSIST_ROOT}/checkpoints" \
  "${PERSIST_ROOT}/bootstrap" \
  "${PERSIST_ROOT}/logs" \
  "${PERSIST_ROOT}/provision" \
  "${PERSIST_ROOT}/cache/torchinductor" \
  "${PERSIST_ROOT}/cache/triton"

# Force the recipe marker to be recreated for this clean run. Offline replay
# archives, team caches, and Hugging Face downloads remain.
rm -f -- "${PERSIST_ROOT}/SNORLAX_V1_EXPERIMENT"
mkdir -p "${PERSIST_ROOT}"

log "Starting a new regularized SnorlaxV1 run; validation runs every 10 epochs"
exec bash "${REPO_DIR}/scripts/setup_and_launch_epoch475_omd_vast.sh"
