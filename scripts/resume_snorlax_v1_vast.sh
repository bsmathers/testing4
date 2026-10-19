#!/usr/bin/env bash
set -Eeuo pipefail

# Fast restart path for an already-running SnorlaxV1 experiment.
# A completed full-state checkpoint is the provisioning certificate: if one
# exists with its matching policy, do not redo downloads, engine builds,
# dataset/team validation, runtime smoke tests, or FIFO-wide preflight scans.

die() { echo "ERROR: $*" >&2; exit 1; }
log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PERSIST_ROOT="${PERSIST_ROOT:-/workspace/snorlax-v1-v2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="${RUN_NAME:-snorlax_v1}"
RUNNER_MODULE="${RUNNER_MODULE:-metamon.rl.ataraxos_online}"

CACHE_DIR="${PERSIST_ROOT}/cache"
SAVE_DIR="${PERSIST_ROOT}/checkpoints"
BUFFER_DIR="${PERSIST_ROOT}/online_buffer"
BOOTSTRAP_DIR="${PERSIST_ROOT}/bootstrap"
LOG_DIR="${PERSIST_ROOT}/logs"
ONLINE_WEIGHTS="${BOOTSTRAP_DIR}/policy_epoch_475_online.pt"
RUN_CONFIG="${REPO_DIR}/metamon/rl/configs/online_runs/snorlax_v1.yaml"

CKPT_ROOT="${SAVE_DIR}/${RUN_NAME}/ckpts"
STATE_ROOT="${CKPT_ROOT}/training_states"
POLICY_ROOT="${CKPT_ROOT}/policy_weights"
LATEST="${CKPT_ROOT}/latest/policy.pt"
BEHAVIOR="${CKPT_ROOT}/latest/behavior_policy.pt"

[[ -d "${STATE_ROOT}" ]] || die "No Snorlax training_states directory: ${STATE_ROOT}"
RESUME_EPOCH="$({ find "${STATE_ROOT}" -mindepth 1 -maxdepth 1 -type d   -name "${RUN_NAME}_epoch_*" -printf '%f\n' 2>/dev/null || true; }   | sed -n 's/.*_epoch_\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1)"
[[ -n "${RESUME_EPOCH}" ]] || die "No complete Snorlax full-state checkpoint found"

RESUME_STATE="${STATE_ROOT}/${RUN_NAME}_epoch_${RESUME_EPOCH}"
RESUME_POLICY="${POLICY_ROOT}/policy_epoch_${RESUME_EPOCH}.pt"
[[ -d "${RESUME_STATE}" ]] || die "Missing resume state: ${RESUME_STATE}"
[[ -s "${RESUME_POLICY}" ]] || die "Checkpoint epoch ${RESUME_EPOCH} has no matching policy"
[[ -s "${ONLINE_WEIGHTS}" ]] || die "Missing immutable Epoch475 anchor: ${ONLINE_WEIGHTS}"
[[ -f "${RUN_CONFIG}" ]] || die "Missing Snorlax run config: ${RUN_CONFIG}"
[[ -f "${PERSIST_ROOT}/provision/snorlax-rollout-schema-v3.done" ]]   || die "Resume state is missing the rollout-schema-v3 marker"
mkdir -p "$(dirname "${LATEST}")" "${LOG_DIR}" "${BUFFER_DIR}/gen1ou"

# This immutable league checkpoint is tiny to repair and avoids re-entering the
# full provisioner merely because an auxiliary copy disappeared.
FROZEN="${SAVE_DIR}/epoch475_frozen/ckpts/policy_weights/policy_epoch_475.pt"
if [[ ! -s "${FROZEN}" ]]; then
  mkdir -p "$(dirname "${FROZEN}")"
  cp -f "${ONLINE_WEIGHTS}" "${FROZEN}"
fi

# Drop policy-only artifacts from interrupted future epochs. A full state is
# the commit record; anything newer than it must not enter the opponent league.
while IFS= read -r policy_path; do
  policy_name="$(basename "${policy_path}")"
  policy_epoch="${policy_name#policy_epoch_}"
  policy_epoch="${policy_epoch%.pt}"
  if [[ "${policy_epoch}" =~ ^[0-9]+$ ]] && (( policy_epoch > RESUME_EPOCH )); then
    rm -f -- "${policy_path}"
    log "Removed incomplete future policy epoch ${policy_epoch}"
  fi
done < <(find "${POLICY_ROOT}" -maxdepth 1 -type f -name 'policy_epoch_*.pt' -print 2>/dev/null)

cp -f "${RESUME_POLICY}" "${LATEST}"

# Runtime variables required by the learner/collector subprocesses. Hardware
# topology is supplied authoritatively by setup_and_launch_epoch475_omd_vast.sh.
export TRAIN_VARIANT=snorlax_v1
export METAMON_CACHE_DIR="${CACHE_DIR}"
export METAMON_SAVE_DIR="${SAVE_DIR}"
export METAMON_BATTLE_HOST="${REPO_DIR}/metamon/env/vectorized/battle_host_engine.js"
export METAMON_PKMN_ENGINE="${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node"
export SNORLAX_RUN_NAME="${RUN_NAME}"
export ATARAXOS_L2SP_ANCHOR_PATH="${ONLINE_WEIGHTS}"
export METAMON_ACTOR_ONLY_COLLECTION=1
export METAMON_COLLECTOR_DTYPE="${COLLECTOR_DTYPE:-bf16}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_XET=1
unset HF_XET_HIGH_PERFORMANCE HF_XET_FIXED_DOWNLOAD_CONCURRENCY
unset HF_XET_FIXED_UPLOAD_CONCURRENCY HF_XET_CACHE
unset HF_XET_CLIENT_RETRY_MAX_ATTEMPTS HF_XET_CLIENT_RETRY_MAX_DURATION
unset HF_XET_CLIENT_READ_TIMEOUT HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY

export METAMON_WANDB_PROJECT="${METAMON_WANDB_PROJECT:-snorlax-v1}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-snorlax-v1-v2}"
export WANDB_NAME="${WANDB_NAME:-SnorlaxV1-v2}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-snorlax-v1}"
export WANDB_TAGS="${WANDB_TAGS:-snorlax-v1,epoch475,ataraxos,schedule-free,gen1ou,807-teams,league}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RESUME=allow
if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  die "WANDB_API_KEY is required when WANDB_MODE=online"
fi

# These are fixed by the Snorlax recipe. Collector topology values are already
# selected by the canonical hardware launcher and intentionally are not
# user-overridable here.
export COLLECTOR_PROCESSES=8
export COLLECTOR_DEVICE=gpu
export COLLECTOR_DTYPE=bf16
export LANES=64
export COLLECTOR_WORKERS=7
export DLOADER_WORKERS=12
export METAMON_PARTIAL_HIDDEN_CHUNK=64
export METAMON_FULL_BATCH_MAX_INACTIVE=8
export MIXED_PRECISION=bf16
export METAMON_LOG_INTERVAL=50
export BATCH_SIZE_PER_GPU=14
export GRAD_ACCUM=1
export GAMES_PER_LEARNER_EPOCH=2000
export TOTAL_LEARNER_EPOCHS=1000000000
export ONLINE_FIFO_SIZE=5000
export VALIDATION_INTERVAL=10
export VALIDATION_TIMESTEPS=1000
export VALIDATION_GAMES=250

export SNORLAX_PERSISTENT_ALTERNATION=1
export SNORLAX_BUFFER_DIR="${BUFFER_DIR}"
export SNORLAX_LOG_DIR="${LOG_DIR}"
export SNORLAX_LATEST_WEIGHTS="${LATEST}"
export SNORLAX_BEHAVIOR_WEIGHTS="${BEHAVIOR}"
export SNORLAX_RUN_CONFIG="${RUN_CONFIG}"
export RUNNER_MODULE

# Reap only stale child phases. Never kill an existing learner implicitly.
stale=()
while IFS= read -r pid; do
  [[ -n "${pid}" ]] && stale+=("${pid}")
done < <(pgrep -f "${RUNNER_MODULE}.*--mode (collect|validate).*--run_name ${RUN_NAME}" 2>/dev/null || true)
if (( ${#stale[@]} > 0 )); then
  log "Stopping ${#stale[@]} stale collector/validation process(es)"
  kill "${stale[@]}" 2>/dev/null || true
  sleep 1
  for pid in "${stale[@]}"; do
    kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null || true
  done
fi

other_learner_pids=()
while IFS= read -r pid; do
  [[ -n "${pid}" && "${pid}" != "$$" ]] && other_learner_pids+=("${pid}")
done < <(pgrep -f "${RUNNER_MODULE}.*--mode learn.*--run_name ${RUN_NAME}" 2>/dev/null || true)
if (( ${#other_learner_pids[@]} > 0 )); then
  die "A Snorlax learner is already running (PID(s): ${other_learner_pids[*]})"
fi

cd "${REPO_DIR}"
log "FAST RESUME: epoch ${RESUME_EPOCH}; skipping provisioning and dataset/runtime rechecks"
log "Collector topology: ${COLLECTOR_PROCESSES} processes x ${LANES} lanes x ${COLLECTOR_WORKERS} Showdown workers; DLOADER_WORKERS=${DLOADER_WORKERS}"

set +e
"${PYTHON_BIN}" -m "${RUNNER_MODULE}"   --run_config "${RUN_CONFIG}" --mode learn   --run_name "${RUN_NAME}" --lanes "${LANES}" --n_workers "${COLLECTOR_WORKERS}"   --save_dir "${SAVE_DIR}" --buffer_dir "${BUFFER_DIR}" --log   --epochs "${TOTAL_LEARNER_EPOCHS}" --ckpt_interval 1 --full_state_ckpt_interval 1   --mixed_precision "${MIXED_PRECISION}" --dloader_workers "${DLOADER_WORKERS}"   --resume_training_state --resume_epoch "${RESUME_EPOCH}"   2>&1 | tee -a "${LOG_DIR}/learner.log"
status="${PIPESTATUS[0]}"
set -e
(( status == 0 )) || die "Persistent Snorlax learner exited with status ${status}"
