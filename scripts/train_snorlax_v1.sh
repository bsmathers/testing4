#!/usr/bin/env bash
set -Eeuo pipefail

# Canonical SnorlaxV1 training entry point. SnorlaxV1 is the Ataraxos-inspired
# schedule-free continuation of Epoch475, with a 5k replay FIFO and the
# intended 0% -> 50% online-mixture ramp over the first 100 learner epochs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PERSIST_ROOT="${PERSIST_ROOT:-/workspace/snorlax-v1-v2}"

if [[ -n "${PERSIST_ROOT}" ]]; then
  marker="${PERSIST_ROOT}/SNORLAX_V1_EXPERIMENT"
  old_marker='SnorlaxV1; replay schema v2; schedule-free r99 beta0.8'
  previous_marker='SnorlaxV1; replay schema v3; schedule-free r99 beta0.8'
  previous_v2_marker='SnorlaxV1-v2; schema v3; onlineramp0to50; advq0to25; criticwarmup100k; schedule-free r99 beta0.8'
  expected_marker='SnorlaxV1-v2; schema v3; onlineramp0to50; advq0to25; criticwarmup100k; dropout0; exactprefix128; behavior-y; schedule-free r99 beta0.8'
  if [[ -d "${PERSIST_ROOT}/online_buffer/gen1ou" ]] \
      && find "${PERSIST_ROOT}/online_buffer/gen1ou" -type f \
        \( -name '*.json' -o -name '*.json.lz4' \) -print -quit | grep -q . \
      && [[ ! -f "${marker}" ]]; then
    echo "ERROR: PERSIST_ROOT contains an unmarked replay buffer; use a fresh PERSIST_ROOT" >&2
    exit 2
  fi
  mkdir -p "${PERSIST_ROOT}"
  if [[ -f "${marker}" ]] \
      && { [[ "$(<"${marker}")" == "${old_marker}" ]] \
        || [[ "$(<"${marker}")" == "${previous_marker}" ]] \
        || [[ "$(<"${marker}")" == "${previous_v2_marker}" ]]; }; then
    state_root="${PERSIST_ROOT}/checkpoints/snorlax_v1/ckpts/training_states"
    if [[ -d "${state_root}" ]] \
        && find "${state_root}" -mindepth 1 -maxdepth 1 -type d \
          -name 'snorlax_v1_epoch_*' -print -quit | grep -q .; then
      echo "ERROR: completed SnorlaxV1 training state predates the corrected v2 recipe; use a fresh PERSIST_ROOT" >&2
      exit 2
    fi
    printf '%s\n' "${expected_marker}" >"${marker}"
    echo "Migrated the untrained SnorlaxV1 workspace marker to the v2 recipe"
  fi
  if [[ -f "${marker}" ]] && [[ "$(<"${marker}")" != "${expected_marker}" ]]; then
    echo "ERROR: PERSIST_ROOT belongs to an incompatible SnorlaxV1 schema; use a fresh PERSIST_ROOT" >&2
    exit 2
  fi
  if [[ ! -f "${marker}" ]]; then
    printf '%s\n' "${expected_marker}" >"${marker}"
  fi
fi

export TRAIN_VARIANT=snorlax_v1
export RUNNER_MODULE=metamon.rl.ataraxos_online
export RUN_NAME="${RUN_NAME:-snorlax_v1}"
export METAMON_WANDB_PROJECT="${METAMON_WANDB_PROJECT:-snorlax-v1}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-snorlax-v1-v2}"
export WANDB_NAME="${WANDB_NAME:-SnorlaxV1-v2}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-snorlax-v1}"
export WANDB_TAGS="${WANDB_TAGS:-snorlax-v1,epoch475,ataraxos,schedule-free,gen1ou,807-teams,league}"

export ONLINE_WEIGHT=0.50
export GRAD_ACCUM=1
export BATCH_SIZE_PER_GPU=14
export DLOADER_WORKERS="${DLOADER_WORKERS:-32}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export METAMON_LOG_INTERVAL="${METAMON_LOG_INTERVAL:-50}"
export COLLECTOR_PROCESSES="${COLLECTOR_PROCESSES:-1}"
export COLLECTOR_DEVICE="${COLLECTOR_DEVICE:-gpu}"
export LANES="${LANES:-96}"
export COLLECTOR_WORKERS="${COLLECTOR_WORKERS:-8}"
export PREFILL_FILES=5000
export ALTERNATE_COLLECT_LEARN=1
export GAMES_PER_LEARNER_EPOCH="${GAMES_PER_LEARNER_EPOCH:-2000}"
export TOTAL_LEARNER_EPOCHS="${TOTAL_LEARNER_EPOCHS:-1000000000}"
export ONLINE_FIFO_SIZE=5000
export VALIDATION_INTERVAL=10
export VALIDATION_TIMESTEPS=1000
export METAMON_ACTOR_ONLY_COLLECTION=1

export GOOD_REPLAYS_DIR="${GOOD_REPLAYS_DIR:-/workspace/good_replays}"

exec bash "${SCRIPT_DIR}/train_smallg1onlinev1a_vast.sh"
