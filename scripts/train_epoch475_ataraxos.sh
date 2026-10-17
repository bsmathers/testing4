#!/usr/bin/env bash
set -Eeuo pipefail

# Full Epoch475 Ataraxos run: 5k-game FIFO, 2k fresh games per learner epoch,
# and a linear 0% -> 50% online trajectory ramp over the first 100 epochs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PERSIST_ROOT:-}" ]]; then
  marker="${PERSIST_ROOT}/ATARAXOS_EXPERIMENT"
  expected_marker='Epoch475 Ataraxos experiment; replay schema v2; schedule-free r99 beta0.8'
  if [[ -d "${PERSIST_ROOT}/online_buffer/gen1ou" ]] \
      && find "${PERSIST_ROOT}/online_buffer/gen1ou" -type f \
        \( -name '*.json' -o -name '*.json.lz4' \) -print -quit | grep -q . \
      && [[ ! -f "${marker}" ]]; then
    echo "ERROR: PERSIST_ROOT contains an unmarked replay buffer; use a fresh PERSIST_ROOT" >&2
    exit 2
  fi
  mkdir -p "${PERSIST_ROOT}"
  if [[ -f "${marker}" ]] && [[ "$(<"${marker}")" != "${expected_marker}" ]]; then
    echo "ERROR: PERSIST_ROOT belongs to an incompatible Ataraxos schema; use a fresh PERSIST_ROOT" >&2
    exit 2
  fi
  if [[ ! -f "${marker}" ]]; then
    printf '%s\n' "${expected_marker}" >"${marker}"
  fi
fi

export TRAIN_VARIANT=epoch475_ataraxos
export RUNNER_MODULE=metamon.rl.ataraxos_online
export RUN_NAME="${RUN_NAME:-epoch475_ataraxos}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-epoch475-ataraxos-sf-v1}"
export WANDB_NAME="${WANDB_NAME:-epoch475-ataraxos-schedule-free}"
export WANDB_TAGS="${WANDB_TAGS:-epoch475,ataraxos,schedule-free,gen1ou,807-teams,league}"

export ONLINE_WEIGHT=0.50
export GRAD_ACCUM=1
export BATCH_SIZE_PER_GPU=14
export DLOADER_WORKERS=0
export COLLECTOR_PROCESSES="${COLLECTOR_PROCESSES:-1}"
export COLLECTOR_DEVICE="${COLLECTOR_DEVICE:-gpu}"
export LANES="${LANES:-96}"
export COLLECTOR_WORKERS="${COLLECTOR_WORKERS:-8}"
export PREFILL_FILES=5000
export ALTERNATE_COLLECT_LEARN=1
export GAMES_PER_LEARNER_EPOCH=2000
export ONLINE_FIFO_SIZE=5000
export METAMON_ACTOR_ONLY_COLLECTION=1

[[ -n "${GOOD_REPLAYS_DIR:-}" ]] || {
  echo "ERROR: GOOD_REPLAYS_DIR must point to a corpus root containing gen1ou/" >&2
  exit 2
}

exec bash "${SCRIPT_DIR}/train_smallg1onlinev1a_vast.sh"
