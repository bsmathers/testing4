#!/usr/bin/env bash
set -Eeuo pipefail

# Epoch475 continuation with source-separated optimistic mirror descent.
#
# Required:
#   GOOD_REPLAYS_DIR=/path/containing/gen1ou/*.json[.lz4]
#
# The first run collects PREFILL_FILES trajectories and writes a plain-text
# throughput calibration before the learner starts. Set MIN_GAMES_PER_HOUR to
# turn that measurement into a fail-fast gate.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_VARIANT=epoch475_omd
export RUNNER_MODULE=metamon.rl.epoch475_omd
export RUN_NAME="${RUN_NAME:-epoch475_omd}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-epoch475-omd-v1}"
export WANDB_NAME="${WANDB_NAME:-epoch475-omd}"
export WANDB_TAGS="${WANDB_TAGS:-epoch475,omd,gen1ou,807-teams,league}"

# OMD defaults: online actor only. Offline optimism is deliberately an ablation.
export EPOCH475_OMD_ONLINE_BETA="${EPOCH475_OMD_ONLINE_BETA:-1.0}"
export EPOCH475_OMD_OFFLINE_BETA="${EPOCH475_OMD_OFFLINE_BETA:-0.0}"
export EPOCH475_OMD_GRADIENT_EMA="${EPOCH475_OMD_GRADIENT_EMA:-0.0}"

export ONLINE_WEIGHT="${ONLINE_WEIGHT:-0.50}"
export GRAD_ACCUM=1
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-14}"
export COLLECTOR_PROCESSES="${COLLECTOR_PROCESSES:-1}"
export COLLECTOR_DEVICE="${COLLECTOR_DEVICE:-gpu}"
export LANES="${LANES:-96}"
export COLLECTOR_WORKERS="${COLLECTOR_WORKERS:-8}"
export PREFILL_FILES="${PREFILL_FILES:-5000}"
export MIN_GAMES_PER_HOUR="${MIN_GAMES_PER_HOUR:-0}"
export CALIBRATION_ONLY="${CALIBRATION_ONLY:-0}"
export ALTERNATE_COLLECT_LEARN="${ALTERNATE_COLLECT_LEARN:-1}"
export GAMES_PER_LEARNER_EPOCH="${GAMES_PER_LEARNER_EPOCH:-2000}"
export ONLINE_FIFO_SIZE="${ONLINE_FIFO_SIZE:-30000}"

[[ -n "${GOOD_REPLAYS_DIR:-}" ]] || {
  echo "ERROR: GOOD_REPLAYS_DIR must point to a corpus root containing gen1ou/" >&2
  exit 2
}

exec bash "${SCRIPT_DIR}/train_smallg1onlinev1a_vast.sh"
