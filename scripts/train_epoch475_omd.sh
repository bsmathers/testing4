#!/usr/bin/env bash
set -Eeuo pipefail

# Compatibility entry point. OMD was discarded; run the corrected Ataraxos job.
#
# Required:
#   GOOD_REPLAYS_DIR=/path/containing/gen1ou/*.json[.lz4]
#
# The first run collects PREFILL_FILES trajectories and writes a plain-text
# throughput calibration before the learner starts. Set MIN_GAMES_PER_HOUR to
# turn that measurement into a fail-fast gate.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_epoch475_ataraxos.sh"
