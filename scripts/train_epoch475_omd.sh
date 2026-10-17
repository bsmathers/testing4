#!/usr/bin/env bash
set -Eeuo pipefail

# Compatibility entry point. OMD was discarded; the run is now SnorlaxV1.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_snorlax_v1.sh"
