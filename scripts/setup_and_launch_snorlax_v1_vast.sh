#!/usr/bin/env bash
set -Eeuo pipefail

# Canonical SnorlaxV1 Vast.ai bootstrap. The compatibility bootstrap contains
# the host checks, dependency setup, repository update, and foreground launch.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/setup_and_launch_epoch475_omd_vast.sh"
