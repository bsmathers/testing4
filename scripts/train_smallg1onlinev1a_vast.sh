#!/usr/bin/env bash
set -Eeuo pipefail

# Launch SmallG1OnlineV1a on a single Vast.ai GPU instance.
#
# Expected layout:
#   REPO_DIR/                         testing4 checkout
#   REPO_DIR/policy/                  3 epoch-475 shards + checksum
#   REPO_DIR/teams_replay/good_teams  807 .txt/.gen1ou_team files
#
# Target Vast host: RTX 5090 (32 GiB), CUDA-capable PyTorch environment,
# 16+ CPU cores, and 64+ GiB system RAM.
# Persistent outputs default to /workspace/smallg1onlinev1a.

die() { echo "ERROR: $*" >&2; exit 1; }
log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PERSIST_ROOT="${PERSIST_ROOT:-/workspace/smallg1onlinev1a}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="${RUN_NAME:-smallg1onlinev1a}"
TRAIN_VARIANT="${TRAIN_VARIANT:-legacy}"
RUNNER_MODULE="${RUNNER_MODULE:-metamon.rl.taurosv1b_online}"
export TRAIN_VARIANT
WANDB_PROJECT="${METAMON_WANDB_PROJECT:-smallg1onlinev1a}"
WANDB_RUN_ID="${WANDB_RUN_ID:-smallg1onlinev1a-vast-v1}"
WANDB_NAME="${WANDB_NAME:-smallg1onlinev1a}"
COLLECTOR_PROCESSES="${COLLECTOR_PROCESSES:-1}"
COLLECTOR_DEVICE="${COLLECTOR_DEVICE:-gpu}"
LANES="${LANES:-96}"
COLLECTOR_WORKERS="${COLLECTOR_WORKERS:-8}"
DLOADER_WORKERS="${DLOADER_WORKERS:-8}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-7}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
PREFILL_FILES="${PREFILL_FILES:-5001}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
ALTERNATE_COLLECT_LEARN="${ALTERNATE_COLLECT_LEARN:-0}"
GAMES_PER_LEARNER_EPOCH="${GAMES_PER_LEARNER_EPOCH:-2000}"
TOTAL_LEARNER_EPOCHS="${TOTAL_LEARNER_EPOCHS:-401}"
COLLECTOR_DTYPE="${COLLECTOR_DTYPE:-}"
ONLINE_FIFO_SIZE="${ONLINE_FIFO_SIZE:-300000}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-0}"
VALIDATION_TIMESTEPS="${VALIDATION_TIMESTEPS:-1000}"
if [[ -z "${COLLECTOR_DTYPE}" ]]; then
  if [[ "${COLLECTOR_DEVICE}" == "gpu" ]]; then
    COLLECTOR_DTYPE=bf16
  else
    COLLECTOR_DTYPE=fp32
  fi
fi

CACHE_DIR="${PERSIST_ROOT}/cache"
SAVE_DIR="${PERSIST_ROOT}/checkpoints"
BUFFER_DIR="${PERSIST_ROOT}/online_buffer"
CONFIG_DIR="${PERSIST_ROOT}/config"
BOOTSTRAP_DIR="${PERSIST_ROOT}/bootstrap"
LOG_DIR="${PERSIST_ROOT}/logs"
RAW_WEIGHTS="${BOOTSTRAP_DIR}/policy_epoch_475.pt"
ONLINE_WEIGHTS="${BOOTSTRAP_DIR}/policy_epoch_475_online.pt"
if [[ "${TRAIN_VARIANT}" == "snorlax_v1" ]]; then
  RUN_CONFIG="metamon/rl/configs/online_runs/snorlax_v1.yaml"
  POOL_CONFIG="metamon/rl/configs/opponent_pools/snorlax_v1.yaml"
  VAL_POOL_CONFIG="metamon/rl/configs/opponent_pools/snorlax_v1_val.yaml"
  EXPERT_SET_NAME=snorlax_v1_expert
  ONLINE_WEIGHT="${ONLINE_WEIGHT:-0.50}"
else
  RUN_CONFIG="${CONFIG_DIR}/smallg1onlinev1a.yaml"
  POOL_CONFIG="${CONFIG_DIR}/hl_gen1ou_smallg1onlinev1a.yaml"
  VAL_POOL_CONFIG="metamon/rl/configs/opponent_pools/taurosv1a_val.yaml"
  EXPERT_SET_NAME=smallg1onlinev1a_expert
  ONLINE_WEIGHT="${ONLINE_WEIGHT:-0.40}"
fi
EXPERT_DIR="${CACHE_DIR}/teams/${EXPERT_SET_NAME}/gen1ou"

[[ -f "${REPO_DIR}/pyproject.toml" ]] || die "REPO_DIR is not testing4: ${REPO_DIR}"
mkdir -p "${CACHE_DIR}" "${SAVE_DIR}" "${BUFFER_DIR}/gen1ou" \
  "${CONFIG_DIR}" "${BOOTSTRAP_DIR}" "${LOG_DIR}"

export METAMON_CACHE_DIR="${CACHE_DIR}"
export METAMON_SAVE_DIR="${SAVE_DIR}"
export METAMON_BATTLE_HOST="${REPO_DIR}/metamon/env/vectorized/battle_host_engine.js"
export METAMON_PKMN_ENGINE="${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node"
if [[ "${TRAIN_VARIANT}" == "snorlax_v1" ]]; then
  export SNORLAX_RUN_NAME="${RUN_NAME}"
  # L2-SP anchors are reconstructed from this immutable policy on learner
  # resume rather than duplicated into every checkpoint.
  export ATARAXOS_L2SP_ANCHOR_PATH="${ONLINE_WEIGHTS}"
else
  export TAUROSV1B_RUN_NAME="epoch475_frozen"
fi
export METAMON_WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_ID WANDB_NAME
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-smallg1onlinev1a}"
export WANDB_TAGS="${WANDB_TAGS:-smallg1onlinev1a,gen1ou,epoch475,807-teams,vast,zig}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_XET_HIGH_PERFORMANCE=1
export METAMON_ACTOR_ONLY_COLLECTION="${METAMON_ACTOR_ONLY_COLLECTION:-1}"
export METAMON_COLLECTOR_DTYPE="${COLLECTOR_DTYPE}"

cd "${REPO_DIR}"
[[ "${COLLECTOR_PROCESSES}" =~ ^[1-9][0-9]*$ ]] || die "COLLECTOR_PROCESSES must be a positive integer"
[[ "${COLLECTOR_DEVICE}" =~ ^(gpu|cpu)$ ]] || die "COLLECTOR_DEVICE must be gpu or cpu"
[[ "${LANES}" =~ ^[1-9][0-9]*$ ]] || die "LANES must be a positive integer"
[[ "${COLLECTOR_WORKERS}" =~ ^[1-9][0-9]*$ ]] || die "COLLECTOR_WORKERS must be a positive integer"
[[ "${DLOADER_WORKERS}" =~ ^[0-9]+$ ]] || die "DLOADER_WORKERS must be a non-negative integer"
[[ "${BATCH_SIZE_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE_PER_GPU must be a positive integer"
[[ "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]] || die "GRAD_ACCUM must be a positive integer"
[[ "${MIXED_PRECISION}" =~ ^(no|fp16|bf16)$ ]] || die "MIXED_PRECISION must be no, fp16, or bf16"
[[ "${ALTERNATE_COLLECT_LEARN}" =~ ^[01]$ ]] || die "ALTERNATE_COLLECT_LEARN must be 0 or 1"
[[ "${GAMES_PER_LEARNER_EPOCH}" =~ ^[1-9][0-9]*$ ]] || die "GAMES_PER_LEARNER_EPOCH must be positive"
[[ "${TOTAL_LEARNER_EPOCHS}" =~ ^[1-9][0-9]*$ ]] || die "TOTAL_LEARNER_EPOCHS must be positive"
[[ "${COLLECTOR_DTYPE}" =~ ^(fp32|bf16|fp16)$ ]] || die "COLLECTOR_DTYPE must be fp32, bf16, or fp16"
[[ "${ONLINE_FIFO_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "ONLINE_FIFO_SIZE must be positive"
[[ "${VALIDATION_INTERVAL}" =~ ^[0-9]+$ ]] || die "VALIDATION_INTERVAL must be non-negative"
[[ "${VALIDATION_TIMESTEPS}" =~ ^[1-9][0-9]*$ ]] || die "VALIDATION_TIMESTEPS must be positive"
if [[ "${COLLECTOR_DEVICE}" == "cpu" && "${COLLECTOR_DTYPE}" != "fp32" ]]; then
  die "CPU collection requires COLLECTOR_DTYPE=fp32"
fi
(( LANES % COLLECTOR_WORKERS == 0 )) \
  || die "LANES (${LANES}) must be divisible by COLLECTOR_WORKERS (${COLLECTOR_WORKERS})"

if [[ "${INSTALL_DEPS}" == "1" && "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null; then
  log "Installing required system build tools"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git curl ca-certificates build-essential xz-utils
fi
for required_command in git curl gcc g++ make; do
  command -v "${required_command}" >/dev/null \
    || die "Missing required command: ${required_command}"
done
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable; start a Vast GPU instance"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h "${PERSIST_ROOT}"
df -i "${PERSIST_ROOT}"

check_runtime() {
  "${PYTHON_BIN}" - <<'PY'
import sys

import torch

print(f"Python: {sys.version.split()[0]}")
print(f"Torch: {torch.__version__}; compiled CUDA: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
print(f"GPU: {torch.cuda.get_device_name(0)}; VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
PY
}

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  log "Installing project dependencies while preserving the image's PyTorch, CUDA, NumPy, and FlashAttention stack"
  "${PYTHON_BIN}" -m pip install -U pip setuptools wheel packaging ninja
  "${PYTHON_BIN}" -m pip install -e . --no-deps
  "${PYTHON_BIN}" -m pip install \
    'gymnasium>=0.26,<=0.29.1' gin-config wandb einops tqdm lz4 termcolor rich \
    huggingface_hub datasets ratarmountcore accelerate orjson requests tabulate psutil \
    'websockets==12.0' \
    'poke-env @ git+https://github.com/UT-Austin-RPL/poke-env.git' \
    'amago @ git+https://github.com/UT-Austin-RPL/amago@v3.4.0'
fi

check_runtime
"${PYTHON_BIN}" - <<'PY'
import amago
import flash_attn
import gin
import gymnasium
import lz4
import wandb
import metamon
print("Required Python imports succeeded")
PY

"${PYTHON_BIN}" - <<'PY'
import torch
from flash_attn import flash_attn_qkvpacked_func, flash_attn_with_kvcache

# Exercise the exact training and collector APIs used by AMAGO 3.4.0 before
# any large downloads. Both calls must execute native sm_120 kernels.
qkv = torch.randn(2, 64, 3, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
out = flash_attn_qkvpacked_func(qkv, causal=True, window_size=(32, 0))
out.float().square().mean().backward()

step_qkv = torch.randn(2, 1, 3, 4, 32, device="cuda", dtype=torch.bfloat16)
q, k, v = step_qkv.unbind(2)
k_cache = torch.randn(2, 64, 4, 32, device="cuda", dtype=torch.bfloat16)
v_cache = torch.randn_like(k_cache)
cache_seqlens = torch.tensor([0, 7], device="cuda", dtype=torch.int32)
cached_out = flash_attn_with_kvcache(
    q=q,
    k_cache=k_cache,
    v_cache=v_cache,
    k=k,
    v=v,
    cache_seqlens=cache_seqlens,
    causal=True,
    window_size=(32, 0),
)
assert cached_out.shape == q.shape and torch.isfinite(cached_out).all()
torch.cuda.synchronize()
print("FlashAttention training and KV-cache smoke tests succeeded")
PY

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  die "WANDB_API_KEY is required when WANDB_MODE=online"
fi
if [[ "${WANDB_MODE}" == "online" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import os
import wandb

if not wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True):
    raise SystemExit("Weights & Biases login failed")
print("Weights & Biases login succeeded")
PY
fi

log "Building and verifying the native Zig battle engine"
bash scripts/setup_pkmn_engine.sh
[[ -s "${METAMON_PKMN_ENGINE}" ]] || die "Zig addon was not created: ${METAMON_PKMN_ENGINE}"
ZIG_PROBE="$(cd metamon/env/vectorized && echo '{"cmd":"close"}' | node battle_host_engine.js 2>&1 >/dev/null || true)"
grep -q 'HIGH-SPEED NATIVE ZIG ENGINE ACTIVE' <<<"${ZIG_PROBE}" \
  || die "battle_host_engine.js did not activate the native Zig engine"

log "Assembling and validating epoch 475"
if [[ ! -s "${RAW_WEIGHTS}" ]]; then
  mapfile -t PARTS < <(find "${REPO_DIR}/policy" -maxdepth 1 -type f \
    -name 'policy_epoch_475.pt.part-*' | sort)
  [[ "${#PARTS[@]}" -eq 3 ]] || die "Expected exactly 3 epoch-475 shards; found ${#PARTS[@]}"
  tmp_weights="${RAW_WEIGHTS}.assembling"
  : >"${tmp_weights}"
  for part in "${PARTS[@]}"; do
    size="$(stat -c '%s' "${part}")"
    (( size <= 104857600 )) || die "Shard exceeds 100 MiB: ${part}"
    cat "${part}" >>"${tmp_weights}"
  done
  expected="$(awk 'NR==1 {print tolower($1)}' policy/policy_epoch_475.pt.sha256)"
  actual="$(sha256sum "${tmp_weights}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || die "Epoch-475 SHA-256 mismatch: ${actual} != ${expected}"
  mv "${tmp_weights}" "${RAW_WEIGHTS}"
fi

"${PYTHON_BIN}" -m metamon.rl.taurosv1b_kl export-online \
  --input "${RAW_WEIGHTS}" --output "${ONLINE_WEIGHTS}"

"${PYTHON_BIN}" - "${ONLINE_WEIGHTS}" <<'PY'
import sys
import torch

path = sys.argv[1]
state = torch.load(path, map_location="cpu")
bad = [k for k in state if k.startswith("_kl_anchor_") or k == "_kl_forward_step"]
assert not bad, f"KL-only keys remain: {bad[:3]}"
assert len(state) == 598, f"Expected 598 state keys, found {len(state)}"
n = sum(v.numel() for v in state.values())
assert n == 34_766_593, f"Expected 34,766,593 parameters, found {n:,}"
print(f"Validated epoch-475 online policy: {len(state)} keys, {n:,} elements")
PY

log "Creating the 807-team expert set without deduplication"
GOOD_TEAMS_DIR="${GOOD_TEAMS_DIR:-${REPO_DIR}/teams_replay/good_teams}"
export GOOD_TEAMS_DIR EXPERT_DIR
"${PYTHON_BIN}" - <<'PY'
import os
import re
import shutil
from pathlib import Path

source = Path(os.environ["GOOD_TEAMS_DIR"])
target = Path(os.environ["EXPERT_DIR"])
teams = sorted(
    p for p in source.rglob("*")
    if p.is_file() and p.name.endswith((".txt", ".gen1ou_team"))
)
if len(teams) != 807:
    raise SystemExit(f"Expected exactly 807 team files, found {len(teams)} in {source}")
if target.exists():
    shutil.rmtree(target)
target.mkdir(parents=True)
names = []
for i, src in enumerate(teams):
    stem = src.name.removesuffix(".gen1ou_team").removesuffix(".txt")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    name = f"{i:04d}__{safe}.gen1ou_team"
    shutil.copyfile(src, target / name)
    names.append(name)
with (target / "index.csv").open("w", encoding="utf-8") as f:
    f.write("filename\n")
    f.writelines(f"{name}\n" for name in names)
print(f"Wrote {len(names)} entries to {target}; duplicate content was preserved")
PY

if [[ "${TRAIN_VARIANT}" == "snorlax_v1" ]]; then
  log "Using committed SnorlaxV1 run, dataset, team, and opponent configs"
  for config_path in \
    "${RUN_CONFIG}" "${POOL_CONFIG}" "${VAL_POOL_CONFIG}" \
    metamon/rl/configs/datasets/snorlax_v1.yaml \
    metamon/rl/configs/team_sets/snorlax_v1.yaml \
    metamon/rl/configs/training/snorlax_v1.gin; do
    [[ -f "${config_path}" ]] || die "Missing SnorlaxV1 config: ${config_path}"
  done
else
log "Writing the legacy SmallG1OnlineV1a opponent pool and run configuration"
cat >"${POOL_CONFIG}" <<'YAML'
# SmallG1OnlineV1.5 roster, with SmallG1OnlineV0 replaced by epoch 475.
defaults:
  team_set: hl_05_26
  battle_backend: metamon
  checkpoints: [null]
  temperatures: [0.5, 1.0, 1.25, 1.5, 1.75, 2.0]
  num_agents: 1
agents:
  TaurosV0:
    model_name: TaurosV0
    num_agents: 2
    checkpoints: "range(50, 63, 2)"
  Kakuna:
    model_name: Kakuna
    num_agents: 2
    checkpoints: "range(24, 35, 2)"
  Epoch475:
    model_name: TaurosV1B
    num_agents: 2
    checkpoints: [475]
  Alakazam:
    model_name: Alakazam
    checkpoints: "range(2, 9, 2)"
  SyntheticRLV2:
    model_name: SyntheticRLV2
    checkpoints: "range(36, 49, 2)"
  V2ADataAblation:
    model_name: V2AGroupedV2DataAblation
    checkpoints: "range(74, 91, 2)"
  Superkazam:
    model_name: Superkazam
    checkpoints: "range(38, 51, 2)"
  Kadabra3:
    model_name: Kadabra3
    checkpoints: "range(10, 21, 2)"
YAML
fi

if [[ "${TRAIN_VARIANT}" == "snorlax_v1" ]]; then
  GOOD_REPLAYS_DIR="${GOOD_REPLAYS_DIR:-}"
  export GOOD_REPLAYS_DIR
  [[ -n "${GOOD_REPLAYS_DIR}" ]] || die "GOOD_REPLAYS_DIR is required for snorlax_v1"
  [[ -d "${GOOD_REPLAYS_DIR}/gen1ou" ]] \
    || die "GOOD_REPLAYS_DIR must contain gen1ou/: ${GOOD_REPLAYS_DIR}"
  [[ "${ONLINE_WEIGHT}" == "0.50" ]] || die "snorlax_v1 requires ONLINE_WEIGHT=0.50"
  [[ "${ONLINE_FIFO_SIZE}" == "5000" ]] || die "snorlax_v1 requires a 5000-game FIFO"
  [[ "${GRAD_ACCUM}" == "1" ]] || die "snorlax_v1 requires GRAD_ACCUM=1"
  [[ "${BATCH_SIZE_PER_GPU}" == "14" ]] || die "snorlax_v1 requires batch size 14"
  [[ "${DLOADER_WORKERS}" == "0" ]] || die "snorlax_v1 requires DLOADER_WORKERS=0 for deterministic sampling"
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import lz4.frame

root = Path(os.environ["GOOD_REPLAYS_DIR"]) / "gen1ou"
files = sorted(list(root.glob("*.json")) + list(root.glob("*.json.lz4")))
if not files:
    raise SystemExit(
        f"GOOD_REPLAYS_DIR/gen1ou contains no parsed .json or .json.lz4 trajectories: {root}"
    )

# A bounded structural probe catches the common and expensive mistake of
# supplying raw Showdown logs (or a corrupt archive) while keeping startup fast
# for a large curated corpus. The learner performs full validation as files are
# sampled.
for path in files[:32]:
    try:
        if path.name.endswith(".lz4"):
            with lz4.frame.open(path, "rb") as f:
                payload = json.loads(f.read().decode("utf-8"))
        else:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unreadable curated trajectory {path}: {exc}") from exc
    states = payload.get("states")
    actions = payload.get("actions")
    if not isinstance(states, list) or not isinstance(actions, list):
        raise SystemExit(
            f"{path} is not a parsed Metamon trajectory (missing states/actions lists)"
        )
    if len(states) < 2 or len(actions) != len(states):
        raise SystemExit(
            f"Malformed curated trajectory {path}: states={len(states)}, actions={len(actions)}"
        )
print(f"Found {len(files)} curated trajectories; structurally checked {min(len(files), 32)}")
PY
else
  DATASET_CONFIG_VALUE="online_selfplay.yaml"
  TRAIN_TEAM_SET_VALUE="smallg1onlinev1a_expert"
  VAL_TEAM_SET_VALUE="smallg1onlinev1a_expert"
  ONLINE_WEIGHT_VALUE="0.4"
  INITIAL_ONLINE_WEIGHT_VALUE="0.0"
  ONLINE_ANNEAL_END_VALUE=20
  CKPT_INTERVAL_VALUE=10
  FULL_STATE_INTERVAL_VALUE=10
  if [[ "${ALTERNATE_COLLECT_LEARN}" == "1" ]]; then
    CKPT_INTERVAL_VALUE=1
    FULL_STATE_INTERVAL_VALUE=1
  fi
  cat >"${RUN_CONFIG}" <<YAML
run_name: ${RUN_NAME}
base_model: TaurosV1A
battle_format: gen1ou
dataset_config: ${DATASET_CONFIG_VALUE}
train_gin_config: grouped_v2_large_isfilter.gin
train_pool: ${POOL_CONFIG}
train_team_set: ${TRAIN_TEAM_SET_VALUE}
val_pool: ${VAL_POOL_CONFIG}
val_team_set: ${VAL_TEAM_SET_VALUE}
dset_max_size: ${ONLINE_FIFO_SIZE}
dset_min_size: 5000
initial_online_weight: ${INITIAL_ONLINE_WEIGHT_VALUE}
online_weight: ${ONLINE_WEIGHT_VALUE}
online_anneal_start_epoch: 0
online_anneal_end_epoch: ${ONLINE_ANNEAL_END_VALUE}
learning_rate: 8.0e-5
lr_warmup_epochs: 20
seq_floor_warmup_epochs: 20
batch_size_per_gpu: ${BATCH_SIZE_PER_GPU}
grad_accum: ${GRAD_ACCUM}
epochs: ${TOTAL_LEARNER_EPOCHS}
steps_per_epoch: 1000
ckpt_interval: ${CKPT_INTERVAL_VALUE}
full_state_ckpt_interval: ${FULL_STATE_INTERVAL_VALUE}
mixed_precision: "${MIXED_PRECISION}"
lanes: ${LANES}
n_workers: ${COLLECTOR_WORKERS}
train_timesteps_per_epoch: 500
temp_low: 1.0
temp_high: 2.0
reward_function: AggressiveShapedReward
val_timesteps: 1000
val_interval: 10
dloader_workers: ${DLOADER_WORKERS}
YAML
fi

log "Downloading offline data, usage statistics, hl_05_26 teams, and opponent weights"
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from metamon.data.download import (
    download_parsed_replays,
    download_self_play_data,
    download_teams,
    download_usage_stats,
)

download_usage_stats(gen=1, version="v5", force_download=False)
download_parsed_replays("gen1ou", version="v6", force_download=False)
subsets = ("pac-tauros", "pac-exploratory") if os.environ.get("TRAIN_VARIANT") == "snorlax_v1" else ("pac-base", "pac-exploratory", "pac-tauros")
for subset in subsets:
    download_self_play_data(subset, "gen1ou", version="main", force_download=False, extract=False)
download_teams("gen1ou", "hl_05_26", version="v5", force_download=False)

from metamon.backend.team_prediction.team import TeamSet
team_dir = Path(os.environ["EXPERT_DIR"])
for path in sorted(team_dir.glob("*.gen1ou_team")):
    team = TeamSet.from_showdown_file(str(path), "gen1ou")
    if len(team.pokemon) != 6:
        raise ValueError(f"{path.name}: parsed {len(team.pokemon)} Pokemon, expected 6")

import metamon.rl.taurosv1b_online  # registers local TaurosV1B
from metamon.rl.pretrained import get_pretrained_model
public = {
    "TaurosV0": range(50, 63, 2),
    "Kakuna": range(24, 35, 2),
    "Alakazam": range(2, 9, 2),
    "SyntheticRLV2": range(36, 49, 2),
    "V2AGroupedV2DataAblation": range(74, 91, 2),
    "Superkazam": range(38, 51, 2),
    "Kadabra3": range(10, 21, 2),
}
if os.environ.get("TRAIN_VARIANT") == "snorlax_v1":
    public = {"TaurosV0": public["TaurosV0"], "Kakuna": public["Kakuna"]}
for model_name, epochs in public.items():
    model = get_pretrained_model(model_name)
    for epoch in epochs:
        path = Path(model.get_path_to_checkpoint(epoch))
        if not path.exists():
            raise FileNotFoundError(f"Failed to fetch {model_name}@{epoch}: {path}")
print("All datasets, 807 expert teams, and public opponents are ready")
PY

FROZEN="${SAVE_DIR}/epoch475_frozen/ckpts/policy_weights/policy_epoch_475.pt"
LATEST="${SAVE_DIR}/${RUN_NAME}/ckpts/latest/policy.pt"
mkdir -p "$(dirname "${FROZEN}")" "$(dirname "${LATEST}")"
cp -f "${ONLINE_WEIGHTS}" "${FROZEN}"
[[ -s "${LATEST}" ]] || cp "${ONLINE_WEIGHTS}" "${LATEST}"
if [[ "${TRAIN_VARIANT}" == "snorlax_v1" ]]; then
  mkdir -p "${SAVE_DIR}/${RUN_NAME}/ckpts/policy_weights"
  BASE_SNAPSHOT="${SAVE_DIR}/${RUN_NAME}/ckpts/policy_weights/policy_epoch_0.pt"
  [[ -s "${BASE_SNAPSHOT}" ]] || cp "${ONLINE_WEIGHTS}" "${BASE_SNAPSHOT}"
fi

CKPT_ROOT="${SAVE_DIR}/${RUN_NAME}/ckpts"
STATE_ROOT="${CKPT_ROOT}/training_states"
POLICY_ROOT="${CKPT_ROOT}/policy_weights"
RESUME_EPOCH=""
if [[ -d "${STATE_ROOT}" ]]; then
  RESUME_EPOCH="$({ find "${STATE_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name "${RUN_NAME}_epoch_*" -printf '%f\n' 2>/dev/null || true; } \
    | sed -n 's/.*_epoch_\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1)"
fi

LEARNER_RESUME_ARGS=()
if [[ -n "${RESUME_EPOCH}" ]]; then
  RESUME_POLICY="${POLICY_ROOT}/policy_epoch_${RESUME_EPOCH}.pt"
  [[ -s "${RESUME_POLICY}" ]] || die "Full state epoch ${RESUME_EPOCH} has no matching policy"
  # A preemption can occur after the atomic policy write but before the atomic
  # optimizer-state directory is published. Such a policy is not a completed
  # epoch and must not enter the discovered opponent lineage on resume.
  while IFS= read -r policy_path; do
    policy_name="$(basename "${policy_path}")"
    policy_epoch="${policy_name#policy_epoch_}"
    policy_epoch="${policy_epoch%.pt}"
    if [[ "${policy_epoch}" =~ ^[0-9]+$ ]] && (( policy_epoch > RESUME_EPOCH )); then
      log "Removing incomplete future policy epoch ${policy_epoch}"
      rm -f -- "${policy_path}"
    fi
  done < <(find "${POLICY_ROOT}" -maxdepth 1 -type f -name 'policy_epoch_*.pt' -print)
  # Keep collection and learning on the same exact full-state epoch.
  cp -f "${RESUME_POLICY}" "${LATEST}"
  LEARNER_RESUME_ARGS=(--resume_training_state --resume_epoch "${RESUME_EPOCH}")
  log "Resuming full learner state from completed epoch ${RESUME_EPOCH}"
else
  if find "${POLICY_ROOT}" -maxdepth 1 -type f -name 'policy_epoch_*.pt' \
      ! -name 'policy_epoch_0.pt' -print -quit 2>/dev/null | grep -q .; then
    die "Policy checkpoints exist but no full training state exists; refusing an optimizer-reset resume"
  fi
  if [[ "${TRAIN_VARIANT}" == "snorlax_v1" ]]; then
    # policy_epoch_0 starts as the lineage bootstrap and is overwritten by the
    # completed first learner epoch. If epoch 0 was interrupted after its policy
    # write, no complete state exists and the bootstrap must be restored.
    cp -f "${ONLINE_WEIGHTS}" "${POLICY_ROOT}/policy_epoch_0.pt"
  fi
  cp -f "${ONLINE_WEIGHTS}" "${LATEST}"
  LEARNER_RESUME_ARGS=(--base_weights "${ONLINE_WEIGHTS}")
  log "Starting ${RUN_NAME} epoch 0 from cleaned epoch-475 weights"
fi

COLLECTOR_PIDS=()
cleanup_collectors() {
  local pid
  if (( ${#COLLECTOR_PIDS[@]} == 0 )); then
    return
  fi
  for pid in "${COLLECTOR_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${COLLECTOR_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  COLLECTOR_PIDS=()
}
trap cleanup_collectors EXIT
trap 'exit 130' INT TERM

start_collectors() {
  local seed_base="$1" phase="$2" i seed
  local -a collector_prefix
  local collector_mp="${COLLECTOR_DTYPE}"
  COLLECTOR_PIDS=()
  collector_prefix=(env)
  if [[ "${COLLECTOR_DEVICE}" == "cpu" ]]; then
    collector_prefix+=(CUDA_VISIBLE_DEVICES= METAMON_CPU_COLLECTOR=1)
  fi
  [[ "${collector_mp}" == "fp32" ]] && collector_mp=no
  for ((i=0; i<COLLECTOR_PROCESSES; i++)); do
    seed=$((seed_base + i))
    "${collector_prefix[@]}" "${PYTHON_BIN}" -m "${RUNNER_MODULE}" \
      --run_config "${RUN_CONFIG}" --mode collect \
      --run_name "${RUN_NAME}" --lanes "${LANES}" --n_workers "${COLLECTOR_WORKERS}" \
      --save_dir "${SAVE_DIR}" --buffer_dir "${BUFFER_DIR}" \
      --base_weights "${LATEST}" --epochs 1000000 --seed "${seed}" \
      --mixed_precision "${collector_mp}" \
      > >(tee -a "${LOG_DIR}/collector-${phase}-${i}.log") 2>&1 &
    COLLECTOR_PIDS+=("$!")
  done
}

fifo_count() {
  find "${BUFFER_DIR}/gen1ou" -type f \( -name '*.json' -o -name '*.json.lz4' \) | wc -l
}

snapshot_fifo_names() {
  local output="$1"
  find "${BUFFER_DIR}/gen1ou" -type f \
    \( -name '*.json' -o -name '*.json.lz4' \) -print | LC_ALL=C sort >"${output}"
}

new_fifo_count() {
  local baseline="$1" current_names
  current_names="${baseline}.current"
  snapshot_fifo_names "${current_names}"
  # Replay filenames contain a random battle id and are never reused. Counting
  # the set difference is reliable even on filesystems with coarse mtimes and
  # while the fixed-size FIFO replaces older files.
  comm -13 "${baseline}" "${current_names}" | wc -l
  rm -f -- "${current_names}"
}

current="$(fifo_count)"
if (( current < PREFILL_FILES )); then
  prefill_start_count="${current}"
  prefill_start_seconds="$(date +%s)"
  log "Prefilling online FIFO: ${current}/${PREFILL_FILES}"
  start_collectors 0 prefill
  while (( current < PREFILL_FILES )); do
    sleep 30
    for pid in "${COLLECTOR_PIDS[@]}"; do
      kill -0 "${pid}" 2>/dev/null || die "A prefill collector exited; inspect ${LOG_DIR}/collector-prefill-*.log"
    done
    current="$(fifo_count)"
    echo "FIFO prefill: ${current}/${PREFILL_FILES}"
  done
  cleanup_collectors
  prefill_end_seconds="$(date +%s)"
  prefill_elapsed=$((prefill_end_seconds - prefill_start_seconds))
  prefill_added=$((current - prefill_start_count))
  if (( prefill_elapsed > 0 && prefill_added > 0 )); then
    games_per_hour=$((prefill_added * 3600 / prefill_elapsed))
    estimated_cycle_minutes="$(awk -v rate="${games_per_hour}" -v games="${GAMES_PER_LEARNER_EPOCH}" 'BEGIN { if (rate > 0) printf "%.1f", games * 60 / rate; else print "unknown" }')"
    {
      echo "${RUN_NAME} collection calibration"
      echo "Games collected: ${prefill_added}"
      echo "Elapsed seconds: ${prefill_elapsed}"
      echo "Games per hour: ${games_per_hour}"
      echo "Estimated minutes per ${GAMES_PER_LEARNER_EPOCH}-game collect phase: ${estimated_cycle_minutes}"
    } | tee "${LOG_DIR}/throughput-calibration.txt"
    if (( ${MIN_GAMES_PER_HOUR:-0} > 0 && games_per_hour < MIN_GAMES_PER_HOUR )); then
      die "Collection throughput ${games_per_hour}/hour is below MIN_GAMES_PER_HOUR=${MIN_GAMES_PER_HOUR}"
    fi
  fi
else
  log "FIFO already contains ${current} files; skipping prefill"
fi

if [[ "${TRAIN_VARIANT}" == "snorlax_v1" ]]; then
  log "Validating exact behavior metadata in the SnorlaxV1 FIFO"
  BUFFER_DIR="${BUFFER_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import lz4.frame
import numpy as np

root = Path(os.environ["BUFFER_DIR"]) / "gen1ou"
files = sorted(list(root.glob("*.json")) + list(root.glob("*.json.lz4")))
if len(files) < 5000:
    raise SystemExit(f"SnorlaxV1 FIFO requires at least 5000 games; found {len(files)}")
required = {"behavior_log_probs", "behavior_values", "behavior_legal_masks"}
for path in files:
    if path.name.endswith(".lz4"):
        with lz4.frame.open(path, "rb") as f:
            payload = json.loads(f.read().decode("utf-8"))
    else:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    missing = required.difference(payload)
    if missing:
        raise SystemExit(f"{path.name} predates SnorlaxV1 metadata; use a fresh PERSIST_ROOT")
    states = payload.get("states")
    actions = payload.get("actions")
    logp = np.asarray(payload["behavior_log_probs"], dtype=np.float32)
    values = np.asarray(payload["behavior_values"], dtype=np.float32)
    legal = np.asarray(payload["behavior_legal_masks"], dtype=bool)
    if not isinstance(states, list) or not isinstance(actions, list) or len(states) < 2:
        raise SystemExit(f"{path.name} has malformed states/actions")
    steps = len(states) - 1
    if len(actions) != len(states) or logp.ndim != 2 or logp.shape[0] != steps:
        raise SystemExit(f"{path.name} has trajectory-misaligned behavior metadata")
    if values.shape != (steps,) or legal.shape != logp.shape:
        raise SystemExit(f"{path.name} has invalid behavior value/legal-mask shapes")
    if not np.isfinite(logp).all() or not np.isfinite(values).all():
        raise SystemExit(f"{path.name} has non-finite behavior metadata")
    if np.any(np.abs(values) > 1.0001):
        raise SystemExit(f"{path.name} has behavior values outside [-1, 1]")
    probs = np.exp(logp)
    if not np.allclose(probs.sum(-1), 1.0, atol=2e-3, rtol=0.0):
        raise SystemExit(f"{path.name} has an unnormalized behavior policy")
    if np.any(probs[~legal] > 1e-8):
        raise SystemExit(f"{path.name} assigns behavior mass to illegal actions")
    chosen = np.asarray(actions[:-1])
    if (
        chosen.shape != (steps,)
        or not np.issubdtype(chosen.dtype, np.integer)
        or np.any(chosen < 0)
        or np.any(chosen >= logp.shape[1])
    ):
        raise SystemExit(f"{path.name} has invalid recorded actions")
    if not legal[np.arange(steps), chosen].all():
        raise SystemExit(f"{path.name} records an action marked illegal")
print(f"Validated behavior metadata in {len(files)} FIFO games")
PY
fi

if [[ "${CALIBRATION_ONLY:-0}" == "1" ]]; then
  log "Calibration-only run completed; learner was not started"
  exit 0
fi

if [[ "${ALTERNATE_COLLECT_LEARN}" == "1" ]]; then
  completed_epoch="${RESUME_EPOCH:--1}"
  next_epoch=$((completed_epoch + 1))
  while (( next_epoch < TOTAL_LEARNER_EPOCHS )); do
    baseline="${LOG_DIR}/.cycle-${next_epoch}-fifo-baseline.txt"
    snapshot_fifo_names "${baseline}"
    phase="epoch-${next_epoch}"
    log "Collect phase for learner epoch ${next_epoch}: target ${GAMES_PER_LEARNER_EPOCH} fresh games"
    start_collectors $((1000 + next_epoch * COLLECTOR_PROCESSES)) "${phase}"
    fresh=0
    while (( fresh < GAMES_PER_LEARNER_EPOCH )); do
      sleep 30
      for pid in "${COLLECTOR_PIDS[@]}"; do
        kill -0 "${pid}" 2>/dev/null \
          || die "A ${phase} collector exited; inspect ${LOG_DIR}/collector-${phase}-*.log"
      done
      fresh="$(new_fifo_count "${baseline}")"
      echo "Cycle ${next_epoch} collection: ${fresh}/${GAMES_PER_LEARNER_EPOCH} fresh games"
    done
    cleanup_collectors
    rm -f -- "${baseline}" "${baseline}.current"

    learner_limit=$((next_epoch + 1))
    if (( completed_epoch >= 0 )); then
      cycle_resume=(--resume_training_state --resume_epoch "${completed_epoch}")
    else
      cycle_resume=(--base_weights "${ONLINE_WEIGHTS}")
    fi
    log "Learn phase: epoch ${next_epoch}, 1000 updates, collectors stopped"
    set +e
    "${PYTHON_BIN}" -m "${RUNNER_MODULE}" \
      --run_config "${RUN_CONFIG}" --mode learn \
      --run_name "${RUN_NAME}" --lanes "${LANES}" --n_workers "${COLLECTOR_WORKERS}" \
      --save_dir "${SAVE_DIR}" --buffer_dir "${BUFFER_DIR}" --log \
      --epochs "${learner_limit}" --ckpt_interval 1 --full_state_ckpt_interval 1 \
      "${cycle_resume[@]}" \
      2>&1 | tee -a "${LOG_DIR}/learner.log"
    learner_status="${PIPESTATUS[0]}"
    set -e
    (( learner_status == 0 )) \
      || die "Learner epoch ${next_epoch} exited with status ${learner_status}"

    completed_epoch="${next_epoch}"
    completed_state="${STATE_ROOT}/${RUN_NAME}_epoch_${completed_epoch}"
    completed_policy="${POLICY_ROOT}/policy_epoch_${completed_epoch}.pt"
    [[ -d "${completed_state}" ]] || die "Missing completed state ${completed_state}"
    [[ -s "${completed_policy}" ]] || die "Missing completed policy ${completed_policy}"
    cp -f "${completed_policy}" "${LATEST}"
    if (( VALIDATION_INTERVAL > 0 && completed_epoch % VALIDATION_INTERVAL == 0 )); then
      log "Validation phase for completed epoch ${completed_epoch}: ${VALIDATION_TIMESTEPS} steps"
      set +e
      "${PYTHON_BIN}" -m "${RUNNER_MODULE}" \
        --run_config "${RUN_CONFIG}" --mode validate --validation_once \
        --validation_epoch "${completed_epoch}" \
        --run_name "${RUN_NAME}" --lanes "${LANES}" --n_workers "${COLLECTOR_WORKERS}" \
        --save_dir "${SAVE_DIR}" --buffer_dir "${BUFFER_DIR}" --log \
        --base_weights "${completed_policy}" \
        --val_timesteps "${VALIDATION_TIMESTEPS}" --val_interval 1 \
        2>&1 | tee -a "${LOG_DIR}/validation.log"
      validation_status="${PIPESTATUS[0]}"
      set -e
      (( validation_status == 0 )) \
        || die "Validation for epoch ${completed_epoch} exited with status ${validation_status}"
    fi
    next_epoch=$((completed_epoch + 1))
  done
  trap - EXIT INT TERM
  log "Alternating collection/training completed through epoch ${completed_epoch}"
  exit 0
fi

start_collectors 1000 train

log "Launching ${RUN_NAME} learner with W&B run ${WANDB_PROJECT}/${WANDB_RUN_ID}"
set +e
"${PYTHON_BIN}" -m "${RUNNER_MODULE}" \
  --run_config "${RUN_CONFIG}" --mode learn \
  --run_name "${RUN_NAME}" --lanes "${LANES}" --n_workers "${COLLECTOR_WORKERS}" \
  --save_dir "${SAVE_DIR}" --buffer_dir "${BUFFER_DIR}" --log \
  "${LEARNER_RESUME_ARGS[@]}" \
  2>&1 | tee -a "${LOG_DIR}/learner.log"
learner_status="${PIPESTATUS[0]}"
set -e
cleanup_collectors
trap - EXIT INT TERM

(( learner_status == 0 )) || die "Learner exited with status ${learner_status}; inspect ${LOG_DIR}/learner.log"
log "${RUN_NAME} training completed"
