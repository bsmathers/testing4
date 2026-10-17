# SnorlaxV1 handoff

Updated: 2026-10-17

## Current status

SnorlaxV1 is ready for its first full remote run. It is the Ataraxos-inspired,
schedule-free continuation of Epoch475. The functional implementation is on
`main` through commit `8dd3588`; this handoff is the only subsequent change.
No production SnorlaxV1 run has been launched yet.

This is a one-shot training run, not a short local experiment. The configured
epoch count is intentionally effectively unbounded. Stop based on evaluation
and learning behavior, not an arbitrary small epoch cap.

## Finalized training recipe

- Initialize the actor and representation from Epoch475. Its legacy critics are
  frozen for checkpoint compatibility. Train a new scalar zero-sum outcome
  value head.
- Prefill a 5,000-game online FIFO. Thereafter alternate 2,000 fresh collected
  games with one learner epoch of 1,000 updates.
- Ramp online data from 0% to 50% over learner epochs 0-100.
- Use batch size 14, no gradient accumulation, no learner mixed precision, and
  deterministic low-discrepancy dataset allocation.
- At the final mixture, sample 50% online, 20% pac-tauros, 15%
  pac-exploratory, 10% curated strong-model trajectories, and 5% human
  trajectories. Pac-exploratory actor labels have weight 0.25; their value
  labels retain full weight.
- Split online teams equally between `hl_05_26` and the committed 807-team
  expert set.
- Train online positions with clipped PPO, reverse KL to the behavior policy,
  a decaying uniform-legal-action magnet, outcome-value regression, and
  top-quartile absolute-advantage filtering. Train offline positions with
  observed-action cross entropy and outcome-value regression.
- Use AdamW group rates of 1e-5 for the representation, 2e-5 for the actor,
  and 1e-4 for the new value head. Hold through 100,000 updates, then decay
  linearly over 400,000 updates to 10% of the original rates.
- Apply recent-weighted schedule-free averaging with beta 0.8 and r 99. Saved
  policies, validation, collection, and opponent play use the averaged model.
- Apply a weak L2-SP anchor to the original Epoch475 encoder and actor, decaying
  to zero over 100,000 updates.

The complete canonical recipe is in `docs/snorlax_v1_training.md`.

## Opponent league

The static half consists of three TaurosV0 and three Kakuna rows. The lineage
half consists of the current policy, nominal offsets -1, -4, -16, and -64, plus
the immutable Epoch475 policy. Missing early lineage slots are filled with
Epoch475.

Checkpoint discovery selects the newest saved epoch at or before each target.
This matters with the retention policy below: after an epoch leaves the latest
ten, a -16 or -64 target will normally resolve to the nearest retained
50-epoch milestone rather than an exact offset. For example, at epoch 123 the
lineage targets resolve from retained epochs 0, 50, 100, and 114-123. This is
not a crash or data-corruption issue, but it changes the age distribution of
self-play opponents. Change retention or league offsets before launch if exact
-16/-64 snapshots are required.

## Epoch retention and resume

Both numbered policy weights and full learner states are saved every epoch.
After a complete state is atomically published, retain:

- every epoch divisible by 50, including epoch 0;
- the ten most recent epochs; and
- separately, the rolling `latest` policy and immutable Epoch475 anchor.

At epoch 123, for example, numbered retention is 0, 50, 100, and 114-123.
Cleanup failures emit a warning and do not terminate training.

Restarting the launcher resumes only from the newest complete, matching policy
and optimizer-state epoch. A policy written after the newest complete state is
treated as an interrupted save and removed. Use the same `PERSIST_ROOT` to
resume. Use a fresh `PERSIST_ROOT` for a genuinely new run; the schema marker
rejects incompatible old replay buffers.

## Important correctness changes already landed

- Online trajectories now contain behavior-time post-mask log probabilities,
  legal-action masks, behavior values, advantages, and value targets.
- Policy metadata and replay shapes/finiteness are validated eagerly. A FIFO
  with missing SnorlaxV1 metadata is rejected before learning.
- BF16 collector probabilities are converted to FP32 and renormalized before
  taking logarithms. This fixes reconstruction failures caused by BF16
  probability mass error.
- CPU collection is allowed only in FP32, avoiding attention KV-cache dtype
  mismatches. The recommended run uses GPU BF16 collection.
- Policies and full learner states are published atomically. Resume requires
  the full optimizer state; an accidental optimizer-reset continuation is
  rejected.
- The L2-SP anchor is reconstructed from the immutable Epoch475 weights on
  resume instead of being duplicated into every checkpoint.
- The collector and learner alternate rather than competing for GPU memory.

## Required remote inputs

Recommended host: one RTX 5090 with 32 GiB VRAM, at least 16 CPU cores, at
least 64 GiB RAM, and 500+ GiB persistent NVMe mounted at `/workspace`. Use a
CUDA development image with `nvcc`; the bootstrap installs and tests the exact
FlashAttention training and KV-cache calls before downloading the large data.

The repository already contains the three Epoch475 weight shards, their
checksum, and the 807 expert team files. The bootstrap downloads public human
replays, pac-tauros, pac-exploratory, `hl_05_26`, TaurosV0, and Kakuna.

The user must provide the curated strong-model corpus at:

```text
/workspace/good_replays/gen1ou/
```

`GOOD_REPLAYS_DIR` must point to the parent directory. These must be parsed
Metamon trajectory files (`.json` or `.json.lz4`), not raw Pokémon Showdown
replay logs. Keep all strong-model trajectories in this single directory; the
dataset deliberately does not separate them by parent model.

Do not resume or recompute the frozen local human teacher corpus at
`replay_bc/cache/tauros_strict_v2_full_20260914`.

## Launch

On a fresh Vast instance:

```bash
export GOOD_REPLAYS_DIR=/workspace/good_replays
export WANDB_API_KEY='...'
git clone https://github.com/bsmathers/testing4.git /workspace/testing4
bash /workspace/testing4/scripts/setup_and_launch_snorlax_v1_vast.sh
```

Optional environment overrides should normally be left at their defaults:

```text
PERSIST_ROOT=/workspace/snorlax-v1
COLLECTOR_DEVICE=gpu
COLLECTOR_DTYPE=bf16
LANES=96
COLLECTOR_WORKERS=8
COLLECTOR_PROCESSES=1
GAMES_PER_LEARNER_EPOCH=2000
```

The bootstrap runs in the foreground. Do not place it under cron or another
scheduler. Relaunch the same command after interruption to resume.

## Outputs and diagnostics

The default persistent tree is `/workspace/snorlax-v1`:

```text
logs/vast-bootstrap-and-training.log
logs/learner.log
logs/collector-prefill-0.log
logs/collector-epoch-<N>-0.log
logs/throughput-calibration.txt
checkpoints/snorlax_v1/ckpts/latest/policy.pt
checkpoints/snorlax_v1/ckpts/policy_weights/policy_epoch_<N>.pt
checkpoints/snorlax_v1/ckpts/training_states/snorlax_v1_epoch_<N>/
online_buffer/gen1ou/
```

Progress logs are plain human-readable text. During prefill, confirm that the
reported game count increases and that throughput is plausible. After the
first learn phase, confirm the presence of both epoch-0 policy and full-state
artifacts before relying on resume.

## Validation completed and remaining limitation

Local checks have covered module import/syntax, exact learner startup, one
offline update, one mixed online/offline update, atomic save, full-state resume,
early opponent-pool expansion, legacy Epoch475 loading, an actual CPU FP32
collector inference, behavior-distribution reconstruction, and epoch-retention
boundaries. The retention test over epochs 0-123 kept exactly 0, 50, 100, and
114-123 while leaving latest and temporary artifacts untouched.

The full CUDA/Vast bootstrap, 5,000-game prefill, and sustained remote training
loop have not yet run end to end. The launcher contains fail-fast checks for
the foreseeable setup and schema failures, but the first remote prefill is the
remaining integration test.

## Canonical files

- `scripts/setup_and_launch_snorlax_v1_vast.sh`
- `scripts/train_snorlax_v1.sh`
- `scripts/train_smallg1onlinev1a_vast.sh`
- `metamon/rl/ataraxos_agent.py`
- `metamon/rl/ataraxos_online.py`
- `metamon/rl/recent_schedule_free.py`
- `metamon/rl/configs/online_runs/snorlax_v1.yaml`
- `metamon/rl/configs/datasets/snorlax_v1.yaml`
- `metamon/rl/configs/team_sets/snorlax_v1.yaml`
- `metamon/rl/configs/opponent_pools/snorlax_v1.yaml`
- `metamon/rl/configs/opponent_pools/snorlax_v1_val.yaml`
- `metamon/rl/configs/training/snorlax_v1.gin`
