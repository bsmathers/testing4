# Epoch475 OMD training pipeline

This pipeline continues the 34.8M-parameter Epoch475 policy with a 50/50
offline/online training mixture and source-separated optimistic updates.

## Data

The offline half is sampled by source, not in proportion to corpus size:

- 40% `pac-tauros`
- 30% parsed Gen 1 OU human replays
- 30% curated strong-model trajectories

`GOOD_REPLAYS_DIR` must contain a `gen1ou/` directory of parsed Metamon
trajectories. Each file must use the perspective of TaurosV1, TaurosV2,
SmallG1OnlineV2, or SmallG1OnlineV3. Do not include the opposing side merely
because it appeared in the same battle; that would imitate an uncontrolled
opponent policy.

Online trajectories are split evenly between the public `hl_05_26` teams and
the local 807-team set. Half of the opponent row mass is TaurosV0/Kakuna. The
other half is a fixed six-slot lineage ladder: the current policy, policies from
1, 4, 16, and 64 learner epochs ago, and the immutable Epoch475 parent. Missing
history slots early in training are filled by Epoch475, keeping the mixture
constant. Each learner epoch is 1,000 optimizer updates.

The online FIFO defaults to 30,000 games. With 2,000 new games per learner
epoch, a game remains eligible for about 15 epochs and is drawn about 3.5 times
in expectation before eviction. This bounds policy lag while retaining useful
replay diversity.

## Optimistic update

The standard AWR/CRR gradient remains intact. By default, optimism is added only
to the online actor component:

```text
g_update = g_standard + online_fraction * beta * (g_online_actor - g_previous)
```

AdamW is the adaptive diagonal mirror map. The previous online actor gradient is
saved in full Accelerate checkpoints, so a resume does not silently reset OMD.
The extra actor-only forward restores PopArt state afterward and therefore does
not update value normalization twice.

Offline optimism is available only as an ablation:

```bash
EPOCH475_OMD_OFFLINE_BETA=1.0
```

The default is zero because the offline corpus is stationary; optimism is meant
to predict movement caused by the evolving online opponent distribution. This
is a practical deep-RL OMD approximation, not a convergence claim for the
non-convex neural policy.

## Throughput calibration

Run a small GPU-backed collection calibration before committing to a large run:

```bash
GOOD_REPLAYS_DIR=/workspace/good_replays \
PREFILL_FILES=5000 \
CALIBRATION_ONLY=1 \
COLLECTOR_DEVICE=gpu \
bash scripts/train_epoch475_omd.sh
```

The result is written as human-readable text to
`$PERSIST_ROOT/logs/throughput-calibration.txt`. It reports games/hour and the
projected time for 250,000 games.

CPU-only collection leaves the 5090 completely available to the learner:

```bash
GOOD_REPLAYS_DIR=/workspace/good_replays \
PREFILL_FILES=5000 \
CALIBRATION_ONLY=1 \
COLLECTOR_DEVICE=cpu \
PERSIST_ROOT=/workspace/epoch475-omd-cpu-calibration \
bash scripts/train_epoch475_omd.sh
```

CPU mode sets `CUDA_VISIBLE_DEVICES=` only on collector processes and activates
`CPUSlidingWindowAttention`; the learner remains GPU-backed. Compare complete
games/hour, CPU utilization, GPU utilization, and learner updates/hour. Kakuna
is large enough that CPU inference may lose despite removing GPU contention.

To enforce a minimum rate:

```bash
MIN_GAMES_PER_HOUR=10000 ... bash scripts/train_epoch475_omd.sh
```

## Full run

For a fresh Vast.ai instance, use the one-command bootstrap. It clones or
safely fast-forwards the checkout, validates the GPU and host resources,
installs FlashAttention if the image lacks it, and starts the foreground
training process:

```bash
export GOOD_REPLAYS_DIR=/workspace/good_replays
export WANDB_API_KEY=...
git clone https://github.com/bsmathers/testing4.git /workspace/testing4
bash /workspace/testing4/scripts/setup_and_launch_epoch475_omd_vast.sh
```

Use a CUDA 12.8+ PyTorch development image so `nvcc` is available if
FlashAttention must be compiled. Training state and the plain-text combined log
are kept under `/workspace/epoch475-omd` by default. The process is intentionally
not daemonized or scheduled; re-run the command to resume after an interruption.

To run without Weights & Biases, set `WANDB_MODE=offline` instead of providing
an API key.

The lower-level launcher remains available when the repository is already set
up:

```bash
GOOD_REPLAYS_DIR=/workspace/good_replays \
PERSIST_ROOT=/workspace/epoch475-omd \
WANDB_API_KEY=... \
bash scripts/train_epoch475_omd.sh
```

The default schedule alternates rather than sharing the GPU:

1. Collect 2,000 fresh games with actor-only BF16 policies.
2. Stop all collectors and release their CUDA contexts.
3. Run one learner epoch of 1,000 updates with batch size 14 and a 50/50
   online/offline mixture.
4. Save complete optimizer, scheduler, RNG, and OMD state, then repeat.

The collection quota is configurable:

```bash
GAMES_PER_LEARNER_EPOCH=3000 ... bash scripts/train_epoch475_omd.sh
```

Collector inference removes critics and target networks, compacts cached
attention history to the causal local window, and defaults to BF16 on CUDA.
Use `COLLECTOR_DTYPE=fp32` or `METAMON_COMPACT_KV_CACHE=0` for controlled
ablations. Keep `COLLECTOR_PROCESSES=1` on a VRAM-limited single-GPU host;
additional processes duplicate both policy weights and KV caches.

The run is resumable from full optimizer state. Start by evaluating checkpoints
after roughly 250k, 1M, and 2M collected games; 250k is an evaluation gate, not
a requirement to pre-collect 250k before learning begins.

OMD currently requires one learner process and `GRAD_ACCUM=1`. Collection can
still use multiple processes through `COLLECTOR_PROCESSES`.

Alternation currently saves a numbered full training-state checkpoint every
epoch so every boundary is safely resumable. These states are not automatically
deleted. Monitor disk use and archive or prune old states only under an explicit
retention policy.
