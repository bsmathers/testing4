# SnorlaxV1 training

SnorlaxV1 is the Ataraxos-inspired, schedule-free continuation of Epoch475.
It is a one-shot remote training run, not a short local experiment.

## Recipe

- Start from Epoch475's actor and representation. Keep its legacy critics only
  for checkpoint compatibility; they are frozen and excluded from optimization.
- Add a fresh scalar head trained on zero-sum game outcomes.
- Alternate 2,000 newly collected games with 1,000 learner updates. Retain a
  5,000-game FIFO, or approximately 2.5 learner epochs of online collection.
- Ramp online sampling linearly from 0% to 50% over the first 100 epochs. The offline corpus is intentionally strong; the ramp is retained rather than treated as a source-quality problem.
- Use batch size 14 with deterministic low-discrepancy source allocation.
- Use clipped PPO, behavior-policy reverse KL, a decaying uniform-policy magnet,
  and outcome-value regression online. Advantage filtering starts disabled and
  ramps over epochs 0-100 to retain the top 75% of online positions by absolute
  advantage (25th-percentile cutoff). Snorlax intentionally omits Ataraxos's
  additional 0.01 absolute-advantage floor so the final gate really retains at
  least 75% of online states. The myopic reference policy is uniform over legal
  actions.
- Use observed-action cross entropy and outcome-value regression offline.
  Pac-exploratory action labels receive weight 0.25; their value labels retain
  full weight.
- At the 50% online cap, total sampling is 20% pac-tauros, 12.5%
  pac-exploratory, 12.5% curated strong-model games, 5% human games, and 50%
  online self-play.
- Online teams are split evenly between `hl_05_26` and the 807-team expert set.
- The opponent league has six TaurosV0/Kakuna rows and six lineage rows:
  current, -1, -4, -16, -64, and immutable Epoch475. Missing early history is
  filled by Epoch475.
- Use grouped AdamW rates of 1e-5 for the representation, 2e-5 for the actor,
  and 1e-4 for the new value head. Hold through 100,000 updates, then linearly
  decay over 400,000 updates to 10% of each rate.
- Apply recent-weighted Schedule-Free averaging with beta 0.8 and r 99.
  Numbered/latest checkpoints, validation, and league play use the averaged
  evaluation iterate x. PPO collection uses a separate atomic
  `latest/behavior_policy.pt` containing the optimizer's gradient-query iterate
  y, so the behavior policy matches the point where Schedule-Free evaluates the
  next PPO gradient.
- Apply a weak L2-SP anchor to the Epoch475 encoder and actor, decaying to zero
  over 100,000 updates.
- The fresh scalar value head trains immediately, but value-loss gradients into
  the shared Epoch475 transformer are ramped from 0% to 100% over the first
  100,000 optimizer updates. This separates critic calibration from early actor
  representation drift while leaving actor/BC gradients unchanged.
- Disable timestep/trajectory dropout for Snorlax PPO. Behavior probabilities
  are recorded in eval mode, so stochastic train-mode dropout would otherwise
  create non-unit PPO ratios before any parameter update. Ataraxos likewise
  uses zero dropout in its move network.
- Online trajectories at most 128 actions are learned in full. For longer
  online games, PPO/value supervision uses the first 128 actions rather than a
  random mid-game slice, preserving the exact preceding transformer history
  used by the collector. Full-game GAE/TD(lambda) targets are computed before
  this prefix restriction. Offline imitation keeps ordinary random 128-step
  windows.
- On the first learner forward after each collection boundary, modules stay in
  eval behavior for that forward only (autograd remains enabled). This prevents
  SigmaReparam's train-only spectral-buffer refresh from changing the policy
  before the first PPO ratio is evaluated; normal train-mode updates resume
  immediately afterward.
- Retain the ten newest 50-epoch milestone policy/full-state checkpoints plus
  the current epoch and previous 64 epochs of each type. This keeps every
  configured lineage offset exact while bounding storage. The rolling latest
  policy and immutable Epoch475 anchor are retained separately.
- Recover the LR/update counter from optimizer state on every full-state
  resume. The alternating launcher starts a fresh learner process each epoch,
  so AMAGO's process-local counter is not a reliable schedule clock.
- After collectors stop, run a one-shot 1,000-step validation at epoch 0 and
  every tenth completed epoch; a validation failure stops the launcher.

The online replay schema stores the exact post-mask behavior log probabilities,
legal-action mask, and behavior-time scalar value for every action. Unresolved
turn-limit/tie trajectories are excluded from this FIFO because they do not have
a valid terminal zero-sum target. A persistent root from an older experiment is
rejected rather than mixed into this run.

## Permanent configuration

- `metamon/rl/configs/online_runs/snorlax_v1.yaml`
- `metamon/rl/configs/datasets/snorlax_v1.yaml`
- `metamon/rl/configs/team_sets/snorlax_v1.yaml`
- `metamon/rl/configs/opponent_pools/snorlax_v1.yaml`
- `metamon/rl/configs/opponent_pools/snorlax_v1_val.yaml`
- `metamon/rl/configs/training/snorlax_v1.gin`

## Launch

```bash
export WANDB_API_KEY=...
git clone https://github.com/bsmathers/testing4.git /workspace/testing4
bash /workspace/testing4/scripts/setup_and_launch_snorlax_v1_vast.sh
```

The launcher downloads the pinned `gen1ou.zip` from
`bsmathers/scraped_replays_parsed` and safely extracts it under
`GOOD_REPLAYS_DIR` (default `/workspace/good_replays`). An already-populated
trajectory directory is reused. The archive URL, revision, checksum, and local
paths can be overridden with the corresponding `STRONG_REPLAYS_*` and
`GOOD_REPLAYS_DIR` environment variables.

The launcher runs in the foreground and resumes only from complete
epoch-boundary model/optimizer states. The expensive first-run Snorlax smoke
block remains in the launcher source for manual debugging but is commented out
and does not execute during production launch. Policy and optimizer-state checkpoints
are published atomically; an orphan policy from an interrupted save is removed
before resumed collection. Use a fresh `PERSIST_ROOT` for SnorlaxV1.

## Overlapped collection and learning

With `SNORLAX_OVERLAP_COLLECTION=1` (the default in `train_snorlax_v1.sh` and
`resume_snorlax_v1_vast.sh`), the persistent learner collects round `e + 1` in
background collector processes while it trains epoch `e`. Round `e + 1` uses
the Schedule-Free y iterate that epoch `e` starts from, written to a versioned
`latest/behavior_policy_round_<r>.pt` so later publishes cannot race it. Each
epoch then costs about `max(collection, learning)` instead of their sum. Set
`SNORLAX_OVERLAP_COLLECTION=0` for the strictly alternating loop.

Training parameters are unchanged (batch 14, 1,000 updates, 5,000-game FIFO).
The trade-off is one more epoch of data age: fresh games are first used about
one epoch after their behavior policy, and the oldest FIFO data is about three
epochs old instead of two. PPO ratios stay exact because behavior log
probabilities are recorded. Each replay now records its behavior policy's
optimizer update (`ataraxos_behavior_update`), and the learner logs Share,
Clip Fraction, Ratio Mean, and Behavior KL by data age under
`Ataraxos Data Age/*`. If old buckets clip heavily, shrink the FIFO
(`dset_max_size`/`dset_min_size` in the run config together with
`ONLINE_FIFO_SIZE` and `PREFILL_FILES`).

Ingest (FIFO rescan and eviction, manifest publish, advantage threshold) still
happens only between epochs, so replay-loader workers never see a file from an
unfinished round. Only one round is ever in flight. A per-round ledger in the
log directory (`.collection-round-<r>-baseline.txt` and `-complete`) makes
restarts safe: on resume, the replays of any round without a completion marker
are deleted and that round is recollected, while completed rounds are reused.
The first epoch of a fresh or resumed run collects its round synchronously.
Validation runs alongside the background round (it never writes to the FIFO).

`[LearnerProfile]` lines show the background round's progress, a failed
collector stops training immediately, and each round logs
`collection/Learner Wait Seconds`. Near-zero wait means learning is now the
bottleneck; a large wait means collection still is.

CPU placement is opt-in. `SNORLAX_LEARNER_CPUS` (for example `0-7`) runs the
learner and its replay-loader workers under `taskset`; collector and
validation children are pinned to `SNORLAX_COLLECTOR_CPUS`, or to every other
CPU when that is unset. Size the learner set so `[LearnerProfile] data_wait`
stays below about 5% and give the rest to the simulators. Collector and learner
kernels share the GPU; if the collection rate drops sharply while the learner
runs, CUDA MPS (if the host allows it) lets their kernels run concurrently.
