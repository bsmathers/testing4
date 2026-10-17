# SnorlaxV1 training

SnorlaxV1 is the Ataraxos-inspired, schedule-free continuation of Epoch475.
It is a one-shot remote training run, not a short local experiment.

## Recipe

- Start from Epoch475's actor and representation. Keep its legacy critics only
  for checkpoint compatibility; they are frozen and excluded from optimization.
- Add a fresh scalar head trained on zero-sum game outcomes.
- Alternate 2,000 newly collected games with 1,000 learner updates. Retain a
  5,000-game FIFO, or approximately 2.5 learner epochs of online collection.
- Ramp online sampling linearly from 0% to 50% over the first 100 epochs.
- Use batch size 14 with deterministic low-discrepancy source allocation.
- Use clipped PPO, behavior-policy reverse KL, a decaying uniform-policy magnet,
  outcome-value regression, and top-quartile absolute-advantage filtering for
  online positions. The myopic reference policy is uniform over legal actions.
- Use observed-action cross entropy and outcome-value regression offline.
  Pac-exploratory action labels receive weight 0.25; their value labels retain
  full weight.
- At the 50% online cap, total sampling is 20% pac-tauros, 15%
  pac-exploratory, 10% curated strong-model games, 5% human games, and 50%
  online self-play.
- Online teams are split evenly between `hl_05_26` and the 807-team expert set.
- The opponent league has six TaurosV0/Kakuna rows and six lineage rows:
  current, -1, -4, -16, -64, and immutable Epoch475. Missing early history is
  filled by Epoch475.
- Use grouped AdamW rates of 1e-5 for the representation, 2e-5 for the actor,
  and 1e-4 for the new value head. Hold through 100,000 updates, then linearly
  decay over 400,000 updates to 10% of each rate.
- Apply recent-weighted Schedule-Free averaging with beta 0.8 and r 99. Raw
  checkpoints, validation, collection, and league play use the averaged model.
- Apply a weak L2-SP anchor to the Epoch475 encoder and actor, decaying to zero
  over 100,000 updates.
- Retain every 50th numbered policy/full-state checkpoint and the ten most
  recent checkpoints of each type. The rolling latest policy and immutable
  Epoch475 anchor are retained separately.

The online replay schema stores the exact post-mask behavior log probabilities,
legal-action mask, and behavior-time scalar value for every action. A persistent
root from an older experiment is rejected rather than mixed into this run.

## Permanent configuration

- `metamon/rl/configs/online_runs/snorlax_v1.yaml`
- `metamon/rl/configs/datasets/snorlax_v1.yaml`
- `metamon/rl/configs/team_sets/snorlax_v1.yaml`
- `metamon/rl/configs/opponent_pools/snorlax_v1.yaml`
- `metamon/rl/configs/opponent_pools/snorlax_v1_val.yaml`
- `metamon/rl/configs/training/snorlax_v1.gin`

## Launch

```bash
export GOOD_REPLAYS_DIR=/workspace/good_replays
export WANDB_API_KEY=...
git clone https://github.com/bsmathers/testing4.git /workspace/testing4
bash /workspace/testing4/scripts/setup_and_launch_snorlax_v1_vast.sh
```

The launcher runs in the foreground and resumes only from complete
epoch-boundary model/optimizer states. Policy and optimizer-state checkpoints
are published atomically; an orphan policy from an interrupted save is removed
before resumed collection. Use a fresh `PERSIST_ROOT` for SnorlaxV1.
