# Epoch475 Ataraxos training

The OMD experiment has been retired. The compatibility launcher
`scripts/train_epoch475_omd.sh` now delegates to the corrected Ataraxos run.

## Experiment

- Start from Epoch475's actor and representation.
- Add a fresh scalar value head trained on zero-sum win/loss outcomes.
- Keep exactly 5,000 online games; collect 2,000 new games between learner epochs.
- Run 1,000 optimizer updates per learner epoch with batch size 14: exactly seven
  online and seven offline trajectories.
- Online loss: clipped PPO, reverse KL to the exact behavior policy, reverse KL
  to uniform over legal actions, and filtered scalar value loss.
- Retain the top quartile by absolute GAE, with a 0.01 floor. The threshold is
  recomputed from the complete FIFO before learning.
- Offline loss: observed-action cross entropy plus outcome-value regression.
- The trainee collects at temperature 1.0. Opponent temperatures remain part of
  the opponent population.

Online replay files contain full post-mask behavior log probabilities, the exact
legal mask, and the behavior-time scalar value for every action. The launcher
refuses an old FIFO that lacks these fields.

The offline half is sampled as 40% pac-tauros, 30% human replays, and 30%
curated strong-model trajectories. Online teams are split evenly between the
strong human set and the 807-team set. Opponents are split evenly between
TaurosV0/Kakuna and the six-slot Epoch475 lineage population: current, -1, -4,
-16, -64, and the original checkpoint.

## Launch

```bash
export GOOD_REPLAYS_DIR=/workspace/good_replays
export WANDB_API_KEY=...
git clone https://github.com/bsmathers/testing4.git /workspace/testing4
bash /workspace/testing4/scripts/setup_and_launch_epoch475_omd_vast.sh
```

The setup script remains under its historical filename for compatibility, but
launches `scripts/train_epoch475_ataraxos.sh`. It runs in the foreground and
alternates collection with learning. Re-running the same command resumes from
the latest complete epoch-boundary checkpoint.

Use a fresh `PERSIST_ROOT` when moving from the abandoned OMD run because those
replays do not contain behavior-policy metadata.
