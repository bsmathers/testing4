# Historical Epoch475 training name

This experiment is now named **SnorlaxV1**. The current recipe and launch
instructions are in `docs/snorlax_v1_training.md`; historical launch filenames
remain as compatibility entry points.

## Archived description

The OMD experiment has been retired. The compatibility launcher
`scripts/train_epoch475_omd.sh` now delegates to the corrected Ataraxos run.

## Experiment

- Start from Epoch475's actor and representation.
- Add a fresh scalar value head trained on zero-sum win/loss outcomes.
- Keep exactly 5,000 online games; collect 2,000 new games between learner epochs.
- Run 1,000 optimizer updates per learner epoch with batch size 14. Online
  sampling ramps linearly from 0% at epoch 0 to 50% at epoch 100; deterministic
  low-discrepancy sampling avoids noisy source counts early in the ramp.
- Online loss: clipped PPO, reverse KL to the exact behavior policy, reverse KL
  to uniform over legal actions, and filtered scalar value loss.
- Preserve Epoch475's aggressive-shaped reward history as an RL2 input while
  computing PPO advantages and scalar-value targets from terminal zero-sum
  outcomes only.
- Retain the top quartile by absolute GAE, with a 0.01 floor. The threshold is
  recomputed from the complete FIFO before learning.
- Offline loss: observed-action cross entropy plus outcome-value regression.
  Pac-exploratory action imitation is weighted by 0.25 while its value target
  keeps full weight.
- Use separate AdamW rates for the pretrained backbone (1e-5), actor (2e-5),
  and fresh scalar value head (1e-4). Hold them through the 100-epoch online
  ramp, then linearly decay over 400 epochs to 10% floors.
- Apply recent-weighted Schedule-Free averaging on top of those unchanged
  rates. The fast iterate retains AdamW momentum 0.9; the x/y interpolation is
  beta=0.8 and polynomial averaging uses r=99. Under constant-gradient drift,
  the deployed average retains 100/101 (99.01%) of the fast iterate's speed,
  while the gradient-query point retains 99.21%, so averaging does not act as
  a materially smaller learning rate. Its mean lag is t/101 updates: roughly
  one learner epoch at update 100k and five at update 500k. Validation,
  collectors, and raw checkpoints always use the averaged x iterate.
  The high r is an intentional non-stationary-RL modification, not the paper's
  long-horizon r=0/r=1 default.
- Apply a weak L2-SP anchor to Epoch475's encoder/actor parameters and linearly
  remove it over the first 100,000 updates. The uniform-policy magnet decays by
  cumulative online trajectory mass rather than offline learner updates.
- The trainee collects at temperature 1.0. Opponent temperatures remain part of
  the opponent population.

Online replay files contain full post-mask behavior log probabilities, the exact
legal mask, and the behavior-time scalar value for every action. The launcher
refuses an old FIFO that lacks these fields.

Built-in validation samples TaurosV0, Kakuna, and the frozen Epoch475 baseline
across both the strong-human and 807-team sets.

Within the offline pool, sampling is 40% pac-tauros, 30% pac-exploratory, 20%
curated strong-model trajectories, and 10% human replays. At the 50% online
cap, these are 20%, 15%, 10%, and 5% of all trajectories. Before the cap, the
same 40/30/20/10 proportions are renormalized over the larger offline share.
Online teams are split evenly between the
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

The historical setup script remains for compatibility, but now launches
`scripts/train_snorlax_v1.sh`. It runs in the foreground and
alternates collection with learning. Re-running the same command resumes from
the latest complete epoch-boundary checkpoint.

Use a fresh `PERSIST_ROOT` when moving from the abandoned OMD run, an earlier
binary-reward Ataraxos draft, or the pre-Schedule-Free optimizer. Those runs
either contain incompatible replay history or optimizer/checkpoint state.
