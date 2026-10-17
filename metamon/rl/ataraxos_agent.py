"""Ataraxos-style actor and loss for the Epoch475 continuation.

The pretrained actor/backbone are retained.  A fresh scalar value head learns
the zero-sum game outcome, avoiding the incompatible shaped-reward critics in
the Epoch475 checkpoint.
"""

from __future__ import annotations

import itertools
import gin
import torch
import torch.nn as nn
from amago.agent import MultiTaskAgent

from metamon.rl import ataraxos_keys as keys


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def _trace(delta: torch.Tensor, done: torch.Tensor, valid: torch.Tensor, lam: float) -> torch.Tensor:
    """Backward lambda trace over (B, T, 1), respecting terminals/padding."""
    out = torch.zeros_like(delta)
    carry = torch.zeros_like(delta[:, 0])
    for t in range(delta.shape[1] - 1, -1, -1):
        carry = delta[:, t] + lam * (1.0 - done[:, t]) * carry
        carry = carry * valid[:, t]
        out[:, t] = carry
    return out


@gin.configurable
class AtaraxosAgent(MultiTaskAgent):
    """Epoch475 actor with source-separated offline and online objectives."""

    def __init__(
        self,
        *args,
        clip_range: float = 0.2,
        behavior_kl_coef: float = 0.1,
        value_coef: float = 1.0,
        offline_policy_coef: float = 1.0,
        offline_value_coef: float = 1.0,
        td_lambda: float = 0.8,
        gae_lambda: float = 0.5,
        magnet_coef: float = 0.05,
        magnet_decay: float = 0.3,
        magnet_floor: float = 0.001,
        magnet_ceil: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ataraxos_value = nn.Linear(self.traj_encoder.emb_dim, 1)
        nn.init.zeros_(self.ataraxos_value.weight)
        nn.init.zeros_(self.ataraxos_value.bias)
        self.clip_range = float(clip_range)
        self.behavior_kl_coef = float(behavior_kl_coef)
        self.value_coef = float(value_coef)
        self.offline_policy_coef = float(offline_policy_coef)
        self.offline_value_coef = float(offline_value_coef)
        self.td_lambda = float(td_lambda)
        self.gae_lambda = float(gae_lambda)
        if self.td_lambda != 0.8 or self.gae_lambda != 0.5:
            raise ValueError(
                "Ataraxos replay traces are precomputed with td_lambda=0.8 "
                "and gae_lambda=0.5"
            )
        self.magnet_coef = float(magnet_coef)
        self.magnet_decay = float(magnet_decay)
        self.magnet_floor = float(magnet_floor)
        self.magnet_ceil = float(magnet_ceil)
        self.register_buffer("_ataraxos_step", torch.zeros((), dtype=torch.long))

    @property
    def trainable_params(self):
        # Epoch475's old shaped-reward critics deliberately receive no updates.
        return itertools.chain(
            self.tstep_encoder.parameters(),
            self.traj_encoder.parameters(),
            self.actor.parameters(),
            self.ataraxos_value.parameters(),
        )

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Old Epoch475 checkpoints predate the outcome-value head.  A zero head is
        # an unbiased baseline and is learned immediately from game outcomes.
        state_dict = dict(state_dict)
        current = self.state_dict()
        for name in ("ataraxos_value.weight", "ataraxos_value.bias", "_ataraxos_step"):
            state_dict.setdefault(name, current[name])
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def get_actions_with_metadata(self, obs, rl2s, time_idxs, hidden_state=None, sample=True):
        """Sample actions and return the exact behavior distribution/value."""
        s, hidden_state = self.get_state_embedding(
            obs=obs, rl2s=rl2s, time_idxs=time_idxs, hidden_state=hidden_state
        )
        dist = self.actor(
            s, straight_from_obs={k: obs[k] for k in self.pass_obs_keys_to_actor}
        )
        actions = dist.sample() if sample else torch.argmax(dist.probs, dim=-1, keepdim=True)
        actions = actions[..., -1, :]
        probs = dist.probs[..., -1, :]
        log_probs = torch.where(
            probs > 0, probs.clamp_min(1e-30).log(), torch.full_like(probs, -70.0)
        )
        value = torch.tanh(self.ataraxos_value(s[..., -1, :]))
        return actions.to(torch.int64), hidden_state, log_probs[..., -1, :], value

    def get_actions(self, obs, rl2s, time_idxs, hidden_state=None, sample=True):
        actions, hidden_state, _, _ = self.get_actions_with_metadata(
            obs, rl2s, time_idxs, hidden_state=hidden_state, sample=sample
        )
        return actions, hidden_state

    def forward(self, batch, log_step: bool):
        self.update_info = {}
        required = (
            keys.SOURCE, keys.BEHAVIOR_LOG_PROBS, keys.BEHAVIOR_VALUE,
            keys.BEHAVIOR_VALID, keys.BEHAVIOR_LEGAL, keys.ADVANTAGE,
            keys.VALUE_TARGET, keys.OUTCOME, keys.ADV_THRESHOLD,
        )
        missing = [k for k in required if k not in batch.obs]
        if missing:
            raise RuntimeError(f"Ataraxos batch missing replay metadata: {missing}")

        o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
        s, _ = self.traj_encoder(seq=o, time_idxs=batch.time_idxs, hidden_state=None)
        dist = self.actor(
            s, straight_from_obs={k: batch.obs[k] for k in self.pass_obs_keys_to_actor}
        )
        probs = dist.probs[..., -1, :]
        log_probs = torch.where(
            probs > 0, probs.clamp_min(1e-30).log(), torch.full_like(probs, -70.0)
        )
        values = torch.tanh(self.ataraxos_value(s))

        T = batch.actions.shape[1]
        chosen = batch.actions.argmax(dim=-1)
        cur_lp = log_probs[:, :T].gather(-1, chosen.unsqueeze(-1))
        old_all_lp = batch.obs[keys.BEHAVIOR_LOG_PROBS][:, :T].float()
        old_lp = old_all_lp.gather(-1, chosen.unsqueeze(-1))
        source = batch.obs[keys.SOURCE][:, :T].bool()
        behavior_valid = batch.obs[keys.BEHAVIOR_VALID][:, :T].bool()
        padded = (batch.rl2s[:, 1 : T + 1] == self.pad_val).all(-1, keepdim=True)
        valid = (~padded).float()
        online = source & behavior_valid & valid.bool()
        offline_states = (~source) & valid.bool()

        # These checks guard the two invariants that make PPO's importance
        # ratio meaningful: the stored behavior distribution is normalized,
        # and the recorded action was legal under that exact distribution.
        behavior_legal = batch.obs[keys.BEHAVIOR_LEGAL][:, :T].bool()
        old_probs = old_all_lp.exp()
        online_rows = online.expand_as(old_probs)
        if not torch.isfinite(old_all_lp[online_rows]).all():
            raise RuntimeError("Ataraxos behavior log probabilities are non-finite")
        old_mass = old_probs.sum(-1, keepdim=True)
        if not torch.allclose(
            old_mass[online], torch.ones_like(old_mass[online]), atol=2e-3, rtol=0.0
        ):
            raise RuntimeError("Ataraxos behavior probabilities do not sum to one")
        if (old_probs[online_rows & ~behavior_legal] > 1e-8).any():
            raise RuntimeError("Ataraxos behavior policy assigns mass to an illegal action")
        chosen_legal = behavior_legal.gather(-1, chosen.unsqueeze(-1))
        if not chosen_legal[online].all():
            raise RuntimeError("Ataraxos replay contains an illegal recorded action")

        # These were computed over the full game before the replay loader took
        # a random sequence window, so the window boundary is not a false
        # terminal for GAE or the TD(lambda) value target.
        advantages = batch.obs[keys.ADVANTAGE][:, :T].float()
        returns = batch.obs[keys.VALUE_TARGET][:, :T].float()

        threshold = batch.obs[keys.ADV_THRESHOLD][:, :T].float()
        selected = online & (advantages.abs() >= threshold)
        if not selected.any():
            raise RuntimeError("Ataraxos online batch has no transition above advantage threshold")

        ratio = torch.exp((cur_lp - old_lp).clamp(-20.0, 20.0))
        unclipped = advantages * ratio
        clipped = advantages * ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
        ppo = -_masked_mean(torch.minimum(unclipped, clipped), selected)

        behavior_kl_step = (probs[:, :T] * (log_probs[:, :T] - old_all_lp)).sum(-1, keepdim=True)
        behavior_kl = _masked_mean(behavior_kl_step, selected)
        legal = ~batch.obs["illegal_actions"][:, :T].bool()
        legal_count = legal.sum(-1, keepdim=True).clamp(min=1)
        uniform_logp = -legal_count.float().log()
        magnet_step = torch.where(
            legal,
            probs[:, :T] * (log_probs[:, :T] - uniform_logp),
            torch.zeros_like(probs[:, :T]),
        ).sum(-1, keepdim=True)
        magnet_kl = _masked_mean(magnet_step, selected)
        online_value = _masked_mean((values[:, :T] - returns.detach()).square(), selected)

        offline_actions = offline_states & (
            ~batch.obs["missing_action_mask"][:, :T].bool()
        )
        offline_lp = cur_lp
        offline_ce = -_masked_mean(offline_lp, offline_actions)
        outcome = batch.obs[keys.OUTCOME][:, :T].float()
        offline_value = _masked_mean(
            (values[:, :T] - outcome).square(), offline_states
        )

        step = int(self._ataraxos_step.item())
        temperature = self.magnet_coef / ((step + 1) ** self.magnet_decay)
        temperature = min(max(temperature, self.magnet_floor), self.magnet_ceil)
        online_loss = (
            ppo + self.behavior_kl_coef * behavior_kl
            + temperature * magnet_kl + self.value_coef * online_value
        )
        offline_loss = (
            self.offline_policy_coef * offline_ce
            + self.offline_value_coef * offline_value
        )
        loss = 0.5 * online_loss + 0.5 * offline_loss
        self._ataraxos_step.add_(1)

        if log_step:
            self.update_info = {
                "Ataraxos/Online Loss": online_loss.detach(),
                "Ataraxos/Offline Loss": offline_loss.detach(),
                "Ataraxos/PPO Loss": ppo.detach(),
                "Ataraxos/Behavior KL": behavior_kl.detach(),
                "Ataraxos/Uniform KL": magnet_kl.detach(),
                "Ataraxos/Value Loss": online_value.detach(),
                "Ataraxos/Offline CE": offline_ce.detach(),
                "Ataraxos/Magnet Coef": float(temperature),
                "Ataraxos/Clip Fraction": _masked_mean(((ratio - 1).abs() > self.clip_range).float(), selected).detach(),
                "Ataraxos/Selected Fraction": selected.float().sum() / online.float().sum().clamp(min=1),
                "Ataraxos/Ratio Mean": _masked_mean(ratio, selected).detach(),
            }
        return loss
