"""Ataraxos-style actor and loss for the Epoch475 continuation.

The pretrained actor/backbone are retained.  A fresh scalar value head learns
the zero-sum game outcome, avoiding the incompatible shaped-reward critics in
the Epoch475 checkpoint.
"""

from __future__ import annotations

import copy
import itertools
import math
import os
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
        l2sp_coef: float = 1e-4,
        l2sp_decay_steps: int = 100_000,
        bootstrap_kl_coef: float = 1.0,
        bootstrap_damping_steps: int = 100_000,
        offline_policy_start_coef: float = 0.1,
        value_backbone_warmup_steps: int = 100_000,
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
        self.l2sp_coef = float(l2sp_coef)
        self.l2sp_decay_steps = int(l2sp_decay_steps)
        self.bootstrap_kl_coef = float(bootstrap_kl_coef)
        self.bootstrap_damping_steps = int(bootstrap_damping_steps)
        self.offline_policy_start_coef = float(offline_policy_start_coef)
        self.value_backbone_warmup_steps = int(value_backbone_warmup_steps)
        self._l2sp_anchor: dict[str, torch.Tensor] | None = None
        # Stored in a plain tuple so PyTorch does not include a second complete
        # policy in Snorlax checkpoints. These modules are reconstructed from
        # the immutable bootstrap after every fresh load/resume.
        self._bootstrap_policy_modules = None
        self.register_buffer("_ataraxos_step", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "_ataraxos_online_mass", torch.zeros((), dtype=torch.float32)
        )
        # The inherited shaped-reward critics remain in the state dict only so
        # Epoch475 checkpoints stay loadable. They are absent from this loss and
        # must not allocate gradients or participate in distributed reduction.
        trainable_modules = (
            self.tstep_encoder,
            self.traj_encoder,
            self.actor,
            self.ataraxos_value,
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    @property
    def trainable_params(self):
        # Epoch475's old shaped-reward critics deliberately receive no updates.
        return itertools.chain(
            self.tstep_encoder.parameters(),
            self.traj_encoder.parameters(),
            self.actor.parameters(),
            self.ataraxos_value.parameters(),
        )

    def soft_sync_targets(self):
        # This objective never reads the inherited target actor/critics. AMAGO
        # invokes this hook on every optimizer update, and copying those frozen
        # networks would consume bandwidth without changing any Snorlax loss or
        # inference output.
        return None

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Old Epoch475 checkpoints predate the outcome-value head.  A zero head is
        # an unbiased baseline and is learned immediately from game outcomes.
        state_dict = dict(state_dict)
        current = self.state_dict()
        ataraxos_names = {
            "ataraxos_value.weight",
            "ataraxos_value.bias",
            "_ataraxos_step",
            "_ataraxos_online_mass",
        }
        present = ataraxos_names.intersection(state_dict)
        if present and present != ataraxos_names:
            raise RuntimeError(
                "Checkpoint has incomplete Ataraxos state: "
                + ", ".join(sorted(ataraxos_names - present))
            )
        # Complete absence is the one supported legacy case: the immutable
        # Epoch475 bootstrap predates the entire Ataraxos extension.
        if not present:
            for name in ataraxos_names:
                state_dict[name] = current[name]
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    @staticmethod
    def _is_l2sp_parameter(name: str) -> bool:
        return name.startswith(("tstep_encoder.", "traj_encoder.", "actor."))

    def _capture_l2sp_anchor(self, state_dict=None) -> None:
        """Keep the immutable Epoch475 policy anchor out of saved checkpoints."""
        anchor = {}
        missing = []
        for name, parameter in self.named_parameters():
            if not self._is_l2sp_parameter(name):
                continue
            source = parameter.detach() if state_dict is None else state_dict.get(name)
            if source is None:
                missing.append(name)
                continue
            anchor[name] = source.detach().to(
                device=parameter.device, dtype=parameter.dtype
            ).clone()
        if missing:
            raise RuntimeError(
                "Epoch475 L2-SP anchor is missing policy parameters: "
                + ", ".join(missing[:5])
            )
        self._l2sp_anchor = anchor

    def _capture_bootstrap_policy(self, state_dict) -> None:
        modules = (
            copy.deepcopy(self.tstep_encoder),
            copy.deepcopy(self.traj_encoder),
            copy.deepcopy(self.actor),
        )
        for prefix, module in zip(
            ("tstep_encoder.", "traj_encoder.", "actor."), modules
        ):
            module_state = {
                name[len(prefix) :]: tensor
                for name, tensor in state_dict.items()
                if name.startswith(prefix)
            }
            module.load_state_dict(module_state, strict=True)
            module.requires_grad_(False)
            module.eval()
        self._bootstrap_policy_modules = modules

    def _bootstrap_damping(self, step: int) -> float:
        """Cosine protection schedule: one at bootstrap and exactly zero later."""
        if self.bootstrap_damping_steps <= 0 or step >= self.bootstrap_damping_steps:
            return 0.0
        progress = max(float(step), 0.0) / float(self.bootstrap_damping_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def on_checkpoint_loaded(self, is_resume: bool = False):
        self._ataraxos_host_step = int(self._ataraxos_step.item())
        # Collectors never evaluate the training loss and should not spend GPU
        # memory on a second frozen copy of the policy parameters.
        if os.environ.get("METAMON_RUN_MODE") != "learn" or (
            self.l2sp_coef <= 0.0 and self.bootstrap_kl_coef <= 0.0
        ):
            self._l2sp_anchor = None
            self._bootstrap_policy_modules = None
            return
        if not is_resume:
            anchor_state = self.state_dict()
        else:
            anchor_path = os.environ.get("ATARAXOS_L2SP_ANCHOR_PATH")
            if not anchor_path or not os.path.isfile(anchor_path):
                raise RuntimeError(
                    "Full-state Ataraxos resume requires ATARAXOS_L2SP_ANCHOR_PATH"
                )
            try:
                anchor_state = torch.load(
                    anchor_path, map_location="cpu", weights_only=True
                )
            except TypeError:
                anchor_state = torch.load(anchor_path, map_location="cpu")
        if self.l2sp_coef > 0.0:
            self._capture_l2sp_anchor(anchor_state)
        if self.bootstrap_kl_coef > 0.0:
            self._capture_bootstrap_policy(anchor_state)

    def _current_l2sp_scale(self) -> float:
        step = getattr(self, "_ataraxos_host_step", None)
        if step is None:
            step = int(self._ataraxos_step.item())
        if self.l2sp_decay_steps <= 0:
            return 0.0
        # All bootstrap protection follows the same smooth schedule and reaches
        # exactly zero, leaving the mature policy unconstrained.
        return self._bootstrap_damping(step)

    @torch.no_grad()
    def add_l2sp_gradients(self) -> None:
        """Add the exact L2-SP gradient after backward and before clipping.

        Building the full parameter-distance penalty into the autograd graph
        duplicated hundreds of large elementwise kernels every update.  The
        analytic gradient is simply coef * scale * (parameter - anchor).
        """
        scale = getattr(self, "_pending_l2sp_scale", None)
        if scale is None:
            scale = self._current_l2sp_scale()
        if scale <= 0.0 or self.l2sp_coef <= 0.0:
            return
        if self._l2sp_anchor is None:
            raise RuntimeError("Ataraxos learner has no Epoch475 L2-SP anchor")
        parameters = []
        anchors = []
        gradients = []
        for name, parameter in self.named_parameters():
            if self._is_l2sp_parameter(name) and parameter.grad is not None:
                parameters.append(parameter)
                anchors.append(self._l2sp_anchor[name])
                gradients.append(parameter.grad)
        if not parameters:
            return
        coefficient = self.l2sp_coef * scale
        if parameters[0].is_cuda:
            # Avoid allocating a full model-sized temporary ``parameter-anchor``
            # list on every update. Two fused multi-tensor additions implement
            # the same analytic gradient while keeping allocator and memory-
            # bandwidth pressure out of the hot path.
            torch._foreach_add_(gradients, parameters, alpha=coefficient)
            torch._foreach_add_(gradients, anchors, alpha=-coefficient)
        else:
            for gradient, parameter, anchor in zip(
                gradients, parameters, anchors
            ):
                gradient.add_(parameter - anchor, alpha=coefficient)

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
        # Collector inference defaults to BF16.  Logging BF16 probabilities in
        # BF16 log-space introduces enough rounding error that reconstructing
        # them can miss unit mass by >0.2%, even for a healthy masked softmax.
        # ``Categorical.sample`` is effectively proportional to these stored
        # probabilities, so FP32 renormalization records its exact effective
        # behavior distribution without changing the sampled action.
        probs = dist.probs[..., -1, :].float()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
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
            keys.OFFLINE_ACTOR_WEIGHT,
        )
        missing = [k for k in required if k not in batch.obs]
        if missing:
            raise RuntimeError(f"Ataraxos batch missing replay metadata: {missing}")
        if batch.actions.ndim != 3:
            raise RuntimeError(
                f"Ataraxos actions must be (batch, time, action), got {batch.actions.shape}"
            )
        if batch.rl2s.ndim != 3 or batch.time_idxs.ndim != 3:
            raise RuntimeError(
                "Ataraxos RL2/time tensors must retain batch and time axes: "
                f"rl2={batch.rl2s.shape}, time={batch.time_idxs.shape}"
            )
        batch_size, action_steps, action_dim = batch.actions.shape
        if action_dim != self.action_dim:
            raise RuntimeError(
                f"Ataraxos action dimension is {action_dim}, expected {self.action_dim}"
            )
        if batch.rl2s.shape[0] != batch_size or batch.rl2s.shape[1] < action_steps + 1:
            raise RuntimeError(
                "Ataraxos state/action alignment is invalid: "
                f"actions={batch.actions.shape}, rl2={batch.rl2s.shape}"
            )
        for name in required:
            tensor = batch.obs[name]
            if tensor.shape[0] != batch_size or tensor.shape[1] < action_steps + 1:
                raise RuntimeError(
                    f"Ataraxos field {name} is not state-aligned: {tensor.shape}"
                )

        o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
        s, _ = self.traj_encoder(seq=o, time_idxs=batch.time_idxs, hidden_state=None)
        dist = self.actor(
            s, straight_from_obs={k: batch.obs[k] for k in self.pass_obs_keys_to_actor}
        )
        probs = dist.probs[..., -1, :]
        log_probs = torch.where(
            probs > 0, probs.clamp_min(1e-30).log(), torch.full_like(probs, -70.0)
        )

        # The scalar outcome head is new, while the actor/trunk come from the
        # already-strong Epoch475 policy. Let the critic calibrate on the frozen
        # representation first, then gradually allow value-loss gradients into
        # the shared transformer. Head gradients are unaffected by this scaling.
        step = getattr(self, "_ataraxos_host_step", None)
        if step is None:
            step = int(self._ataraxos_step.item())
        damping = self._bootstrap_damping(step)
        if self.value_backbone_warmup_steps <= 0:
            value_backbone_scale = 1.0
        else:
            value_backbone_scale = 1.0 - damping
        value_features = (
            s.detach() + value_backbone_scale * (s - s.detach())
        )
        values = torch.tanh(self.ataraxos_value(value_features))

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

        # Replay loading already validates these invariants, and the launcher
        # fully validates the FIFO before training. Repeating four device-to-host
        # assertions every update serializes the CUDA pipeline. Keep an opt-in
        # diagnostic path for corruption investigations.
        if os.environ.get("METAMON_STRICT_BATCH_CHECKS") == "1":
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
        offline_actor_weight = batch.obs[keys.OFFLINE_ACTOR_WEIGHT][
            :, :T
        ].float()
        offline_ce = -_masked_mean(
            offline_actor_weight * offline_lp, offline_actions
        )
        outcome = batch.obs[keys.OUTCOME][:, :T].float()
        offline_value = _masked_mean(
            (values[:, :T] - outcome).square(), offline_states
        )

        bootstrap_kl = torch.zeros((), device=probs.device)
        bootstrap_kl_coefficient = self.bootstrap_kl_coef * damping
        if bootstrap_kl_coefficient > 0.0:
            if self._bootstrap_policy_modules is None:
                raise RuntimeError("Ataraxos learner has no model-475 policy anchor")
            bootstrap_tstep, bootstrap_traj, bootstrap_actor = (
                self._bootstrap_policy_modules
            )
            with torch.no_grad():
                bootstrap_o = bootstrap_tstep(obs=batch.obs, rl2s=batch.rl2s)
                bootstrap_s, _ = bootstrap_traj(
                    seq=bootstrap_o,
                    time_idxs=batch.time_idxs,
                    hidden_state=None,
                )
                bootstrap_dist = bootstrap_actor(
                    bootstrap_s,
                    straight_from_obs={
                        key: batch.obs[key] for key in self.pass_obs_keys_to_actor
                    },
                )
                bootstrap_probs = bootstrap_dist.probs[..., -1, :].float()
                bootstrap_log_probs = torch.where(
                    bootstrap_probs > 0,
                    bootstrap_probs.clamp_min(1e-30).log(),
                    torch.full_like(bootstrap_probs, -70.0),
                )
            bootstrap_kl_step = (
                bootstrap_probs[:, :T]
                * (bootstrap_log_probs[:, :T] - log_probs[:, :T])
            ).sum(-1, keepdim=True)
            bootstrap_kl = _masked_mean(bootstrap_kl_step, valid.bool())

        temperature = (
            self.magnet_coef
            / (self._ataraxos_online_mass.float() + 1.0).pow(self.magnet_decay)
        ).clamp(self.magnet_floor, self.magnet_ceil)
        online_loss = (
            ppo + self.behavior_kl_coef * behavior_kl
            + temperature * magnet_kl + self.value_coef * online_value
        )
        effective_offline_policy_coef = self.offline_policy_coef * (
            1.0 - (1.0 - self.offline_policy_start_coef) * damping
        )
        offline_loss = (
            effective_offline_policy_coef * offline_ce
            + self.offline_value_coef * offline_value
        )
        # Weight by the actual trajectory composition of this batch. Production
        # Snorlax ramps from 25/75 to a 50/50 online/offline source mixture, while
        # ablations may use other source ratios.
        batch_online_fraction = source[:, 0, 0].float().mean()
        loss = (
            batch_online_fraction * online_loss
            + (1.0 - batch_online_fraction) * offline_loss
            + bootstrap_kl_coefficient * bootstrap_kl
        )
        l2sp_scale = self._current_l2sp_scale()
        self._pending_l2sp_scale = l2sp_scale
        l2sp_penalty = torch.zeros((), device=loss.device)
        if log_step and l2sp_scale > 0.0 and self.l2sp_coef > 0.0:
            if self._l2sp_anchor is None:
                raise RuntimeError("Ataraxos learner has no Epoch475 L2-SP anchor")
            with torch.no_grad():
                for name, parameter in self.named_parameters():
                    if self._is_l2sp_parameter(name):
                        l2sp_penalty.add_(
                            (parameter - self._l2sp_anchor[name]).square().sum(),
                            alpha=0.5,
                        )
        self._ataraxos_step.add_(1)
        self._ataraxos_online_mass.add_(batch_online_fraction.detach())
        self._ataraxos_host_step = step + 1

        if log_step:
            self.update_info = {
                "Ataraxos/Online Loss": online_loss.detach(),
                "Ataraxos/Offline Loss": offline_loss.detach(),
                "Ataraxos/PPO Loss": ppo.detach(),
                "Ataraxos/Behavior KL": behavior_kl.detach(),
                "Ataraxos/Uniform KL": magnet_kl.detach(),
                "Ataraxos/Value Loss": online_value.detach(),
                "Ataraxos/Offline Value Loss": offline_value.detach(),
                "Ataraxos/Offline CE": offline_ce.detach(),
                "Ataraxos/Bootstrap KL": bootstrap_kl.detach(),
                "Ataraxos/Bootstrap KL Coef": float(bootstrap_kl_coefficient),
                "Ataraxos/Bootstrap Damping": float(damping),
                "Ataraxos/Effective Offline Policy Coef": float(
                    effective_offline_policy_coef
                ),
                "Ataraxos/Value Backbone Scale": float(value_backbone_scale),
                "Ataraxos/Magnet Coef": temperature.detach(),
                "Ataraxos/Online Mass": self._ataraxos_online_mass.detach(),
                "Ataraxos/Batch Online Fraction": batch_online_fraction.detach(),
                "Ataraxos/Offline Actor Weight": _masked_mean(
                    offline_actor_weight, offline_states
                ).detach(),
                "Ataraxos/L2SP Penalty": l2sp_penalty.detach(),
                "Ataraxos/L2SP Coef": float(self.l2sp_coef * l2sp_scale),
                "Ataraxos/Clip Fraction": _masked_mean(((ratio - 1).abs() > self.clip_range).float(), selected).detach(),
                "Ataraxos/Selected Fraction": selected.float().sum() / online.float().sum().clamp(min=1),
                "Ataraxos/Ratio Mean": _masked_mean(ratio, selected).detach(),
            }
        return loss
