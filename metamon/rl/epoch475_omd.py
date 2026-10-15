"""Epoch-475 online finetuning with source-separated optimistic updates.

This runner reuses the audited TaurosV1B online pipeline, but tags trajectories
as offline or online and applies optimism only to the online actor gradient by
default.  The ordinary gradient is always retained, so critics and offline
behavior regression follow the established AWR/CRR recipe.

For online actor gradient ``g_t`` and the previous (optionally EMA-smoothed)
online actor gradient ``m_{t-1}``, the update direction is::

    g_total + beta_online * (g_t - m_{t-1})

With ``beta_online=1`` and no EMA this is the usual ``2 g_t - g_{t-1}``
optimistic-gradient form. AdamW supplies the adaptive diagonal mirror map.  An
offline optimism coefficient exists only as an ablation and defaults to zero.

The implementation is intentionally single-GPU and requires grad_accum=1. This
keeps gradient provenance exact and makes resume state unambiguous.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from amago.loading import Batch, RLDataset

from metamon.rl import taurosv1b_online as v1b


SOURCE_KEY = "_epoch475_omd_online_source"


class SourceTaggedDataset(RLDataset):
    """Attach a per-trajectory source flag without changing model features."""

    def __init__(self, dataset: RLDataset, online: bool):
        super().__init__(dset_name=dataset.dset_name)
        self.dataset = dataset
        self.online = bool(online)

    @property
    def save_new_trajs_to(self):
        return self.dataset.save_new_trajs_to

    @property
    def ready_for_training(self):
        return self.dataset.ready_for_training

    def configure_from_experiment(self, experiment):
        RLDataset.configure_from_experiment(self, experiment)
        self.dataset.configure_from_experiment(experiment)

    def on_end_of_collection(self, experiment):
        return self.dataset.on_end_of_collection(experiment)

    def delete(self):
        return self.dataset.delete()

    def get_description(self):
        source = "online" if self.online else "offline"
        return f"{self.dataset.get_description()}\nOMD source: {source}"

    def sample_random_trajectory(self):
        data = self.dataset.sample_random_trajectory()
        data.obs[SOURCE_KEY] = torch.full(
            (len(data.time_idxs), 1),
            1 if self.online else 0,
            dtype=torch.uint8,
        )
        return data


class SourceTaggedReadyAwareMixture(v1b.ReadyAwareOnlineMixture):
    """Ready-aware online/offline mixture whose samples retain provenance."""

    def __init__(self, *, fifo, offline, **kwargs):
        super().__init__(
            fifo=SourceTaggedDataset(fifo, online=True),
            offline=SourceTaggedDataset(offline, online=False),
            **kwargs,
        )


@dataclass
class OptimisticGradientState:
    """Checkpointable previous actor gradients."""

    ema: float
    online: dict[int, torch.Tensor] = field(default_factory=dict)
    offline: dict[int, torch.Tensor] = field(default_factory=dict)
    steps: int = 0

    def state_dict(self):
        return {
            "ema": float(self.ema),
            "steps": int(self.steps),
            "online": {i: g.detach().cpu() for i, g in self.online.items()},
            "offline": {i: g.detach().cpu() for i, g in self.offline.items()},
        }

    def load_state_dict(self, state):
        self.ema = float(state.get("ema", self.ema))
        self.steps = int(state.get("steps", 0))
        self.online = dict(state.get("online", {}))
        self.offline = dict(state.get("offline", {}))


def _slice_batch(batch: Batch, select: torch.Tensor) -> Batch:
    return Batch(
        obs={key: value[select] for key, value in batch.obs.items()},
        rl2s=batch.rl2s[select],
        rews=batch.rews[select],
        dones=batch.dones[select],
        actions=batch.actions[select],
        time_idxs=batch.time_idxs[select],
    )


class Epoch475OMDExperiment(v1b.TaurosV1BOnlineExperiment):
    """Audited online experiment with source-separated actor optimism."""

    beta_online = 1.0
    beta_offline = 0.0
    gradient_ema = 0.0

    def init_model(self) -> None:
        super().init_model()
        if self.accelerator.num_processes != 1:
            raise ValueError("Epoch475 OMD currently supports one learner process only")
        if self.batches_per_update != 1:
            raise ValueError("Epoch475 OMD requires grad_accum/batches_per_update=1")
        self.omd_state = OptimisticGradientState(ema=self.gradient_ema)
        self.accelerator.register_for_checkpointing(self.omd_state)

    def _actor_only_loss(self, batch: Batch, log_step: bool = False):
        policy = self.policy
        old_critic_weight = policy.critic_loss_weight
        old_online_coeff = policy.online_coeff
        popart_state = {
            key: value.detach().clone() for key, value in policy.popart.state_dict().items()
        }
        try:
            policy.critic_loss_weight = 0.0
            policy.online_coeff = 0.0
            return self.policy_aclr(batch, log_step=log_step)
        finally:
            policy.critic_loss_weight = old_critic_weight
            policy.online_coeff = old_online_coeff
            # The extra actor-only forward must not update PopArt a second time.
            policy.popart.load_state_dict(popart_state)

    @staticmethod
    def _capture_gradients(parameters):
        return [None if p.grad is None else p.grad.detach().clone() for p in parameters]

    def _add_optimistic_correction(
        self,
        parameters,
        actor_grads,
        previous,
        beta: float,
        source_fraction: float,
    ):
        next_previous = {}
        for idx, (parameter, current) in enumerate(zip(parameters, actor_grads)):
            ordinary = parameter.grad
            if current is None:
                continue
            old = previous.get(idx)
            if old is None:
                correction = torch.zeros_like(current)
                smoothed = current
            else:
                old = old.to(device=current.device, dtype=current.dtype)
                correction = beta * source_fraction * (current - old)
                smoothed = self.omd_state.ema * old + (1.0 - self.omd_state.ema) * current
            if ordinary is None:
                parameter.grad = correction
            else:
                ordinary.add_(correction)
            next_previous[idx] = smoothed.detach()
        return next_previous

    def train_step(self, batch: Batch, log_step: bool):
        source = batch.obs.get(SOURCE_KEY)
        if source is None:
            raise RuntimeError(
                f"OMD batch is missing {SOURCE_KEY}; use SourceTaggedReadyAwareMixture"
            )
        online_rows = source[:, 0, 0].bool()
        offline_rows = ~online_rows
        parameters = [p for p in self.policy_aclr.parameters() if p.requires_grad]

        with self.accelerator.accumulate(self.policy_aclr):
            self.optimizer.zero_grad(set_to_none=True)
            ordinary_loss = self.policy_aclr(batch, log_step=log_step)
            metrics = {"Loss": ordinary_loss} | self.policy.update_info
            self.accelerator.backward(ordinary_loss)
            ordinary_grads = self._capture_gradients(parameters)

            corrections = []
            online_actor_loss: Optional[torch.Tensor] = None
            offline_actor_loss: Optional[torch.Tensor] = None

            if online_rows.any() and self.beta_online != 0.0:
                self.optimizer.zero_grad(set_to_none=True)
                online_actor_loss = self._actor_only_loss(_slice_batch(batch, online_rows))
                self.accelerator.backward(online_actor_loss)
                corrections.append(
                    (
                        self._capture_gradients(parameters),
                        "online",
                        self.beta_online,
                        float(online_rows.float().mean().item()),
                    )
                )

            if offline_rows.any() and self.beta_offline != 0.0:
                self.optimizer.zero_grad(set_to_none=True)
                offline_actor_loss = self._actor_only_loss(_slice_batch(batch, offline_rows))
                self.accelerator.backward(offline_actor_loss)
                corrections.append(
                    (
                        self._capture_gradients(parameters),
                        "offline",
                        self.beta_offline,
                        float(offline_rows.float().mean().item()),
                    )
                )

            # Restore the ordinary gradient, then add each source's optimistic term.
            for parameter, ordinary in zip(parameters, ordinary_grads):
                parameter.grad = ordinary
            for actor_grads, source_name, beta, source_fraction in corrections:
                previous = getattr(self.omd_state, source_name)
                updated = self._add_optimistic_correction(
                    parameters,
                    actor_grads,
                    previous,
                    beta,
                    source_fraction,
                )
                setattr(self.omd_state, source_name, updated)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(parameters, self.grad_clip)
                self.policy.soft_sync_targets()
                self.grad_update_counter += 1
                self.omd_state.steps += 1
                if log_step:
                    metrics.update(
                        {
                            "Learning Rate": self.lr_schedule.get_last_lr()[0],
                            "OMD/Online Rows": int(online_rows.sum().item()),
                            "OMD/Offline Rows": int(offline_rows.sum().item()),
                            "OMD/Online Beta": float(self.beta_online),
                            "OMD/Offline Beta": float(self.beta_offline),
                        }
                        | self.policy.get_grad_norms()
                    )
                    if online_actor_loss is not None:
                        metrics["OMD/Online Actor Loss"] = online_actor_loss.detach()
                    if offline_actor_loss is not None:
                        metrics["OMD/Offline Actor Loss"] = offline_actor_loss.detach()
            self.optimizer.step()
            self.lr_schedule.step()
        return metrics


def main() -> None:
    Epoch475OMDExperiment.beta_online = float(
        os.environ.get("EPOCH475_OMD_ONLINE_BETA", "1.0")
    )
    Epoch475OMDExperiment.beta_offline = float(
        os.environ.get("EPOCH475_OMD_OFFLINE_BETA", "0.0")
    )
    Epoch475OMDExperiment.gradient_ema = float(
        os.environ.get("EPOCH475_OMD_GRADIENT_EMA", "0.0")
    )
    if not 0.0 <= Epoch475OMDExperiment.gradient_ema < 1.0:
        raise ValueError("EPOCH475_OMD_GRADIENT_EMA must be in [0, 1)")

    original_mixture = v1b.ReadyAwareOnlineMixture
    original_experiment = v1b.TaurosV1BOnlineExperiment
    v1b.ReadyAwareOnlineMixture = SourceTaggedReadyAwareMixture
    v1b.TaurosV1BOnlineExperiment = Epoch475OMDExperiment
    try:
        v1b.main()
    finally:
        v1b.ReadyAwareOnlineMixture = original_mixture
        v1b.TaurosV1BOnlineExperiment = original_experiment


if __name__ == "__main__":
    main()
