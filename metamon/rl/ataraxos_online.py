"""One-shot Epoch475 continuation with the Ataraxos training recipe."""

from __future__ import annotations

import json
import os
import shutil

import amago
import lz4.frame
import numpy as np
import torch
from amago.loading import RLDataset

from metamon.rl import ataraxos_keys as keys
from metamon.rl import taurosv1b_online as v1b
from metamon.rl.recent_schedule_free import RecentScheduleFreeAdamW


class TaggedDataset(RLDataset):
    def __init__(self, dataset: RLDataset, online: bool, owner=None):
        super().__init__(dset_name=dataset.dset_name)
        self.dataset = dataset
        self.online = bool(online)
        self.owner = owner

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
        return self.dataset.get_description() + ("\nAtaraxos online" if self.online else "\nAtaraxos offline")

    def sample_random_trajectory(self):
        data = self.dataset.sample_random_trajectory()
        T = len(data.time_idxs)
        action_dim = data.actions.shape[-1]
        data.obs[keys.SOURCE] = torch.full((T, 1), self.online, dtype=torch.bool)
        if self.online:
            required = (
                keys.BEHAVIOR_LOG_PROBS, keys.BEHAVIOR_VALUE,
                keys.BEHAVIOR_VALID, keys.BEHAVIOR_LEGAL, keys.ADVANTAGE,
                keys.VALUE_TARGET,
            )
            missing = [k for k in required if k not in data.obs]
            if missing:
                raise RuntimeError(f"online replay is missing Ataraxos metadata: {missing}")
        else:
            data.obs[keys.BEHAVIOR_LOG_PROBS] = torch.zeros(T, action_dim)
            data.obs[keys.BEHAVIOR_VALUE] = torch.zeros(T, 1)
            data.obs[keys.BEHAVIOR_VALID] = torch.zeros(T, 1, dtype=torch.bool)
            data.obs[keys.BEHAVIOR_LEGAL] = torch.zeros(
                T, action_dim, dtype=torch.bool
            )
            data.obs[keys.ADVANTAGE] = torch.zeros(T, 1)
            data.obs[keys.VALUE_TARGET] = torch.zeros(T, 1)
        threshold = 0.01 if self.owner is None else self.owner.adv_threshold
        data.obs[keys.ADV_THRESHOLD] = torch.full((T, 1), float(threshold))
        return data


class BalancedAtaraxosMixture(v1b.ReadyAwareOnlineMixture):
    """Low-discrepancy sampling for the configured online-weight ramp."""

    def __init__(self, *, fifo, offline, **kwargs):
        self.adv_threshold = 0.01
        self.current_online_weight = 0.0
        self._sample_idx = 0
        tagged_fifo = TaggedDataset(fifo, online=True, owner=self)
        tagged_offline = TaggedDataset(offline, online=False, owner=self)
        super().__init__(fifo=tagged_fifo, offline=tagged_offline, **kwargs)

    def sample_random_trajectory(self):
        # Deterministic low-discrepancy allocation avoids Bernoulli noise when
        # the first ramp epochs contain only a handful of online trajectories.
        # At the 0.5 cap this alternates sources and yields exactly 7/7 in every
        # batch of 14.
        before = int(self._sample_idx * self.current_online_weight + 1e-9)
        after = int((self._sample_idx + 1) * self.current_online_weight + 1e-9)
        dataset = self.fifo if after > before else self.offline
        self._sample_idx += 1
        if not self.fifo.ready_for_training:
            raise RuntimeError("Ataraxos learner started before the 5,000-game FIFO was ready")
        self._sampling_metrics[dataset.dset_name] += 1
        return dataset.sample_random_trajectory()

    def update_dset_weights(self, epoch: int):
        super().update_dset_weights(epoch)
        self.current_online_weight = next(
            weight for dataset, weight in self._available_datasets
            if dataset is self.fifo
        )
        self._sample_idx = 0
        if self.fifo.ready_for_training and self.current_online_weight > 0.0:
            self.adv_threshold = self._compute_advantage_threshold()
        print(
            f"  [Ataraxos] epoch={epoch} "
            f"online={self.current_online_weight:.4f} "
            f"offline={1.0 - self.current_online_weight:.4f} "
            f"adv_threshold={self.adv_threshold:.6f}", flush=True
        )

    def _compute_advantage_threshold(self) -> float:
        parsed = self.fifo.dataset.parsed_replay_dset
        values = []
        filenames = list(parsed.filenames)
        for filename in filenames:
            try:
                if filename.endswith(".lz4"):
                    with lz4.frame.open(filename, "rb") as f:
                        payload = json.loads(f.read().decode("utf-8"))
                else:
                    with open(filename, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                v = np.asarray(payload["behavior_values"], dtype=np.float32)
                states = payload["states"]
                actions = payload["actions"]
                if v.ndim != 1 or v.size == 0:
                    raise ValueError("behavior_values must be a non-empty vector")
                if not np.isfinite(v).all():
                    raise ValueError("behavior_values contain non-finite entries")
                if len(states) != v.size + 1 or len(actions) != len(states):
                    raise ValueError("behavior metadata is not trajectory-aligned")
                state_values = np.concatenate([v, np.zeros(1, dtype=np.float32)])
                final = states[-1]
                outcome = 1.0 if final.get("battle_won") else (-1.0 if final.get("battle_lost") else 0.0)
                rewards = np.zeros(v.size, dtype=np.float32)
                rewards[-1] = outcome
                done = np.zeros(v.size, dtype=np.float32)
                done[-1] = 1.0
                delta = rewards + (1.0 - done) * state_values[1:] - state_values[:-1]
                carry = 0.0
                adv = np.empty_like(delta)
                for t in range(delta.size - 1, -1, -1):
                    carry = float(delta[t]) + 0.5 * (1.0 - float(done[t])) * carry
                    adv[t] = carry
                values.extend(np.abs(adv).tolist())
            except (
                OSError,
                KeyError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                raise RuntimeError(
                    f"invalid online FIFO replay: {filename}: {exc}"
                ) from exc
        if not values:
            raise RuntimeError("could not compute Ataraxos advantage threshold from FIFO")
        return max(float(np.quantile(np.asarray(values), 0.75)), 0.01)


class AtaraxosExperiment(v1b.TaurosV1BOnlineExperiment):
    """Ataraxos with grouped LRs and responsive Schedule-Free averaging."""

    _GROUP_LRS = {
        "backbone": 1.0e-5,
        "actor": 2.0e-5,
        "value": 1.0e-4,
    }
    _LR_HOLD_STEPS = 100_000
    _LR_DECAY_STEPS = 400_000
    _LR_FLOOR_SCALE = 0.1
    _SF_BETA = 0.8
    _SF_R = 99.0
    _SF_INNER_MOMENTUM = 0.9

    def init_optimizer(self, policy):
        groups = [
            {
                "params": list(policy.tstep_encoder.parameters())
                + list(policy.traj_encoder.parameters()),
                "lr": self._GROUP_LRS["backbone"],
                "ataraxos_name": "backbone",
                "ataraxos_base_lr": self._GROUP_LRS["backbone"],
            },
            {
                "params": list(policy.actor.parameters()),
                "lr": self._GROUP_LRS["actor"],
                "ataraxos_name": "actor",
                "ataraxos_base_lr": self._GROUP_LRS["actor"],
            },
            {
                "params": list(policy.ataraxos_value.parameters()),
                "lr": self._GROUP_LRS["value"],
                "ataraxos_name": "value",
                "ataraxos_base_lr": self._GROUP_LRS["value"],
            },
        ]
        grouped = [p for group in groups for p in group["params"]]
        grouped_ids = {id(p) for p in grouped}
        trainable = list(policy.trainable_params)
        if len(grouped) != len(grouped_ids) or grouped_ids != {id(p) for p in trainable}:
            raise RuntimeError("Ataraxos optimizer groups do not match trainable parameters")
        return RecentScheduleFreeAdamW(
            groups,
            lr=self.learning_rate,
            beta=self._SF_BETA,
            inner_momentum=self._SF_INNER_MOMENTUM,
            r=self._SF_R,
            # Averaging is layered on the existing LR schedule rather than
            # changing its weights when that schedule begins to decay.
            weight_lr_power=0.0,
            weight_decay=self.l2_coeff,
        )

    def _lr_scale(self, step: int) -> float:
        if step <= self._LR_HOLD_STEPS:
            return 1.0
        progress = min(
            (step - self._LR_HOLD_STEPS) / float(self._LR_DECAY_STEPS), 1.0
        )
        return 1.0 - progress * (1.0 - self._LR_FLOOR_SCALE)

    def interact(self, envs, timesteps, hidden_state=None, render=False, save_on_done=False, episodes=None, sample=True):
        # This is the stock AMAGO loop with one deliberate addition before
        # env.step: attach the exact behavior distribution/value to the battle.
        self.optimizer.eval()
        policy = self.policy
        policy.eval()
        amago.utils.call_async_env(envs, "reset_stats")
        if hidden_state is None:
            hidden_state = policy.traj_encoder.init_hidden_state(self.parallel_actors, self.DEVICE)

        def get_t():
            packed = amago.utils.call_async_env(envs, "current_timestep")
            obs, rl2s, tids = zip(*packed)
            obs = amago.utils.stack_list_array_dicts(list(obs), axis=0, cat=True)
            rl2s = np.concatenate(rl2s, axis=0)
            tids = np.concatenate(tids, axis=0)
            obs = {k: torch.from_numpy(v).to(self.DEVICE).unsqueeze(1) for k, v in obs.items()}
            return obs, torch.from_numpy(rl2s).to(self.DEVICE).unsqueeze(1), torch.from_numpy(tids).to(self.DEVICE).unsqueeze(1)

        obs, rl2s, tids = get_t()
        episodes_finished = 0
        for _ in range(timesteps):
            self.on_interact_step(obs)
            with torch.no_grad(), self.caster():
                actions, hidden_state, behavior_lp, behavior_v = policy.get_actions_with_metadata(
                    obs=obs, rl2s=rl2s, time_idxs=tids,
                    sample=sample, hidden_state=hidden_state,
                )
            target = envs.env
            while not hasattr(target, "record_policy_metadata") and hasattr(target, "env"):
                target = target.env
            if not hasattr(target, "record_policy_metadata"):
                raise RuntimeError("Ataraxos collection env cannot record policy metadata")
            target.record_policy_metadata(
                behavior_lp.float().cpu().numpy(),
                behavior_v.float().cpu().numpy(),
                (~obs["illegal_actions"][:, -1].bool()).cpu().numpy(),
            )
            *_, terminated, truncated, _ = envs.step(actions.squeeze(1).cpu().numpy())
            done = terminated | truncated
            if done.ndim == 2:
                done = done.squeeze(1)
            if done.any():
                if save_on_done:
                    amago.utils.call_async_env(envs, "save_finished_trajs")
                episodes_finished += int(done.sum())
            obs, rl2s, tids = get_t()
            hidden_state = policy.traj_encoder.reset_hidden_state(hidden_state, done)
            if render:
                envs.render()
            if episodes is not None and episodes_finished >= episodes:
                break
        return_history = amago.utils.call_async_env(envs, "return_history")
        special_history = amago.utils.call_async_env(envs, "special_history")
        return hidden_state, (return_history, special_history)

    def train_step(self, batch, log_step: bool):
        self.optimizer.train()
        step = max(int(self.grad_update_counter), 0)
        scale = self._lr_scale(step)
        for group in self.optimizer.param_groups:
            group["lr"] = group["ataraxos_base_lr"] * scale
        metrics = super().train_step(batch, log_step)
        if log_step:
            # The stock logger reads the dormant constant scheduler. Record all
            # three manually managed parameter-group rates explicitly.
            for group in self.optimizer.param_groups:
                metrics[
                    f"Learning Rate/{group['ataraxos_name']}"
                ] = group["lr"]
            metrics["Learning Rate"] = next(
                group["lr"]
                for group in self.optimizer.param_groups
                if group["ataraxos_name"] == "actor"
            )
            metrics["ScheduleFree/Beta"] = self._SF_BETA
            metrics["ScheduleFree/R"] = self._SF_R
            metrics["ScheduleFree/Constant Drift Retention"] = (
                self._SF_R + 1.0
            ) / (self._SF_R + 2.0)
            metrics["ScheduleFree/Query Drift Retention"] = 1.0 - (
                self._SF_BETA / (self._SF_R + 2.0)
            )
            for group in self.optimizer.param_groups:
                metrics[
                    f"ScheduleFree/Average Coef/{group['ataraxos_name']}"
                ] = group["sf_ckp1"]
        return metrics

    def save_checkpoint(self) -> None:
        # Raw policy checkpoints and full-state model weights must contain x;
        # optimizer state retains z and reconstructs y when training resumes.
        # Both artifacts are published atomically: the launcher treats a visible
        # epoch directory as complete, so a preemption during save must leave
        # only a .tmp path that resume discovery ignores.
        self.optimizer.eval()
        policy_dir = os.path.join(self.ckpt_dir, "policy_weights")
        policy_path = os.path.join(policy_dir, f"policy_epoch_{self.epoch}.pt")
        policy_tmp = policy_path + ".tmp"
        if self.accelerator.is_main_process:
            torch.save(self.policy.state_dict(), policy_tmp)
            os.replace(policy_tmp, policy_path)

        interval = self.full_state_ckpt_interval
        if interval is None or interval <= 0 or self.epoch % interval:
            return

        state_root = os.path.join(self.ckpt_dir, "training_states")
        state_name = f"{self.run_name}_epoch_{self.epoch}"
        state_path = os.path.join(state_root, state_name)
        state_tmp = state_path + ".tmp"
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if os.path.isdir(state_tmp):
                shutil.rmtree(state_tmp)
            elif os.path.exists(state_tmp):
                os.remove(state_tmp)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(state_tmp, safe_serialization=True)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if os.path.exists(state_path):
                shutil.rmtree(state_path)
            os.replace(state_tmp, state_path)
        self.accelerator.wait_for_everyone()

    def write_latest_policy(self) -> None:
        # Collectors and league opponents always consume the averaged policy.
        self.optimizer.eval()
        return super().write_latest_policy()


def main():
    original_mixture = v1b.ReadyAwareOnlineMixture
    original_experiment = v1b.TaurosV1BOnlineExperiment
    v1b.ReadyAwareOnlineMixture = BalancedAtaraxosMixture
    v1b.TaurosV1BOnlineExperiment = AtaraxosExperiment
    try:
        v1b.main()
    finally:
        v1b.ReadyAwareOnlineMixture = original_mixture
        v1b.TaurosV1BOnlineExperiment = original_experiment


if __name__ == "__main__":
    main()
