"""One-shot Epoch475 continuation with the Ataraxos training recipe."""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import amago
import lz4.frame
import numpy as np
import orjson
import torch
from amago.loading import MAGIC_PAD_VAL, RLDataset, RLData_pad_collate
from torch.utils.data import DataLoader

from metamon.rl import ataraxos_keys as keys
from metamon.rl import taurosv1b_online as v1b
from metamon.rl.recent_schedule_free import RecentScheduleFreeAdamW


def _fixed_length_collate(samples, *, max_seq_len: int):
    """Right-pad every batch to one compile-stable sequence shape."""
    batch = RLData_pad_collate(samples)

    def pad_time(tensor: torch.Tensor, target: int) -> torch.Tensor:
        missing = target - tensor.shape[1]
        if missing < 0:
            raise RuntimeError(
                f"collated sequence length {tensor.shape[1]} exceeds target {target}"
            )
        if missing == 0:
            return tensor
        shape = list(tensor.shape)
        shape[1] = missing
        padding = tensor.new_full(shape, MAGIC_PAD_VAL)
        return torch.cat((tensor, padding), dim=1)

    state_len = max_seq_len + 1
    batch.obs = {key: pad_time(value, state_len) for key, value in batch.obs.items()}
    batch.rl2s = pad_time(batch.rl2s, state_len)
    batch.time_idxs = pad_time(batch.time_idxs, state_len)
    batch.rews = pad_time(batch.rews, max_seq_len)
    batch.dones = pad_time(batch.dones, max_seq_len)
    batch.actions = pad_time(batch.actions, max_seq_len)
    return batch


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
                        payload = orjson.loads(f.read())
                else:
                    with open(filename, "rb") as f:
                        payload = orjson.loads(f.read())
                v = np.asarray(payload["behavior_values"], dtype=np.float32)
                states = payload["states"]
                actions = payload["actions"]
                logp = np.asarray(
                    payload["behavior_log_probs"], dtype=np.float32
                )
                legal = np.asarray(
                    payload["behavior_legal_masks"], dtype=bool
                )
                if v.ndim != 1 or v.size == 0:
                    raise ValueError("behavior_values must be a non-empty vector")
                if not np.isfinite(v).all():
                    raise ValueError("behavior_values contain non-finite entries")
                if np.any(np.abs(v) > 1.0001):
                    raise ValueError("behavior_values fall outside [-1, 1]")
                if len(states) != v.size + 1 or len(actions) != len(states):
                    raise ValueError("behavior metadata is not trajectory-aligned")
                if logp.ndim != 2 or logp.shape[0] != v.size:
                    raise ValueError("behavior_log_probs are not trajectory-aligned")
                if legal.shape != logp.shape:
                    raise ValueError("behavior_legal_masks have the wrong shape")
                if not np.isfinite(logp).all():
                    raise ValueError("behavior_log_probs contain non-finite entries")
                probs = np.exp(logp)
                if not np.allclose(probs.sum(-1), 1.0, atol=2e-3, rtol=0.0):
                    raise ValueError("behavior probabilities do not sum to one")
                if np.any(probs[~legal] > 1e-8):
                    raise ValueError("behavior policy assigns mass to an illegal action")
                chosen = np.asarray(actions[:-1])
                if (
                    chosen.ndim != 1
                    or not np.issubdtype(chosen.dtype, np.integer)
                    or np.any(chosen < 0)
                    or np.any(chosen >= logp.shape[1])
                ):
                    raise ValueError("recorded online actions are invalid")
                if not legal[np.arange(v.size), chosen].all():
                    raise ValueError("recorded online action is illegal")
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
                orjson.JSONDecodeError,
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
    _CHECKPOINT_MILESTONE_INTERVAL = 50
    _CHECKPOINT_MILESTONE_COUNT = 10
    # Keep the current epoch plus the previous 64 so every configured lineage
    # offset (through -64) resolves to its exact policy checkpoint.
    _CHECKPOINT_RECENT_COUNT = 65

    def init_dloaders(self):
        # AMAGO normally pads only to the longest item in each batch. That makes
        # the sequence dimension vary as the FIFO changes and causes torch.compile
        # to spend about a minute compiling a new forward/backward graph at the
        # start of learner phases. A fixed right-padded shape is loss-equivalent:
        # Ataraxos already masks every MAGIC_PAD_VAL timestep.
        train_dloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            collate_fn=partial(
                _fixed_length_collate, max_seq_len=int(self.max_seq_len)
            ),
            pin_memory=True,
        )
        self.train_dloader = self.accelerator.prepare(train_dloader)
        return self.train_dloader

    @staticmethod
    def _replay_names(root: Path) -> set[str]:
        return {
            entry.name
            for entry in os.scandir(root)
            if entry.is_file()
            and (entry.name.endswith(".json") or entry.name.endswith(".json.lz4"))
        }

    def _stop_external_collectors(self) -> None:
        children = getattr(self, "_persistent_collectors", [])
        self._persistent_collectors = []
        for child, _stream in children:
            if child.poll() is None:
                child.terminate()
        for child, stream in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
            stream.close()

    def _ensure_external_collectors(self):
        buffer_dir = Path(os.environ["SNORLAX_BUFFER_DIR"])
        log_dir = Path(os.environ["SNORLAX_LOG_DIR"])
        latest = os.environ["SNORLAX_LATEST_WEIGHTS"]
        processes = int(os.environ.get("COLLECTOR_PROCESSES", "8"))
        lanes = int(os.environ.get("LANES", "64"))
        workers = int(os.environ.get("COLLECTOR_WORKERS", "8"))
        dtype = os.environ.get("COLLECTOR_DTYPE", "bf16")
        collector_mp = "no" if dtype == "fp32" else dtype
        run_config = os.environ["SNORLAX_RUN_CONFIG"]
        module = os.environ.get("RUNNER_MODULE", "metamon.rl.ataraxos_online")
        save_dir = os.environ["METAMON_SAVE_DIR"]
        pause_file = log_dir / ".persistent-collectors-paused"
        ack_dir = log_dir / ".persistent-collector-acks"
        ack_dir.mkdir(parents=True, exist_ok=True)
        children = getattr(self, "_persistent_collectors", [])
        if children and all(child.poll() is None for child, _stream in children):
            return children, pause_file, ack_dir
        if children:
            self._stop_external_collectors()
        pause_file.unlink(missing_ok=True)
        for ack in ack_dir.glob("*.ready"):
            ack.unlink()
        children = []
        try:
            for idx in range(processes):
                log_path = log_dir / f"collector-persistent-{idx}.log"
                stream = log_path.open("a", encoding="utf-8", buffering=1)
                child_env = dict(os.environ)
                child_env.pop("SNORLAX_PERSISTENT_ALTERNATION", None)
                child_env.update(
                    METAMON_RUN_MODE="collect",
                    SNORLAX_COLLECT_PAUSE_FILE=str(pause_file),
                    SNORLAX_COLLECT_ACK_FILE=str(ack_dir / f"{idx}.ready"),
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    NUMEXPR_NUM_THREADS="1",
                )
                command = [
                    sys.executable, "-m", module,
                    "--run_config", run_config,
                    "--mode", "collect",
                    "--run_name", self.run_name,
                    "--lanes", str(lanes),
                    "--n_workers", str(workers),
                    "--save_dir", save_dir,
                    "--buffer_dir", str(buffer_dir),
                    "--base_weights", latest,
                    "--epochs", "1000000",
                    "--seed", str(1000 + idx),
                    "--mixed_precision", collector_mp,
                ]
                children.append(
                    (subprocess.Popen(command, env=child_env, stdout=stream, stderr=subprocess.STDOUT), stream)
                )
        except BaseException:
            self._persistent_collectors = children
            self._stop_external_collectors()
            raise
        self._persistent_collectors = children
        if not getattr(self, "_collector_cleanup_registered", False):
            atexit.register(self._stop_external_collectors)
            self._collector_cleanup_registered = True
        return children, pause_file, ack_dir

    def _external_collect_phase(self, epoch: int) -> None:
        """Collect with GPU/env processes retained across learner phases."""
        buffer_dir = Path(os.environ["SNORLAX_BUFFER_DIR"])
        replay_dir = buffer_dir / "gen1ou"
        log_dir = Path(os.environ["SNORLAX_LOG_DIR"])
        target = int(os.environ.get("GAMES_PER_LEARNER_EPOCH", "2000"))
        processes = int(os.environ.get("COLLECTOR_PROCESSES", "8"))
        lanes = int(os.environ.get("LANES", "64"))
        baseline = self._replay_names(replay_dir)
        self.optimizer.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        children, pause_file, ack_dir = self._ensure_external_collectors()
        for ack in ack_dir.glob("*.ready"):
            ack.unlink()
        pause_file.unlink(missing_ok=True)
        print(
            f"[PersistentCycle] collect epoch={epoch} target={target} "
            f"processes={processes} lanes={lanes}",
            flush=True,
        )
        try:
            fresh = 0
            while fresh < target:
                time.sleep(10)
                for child, _stream in children:
                    status = child.poll()
                    if status is not None:
                        raise RuntimeError(
                            f"collector exited with status {status}; inspect "
                            f"{log_dir}/collector-persistent-*.log"
                        )
                fresh = len(self._replay_names(replay_dir) - baseline)
                print(f"[PersistentCycle] collect epoch={epoch} {fresh}/{target}", flush=True)
            pause_file.touch()
            deadline = time.monotonic() + 30.0
            while len(list(ack_dir.glob("*.ready"))) < processes:
                for child, _stream in children:
                    if child.poll() is not None:
                        raise RuntimeError("collector exited while entering standby")
                if time.monotonic() >= deadline:
                    raise RuntimeError("collectors did not enter standby within 30 seconds")
                time.sleep(0.25)
        except BaseException:
            self._stop_external_collectors()
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _external_validation_phase(self, epoch: int) -> None:
        interval = int(os.environ.get("VALIDATION_INTERVAL", "0"))
        if interval <= 0 or epoch % interval:
            return
        latest = os.environ["SNORLAX_LATEST_WEIGHTS"]
        log_dir = Path(os.environ["SNORLAX_LOG_DIR"])
        child_env = dict(os.environ)
        child_env.pop("SNORLAX_PERSISTENT_ALTERNATION", None)
        child_env["METAMON_RUN_MODE"] = "validate"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        command = [
            sys.executable,
            "-m",
            os.environ.get("RUNNER_MODULE", "metamon.rl.ataraxos_online"),
            "--run_config",
            os.environ["SNORLAX_RUN_CONFIG"],
            "--mode",
            "validate",
            "--validation_once",
            "--validation_epoch",
            str(epoch),
            "--run_name",
            self.run_name,
            "--lanes",
            os.environ.get("LANES", "64"),
            "--n_workers",
            os.environ.get("COLLECTOR_WORKERS", "8"),
            "--save_dir",
            os.environ["METAMON_SAVE_DIR"],
            "--buffer_dir",
            os.environ["SNORLAX_BUFFER_DIR"],
            "--base_weights",
            latest,
            "--val_timesteps",
            os.environ.get("VALIDATION_TIMESTEPS", "1000"),
            "--val_interval",
            "1",
            "--mixed_precision",
            os.environ.get("MIXED_PRECISION", "bf16"),
            "--log",
        ]
        print(f"[PersistentCycle] validation epoch={epoch}", flush=True)
        validation_log = log_dir / "validation.log"
        with validation_log.open("a", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                env=child_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            # Validation is observability, not training state. The policy and
            # full optimizer checkpoint have already been published atomically;
            # do not terminate a long training run because an evaluator failed.
            print(
                f"[PersistentCycle] WARNING validation epoch={epoch} exited "
                f"with status {result.returncode}; training will continue; "
                f"inspect {validation_log}",
                flush=True,
            )

    def learn(self):
        if os.environ.get("SNORLAX_PERSISTENT_ALTERNATION") != "1":
            return super().learn()

        from tqdm import tqdm

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.epochs):
            self.epoch = epoch
            self.policy_aclr.eval()
            self._external_collect_phase(epoch)
            self.accelerator.wait_for_everyone()

            dset_log = self.dataset.on_end_of_collection(experiment=self)
            self.log(dset_log, key="dataset")
            self.init_dloaders()
            if not self.dataset.ready_for_training:
                raise RuntimeError("Snorlax FIFO became unavailable after collection")

            self.policy_aclr.train()
            loader = iter(self.train_dloader)
            progress = tqdm(
                range(self.train_batches_per_epoch),
                desc=f"{self.run_name} Epoch {epoch} Train",
                colour="green",
            ) if self.verbose else range(self.train_batches_per_epoch)
            block_started = time.perf_counter()
            data_seconds = 0.0
            for train_step in progress:
                data_started = time.perf_counter()
                batch = next(loader)
                data_seconds += time.perf_counter() - data_started
                total_step = epoch * self.train_batches_per_epoch + train_step
                log_step = total_step % self.log_interval == 0
                loss_dict = self.train_step(batch, log_step=log_step)
                if log_step:
                    self.log(loss_dict, key="train-update")
                if (train_step + 1) % 20 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - block_started
                    component_profile = self._consume_learner_profile()
                    component_text = ""
                    if component_profile:
                        component_text = " " + " ".join(
                            f"{name}={milliseconds:.1f}ms"
                            for name, milliseconds in component_profile.items()
                        )
                    print(
                        f"[LearnerProfile] epoch={epoch} step={train_step + 1} "
                        f"rate={20.0 / elapsed:.2f}it/s "
                        f"data_wait={100.0 * data_seconds / elapsed:.1f}%"
                        f"{component_text}",
                        flush=True,
                    )
                    block_started = time.perf_counter()
                    data_seconds = 0.0
            self.accelerator.wait_for_everyone()
            del loader
            del self.train_dloader

            if self.ckpt_interval and epoch % self.ckpt_interval == 0:
                self.save_checkpoint()
            if self.always_save_latest:
                self.write_latest_policy()
            self._external_validation_phase(epoch)

    @classmethod
    def _prune_numbered_checkpoints(
        cls,
        root: str,
        *,
        prefix: str,
        suffix: str,
        directories: bool,
    ) -> int:
        """Keep ten 50-epoch milestones and current plus previous 64."""
        try:
            names = os.listdir(root)
        except FileNotFoundError:
            return 0

        entries = []
        for name in names:
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            epoch_text = name[len(prefix) : len(name) - len(suffix) if suffix else None]
            if not epoch_text.isdigit():
                continue
            path = os.path.join(root, name)
            if directories != os.path.isdir(path):
                continue
            entries.append((int(epoch_text), path))

        recent = {
            epoch
            for epoch, _ in sorted(entries, key=lambda item: item[0])[
                -cls._CHECKPOINT_RECENT_COUNT :
            ]
        }
        milestones = sorted(
            epoch
            for epoch, _ in entries
            if epoch % cls._CHECKPOINT_MILESTONE_INTERVAL == 0
        )[-cls._CHECKPOINT_MILESTONE_COUNT :]
        keep = recent | set(milestones)
        removed = 0
        for epoch, path in entries:
            if epoch in keep:
                continue
            try:
                if directories:
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed += 1
            except OSError as exc:
                # Retention is housekeeping, not a reason to lose a long run.
                # Surface the failure and retry it after the next checkpoint.
                print(
                    f"  [Ataraxos] warning: could not remove old checkpoint "
                    f"{path}: {exc}",
                    flush=True,
                )
        return removed

    def _prune_checkpoints(self) -> None:
        policy_removed = self._prune_numbered_checkpoints(
            os.path.join(self.ckpt_dir, "policy_weights"),
            prefix="policy_epoch_",
            suffix=".pt",
            directories=False,
        )
        state_removed = self._prune_numbered_checkpoints(
            os.path.join(self.ckpt_dir, "training_states"),
            prefix=f"{self.run_name}_epoch_",
            suffix="",
            directories=True,
        )
        if policy_removed or state_removed:
            print(
                "  [Ataraxos] checkpoint retention removed "
                f"{policy_removed} policy and {state_removed} training-state "
                "checkpoint(s)",
                flush=True,
            )

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

    def _optimizer_update_step(self) -> int:
        """Read the update count persisted in every optimizer parameter group."""
        steps = set()
        for group in self.optimizer.param_groups:
            if "k" not in group:
                raise RuntimeError("Ataraxos optimizer group is missing update count k")
            steps.add(int(group["k"]))
        if len(steps) != 1:
            raise RuntimeError(
                "Ataraxos optimizer parameter groups disagree on update count: "
                f"{sorted(steps)}"
            )
        return steps.pop()

    def load_checkpoint(self, epoch: int, resume_training_state: bool = True) -> None:
        result = super().load_checkpoint(
            epoch, resume_training_state=resume_training_state
        )
        if resume_training_state:
            # AMAGO labels a state with the epoch that has already completed,
            # but its stock loader assigns that same value to self.epoch.
            # learn() treats self.epoch as the next epoch to run, so advance it
            # here instead of silently repeating a completed 1,000-step block.
            self.epoch = epoch + 1
            optimizer_step = self._optimizer_update_step()
            policy_step = int(self.policy._ataraxos_step.item())
            if int(getattr(self, "batches_per_update", 1)) == 1:
                if policy_step != optimizer_step:
                    raise RuntimeError(
                        "Ataraxos resume step mismatch: policy has "
                        f"{policy_step}, optimizer has {optimizer_step}"
                    )
            self.grad_update_counter = optimizer_step
            print(
                f"  [Ataraxos] restored {optimizer_step} optimizer updates",
                flush=True,
            )
        return result

    def collect_new_training_data(self) -> None:
        pause_file = os.environ.get("SNORLAX_COLLECT_PAUSE_FILE")
        ack_file = os.environ.get("SNORLAX_COLLECT_ACK_FILE")
        resumed = False
        if pause_file and ack_file and os.path.exists(pause_file):
            Path(ack_file).touch()
            print("[PersistentCollector] standby", flush=True)
            while os.path.exists(pause_file):
                time.sleep(0.25)
            Path(ack_file).unlink(missing_ok=True)
            resumed = True
        if resumed:
            # The learner atomically published a new averaged policy while this
            # collector was paused. Do not carry an old-policy hidden state or a
            # partially played battle into the new collection phase.
            self.read_latest_policy()
            self.train_envs.reset()
            self.hidden_state = None
            print("[PersistentCollector] resumed with latest policy", flush=True)
        return super().collect_new_training_data()

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
        target = envs.env
        while not hasattr(target, "record_policy_metadata") and hasattr(target, "env"):
            target = target.env
        if not hasattr(target, "record_policy_metadata"):
            raise RuntimeError("Ataraxos collection env cannot record policy metadata")
        profile = os.environ.get("METAMON_COLLECT_PROFILE") == "1"
        profile_interval = max(
            1, int(os.environ.get("METAMON_COLLECT_PROFILE_INTERVAL", "100"))
        )
        profile_times = {name: 0.0 for name in ("hook", "policy_gpu", "metadata", "env_step", "next_obs")}
        profile_steps = 0
        for _ in range(timesteps):
            pause_file = os.environ.get("SNORLAX_COLLECT_PAUSE_FILE")
            if pause_file and os.path.exists(pause_file):
                break
            started = time.perf_counter()
            self.on_interact_step(obs)
            if profile:
                profile_times["hook"] += time.perf_counter() - started
                started = time.perf_counter()
            with torch.inference_mode(), self.caster():
                actions, hidden_state, behavior_lp, behavior_v = policy.get_actions_with_metadata(
                    obs=obs, rl2s=rl2s, time_idxs=tids,
                    sample=sample, hidden_state=hidden_state,
                )
            # Collection needs these four small GPU results on the host. Packing
            # them into one transfer avoids four separate CUDA synchronizations
            # per environment step, which dominates short cached forwards.
            action_dim = behavior_lp.shape[-1]
            host = torch.cat(
                (
                    actions.reshape(-1, 1).float(),
                    behavior_lp.float(),
                    behavior_v.reshape(-1, 1).float(),
                    (~obs["illegal_actions"][:, -1].bool()).float(),
                ),
                dim=-1,
            ).cpu().numpy()
            host_actions = host[:, 0].astype(np.int64, copy=False)
            if profile:
                profile_times["policy_gpu"] += time.perf_counter() - started
                started = time.perf_counter()
            target.record_policy_metadata(
                host[:, 1 : 1 + action_dim],
                host[:, 1 + action_dim],
                host[:, 2 + action_dim :].astype(bool, copy=False),
            )
            if profile:
                profile_times["metadata"] += time.perf_counter() - started
                started = time.perf_counter()
            *_, terminated, truncated, _ = envs.step(host_actions)
            if profile:
                profile_times["env_step"] += time.perf_counter() - started
                started = time.perf_counter()
            done = terminated | truncated
            if done.ndim == 2:
                done = done.squeeze(1)
            if done.any():
                if save_on_done:
                    amago.utils.call_async_env(envs, "save_finished_trajs")
                episodes_finished += int(done.sum())
            obs, rl2s, tids = get_t()
            hidden_state = policy.traj_encoder.reset_hidden_state(hidden_state, done)
            if profile:
                profile_times["next_obs"] += time.perf_counter() - started
                profile_steps += 1
                if profile_steps % profile_interval == 0:
                    total = sum(profile_times.values())
                    fields = " ".join(
                        f"{name}={seconds / profile_steps * 1000:.1f}ms/{100.0 * seconds / total:.0f}%"
                        for name, seconds in profile_times.items()
                    )
                    print(f"[CollectorBoundary] steps={profile_steps} {fields}", flush=True)
                    if hasattr(target, "profile_report"):
                        print(f"[VectorBoundary]\n{target.profile_report()}", flush=True)
            if render:
                envs.render()
            if episodes is not None and episodes_finished >= episodes:
                break
        return_history = amago.utils.call_async_env(envs, "return_history")
        special_history = amago.utils.call_async_env(envs, "special_history")
        return hidden_state, (return_history, special_history)

    def train_step(self, batch, log_step: bool):
        self.optimizer.train()
        # ``grad_update_counter`` is process-local in AMAGO. The optimizer's
        # group counter is checkpointed and therefore remains authoritative
        # across the one-process-per-epoch alternating launcher.
        step = self._optimizer_update_step()
        scale = self._lr_scale(step)
        for group in self.optimizer.param_groups:
            group["lr"] = group["ataraxos_base_lr"] * scale
        profile = (
            torch.cuda.is_available()
            and getattr(self, "_learner_profile_recorded", 0)
            < int(os.environ.get("METAMON_LEARN_PROFILE_STEPS", "200"))
        )
        events = None
        if profile:
            events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            events[0].record()
        # The policy loss used to build L2-SP into the autograd graph. Its exact
        # analytic gradient is cheaper to inject after backward and before
        # gradient clipping, without changing the update.
        with self.accelerator.accumulate(self.policy_aclr):
            self.optimizer.zero_grad()
            loss = self.policy_aclr(batch, log_step=log_step)
            if events:
                events[1].record()
            reported_loss = loss
            if log_step:
                reported_loss = loss.detach() + (
                    self.policy.update_info["Ataraxos/L2SP Penalty"]
                    * self.policy.update_info["Ataraxos/L2SP Coef"]
                )
            metrics = {"Loss": reported_loss} | self.policy.update_info
            self.accelerator.backward(loss)
            if events:
                events[2].record()
            if self.accelerator.sync_gradients:
                self.policy.add_l2sp_gradients()
                self.accelerator.clip_grad_norm_(
                    self.policy_aclr.parameters(), self.grad_clip
                )
                self.policy.soft_sync_targets()
                self.grad_update_counter += 1
                if log_step:
                    metrics.update(self.policy.get_grad_norms())
            if events:
                events[3].record()
            self.optimizer.step()
            self.lr_schedule.step()
            if events:
                events[4].record()
                if not hasattr(self, "_learner_profile_events"):
                    self._learner_profile_events = []
                self._learner_profile_events.append(events)
                self._learner_profile_recorded = (
                    getattr(self, "_learner_profile_recorded", 0) + 1
                )
        # AMAGO still advances its registered constant scheduler after the
        # optimizer update. It does not affect the update above, but it resets
        # the visible group LRs to their initial values. Reinstall the managed
        # values so logs and serialized optimizer metadata reflect this step.
        for group in self.optimizer.param_groups:
            group["lr"] = group["ataraxos_base_lr"] * scale
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

    def _consume_learner_profile(self) -> dict[str, float]:
        events = getattr(self, "_learner_profile_events", [])
        if not events:
            return {}
        names = ("forward", "backward", "grad", "optimizer")
        totals = {name: 0.0 for name in names}
        for sample in events:
            for index, name in enumerate(names):
                totals[name] += sample[index].elapsed_time(sample[index + 1])
        self._learner_profile_events = []
        return {name: value / len(events) for name, value in totals.items()}

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
            self._prune_checkpoints()
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
