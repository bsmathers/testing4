"""One-shot Epoch475 continuation with the Ataraxos training recipe."""

from __future__ import annotations

import os
import random
import re
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
from torch.utils.data import DataLoader, get_worker_info

from metamon.rl import ataraxos_keys as keys
from metamon.rl import taurosv1b_online as v1b
from metamon.rl.recent_schedule_free import RecentScheduleFreeAdamW
from metamon.env.vectorized.amago_policy import normalize_discrete_policy_actions


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
        threshold = 0.0 if self.owner is None else self.owner.adv_threshold
        data.obs[keys.ADV_THRESHOLD] = torch.full((T, 1), float(threshold))
        return data


class BalancedAtaraxosMixture(v1b.ReadyAwareOnlineMixture):
    """Low-discrepancy sampling with a conservative advantage-filter ramp."""

    _ADV_QUANTILE_START = 0.0
    _ADV_QUANTILE_END = 0.25
    _ADV_FLOOR_START = 0.0
    _ADV_FLOOR_END = 0.0
    _ADV_RAMP_END_EPOCH = 100

    def __init__(self, *, fifo, offline, **kwargs):
        self.adv_threshold = 0.0
        self.adv_quantile = self._ADV_QUANTILE_START
        self.adv_floor = self._ADV_FLOOR_START
        self.current_online_weight = 0.0
        self._sample_idx = 0
        tagged_fifo = TaggedDataset(fifo, online=True, owner=self)
        tagged_offline = TaggedDataset(offline, online=False, owner=self)
        super().__init__(fifo=tagged_fifo, offline=tagged_offline, **kwargs)

    _ONLINE_CONTEXT_BURN_IN = 32

    def __getitem__(self, i):
        """Sample with a context burn-in for online PPO trajectories.

        AMAGO normally takes an arbitrary max-seq-len window. For a transformer
        policy that is incorrect for stored PPO behavior probabilities at the
        left edge of a mid-episode window: the collector had the preceding
        attention context, while the learner would not. Snorlax uses a 32-step
        left attention window, so prepend 32 context steps and exclude them from
        behavior-valid losses. Offline imitation keeps AMAGO's ordinary sampler.
        """
        self.check_configured()
        data = self.sample_random_trajectory()
        if self.max_seq_len is None:
            return data
        is_online = bool(data.obs[keys.SOURCE][0, 0].item())
        if not is_online or len(data) <= self.max_seq_len:
            return data.random_slice(
                length=self.max_seq_len, padded_sampling=self.padded_sampling
            )

        burn_in = min(self._ONLINE_CONTEXT_BURN_IN, self.max_seq_len - 1)
        train_len = self.max_seq_len - burn_in
        # Choose the supervised region uniformly, including the terminal end of
        # long games. The physical slice starts up to burn_in steps earlier.
        train_start = random.randrange(0, len(data) - train_len + 1)
        slice_start = max(0, train_start - burn_in)
        slice_stop = slice_start + self.max_seq_len
        valid_start = train_start - slice_start
        valid_stop = valid_start + train_len

        tcp = slice(slice_start, slice_stop + 1)
        tc = slice(slice_start, slice_stop)
        data.obs = {k: v[tcp] for k, v in data.obs.items()}
        data.rl2s = data.rl2s[tcp]
        data.time_idxs = data.time_idxs[tcp]
        data.dones = data.dones[tc]
        data.rews = data.rews[tc]
        data.actions = data.actions[tc]

        behavior_valid = data.obs[keys.BEHAVIOR_VALID]
        behavior_valid[:valid_start] = False
        behavior_valid[valid_stop:] = False
        return data

    def sample_random_trajectory(self):
        # Deterministic low-discrepancy allocation avoids Bernoulli source-count
        # noise. At the configured 0.5 weight this alternates sources and yields
        # exactly 7/7 in every batch of 14.
        worker = get_worker_info()
        # Each loader worker owns a private counter. Stagger their phases so
        # arbitrary configured weights remain well distributed across workers.
        # The 50% setting remains exactly 7/7 in every 14-sample batch.
        phase = 0.0 if worker is None else worker.id / float(worker.num_workers)
        before = int(
            self._sample_idx * self.current_online_weight + phase + 1e-9
        )
        after = int(
            (self._sample_idx + 1) * self.current_online_weight + phase + 1e-9
        )
        dataset = self.fifo if after > before else self.offline
        self._sample_idx += 1
        if not self.fifo.ready_for_training:
            raise RuntimeError("Ataraxos learner started before the 5,000-game FIFO was ready")
        self._sampling_metrics[dataset.dset_name] += 1
        return dataset.sample_random_trajectory()

    @classmethod
    def _advantage_filter_params(cls, epoch: int) -> tuple[float, float]:
        """Ramp from no filtering to retaining the top 75% of online states."""
        end = max(int(cls._ADV_RAMP_END_EPOCH), 1)
        progress = min(max(float(epoch) / end, 0.0), 1.0)
        quantile = (
            cls._ADV_QUANTILE_START
            + progress * (cls._ADV_QUANTILE_END - cls._ADV_QUANTILE_START)
        )
        floor = (
            cls._ADV_FLOOR_START
            + progress * (cls._ADV_FLOOR_END - cls._ADV_FLOOR_START)
        )
        return float(quantile), float(floor)

    def update_dset_weights(self, epoch: int):
        super().update_dset_weights(epoch)
        self.current_online_weight = next(
            weight for dataset, weight in self._available_datasets
            if dataset is self.fifo
        )
        self._sample_idx = 0
        self.adv_quantile, self.adv_floor = self._advantage_filter_params(epoch)
        if self.fifo.ready_for_training and self.current_online_weight > 0.0:
            self.adv_threshold = self._compute_advantage_threshold(
                quantile=self.adv_quantile,
                floor=self.adv_floor,
            )
        print(
            f"  [Ataraxos] epoch={epoch} "
            f"online={self.current_online_weight:.4f} "
            f"offline={1.0 - self.current_online_weight:.4f} "
            f"adv_quantile={self.adv_quantile:.4f} "
            f"adv_floor={self.adv_floor:.6f} "
            f"adv_threshold={self.adv_threshold:.6f}", flush=True
        )

    def _compute_advantage_threshold(
        self, *, quantile: float, floor: float
    ) -> float:
        parsed = self.fifo.dataset.parsed_replay_dset
        values = []
        wins = losses = unresolved = 0
        filenames = list(parsed.filenames)
        for filename in filenames:
            try:
                if filename.endswith(".lz4"):
                    with lz4.frame.open(filename, "rb") as f:
                        payload = orjson.loads(f.read())
                else:
                    with open(filename, "rb") as f:
                        payload = orjson.loads(f.read())
                if payload.get("ataraxos_collector_version") != 3:
                    raise ValueError(
                        "replay predates the native outcome and lane/action fixes"
                    )
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
                won = bool(final.get("battle_won"))
                lost = bool(final.get("battle_lost"))
                if won and lost:
                    raise ValueError("final state is both won and lost")
                if won:
                    wins += 1
                    outcome = 1.0
                elif lost:
                    losses += 1
                    outcome = -1.0
                else:
                    unresolved += 1
                    outcome = 0.0
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
        if unresolved:
            raise RuntimeError(
                "Ataraxos FIFO contains unresolved trajectories without a valid "
                f"terminal zero-sum target: {unresolved}"
            )
        if not values:
            raise RuntimeError("could not compute Ataraxos advantage threshold from FIFO")
        decisive = wins + losses
        if decisive >= 100 and (wins == 0 or losses == 0):
            raise RuntimeError(
                "Ataraxos FIFO has a one-sided outcome collapse: "
                f"wins={wins}, losses={losses}, unresolved={unresolved}"
            )
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"advantage-filter quantile must be in [0, 1], got {quantile}")
        if floor < 0.0:
            raise ValueError(f"advantage-filter floor must be nonnegative, got {floor}")
        if quantile == 0.0 and floor == 0.0:
            return 0.0
        return max(float(np.quantile(np.asarray(values), quantile)), float(floor))


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

    @staticmethod
    def _collection_ready_marker() -> Path:
        return Path(os.environ["SNORLAX_LOG_DIR"]) / ".collection-ready"

    @staticmethod
    def _collection_outcome_report(epoch: int) -> Path:
        return (
            Path(os.environ["SNORLAX_LOG_DIR"])
            / f"collection-epoch-{epoch}-outcomes.txt"
        )

    @classmethod
    def _purge_unresolved_online_replays(cls, replay_dir: Path) -> int:
        """Remove schema-v3 trajectories without a decisive terminal outcome.

        Earlier Snorlax collectors persisted turn-limit boundaries and simulator
        ties. The learner then interpreted their missing terminal result as a
        zero payoff, which is not a valid target for an unfinished game. New
        collectors no longer write these files; this cleanup makes an existing
        FIFO safe to resume after upgrading.
        """
        removed = 0
        for name in cls._replay_names(replay_dir):
            path = replay_dir / name
            try:
                if name.endswith(".lz4"):
                    with lz4.frame.open(path, "rb") as stream:
                        payload = orjson.loads(stream.read())
                else:
                    with path.open("rb") as stream:
                        payload = orjson.loads(stream.read())
                if payload.get("ataraxos_collector_version") != 3:
                    continue
                final = payload["states"][-1]
                won = bool(final.get("battle_won"))
                lost = bool(final.get("battle_lost"))
                if won and lost:
                    raise ValueError("final state is both won and lost")
                if not won and not lost:
                    path.unlink()
                    removed += 1
            except (
                OSError,
                IndexError,
                KeyError,
                TypeError,
                ValueError,
                orjson.JSONDecodeError,
            ) as exc:
                raise RuntimeError(
                    f"cannot validate Snorlax FIFO outcome for {path}: {exc}"
                ) from exc
        return removed

    @classmethod
    def _count_collection_outcomes(
        cls, replay_dir: Path, replay_names: set[str]
    ) -> dict[str, int | float]:
        wins = losses = unresolved = 0
        for name in replay_names:
            path = replay_dir / name
            try:
                if name.endswith(".lz4"):
                    with lz4.frame.open(path, "rb") as stream:
                        payload = orjson.loads(stream.read())
                else:
                    with path.open("rb") as stream:
                        payload = orjson.loads(stream.read())
                if payload.get("ataraxos_collector_version") != 3:
                    raise ValueError("trajectory is not Snorlax collector schema v3")
                final = payload["states"][-1]
                won = bool(final.get("battle_won"))
                lost = bool(final.get("battle_lost"))
                if won and lost:
                    raise ValueError("final state is both won and lost")
                if won:
                    wins += 1
                elif lost:
                    losses += 1
                else:
                    unresolved += 1
            except (
                OSError,
                IndexError,
                KeyError,
                TypeError,
                ValueError,
                orjson.JSONDecodeError,
            ) as exc:
                raise RuntimeError(
                    f"cannot determine collection outcome for {path}: {exc}"
                ) from exc
        games = wins + losses + unresolved
        decisive = wins + losses
        return {
            "Games": games,
            "Wins": wins,
            "Losses": losses,
            "Unresolved": unresolved,
            "Win Rate": wins / decisive if decisive else 0.0,
            "Decisive Fraction": decisive / games if games else 0.0,
        }

    @classmethod
    def _write_collection_outcome_report(
        cls, epoch: int, metrics: dict[str, int | float]
    ) -> None:
        report = cls._collection_outcome_report(epoch)
        temporary = report.with_suffix(".tmp")
        lines = [f"Collection epoch: {epoch}"]
        lines.extend(f"{name}: {value}" for name, value in metrics.items())
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, report)

    @classmethod
    def _read_collection_outcome_report(
        cls, epoch: int
    ) -> dict[str, int | float] | None:
        report = cls._collection_outcome_report(epoch)
        try:
            lines = report.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return None
        parsed: dict[str, int | float] = {}
        integer_names = {"Games", "Wins", "Losses", "Unresolved"}
        for line in lines[1:]:
            name, separator, raw_value = line.partition(": ")
            if not separator:
                raise RuntimeError(f"malformed collection report: {report}")
            parsed[name] = (
                int(raw_value) if name in integer_names else float(raw_value)
            )
        required = integer_names | {"Win Rate", "Decisive Fraction"}
        if set(parsed) != required:
            raise RuntimeError(f"incomplete collection report: {report}")
        return parsed

    def _collection_is_ready(self, epoch: int) -> bool:
        marker = self._collection_ready_marker()
        expected = f"collection complete for learner epoch {epoch}"
        try:
            ready = marker.read_text(encoding="utf-8").strip() == expected
        except FileNotFoundError:
            ready = False
        if not ready:
            # Recover runs made before the durable marker existed. The learner
            # log records the exact completed count before entering training.
            log_path = Path(os.environ["SNORLAX_LOG_DIR"]) / "learner.log"
            try:
                with log_path.open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    size = stream.tell()
                    stream.seek(max(0, size - 1_000_000))
                    tail = stream.read().decode("utf-8", errors="replace")
            except FileNotFoundError:
                tail = ""
            matches = re.findall(
                rf"\[PersistentCycle\] collect epoch={epoch} (\d+)/(\d+)", tail
            )
            ready = bool(matches and int(matches[-1][0]) >= int(matches[-1][1]))
            if ready:
                self._mark_collection_ready(epoch)
        # Do not trust a retained marker/log after the replay buffer was erased
        # or only partially restored on a replacement instance.
        min_fifo = int(os.environ.get("ONLINE_FIFO_SIZE", "5000"))
        replay_dir = Path(os.environ["SNORLAX_BUFFER_DIR"]) / "gen1ou"
        try:
            replay_count = len(self._replay_names(replay_dir))
        except FileNotFoundError:
            replay_count = 0
        if ready and replay_count < min_fifo:
            ready = False
        if not ready:
            marker.unlink(missing_ok=True)
        return ready

    def _mark_collection_ready(self, epoch: int) -> None:
        marker = self._collection_ready_marker()
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(
            f"collection complete for learner epoch {epoch}\n", encoding="utf-8"
        )
        os.replace(temporary, marker)

    def _write_behavior_policy(self) -> str:
        """Publish the Schedule-Free y iterate used for PPO data collection.

        Schedule-Free evaluates gradients at y but validation/checkpoints at the
        averaged x iterate. In on-policy RL the rollout distribution is part of
        the objective, so collectors must use y as well; otherwise the very first
        PPO ratio after collection already includes an optimizer-internal x->y
        policy change. This conversion is reversible and does not advance the
        optimizer.
        """
        path = os.environ.get("SNORLAX_BEHAVIOR_WEIGHTS")
        if not path:
            latest = Path(os.environ["SNORLAX_LATEST_WEIGHTS"])
            path = str(latest.with_name("behavior_policy.pt"))
        temporary = path + ".tmp"

        # Persistent collection boundaries leave the optimizer in eval/x mode.
        # Be defensive on manual entry points so the saved artifact is always y.
        self.optimizer.eval()
        self.optimizer.train()
        try:
            if self.accelerator.is_main_process:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(self.policy.state_dict(), temporary)
                os.replace(temporary, path)
            self.accelerator.wait_for_everyone()
        finally:
            # Restore x for validation/checkpoint semantics and so repeated calls
            # are idempotent.
            self.optimizer.eval()
        return path

    def _external_collect_phase(self, epoch: int) -> dict[str, int | float] | None:
        """Run direct collector subprocesses, matching the fast pre-warm design."""
        buffer_dir = Path(os.environ["SNORLAX_BUFFER_DIR"])
        replay_dir = buffer_dir / "gen1ou"
        log_dir = Path(os.environ["SNORLAX_LOG_DIR"])
        behavior_weights = os.environ.get(
            "SNORLAX_BEHAVIOR_WEIGHTS", os.environ["SNORLAX_LATEST_WEIGHTS"]
        )
        target = int(os.environ.get("GAMES_PER_LEARNER_EPOCH", "2000"))
        removed_unresolved = self._purge_unresolved_online_replays(replay_dir)
        if removed_unresolved:
            print(
                f"[PersistentCycle] removed {removed_unresolved} unresolved "
                "schema-v3 replay(s) from the FIFO",
                flush=True,
            )
        min_fifo = int(os.environ.get("ONLINE_FIFO_SIZE", "5000"))
        current_fifo = len(self._replay_names(replay_dir))
        target = max(target, max(min_fifo - current_fifo, 0))
        processes = int(os.environ.get("COLLECTOR_PROCESSES", "8"))
        lanes = int(os.environ.get("LANES", "64"))
        workers = int(os.environ.get("COLLECTOR_WORKERS", "8"))
        dtype = os.environ.get("COLLECTOR_DTYPE", "bf16")
        collector_mp = "no" if dtype == "fp32" else dtype
        run_config = os.environ["SNORLAX_RUN_CONFIG"]
        module = os.environ.get("RUNNER_MODULE", "metamon.rl.ataraxos_online")
        save_dir = os.environ["METAMON_SAVE_DIR"]
        if self._collection_is_ready(epoch):
            print(
                f"[PersistentCycle] collect epoch={epoch} already complete; "
                "resuming learner",
                flush=True,
            )
            return self._read_collection_outcome_report(epoch)
        baseline = self._replay_names(replay_dir)
        self.optimizer.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        children: list[tuple[subprocess.Popen, object]] = []
        print(
            f"[PersistentCycle] collect epoch={epoch} target={target} "
            f"processes={processes} lanes={lanes}",
            flush=True,
        )
        try:
            for idx in range(processes):
                log_path = log_dir / f"collector-epoch-{epoch}-{idx}.log"
                stream = log_path.open("a", encoding="utf-8", buffering=1)
                child_env = dict(os.environ)
                child_env.pop("SNORLAX_PERSISTENT_ALTERNATION", None)
                child_env.update(
                    METAMON_RUN_MODE="collect",
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
                    "--base_weights", behavior_weights,
                    "--epochs", "1000000",
                    "--seed", str(1000 + epoch * processes + idx),
                    "--mixed_precision", collector_mp,
                ]
                children.append(
                    (
                        subprocess.Popen(
                            command,
                            env=child_env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                        ),
                        stream,
                    )
                )
            fresh = 0
            while fresh < target:
                time.sleep(10)
                for child, _stream in children:
                    status = child.poll()
                    if status is not None:
                        raise RuntimeError(
                            f"collector exited with status {status}; inspect "
                            f"{log_dir}/collector-epoch-{epoch}-*.log"
                        )
                fresh = len(self._replay_names(replay_dir) - baseline)
                print(f"[PersistentCycle] collect epoch={epoch} {fresh}/{target}", flush=True)
        finally:
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
        fresh_names = self._replay_names(replay_dir) - baseline
        metrics = self._count_collection_outcomes(replay_dir, fresh_names)
        self._write_collection_outcome_report(epoch, metrics)
        print(
            f"[PersistentCycle] outcomes epoch={epoch} "
            f"wins={metrics['Wins']} losses={metrics['Losses']} "
            f"unresolved={metrics['Unresolved']} "
            f"win_rate={metrics['Win Rate']:.4f}",
            flush=True,
        )
        self._mark_collection_ready(epoch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

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
            raise RuntimeError(
                f"validation epoch={epoch} exited with status {result.returncode}; "
                f"inspect {validation_log}"
            )

    def learn(self):
        if os.environ.get("SNORLAX_PERSISTENT_ALTERNATION") != "1":
            return super().learn()

        from tqdm import tqdm

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.epochs):
            self.epoch = epoch
            self.policy_aclr.eval()
            behavior_path = self._write_behavior_policy()
            print(
                f"[PersistentCycle] behavior policy epoch={epoch} {behavior_path}",
                flush=True,
            )
            collection_log = self._external_collect_phase(epoch)
            self.accelerator.wait_for_everyone()

            if collection_log is not None:
                self.log(collection_log, key="collection")
            else:
                print(
                    f"[PersistentCycle] collection epoch={epoch} predates "
                    "per-round outcome logging; W&B starts next round",
                    flush=True,
                )

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
            self._collection_ready_marker().unlink(missing_ok=True)
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
            if rl2s.shape != (self.parallel_actors, self.policy.action_dim + 1):
                raise RuntimeError(
                    f"collector RL2 batch has shape {rl2s.shape}, expected "
                    f"{(self.parallel_actors, self.policy.action_dim + 1)}"
                )
            if tids.shape != (self.parallel_actors, 1):
                raise RuntimeError(
                    f"collector time-index batch has shape {tids.shape}, expected "
                    f"{(self.parallel_actors, 1)}"
                )
            bad_obs = {
                key: value.shape
                for key, value in obs.items()
                if value.shape[0] != self.parallel_actors
            }
            if bad_obs:
                raise RuntimeError(f"collector observations lost their lane axis: {bad_obs}")
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
            actions = normalize_discrete_policy_actions(
                actions, self.parallel_actors, context="collector policy"
            )
            expected_logp = (self.parallel_actors, self.policy.action_dim)
            expected_values = (self.parallel_actors, 1)
            if tuple(behavior_lp.shape) != expected_logp:
                raise RuntimeError(
                    f"collector behavior logits have shape {tuple(behavior_lp.shape)}, "
                    f"expected {expected_logp}"
                )
            if tuple(behavior_v.shape) != expected_values:
                raise RuntimeError(
                    f"collector values have shape {tuple(behavior_v.shape)}, "
                    f"expected {expected_values}"
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
            # AMAGOEnv.make_action_rep indexes ``action[..., 0]``.  Preserve
            # the discrete action's trailing singleton dimension here; passing
            # a flat (lanes,) array makes it select lane 0's action once and
            # write that same previous-action one-hot into every lane's RL2
            # input.  The raw vector env accepts either shape after reshape().
            env_actions = host[:, 0:1].astype(np.int64, copy=False)
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
            *_, terminated, truncated, _ = envs.step(env_actions)
            if profile:
                profile_times["env_step"] += time.perf_counter() - started
                started = time.perf_counter()
            done = terminated | truncated
            if done.ndim == 2:
                if done.shape != (self.parallel_actors, 1):
                    raise RuntimeError(
                        f"collector done batch has shape {done.shape}, expected "
                        f"{(self.parallel_actors, 1)}"
                    )
                done = done.squeeze(1)
            if done.shape != (self.parallel_actors,):
                raise RuntimeError(
                    f"collector done batch has shape {done.shape}, expected "
                    f"{(self.parallel_actors,)}"
                )
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
        strict_numerics = os.environ.get("METAMON_STRICT_NUMERICS") == "1"
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
            if strict_numerics and not bool(torch.isfinite(loss).all()):
                raise RuntimeError("Ataraxos smoke update produced a non-finite loss")
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
                if strict_numerics:
                    for name, parameter in self.policy.named_parameters():
                        gradient = parameter.grad
                        if gradient is not None and not bool(torch.isfinite(gradient).all()):
                            raise RuntimeError(
                                f"Ataraxos smoke update produced a non-finite gradient: {name}"
                            )
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
            if strict_numerics and self.accelerator.sync_gradients:
                for name, parameter in self.policy.named_parameters():
                    if not bool(torch.isfinite(parameter).all()):
                        raise RuntimeError(
                            f"Ataraxos smoke update produced a non-finite parameter: {name}"
                        )
                for state in self.optimizer.state.values():
                    for value in state.values():
                        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                            raise RuntimeError(
                                "Ataraxos smoke update produced non-finite optimizer state"
                            )
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
