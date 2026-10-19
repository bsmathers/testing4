"""Batched in-the-loop opponents for the vectorized Showdown env.

The env synchronizes all lanes at each decision cycle and asks the opponent for
one action per *active* lane in a single call, so opponent inference is amortized
across the batch. Two implementations are provided:

  * :class:`RandomBatchedOpponent` — no NN; returns random action indices (the env
    repairs any illegal pick against the legal mask). Useful for smoke tests.
  * :class:`AmagoBatchedOpponent` — wraps a metamon policy via
    :class:`~metamon.env.vectorized.amago_policy.AmagoLadderPolicyDriver`, using
    the same ``rl2`` / ``time_idx`` bookkeeping as ``QueueOnLocalLadder``.
  * :class:`ConfigBatchedOpponent` — one shared policy for all lanes; on full env
    ``reset()``, sample an opponent from an :class:`~metamon.rl.evaluate.opponent_pool.OpponentPoolConfig`.
    Pool entries may mix action dimensions: each cached policy bundle owns its own
    ``rl2`` buffer and hidden state; ``configure()`` swaps bundles and reinitializes.
    ``configure_groups()`` instead partitions the lanes into groups that each
    play a different opponent, so one collector process covers several league
    rows at once. Each group owns a driver sized to its own lanes (exact
    per-lane recurrent state); groups with the same model, checkpoint, and
    temperature share one set of policy weights.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch

from .amago_policy import AmagoLadderPolicyDriver, optimize_policy_for_collection

if TYPE_CHECKING:
    from metamon.env.wrappers import TeamSet
    from metamon.rl.evaluate.common import PolicySpec


class BatchedOpponent(ABC):
    """Interface the env uses to drive the in-the-loop opponent."""

    @abstractmethod
    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        """Return an action index per lane.

        Only entries where ``active[i]`` is True are guaranteed meaningful (and
        only those lanes advance any internal recurrent state). ``obs_list`` has
        one obs dict per lane (the env supplies a cached obs for inactive lanes).
        """

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        """Record the reward/action for ``lane_idx`` (rl2 bookkeeping)."""

    def reset_lanes(self, done_mask: np.ndarray) -> None:
        """Reset per-lane recurrent state for finished lanes."""

    def reset_all(self) -> None:
        """Reset all per-lane state (called on env.reset)."""


class RandomBatchedOpponent(BatchedOpponent):
    def __init__(
        self, num_lanes: int, action_dim: int, rng: Optional[np.random.Generator] = None
    ):
        self.num_lanes = num_lanes
        self.action_dim = action_dim
        self.rng = rng or np.random.default_rng()

    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        return self.rng.integers(0, self.action_dim, size=self.num_lanes).astype(
            np.int64
        )


class AmagoBatchedOpponent(BatchedOpponent):
    """Wrap an AMAGO policy with ladder-identical rollout bookkeeping."""

    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        num_lanes: int,
        action_dim: int,
        hidden_state=None,
        sample: bool = True,
    ):
        self._driver = AmagoLadderPolicyDriver(
            policy=policy,
            device=device,
            num_lanes=num_lanes,
            action_dim=action_dim,
            hidden_state=hidden_state,
            sample=sample,
        )
        self.action_dim = int(action_dim)

    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        return self._driver.act(active, obs_list)

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        self._driver.observe(lane_idx, reward, action_idx)

    def reset_lanes(self, done_mask: np.ndarray) -> None:
        self._driver.reset_lanes(done_mask)

    def reset_all(self) -> None:
        self._driver.reset_all()


class ConfigBatchedOpponent(BatchedOpponent):
    """One opponent shared by all lanes; resample from config on env ``reset()`` only."""

    def __init__(
        self,
        config: "OpponentPoolConfig",
        num_lanes: int,
        device: torch.device,
        sample: bool = True,
        cache_size: int = 1,
    ):
        from metamon.rl.evaluate.opponent_pool import OpponentPoolConfig

        if not isinstance(config, OpponentPoolConfig):
            raise TypeError(f"config must be OpponentPoolConfig, got {type(config)}")
        self.config = config
        self.num_lanes = int(num_lanes)
        self.device = device
        self.sample = sample
        # Bound how many opponent policies stay resident on the GPU. Each cached
        # bundle holds a full policy (60-200M params) plus per-lane KV caches, so
        # an unbounded cache OOMs collectors that resample a new opponent every
        # epoch. LRU-evict and free GPU memory beyond this many distinct opponents.
        self._cache_size = max(1, int(cache_size))
        self.current_spec: Optional["PolicySpec"] = None
        self.current_team: Optional["TeamSet"] = None
        self._active_key: Optional[str] = None
        self._bundle: Optional[AmagoBatchedOpponent] = None
        self._cache: "OrderedDict[str, AmagoBatchedOpponent]" = OrderedDict()
        # Lane-group mode (see ``configure_groups``).
        self._groups: Optional[List[dict]] = None
        self._lane_group: Optional[np.ndarray] = None
        self._lane_local: Optional[np.ndarray] = None

    def _load_policy(self, spec: "PolicySpec"):
        from metamon.rl.pretrained import get_pretrained_model

        model = get_pretrained_model(spec.model_name)
        agent = model.initialize_agent(
            checkpoint=spec.checkpoint,
            log=False,
            action_temperature=spec.temperature,
        )
        action_dim = model.action_space.gym_space.n
        policy = optimize_policy_for_collection(agent.policy, self.device)
        return policy, action_dim

    def _make_bundle(
        self, spec: "PolicySpec", num_lanes: Optional[int] = None, loaded=None
    ) -> AmagoBatchedOpponent:
        policy, action_dim = loaded if loaded is not None else self._load_policy(spec)
        return AmagoBatchedOpponent(
            policy=policy,
            device=self.device,
            num_lanes=self.num_lanes if num_lanes is None else int(num_lanes),
            action_dim=action_dim,
            sample=self.sample,
        )

    @staticmethod
    def _weights_key(spec: "PolicySpec") -> str:
        # Team set does not change the policy; share weights across it.
        return (
            f"{spec.model_name}_ckpt{spec.checkpoint}_t{spec.temperature}_"
            f"{spec.battle_backend}"
        )

    def _clear_groups(self) -> None:
        if self._groups is None:
            return
        for group in self._groups:
            self._free_bundle(group["bundle"])
        self._groups = None
        self._lane_group = None
        self._lane_local = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _clear_cache(self) -> None:
        while self._cache:
            _, bundle = self._cache.popitem(last=False)
            self._free_bundle(bundle)
        self._bundle = None
        self._active_key = None

    @property
    def grouped(self) -> bool:
        return self._groups is not None

    def configure_groups(
        self, specs: List["PolicySpec"], lane_indices: List[List[int]]
    ) -> List["PolicySpec"]:
        """Give each disjoint lane group its own opponent until the next configure."""
        if not specs or len(specs) != len(lane_indices):
            raise ValueError("configure_groups needs one lane list per opponent spec")
        flat = sorted(int(i) for lanes in lane_indices for i in lanes)
        if flat != list(range(self.num_lanes)):
            raise ValueError(
                "opponent lane groups must partition all "
                f"{self.num_lanes} lanes exactly once"
            )
        if any(len(lanes) == 0 for lanes in lane_indices):
            raise ValueError("opponent lane groups must be non-empty")
        self._clear_groups()
        # Grouped mode replaces the single-opponent cache; free it first so
        # old and new opponents never coexist on the GPU.
        self._clear_cache()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        loaded: dict = {}
        groups: List[dict] = []
        lane_group = np.full((self.num_lanes,), -1, dtype=np.int64)
        lane_local = np.full((self.num_lanes,), -1, dtype=np.int64)
        for index, (spec, lanes) in enumerate(zip(specs, lane_indices)):
            lanes_arr = np.asarray(sorted(int(i) for i in lanes), dtype=np.int64)
            key = self._weights_key(spec)
            if key not in loaded:
                loaded[key] = self._load_policy(spec)
            bundle = self._make_bundle(spec, num_lanes=len(lanes_arr), loaded=loaded[key])
            groups.append(
                {
                    "spec": spec,
                    "lanes": lanes_arr,
                    "bundle": bundle,
                    "team": self.config.team_set_for(spec.team_set),
                }
            )
            lane_group[lanes_arr] = index
            lane_local[lanes_arr] = np.arange(len(lanes_arr), dtype=np.int64)
        self._groups = groups
        self._lane_group = lane_group
        self._lane_local = lane_local
        self.current_spec = groups[0]["spec"]
        self.current_team = groups[0]["team"]
        return [group["spec"] for group in groups]

    def group_team(self, index: int) -> "TeamSet":
        if self._groups is None:
            raise RuntimeError("opponent is not in lane-group mode")
        return self._groups[index]["team"]

    def _free_bundle(self, bundle: AmagoBatchedOpponent) -> None:
        """Drop a bundle's GPU tensors (policy weights + per-lane KV caches)."""
        driver = getattr(bundle, "_driver", None)
        if driver is not None:
            driver.hidden_state = None
            driver._scratch_hidden.clear()
            driver.policy = None

    def configure(self, spec: Optional["PolicySpec"] = None) -> "PolicySpec":
        """Activate one sampled (or explicit) opponent for all lanes."""
        self._clear_groups()
        if spec is None:
            spec = self.config.sample_opponent()
        self.current_spec = spec
        self.current_team = self.config.team_set_for(spec.team_set)
        key = spec.unique_key

        from metamon.rl.pretrained import LATEST_CHECKPOINT

        if spec.checkpoint == LATEST_CHECKPOINT and key in self._cache:
            # The rolling ``latest/policy.pt`` is overwritten by the learner each
            # epoch, but ``unique_key`` is constant (``..._ckpt-1_...``). Drop any
            # cached bundle for it so the miss path below reloads fresh weights
            # from disk. Numbered epochs are immutable and stay LRU-cached.
            stale = self._cache.pop(key)
            self._free_bundle(stale)
            del stale
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        made_new = False
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            # Make room before loading so two full policies and KV caches never
            # coexist merely because the league changed opponents.
            while len(self._cache) >= self._cache_size:
                _, old_bundle = self._cache.popitem(last=False)
                if old_bundle is self._bundle:
                    self._bundle = None
                self._free_bundle(old_bundle)
                del old_bundle
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._cache[key] = self._make_bundle(spec)
            made_new = True
        self._bundle = self._cache[key]
        self._active_key = key
        if not made_new:
            self._bundle.reset_all()
        return spec

    def _require_bundle(self) -> AmagoBatchedOpponent:
        if self._bundle is None:
            raise RuntimeError(
                "ConfigBatchedOpponent.configure() must run before act()"
            )
        return self._bundle

    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        if self._groups is None:
            return self._require_bundle().act(active, obs_list)
        active = np.asarray(active, dtype=bool)
        actions = np.zeros((self.num_lanes,), dtype=np.int64)
        for group in self._groups:
            lanes = group["lanes"]
            sub_active = active[lanes]
            if not sub_active.any():
                continue
            # Each group's driver only ever sees its own lanes, so mixed
            # observation formats across groups never get stacked together.
            sub_obs = [obs_list[int(i)] for i in lanes]
            actions[lanes] = group["bundle"].act(sub_active, sub_obs)
        return actions

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        if self._groups is None:
            self._require_bundle().observe(lane_idx, reward, action_idx)
            return
        group = self._groups[int(self._lane_group[lane_idx])]
        group["bundle"].observe(int(self._lane_local[lane_idx]), reward, action_idx)

    def reset_lanes(self, done_mask: np.ndarray) -> None:
        if self._groups is not None:
            done_mask = np.asarray(done_mask, dtype=bool)
            for group in self._groups:
                group["bundle"].reset_lanes(done_mask[group["lanes"]])
            return
        if self._bundle is not None:
            self._bundle.reset_lanes(done_mask)

    def reset_all(self) -> None:
        if self._groups is not None:
            for group in self._groups:
                group["bundle"].reset_all()
            return
        if self._bundle is not None:
            self._bundle.reset_all()
