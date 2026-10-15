"""AMAGO policy drivers that mirror QueueOnLocalLadder / MetamonAMAGOWrapper semantics.

Each :class:`AmagoLadderPolicyDriver` tracks per-lane ``rl2`` and ``time_idx`` the same
way ``AMAGOEnv`` + ``SequenceWrapper.current_timestep`` do during
``Experiment.interact`` (``rl2 = concat(reward, prev_action_one_hot)``,
``time_idx = step_count``).  Vectorized eval uses one driver for the evaluated agent
and the opponent stack uses the same class so both policies see identical inputs.
"""

from __future__ import annotations

import gc
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .obs_utils import numpy_obs_to_torch, stack_obs_dicts, unstack_obs_dicts


_INFERENCE_CHILDREN = frozenset({"tstep_encoder", "traj_encoder", "actor"})


def collector_inference_dtype(device: torch.device) -> torch.dtype:
    """Resolve the explicitly requested rollout dtype."""
    requested = os.environ.get("METAMON_COLLECTOR_DTYPE", "").strip().lower()
    if not requested:
        requested = "bf16" if device.type == "cuda" else "fp32"
    choices = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if requested not in choices:
        raise ValueError(
            "METAMON_COLLECTOR_DTYPE must be fp32, bf16, or fp16; "
            f"got {requested!r}"
        )
    dtype = choices[requested]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 collector inference is unsupported on CPU")
    return dtype


def optimize_policy_for_collection(
    policy: torch.nn.Module, device: torch.device
) -> torch.nn.Module:
    """Strip training-only children and cast a policy for action inference."""
    if getattr(policy, "_metamon_collection_optimized", False):
        return policy
    removed = []
    for name, _child in list(policy.named_children()):
        if name not in _INFERENCE_CHILDREN:
            setattr(policy, name, None)
            removed.append(name)
    # FlashAttention's causal local window only reads ``left`` prior keys plus
    # the current key. Retaining the training max sequence beyond left + 1 is
    # therefore redundant during cached one-step inference. AMAGO's cache rolls
    # while preserving absolute ``time_idxs``, so this is behavior-preserving.
    compact_cache = os.environ.get("METAMON_COMPACT_KV_CACHE", "1") == "1"
    cache_len = None
    traj_encoder = getattr(policy, "traj_encoder", None)
    if compact_cache and traj_encoder is not None:
        left_windows = []
        for module in traj_encoder.modules():
            window = getattr(module, "window_size", None)
            if (
                isinstance(window, (tuple, list))
                and len(window) == 2
                and isinstance(window[0], int)
                and window[0] >= 0
            ):
                left_windows.append(int(window[0]))
        if left_windows and hasattr(traj_encoder, "max_seq_len"):
            cache_len = max(left_windows) + 1
            traj_encoder.max_seq_len = min(
                int(traj_encoder.max_seq_len), int(cache_len)
            )
    dtype = collector_inference_dtype(device)
    policy.to(device=device, dtype=dtype)
    policy.eval()
    policy._metamon_collection_optimized = True
    policy._metamon_collection_dtype = dtype
    policy._metamon_removed_children = tuple(removed)
    policy._metamon_collection_cache_len = cache_len
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return policy


def load_collection_policy_state(policy: torch.nn.Module, state: dict) -> None:
    """Load a full learner checkpoint into an actor-only collection policy."""
    current = policy.state_dict()
    selected = {key: value for key, value in state.items() if key in current}
    missing = sorted(set(current) - set(selected))
    if missing:
        raise RuntimeError(
            "Collection checkpoint is missing actor-path tensors: "
            + ", ".join(missing[:8])
        )
    policy.load_state_dict(selected, strict=True)


class AmagoLadderPolicyDriver:
    """Batched policy forward matching AMAGO ladder rollouts."""

    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        num_lanes: int,
        action_dim: int,
        hidden_state=None,
        sample: bool = True,
    ):
        self.policy = policy
        self.device = device
        self.num_lanes = int(num_lanes)
        self.action_dim = int(action_dim)
        self.sample = sample
        self.rl2s = np.zeros((self.num_lanes, self.action_dim + 1), dtype=np.float32)
        self.step_counts = np.zeros((self.num_lanes,), dtype=np.int64)
        if hidden_state is None:
            hidden_state = self.policy.traj_encoder.init_hidden_state(
                self.num_lanes, self.device
            )
        self.hidden_state = hidden_state

    # Keep partial-lane inference memory bounded.  A Tauros-sized transformer KV
    # cache is large enough that cloning dozens of inactive lanes can OOM a 16 GB
    # learner/collector GPU.  Eight lanes keeps temporary cache storage small while
    # preserving a useful amount of batching.
    _PARTIAL_HIDDEN_CHUNK = 8

    def _snapshot_hidden(self, idx: np.ndarray) -> Optional[dict]:
        """Save a *small* set of lanes before a full-batch policy forward."""
        if len(idx) == 0:
            return None
        hs = self.hidden_state
        return {
            "idx": idx,
            "seq_lens": hs.seq_lens[idx].clone(),
            "key": hs.key_cache.data[:, idx].clone(),
            "val": hs.val_cache.data[:, idx].clone(),
        }

    def _restore_hidden(self, saved: Optional[dict]) -> None:
        if saved is None:
            return
        hs = self.hidden_state
        idx = saved["idx"]
        hs.seq_lens[idx] = saved["seq_lens"]
        hs.key_cache.data[:, idx] = saved["key"]
        hs.val_cache.data[:, idx] = saved["val"]

    @staticmethod
    def _copy_hidden_lanes(src, dst, src_idxs: np.ndarray) -> None:
        """Copy selected source lanes into a compact scratch hidden state."""
        for dst_i, src_i in enumerate(src_idxs):
            dst.seq_lens[dst_i].copy_(src.seq_lens[int(src_i)])
            dst.key_cache.data[:, dst_i].copy_(src.key_cache.data[:, int(src_i)])
            dst.val_cache.data[:, dst_i].copy_(src.val_cache.data[:, int(src_i)])

    @staticmethod
    def _copy_hidden_lanes_back(src, dst, dst_idxs: np.ndarray) -> None:
        """Scatter a compact scratch hidden state back into selected lanes."""
        for src_i, dst_i in enumerate(dst_idxs):
            dst.seq_lens[int(dst_i)].copy_(src.seq_lens[src_i])
            dst.key_cache.data[:, int(dst_i)].copy_(src.key_cache.data[:, src_i])
            dst.val_cache.data[:, int(dst_i)].copy_(src.val_cache.data[:, src_i])

    def _policy_act(self, obs_list: List[dict], lane_idxs: np.ndarray, hidden_state):
        obs_batch = stack_obs_dicts([obs_list[int(i)] for i in lane_idxs])
        torch_obs = numpy_obs_to_torch(obs_batch, self.device)
        rl2s = (
            torch.from_numpy(self.rl2s[lane_idxs]).to(self.device).unsqueeze(1)
        )
        time_idxs = (
            torch.from_numpy(self.step_counts[lane_idxs])
            .to(self.device)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        dtype = getattr(self.policy, "_metamon_collection_dtype", torch.float32)
        autocast = (
            torch.autocast(device_type=self.device.type, dtype=dtype)
            if dtype != torch.float32 and self.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            act_out, hidden_state = self.policy.get_actions(
                obs=torch_obs,
                rl2s=rl2s,
                time_idxs=time_idxs,
                hidden_state=hidden_state,
                sample=self.sample,
            )
        out = act_out.squeeze(1).cpu().numpy().astype(np.int64).reshape(-1)
        return out, hidden_state

    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        """Return action indices; only ``active`` lanes advance recurrent state."""
        actions = np.zeros((self.num_lanes,), dtype=np.int64)
        active_idxs = np.flatnonzero(active)
        if len(active_idxs) == 0:
            return actions

        # Fast path: all lanes advance, so there is nothing to preserve and no
        # temporary KV-cache allocation is required.
        if len(active_idxs) == self.num_lanes:
            out, self.hidden_state = self._policy_act(
                obs_list, active_idxs, self.hidden_state
            )
            actions[:] = out
            return actions

        inactive_idxs = np.flatnonzero(~active)

        # If only a handful of lanes are inactive, a single full-batch forward is
        # faster than compacting the active side.  Crucially, cap the snapshot at
        # _PARTIAL_HIDDEN_CHUNK lanes so this path can never recreate the old
        # hundreds-of-MiB transient allocation.
        if len(inactive_idxs) <= self._PARTIAL_HIDDEN_CHUNK:
            saved = self._snapshot_hidden(inactive_idxs)
            all_idxs = np.arange(self.num_lanes, dtype=np.int64)
            out, self.hidden_state = self._policy_act(
                obs_list, all_idxs, self.hidden_state
            )
            self._restore_hidden(saved)
            actions[active_idxs] = out[active_idxs]
            return actions

        # Otherwise run only active lanes, in bounded-size chunks.  AMAGO's
        # TformerHiddenState has a preallocated per-batch KV cache, so initialize
        # a compact scratch state and copy lanes one at a time (copy_ from a view)
        # rather than materializing a large advanced-indexing gather.
        for start in range(0, len(active_idxs), self._PARTIAL_HIDDEN_CHUNK):
            chunk = active_idxs[start : start + self._PARTIAL_HIDDEN_CHUNK]
            scratch = self.policy.traj_encoder.init_hidden_state(
                len(chunk), self.device
            )
            self._copy_hidden_lanes(self.hidden_state, scratch, chunk)
            out, scratch = self._policy_act(obs_list, chunk, scratch)
            self._copy_hidden_lanes_back(scratch, self.hidden_state, chunk)
            actions[chunk] = out
            del scratch

        return actions

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        """Record env feedback after a committed decision (matches AMAGOEnv.step)."""
        self.step_counts[lane_idx] += 1
        self.rl2s[lane_idx] = 0.0
        self.rl2s[lane_idx, 0] = float(reward)
        if 0 <= action_idx < self.action_dim:
            self.rl2s[lane_idx, 1 + action_idx] = 1.0

    def reset_lanes(self, done_mask: np.ndarray) -> None:
        if not done_mask.any():
            return
        for i in np.where(done_mask)[0]:
            self.step_counts[i] = 0
            self.rl2s[i] = 0.0
        self.hidden_state = self.policy.traj_encoder.reset_hidden_state(
            self.hidden_state, torch.as_tensor(done_mask, device=self.device)
        )

    def reset_all(self) -> None:
        self.step_counts[:] = 0
        self.rl2s[:] = 0.0
        # Reuse the preallocated KV cache instead of transiently holding two.
        hs = self.hidden_state
        try:
            hs.seq_lens.zero_()
            hs.key_cache.data.zero_()
            hs.val_cache.data.zero_()
        except (AttributeError, RuntimeError):
            self.hidden_state = self.policy.traj_encoder.init_hidden_state(
                self.num_lanes, self.device
            )


def vectorized_ladder_eval(
    policy: torch.nn.Module,
    device: torch.device,
    make_env: callable,
    total_battles: int,
    action_dim: int,
    sample: bool = True,
    timesteps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run vectorized Showdown eval with symmetric ladder-style policy drivers.

    The evaluated agent and in-the-loop opponent both use :class:`AmagoLadderPolicyDriver`
    with identical ``get_actions`` bookkeeping.  This replaces AMAGO ``interact`` for
    batched Showdown eval so the eval agent is not on a different code path than the
    opponent.
    """
    from .opponent import AmagoBatchedOpponent

    wrapped = make_env()
    env = wrapped._metamon_env if hasattr(wrapped, "_metamon_env") else wrapped.env
    num_lanes = env.batched_envs
    eval_actor = AmagoBatchedOpponent(
        policy=policy,
        device=device,
        num_lanes=num_lanes,
        action_dim=action_dim,
        sample=sample,
    )
    policy.eval()

    if timesteps is None:
        timesteps = max(total_battles * 250 // num_lanes, 250)

    obs, _info = env.reset()
    episodes_done = 0
    steps = 0
    returns: List[float] = []
    wins: List[float] = []
    valid_ratios: List[float] = []
    episode_return = np.zeros((num_lanes,), dtype=np.float64)

    while episodes_done < total_battles and steps < timesteps:
        obs_list = unstack_obs_dicts(obs)
        active = np.ones((num_lanes,), dtype=bool)
        actions = eval_actor.act(active, obs_list)
        obs, rewards, terminated, truncated, info = env.step(actions)
        steps += 1
        episode_return += rewards

        done = terminated | truncated
        if done.any():
            done_idx = np.where(done)[0]
            for i in done_idx:
                episodes_done += 1
                returns.append(float(episode_return[i]))
                episode_return[i] = 0.0
                won = info.get("won")
                if isinstance(won, list):
                    if won[i] is not None:
                        wins.append(float(won[i]))
                elif won is not None:
                    wins.append(float(won))
                valid = info.get("valid_action_count")
                invalid = info.get("invalid_action_count")
                if (
                    isinstance(valid, list)
                    and isinstance(invalid, list)
                    and valid[i] is not None
                    and invalid[i] is not None
                ):
                    denom = valid[i] + invalid[i]
                    if denom > 0:
                        valid_ratios.append(valid[i] / denom)

    env.close()

    env_name = getattr(wrapped, "env_name", "metamon")
    results: Dict[str, Any] = {
        f"Average Total Return in {env_name}": (
            float(np.mean(returns)) if returns else 0.0
        ),
        "Average Total Return (Across All Env Names)": (
            float(np.mean(returns)) if returns else 0.0
        ),
    }
    if returns:
        results[f"Bottom Quintile Total Return in {env_name}"] = float(
            np.percentile(returns, 20)
        )
    if wins:
        results[f"Average Win Rate in {env_name}"] = float(np.mean(wins))
    if valid_ratios:
        results[f"Average Valid Actions in {env_name}"] = float(np.mean(valid_ratios))
    return results
