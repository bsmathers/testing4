#!/usr/bin/env python3
"""Small, deterministic checks used by the first-run Snorlax launcher smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lz4.frame
import numpy as np


def _read(path: Path) -> dict:
    if path.name.endswith(".lz4"):
        with lz4.frame.open(path, "rb") as stream:
            return json.loads(stream.read().decode("utf-8"))
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def preflight() -> None:
    # Import the exact entry point so registration and dependency failures occur
    # before the expensive collection block.
    import metamon.rl.ataraxos_online  # noqa: F401
    import torch
    from metamon.env.vectorized.amago_policy import normalize_discrete_policy_actions
    from metamon.rl.pretrained import get_pretrained_model

    model = get_pretrained_model("TaurosV1A")
    action_dim = model.action_space.gym_space.n
    if action_dim <= 1:
        raise RuntimeError(f"invalid Snorlax action dimension: {action_dim}")
    if getattr(model.observation_space, "gym_space", None) is None:
        raise RuntimeError("Snorlax observation space was not initialized")
    lane_actions = torch.tensor([2, 5, 1, 7]).reshape(4, 1, 1)
    normalized = normalize_discrete_policy_actions(
        lane_actions, 4, context="smoke policy"
    )
    if normalized.shape != (4, 1) or normalized[:, 0].tolist() != [2, 5, 1, 7]:
        raise RuntimeError("AMAGO action normalization lost or mixed a lane")
    print(
        "PASS: imports, Snorlax registration, observation space, and "
        f"{action_dim}-action interface",
        flush=True,
    )


def validate_buffer(root: Path) -> None:
    from metamon.data.parsed_replay_dset import MetamonDataset
    from metamon.rl.metamon_to_amago import MetamonAMAGODataset
    from metamon.rl.pretrained import get_pretrained_model

    replay_dir = root / "gen1ou"
    files = sorted(
        path
        for path in replay_dir.glob("*")
        if path.name.endswith((".json", ".json.lz4"))
    )
    if len(files) < 2:
        raise RuntimeError(
            f"collector produced {len(files)} replay(s); at least two are required"
        )

    for path in files:
        payload = _read(path)
        states = payload.get("states")
        actions = payload.get("actions")
        log_probs = payload.get("behavior_log_probs")
        values = payload.get("behavior_values")
        legal = payload.get("behavior_legal_masks")
        if payload.get("ataraxos_collector_version") != 2:
            raise RuntimeError(f"{path.name}: collector schema is not version 2")
        if not isinstance(states, list) or len(states) < 2:
            raise RuntimeError(f"{path.name}: invalid state sequence")
        action_steps = len(states) - 1
        if not isinstance(actions, list) or len(actions) != len(states) or actions[-1] != -1:
            raise RuntimeError(
                f"{path.name}: actions must align with states and end in the -1 sentinel"
            )
        lengths = {
            "log probabilities": len(log_probs or []),
            "values": len(values or []),
            "legal masks": len(legal or []),
        }
        bad = {name: length for name, length in lengths.items() if length != action_steps}
        if bad:
            raise RuntimeError(
                f"{path.name}: state/action alignment failed; expected "
                f"{action_steps}, got {bad}"
            )
        final = states[-1]
        if bool(final.get("battle_won")) and bool(final.get("battle_lost")):
            raise RuntimeError(f"{path.name}: impossible win-and-loss outcome")
        logp = np.asarray(log_probs, dtype=np.float32)
        mask = np.asarray(legal, dtype=bool)
        value = np.asarray(values, dtype=np.float32)
        if logp.ndim != 2 or mask.shape != logp.shape or value.shape != (action_steps,):
            raise RuntimeError(f"{path.name}: malformed behavior tensor dimensions")
        if not np.isfinite(logp).all() or not np.isfinite(value).all():
            raise RuntimeError(f"{path.name}: non-finite behavior metadata")
        if not np.allclose(np.exp(logp).sum(-1), 1.0, atol=2e-3):
            raise RuntimeError(f"{path.name}: behavior probabilities do not sum to one")
        if np.any(np.exp(logp)[~mask] > 1e-8):
            raise RuntimeError(f"{path.name}: illegal action has probability mass")
        chosen = np.asarray(actions[:-1])
        if (
            chosen.shape != (action_steps,)
            or not np.issubdtype(chosen.dtype, np.integer)
            or np.any(chosen < 0)
            or np.any(chosen >= logp.shape[1])
            or not mask[np.arange(action_steps), chosen].all()
        ):
            raise RuntimeError(f"{path.name}: recorded action is not legal in its lane")

    # Parse every generated replay through the same dataset adapter used by the
    # learner. This catches nested observation and 2-D/1-D conversion errors.
    model = get_pretrained_model("TaurosV1A")
    parsed = MetamonDataset(
        dset_root=str(root),
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        formats=["gen1ou"],
        shuffle=False,
        use_cached_filenames=False,
        write_index_cache=False,
    )
    wrapped = MetamonAMAGODataset(parsed)
    if len(parsed) != len(files):
        raise RuntimeError(f"dataset indexed {len(parsed)} of {len(files)} smoke replays")
    for filename in parsed.filenames:
        item = wrapped._process_data(parsed.load_filename(filename))
        steps = item.actions.shape[0]
        if item.actions.ndim != 2 or item.time_idxs.shape != (steps + 1, 1):
            raise RuntimeError(
                f"learner adapter lost a sequence/action axis for {Path(filename).name}"
            )
        for key, tensor in item.obs.items():
            if tensor.shape[0] != steps + 1:
                raise RuntimeError(
                    f"learner observation {key!r} is misaligned for {Path(filename).name}"
                )
    print(
        f"PASS: {len(files)} schema-v2 trajectories passed structural and learner parsing checks",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    validate = subparsers.add_parser("validate-buffer")
    validate.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
    else:
        validate_buffer(args.root)


if __name__ == "__main__":
    main()
