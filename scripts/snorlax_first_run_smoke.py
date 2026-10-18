#!/usr/bin/env python3
"""Small, deterministic checks used by the first-run Snorlax launcher smoke test."""

from __future__ import annotations

import argparse
import json
import tempfile
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

    # Retention must preserve the exact current/-1/-4/-16/-64 lineage, while
    # early runs must not silently substitute a newer checkpoint for an older
    # nominal offset.
    from metamon.rl.ataraxos_online import AtaraxosExperiment
    from metamon.rl.evaluate import opponent_pool
    from metamon.rl import pretrained as pretrained_module

    with tempfile.TemporaryDirectory(prefix="snorlax-retention-smoke-") as tmp:
        ckpt_root = Path(tmp)
        policy_dir = ckpt_root / "policy_weights"
        policy_dir.mkdir()
        for epoch in range(131):
            (policy_dir / f"policy_epoch_{epoch}.pt").touch()
        removed = AtaraxosExperiment._prune_numbered_checkpoints(
            str(policy_dir),
            prefix="policy_epoch_",
            suffix=".pt",
            directories=False,
        )
        if removed != 64:
            raise RuntimeError(f"checkpoint retention removed {removed}, expected 64")

        class _FakeModel:
            local_ckpt_dir = str(ckpt_root)

            @staticmethod
            def get_path_to_checkpoint(checkpoint: int) -> str:
                return str(policy_dir / f"policy_epoch_{checkpoint}.pt")

        original_get = pretrained_module.get_pretrained_model
        pretrained_module.get_pretrained_model = lambda _name: _FakeModel()
        try:
            discovered = opponent_pool._discover_checkpoints(
                "SnorlaxV1",
                include_latest=False,
                min_epoch=None,
                max_epoch=None,
                step=None,
                offsets=[0, 1, 4, 16, 64],
                update_interval=1,
            )
            for path in policy_dir.glob("policy_epoch_*.pt"):
                path.unlink()
            for epoch in range(11):
                (policy_dir / f"policy_epoch_{epoch}.pt").touch()
            early = opponent_pool._discover_checkpoints(
                "SnorlaxV1",
                include_latest=False,
                min_epoch=None,
                max_epoch=None,
                step=None,
                offsets=[0, 1, 4, 16, 64],
                update_interval=1,
            )
        finally:
            pretrained_module.get_pretrained_model = original_get
        if discovered != [66, 114, 126, 129, 130]:
            raise RuntimeError(f"wrong retained Snorlax offset lineage: {discovered}")
        if early != [6, 9, 10]:
            raise RuntimeError(f"early Snorlax offsets were silently substituted: {early}")
    print(
        "PASS: imports, lane-preserving action conversion, exact retention/offset "
        f"lineage, observation space, and {action_dim}-action interface",
        flush=True,
    )


def simulator_transitions() -> None:
    from metamon.env.vectorized.lane import KIND_FORCESWITCH
    from metamon.env.vectorized.opponent import BatchedOpponent
    from metamon.env.vectorized.vector_env import VectorizedShowdownEnv
    from metamon.env.wrappers import get_metamon_teams
    from metamon.rl.pretrained import get_pretrained_model

    model = get_pretrained_model("TaurosV1A")
    action_dim = model.action_space.gym_space.n

    class _CountingOpponent(BatchedOpponent):
        def __init__(self, lanes: int, seed: int):
            self.lanes = lanes
            self.rng = np.random.default_rng(seed)
            self.acts = np.zeros(lanes, dtype=np.int64)
            self.observes = np.zeros(lanes, dtype=np.int64)
            self.partial_batches = 0
            self.reset_count = 0

        def act(self, active: np.ndarray, obs_list: list[dict]) -> np.ndarray:
            active = np.asarray(active, dtype=bool)
            if active.any() and not active.all():
                self.partial_batches += 1
            self.acts += active.astype(np.int64)
            choices = np.zeros(self.lanes, dtype=np.int64)
            for lane in np.flatnonzero(active):
                illegal = np.asarray(obs_list[int(lane)]["illegal_actions"], dtype=bool)
                legal = np.flatnonzero(~illegal)
                if len(legal) == 0:
                    raise RuntimeError(f"opponent lane {lane} has no legal action")
                # Repeating the first move can create pathological Gen 1 loops
                # that are irrelevant to the production policy path.
                choices[lane] = int(self.rng.choice(legal))
            return choices

        def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
            del reward
            if not 0 <= int(action_idx) < action_dim:
                raise RuntimeError(f"observed invalid opponent action {action_idx}")
            self.observes[lane_idx] += 1

        def reset_lanes(self, done_mask: np.ndarray) -> None:
            self.reset_count += int(np.asarray(done_mask, dtype=bool).sum())

        def reset_all(self) -> None:
            return None

    rng = np.random.default_rng(24680)
    total_forced = 0
    total_done = 0
    total_truncated = 0
    total_partial = 0
    with tempfile.TemporaryDirectory(prefix="snorlax-simulator-smoke-") as tmp:
        trajectory_root = Path(tmp)
        for eval_side in (0, 1):
            done_before_side = total_done
            lanes = 4
            opponent = _CountingOpponent(lanes, seed=97531 + eval_side)
            side_root = trajectory_root / f"side-{eval_side}"
            # The native simulator cannot generate random teams from null team
            # specs. Exercise the same explicit format/team path used by the
            # real Snorlax run instead.
            player_teams = get_metamon_teams("gen1ou", "snorlax_v1_expert")
            opponent_teams = get_metamon_teams("gen1ou", "snorlax_v1_expert")
            env = VectorizedShowdownEnv(
                player_team_set=player_teams,
                opponent_team_set=opponent_teams,
                opponent=opponent,
                opponent_obs_space=model.observation_space,
                opponent_action_space=model.action_space,
                eval_obs_space=model.observation_space,
                eval_action_space=model.action_space,
                eval_reward_function=model.reward_function,
                opponent_reward_function=model.reward_function,
                batched_envs=lanes,
                battle_format="gen1ou",
                turn_limit=12,
                save_trajectories_to=str(side_root),
                n_workers=1,
                seed=13579 + eval_side,
                eval_player_side=eval_side,
            )
            try:
                _obs, info = env.reset()
                for _step in range(16):
                    total_forced += sum(
                        lane.request_kind(env.eval_side) == KIND_FORCESWITCH
                        for lane in env.lanes
                    )
                    actions = np.asarray(
                        [int(rng.choice(legal)) for legal in info["legal_actions"]],
                        dtype=np.int64,
                    )
                    _obs, _rewards, terminated, truncated, info = env.step(actions)
                    if not np.array_equal(opponent.acts, opponent.observes):
                        raise RuntimeError(
                            "opponent recurrent state advanced without exactly one observe"
                        )
                    total_done += int(np.asarray(terminated, dtype=bool).sum())
                    total_truncated += int(np.asarray(truncated, dtype=bool).sum())
                total_partial += opponent.partial_batches
                # Time-limit lanes are reported as both terminated and
                # truncated so learning cannot bootstrap across the reset.
                # Therefore terminated alone is the completed-lane count.
                if opponent.reset_count != total_done - done_before_side:
                    raise RuntimeError("opponent reset count disagrees with completed lanes")
            finally:
                env.close()

        files = sorted(trajectory_root.rglob("*.json.lz4"))
        if not files:
            raise RuntimeError("simulator transition smoke produced no trajectories")
        for path in files:
            payload = _read(path)
            final = payload["states"][-1]
            won = bool(final.get("battle_won"))
            lost = bool(final.get("battle_lost"))
            expected = "WIN" if won else ("LOSS" if lost else "UNRESOLVED")
            if won and lost or not path.name.endswith(f"_{expected}.json.lz4"):
                raise RuntimeError(f"POV outcome disagrees with filename: {path.name}")

    if total_done == 0 and total_truncated == 0:
        raise RuntimeError("simulator smoke did not exercise a terminal or timeout reset")
    print(
        "PASS: short both-POV simulator transitions and act/observe accounting "
        f"({total_done} terminal, {total_truncated} timeout, "
        f"{total_forced} forced-switch, {total_partial} partial-batch events)",
        flush=True,
    )


def validate_buffer(root: Path) -> None:
    from metamon.data.parsed_replay_dset import MetamonDataset
    from metamon.rl import ataraxos_keys
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

    wins = losses = unresolved = 0
    for path in files:
        payload = _read(path)
        states = payload.get("states")
        actions = payload.get("actions")
        log_probs = payload.get("behavior_log_probs")
        values = payload.get("behavior_values")
        legal = payload.get("behavior_legal_masks")
        if payload.get("ataraxos_collector_version") != 3:
            raise RuntimeError(f"{path.name}: collector schema is not version 3")
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
        won = bool(final.get("battle_won"))
        lost = bool(final.get("battle_lost"))
        if won and lost:
            raise RuntimeError(f"{path.name}: impossible win-and-loss outcome")
        if won:
            wins += 1
        elif lost:
            losses += 1
        else:
            unresolved += 1
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
        payload = _read(Path(filename))
        final = payload["states"][-1]
        expected_outcome = 1.0 if final.get("battle_won") else (
            -1.0 if final.get("battle_lost") else 0.0
        )
        outcomes = item.obs[ataraxos_keys.OUTCOME]
        if not np.all(np.asarray(outcomes) == expected_outcome):
            raise RuntimeError(
                f"learner outcome target has the wrong POV sign for {Path(filename).name}"
            )
    decisive = wins + losses
    if decisive < 16:
        raise RuntimeError(
            "collector smoke produced too few decisive games to validate outcomes: "
            f"wins={wins}, losses={losses}, unresolved={unresolved}"
        )
    if wins == 0 or losses == 0:
        raise RuntimeError(
            "collector smoke has a one-sided outcome collapse: "
            f"wins={wins}, losses={losses}, unresolved={unresolved}"
        )
    print(
        f"PASS: {len(files)} schema-v3 trajectories passed structural, outcome, "
        f"and learner parsing checks (wins={wins}, losses={losses}, "
        f"unresolved={unresolved})",
        flush=True,
    )


def validate_resume(ckpt_root: Path) -> None:
    policies = ckpt_root / "policy_weights"
    states = ckpt_root / "training_states"
    policy0 = policies / "policy_epoch_0.pt"
    policy1 = policies / "policy_epoch_1.pt"
    state0 = states / "snorlax_first_run_learn_epoch_0"
    state1 = states / "snorlax_first_run_learn_epoch_1"
    required = (policy0, policy1, state0, state1)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"checkpoint/resume smoke is missing artifacts: {missing}")
    if list(ckpt_root.rglob("*.tmp")):
        raise RuntimeError("checkpoint/resume smoke left an incomplete temporary artifact")

    for state_dir in (state0, state1):
        if not any(path.is_file() and path.stat().st_size > 0 for path in state_dir.rglob("*")):
            raise RuntimeError(f"full training state is empty: {state_dir}")
    print(
        "PASS: resumed run produced its next policy and full-state artifacts",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("simulator-transitions")
    validate = subparsers.add_parser("validate-buffer")
    validate.add_argument("root", type=Path)
    resume = subparsers.add_parser("validate-resume")
    resume.add_argument("ckpt_root", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
    elif args.command == "simulator-transitions":
        simulator_transitions()
    elif args.command == "validate-resume":
        validate_resume(args.ckpt_root)
    else:
        validate_buffer(args.root)


if __name__ == "__main__":
    main()
