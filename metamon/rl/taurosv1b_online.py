"""TaurosV1B online-RL runner with the audit fixes required by the public-only run.

This module intentionally reuses the mature environment/model construction in
``metamon.rl.online_rl`` while fixing the semantics that matter for a long V1B run:

* online FIFO data is never sampled before ``ready_for_training``;
* epoch 0 is a valid anneal start (no truthiness fallback to 170);
* checkpoint id 0 is a valid explicit checkpoint;
* full-state resume continues at N+1 rather than repeating epoch N;
* resume never rewrites the configured dataset schedule;
* policy checkpoints and full optimizer states have separate cadences;
* mixed precision is configurable instead of being hard-coded to FP32;
* the advantage filter has safe statistics for empty/constant masks.

Run with ``python -m metamon.rl.taurosv1b_online``.  The ordinary online-RL
module is left available for historical reproducibility of existing runs.
"""

from __future__ import annotations

import collections
import os
from argparse import ArgumentParser
from typing import Any, Optional

import amago
import gin
import torch
import wandb

import metamon
from metamon.data import MetamonDataset
from metamon.interface import (
    get_action_space,
    get_observation_space,
    get_reward_function,
    get_reward_function_names,
)
from metamon.tokenizer import get_tokenizer
from metamon.rl import online_rl as legacy
from metamon.rl.custom_agent import ISAdvantageFilter
from metamon.rl.dataset_config import flatten_config, load_dataset_config, save_dataset_config
from metamon.rl.metamon_to_amago import MetamonFIFODataset, MetamonOnlineExperiment
from metamon.rl.pretrained import (
    LATEST_CHECKPOINT,
    LocalPretrainedModel,
    get_pretrained_model,
    get_pretrained_model_names,
    pretrained_model,
)


# ---------------------------------------------------------------------------
# Local V1B registry entry used only by the phase-F dynamic self-play pool.
# The actual trainee still uses TaurosV1A as an *architecture descriptor* and
# loads the distilled V1B weights explicitly with --base_weights.
# ---------------------------------------------------------------------------

class _LocalV1BModel(LocalPretrainedModel):
    """Shared architecture metadata for evolving and frozen V1B policies."""

    def __init__(
        self,
        run_name: str,
        default_checkpoint: int,
        train_gin_config: str = "grouped_v2_large_robust_awr_beta3.gin",
    ):
        save_dir = os.environ.get("METAMON_SAVE_DIR", os.path.abspath("checkpoints"))
        super().__init__(
            amago_ckpt_dir=save_dir,
            model_name=run_name,
            model_gin_config="grouped_v2_medium.gin",
            train_gin_config=train_gin_config,
            default_checkpoint=default_checkpoint,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("GroupedObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            dataset_config="online_selfplay_taurosv1b.yaml",
            gin_overrides={
                "MetamonGroupedTstepEncoderV2.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                )
            },
        )


@pretrained_model("TaurosV1B")
class TaurosV1B(_LocalV1BModel):
    """Local 35M V1B run, primarily for ``discover: true`` opponent pools."""

    def __init__(self):
        run_name = os.environ.get("TAUROSV1B_RUN_NAME", "taurosv1b_phase_f")
        super().__init__(run_name=run_name, default_checkpoint=0)

    def get_path_to_checkpoint(self, checkpoint: int) -> str:
        if checkpoint == LATEST_CHECKPOINT:
            return os.path.join(self.local_ckpt_dir, "latest", "policy.pt")
        return super().get_path_to_checkpoint(checkpoint)


@pretrained_model("SnorlaxV1")
class SnorlaxV1(_LocalV1BModel):
    """Evolving SnorlaxV1 policy used by its own self-play lineage."""

    def __init__(self):
        run_name = os.environ.get("SNORLAX_RUN_NAME", "snorlax_v1")
        super().__init__(
            run_name=run_name,
            default_checkpoint=0,
            train_gin_config="snorlax_v1.gin",
        )

    def get_path_to_checkpoint(self, checkpoint: int) -> str:
        if checkpoint == LATEST_CHECKPOINT:
            return os.path.join(self.local_ckpt_dir, "latest", "policy.pt")
        return super().get_path_to_checkpoint(checkpoint)

    def initialize_agent(
        self,
        checkpoint: Optional[int] = None,
        log: bool = False,
        action_temperature: float = 1.0,
    ):
        # LocalPretrainedModel reserves checkpoint 0 to mean random
        # initialization.  Here policy_epoch_0.pt is a real immutable snapshot,
        # so explicitly load it after constructing the correctly shaped agent.
        requested = self.default_checkpoint if checkpoint is None else checkpoint
        experiment = super().initialize_agent(
            checkpoint=requested,
            log=log,
            action_temperature=action_temperature,
        )
        if requested == 0:
            path = self.get_path_to_checkpoint(0)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            state = torch.load(path, map_location="cpu")
            experiment.policy.load_state_dict(state, strict=True)
        return experiment


@pretrained_model("Epoch475Frozen")
class Epoch475Frozen(_LocalV1BModel):
    """Immutable Epoch475 anchor used to fill unavailable history slots."""

    def __init__(self):
        super().__init__(run_name="epoch475_frozen", default_checkpoint=475)


@gin.configurable
class RobustAdvantageFilter(ISAdvantageFilter):
    """Numerically safe normalized exponential AWR filter.

    Despite the historical parent class name, V1B does *not* inject a behavior-
    policy log-ratio; this is normalized exponential advantage-weighted behavior
    regression.  ``delta_log`` remains supported if a future caller explicitly
    supplies one.

    The original implementation uses unbiased ``std()`` directly on the masked
    values.  A one-element mask therefore produces NaN and an empty mask has no
    defined statistics.  Here constant/small batches fall back to unit scale so
    the filter becomes neutral instead of corrupting the update.
    """

    def __call__(self, adv: torch.Tensor) -> torch.Tensor:
        mask = self._mask
        self._mask = None
        delta_log = self._delta_log
        self._delta_log = None

        adv_f = adv.float()
        if mask is not None:
            mask = mask[:, : adv.shape[1], ...]
            while mask.ndim < adv.ndim:
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(adv).bool()
            valid = adv_f[mask]
        else:
            valid = adv_f.reshape(-1)

        if valid.numel() == 0:
            mu = torch.zeros((), device=adv.device, dtype=adv_f.dtype)
            sigma = torch.ones((), device=adv.device, dtype=adv_f.dtype)
        else:
            mu = valid.mean()
            if valid.numel() <= 1:
                sigma = torch.ones_like(mu)
            else:
                sigma = valid.std(unbiased=False)
                sigma = torch.where(
                    torch.isfinite(sigma) & (sigma > self.eps),
                    sigma,
                    torch.ones_like(sigma),
                )

        adv_norm = (adv_f - mu) / sigma
        exponent = self.beta * adv_norm
        if delta_log is not None:
            delta_log = delta_log[:, : adv.shape[1], ...].float()
            exponent = exponent + torch.clamp(
                delta_log, -self.clip_delta, self.clip_delta
            )
        exponent = torch.nan_to_num(exponent, nan=0.0, posinf=20.0, neginf=-20.0)

        weights = torch.exp(exponent).to(dtype=adv.dtype)
        if self.clip_weights_low is not None or self.clip_weights_high is not None:
            weights = torch.clamp(
                weights, min=self.clip_weights_low, max=self.clip_weights_high
            )
        if self.seq_enabled:
            weights = weights * self._compute_seq_weights(adv)
        return weights


class ReadyAwareOnlineMixture(legacy.OnlineMixtureOfDatasets):
    """Online/offline mixture that never samples an unreadied FIFO.

    The configured ramp begins at ``max(start_epoch, fifo_ready_epoch)``.  If the
    buffer was prefilled, this is exactly the requested absolute schedule.  If it
    was not, the entire ramp is shifted rather than jumping part-way through it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fifo_ready_epoch: Optional[int] = None

    def update_dset_weights(self, epoch: int):
        self.check_configured()
        fifo_ready = bool(self.fifo.ready_for_training)
        if fifo_ready and self._fifo_ready_epoch is None:
            self._fifo_ready_epoch = int(epoch)

        if not fifo_ready:
            curr_online = 0.0
            effective_start = None
            effective_end = None
        else:
            ready_epoch = self._fifo_ready_epoch if self._fifo_ready_epoch is not None else epoch
            duration = max(self.end_epoch - self.start_epoch, 0)
            effective_start = max(self.start_epoch, int(ready_epoch))
            effective_end = effective_start + duration
            if effective_end <= effective_start:
                curr_online = self.final_online_weight
            elif epoch <= effective_start:
                curr_online = self.initial_online_weight
            elif epoch >= effective_end:
                curr_online = self.final_online_weight
            else:
                progress = (epoch - effective_start) / (effective_end - effective_start)
                curr_online = self.initial_online_weight + progress * (
                    self.final_online_weight - self.initial_online_weight
                )

        curr_online = float(min(max(curr_online, 0.0), 1.0))
        curr_offline = 1.0 - curr_online
        self._available_datasets = [
            (self.fifo, curr_online),
            (self.offline, curr_offline),
        ]
        accelerator = getattr(getattr(self, "experiment", None), "accelerator", None)
        if accelerator is None or accelerator.is_main_process:
            if effective_start is None:
                schedule = "waiting for FIFO readiness"
            else:
                schedule = f"effective epochs {effective_start} -> {effective_end}"
            print(
                f"  [ReadyAwareOnlineMixture] Epoch {epoch}: online={curr_online:.4f}, "
                f"offline={curr_offline:.4f} ({schedule})",
                flush=True,
            )


class TaurosV1BOnlineExperiment(MetamonOnlineExperiment):
    """Online experiment with cheap policy checkpoints and sparse full states."""

    full_state_ckpt_interval: Optional[int] = 100
    requested_mixed_precision: str = "no"
    requested_lr_warmup_start_lr: Optional[float] = None

    def __init__(self, *args, **kwargs):
        kwargs["mixed_precision"] = self.requested_mixed_precision
        super().__init__(*args, **kwargs)

    def init_model(self) -> None:
        """Build the policy with an optional nonzero linear LR warmup.

        AMAGO's stock warmup always starts at zero. Phase E needs to preserve the
        already-strong C policy, so it starts at 5e-6 and linearly reaches the
        configured peak LR over the requested number of optimizer updates.
        """
        start_lr = self.requested_lr_warmup_start_lr
        if start_lr is None:
            super().init_model()
            return

        peak_lr = float(self.learning_rate)
        start_lr = float(start_lr)
        if peak_lr <= 0.0:
            raise ValueError(f"learning_rate must be positive; got {peak_lr}")
        if not (0.0 < start_lr <= peak_lr):
            raise ValueError(
                "lr_warmup_start_lr must satisfy 0 < start <= learning_rate; "
                f"got start={start_lr}, peak={peak_lr}"
            )
        if int(self.lr_warmup_steps) <= 0 and start_lr != peak_lr:
            raise ValueError(
                "A nontrivial lr_warmup_start_lr requires lr_warmup_epochs > 0."
            )

        policy_kwargs = {
            "tstep_encoder_type": self.tstep_encoder_type,
            "traj_encoder_type": self.traj_encoder_type,
            "obs_space": self.rl2_space["obs"],
            "rl2_space": self.rl2_space["rl2"],
            "action_space": self.train_envs.single_action_space,
            "max_seq_len": self.max_seq_len,
        }
        policy = self.agent_type(**policy_kwargs)
        optimizer = self.init_optimizer(policy)

        start_factor = start_lr / peak_lr
        warmup_steps = max(int(self.lr_warmup_steps), 1)

        def lr_lambda(current_step: int) -> float:
            if current_step >= warmup_steps:
                return 1.0
            progress = float(current_step) / float(warmup_steps)
            return start_factor + (1.0 - start_factor) * progress

        lr_schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambda,
        )
        self.policy_aclr, self.optimizer, self.lr_schedule = self.accelerator.prepare(
            policy, optimizer, lr_schedule
        )
        self.accelerator.register_for_checkpointing(self.lr_schedule)
        self.grad_update_counter = 0

    def save_checkpoint(self) -> None:
        """Called at ``ckpt_interval``; always save policy, full state sparsely."""
        if self.accelerator.is_main_process:
            path = os.path.join(
                self.ckpt_dir,
                "policy_weights",
                f"policy_epoch_{self.epoch}.pt",
            )
            torch.save(self.policy.state_dict(), path)

        interval = self.full_state_ckpt_interval
        if interval is not None and interval > 0 and self.epoch % interval == 0:
            ckpt_name = f"{self.run_name}_epoch_{self.epoch}"
            self.accelerator.save_state(
                os.path.join(self.ckpt_dir, "training_states", ckpt_name),
                safe_serialization=True,
            )


def build_ready_aware_online_dataset(
    *,
    pretrained,
    buffer_dir: str,
    dataset_config_path: str,
    online_weight: float,
    dset_max_size: int,
    dset_min_size: int,
    online_anneal_epochs: int,
    battle_format: str,
    reward_function,
    stats_dropout_prob: float = 0.0,
    initial_online_weight: Optional[float] = None,
    online_anneal_start_epoch: Optional[int] = None,
    online_anneal_end_epoch: Optional[int] = None,
):
    """Build the audited online/offline mixture used by V1B."""
    config = load_dataset_config(dataset_config_path)
    formats = config.formats or [battle_format]
    fifo_root = os.path.abspath(buffer_dir)
    os.makedirs(os.path.join(fifo_root, battle_format), exist_ok=True)

    fifo_obs_space = pretrained.observation_space
    if stats_dropout_prob > 0.0:
        fifo_obs_space = legacy.StatsDropoutObservationSpace(
            base_obs_space=fifo_obs_space, dropout_prob=stats_dropout_prob
        )
    fifo_metamon = MetamonDataset(
        dset_root=fifo_root,
        observation_space=fifo_obs_space,
        action_space=pretrained.action_space,
        reward_function=reward_function,
        formats=formats,
        shuffle=True,
        verbose=False,
        write_index_cache=False,
    )
    fifo = MetamonFIFODataset(
        parsed_replay_dset=fifo_metamon,
        dset_max_size=dset_max_size,
        dset_min_size=dset_min_size,
        dset_name="Online FIFO Buffer",
    )

    if online_weight >= 1.0:
        return fifo
    offline = legacy.build_dataset(
        config=config,
        obs_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=reward_function,
    )
    if online_weight <= 0.0:
        return offline

    init_online = 0.0 if initial_online_weight is None else initial_online_weight
    start_ep = 170 if online_anneal_start_epoch is None else online_anneal_start_epoch
    end_ep = (
        start_ep + online_anneal_epochs
        if online_anneal_end_epoch is None
        else online_anneal_end_epoch
    )
    return ReadyAwareOnlineMixture(
        fifo=fifo,
        offline=offline,
        initial_online_weight=init_online,
        final_online_weight=online_weight,
        start_epoch=start_ep,
        end_epoch=end_ep,
        dset_name="Online + Offline Mixture",
    )


def _create_experiment_with_safe_class(
    *,
    mixed_precision: str,
    full_state_interval: int,
    lr_warmup_start_lr: Optional[float] = None,
    **kwargs,
):
    """Reuse the legacy experiment builder but substitute the audited subclass."""
    old_cls = legacy.MetamonOnlineExperiment
    TaurosV1BOnlineExperiment.requested_mixed_precision = mixed_precision
    TaurosV1BOnlineExperiment.full_state_ckpt_interval = full_state_interval
    TaurosV1BOnlineExperiment.requested_lr_warmup_start_lr = lr_warmup_start_lr
    legacy.MetamonOnlineExperiment = TaurosV1BOnlineExperiment
    try:
        return legacy.create_online_experiment(**kwargs)
    finally:
        legacy.MetamonOnlineExperiment = old_cls


def _explicit_checkpoint_path(args, pretrained) -> str:
    if args.prev_run_dir is not None:
        if args.prev_run_name is None or args.prev_checkpoint is None:
            raise ValueError("--prev_run_name and --prev_checkpoint are required with --prev_run_dir")
        return os.path.join(
            args.prev_run_dir,
            args.prev_run_name,
            "ckpts",
            "policy_weights",
            f"policy_epoch_{args.prev_checkpoint}.pt",
        )
    ckpt = pretrained.default_checkpoint if args.base_checkpoint is None else args.base_checkpoint
    return pretrained.get_path_to_checkpoint(ckpt)


def add_cli(parser: ArgumentParser) -> ArgumentParser:
    legacy.add_cli(parser)
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        help="Raw policy state_dict to load into the --base_model architecture. "
        "Used for the distilled 35M V1B bootstrap.",
    )
    parser.add_argument(
        "--full_state_ckpt_interval",
        type=int,
        default=100,
        help="Save optimizer/scheduler/RNG state every N epochs; raw policy checkpoints "
        "continue to use --ckpt_interval.",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="no",
        help="Accelerate mixed-precision mode. V1B defaults to FP32; bf16 must be smoke-tested.",
    )
    parser.add_argument(
        "--lr_warmup_start_lr",
        type=float,
        default=None,
        help="Optional nonzero starting LR for a linear warmup to --learning_rate. "
        "If omitted, use AMAGO's ordinary zero-to-peak warmup.",
    )
    parser.add_argument(
        "--validation_once",
        action="store_true",
        help="Run one validation pass and exit (requires --mode validate).",
    )
    parser.add_argument(
        "--validation_epoch",
        type=int,
        default=None,
        help="Epoch value attached to a --validation_once metrics record.",
    )
    return parser


def main() -> None:
    pre = ArgumentParser(add_help=False)
    pre.add_argument("--run_config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    parser = add_cli(ArgumentParser(description="Audited TaurosV1B online RL runner."))
    if pre_args.run_config:
        cfg = legacy._load_run_config(pre_args.run_config)
        known = {a.dest for a in parser._actions if a.dest != "help"}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise SystemExit(f"Unknown key(s) in --run_config: {', '.join(unknown)}")
        parser.set_defaults(**{k: cfg[k] for k in cfg if k in known})
        for action in parser._actions:
            if action.dest in cfg:
                action.required = False
    args = parser.parse_args()
    os.environ["METAMON_RUN_MODE"] = args.mode

    # E/F rely on the periodic TaurosV0 monitor as a training health check.
    # Install it explicitly here rather than relying only on the gin import side
    # effect, and fail fast if someone launches a learner without W&B logging or
    # changes the checkpoint cadence so the 5-epoch monitor would not fire.
    if args.mode == "learn" and args.run_name in {"taurosv1b_phase_e", "taurosv1b_phase_f"}:
        from metamon.rl import taurosv1b_tournament as v0_monitor

        v0_monitor._install_patch()
        if not args.log:
            raise ValueError(
                "TaurosV1B E/F learner requires --log so periodic TaurosV0 "
                "tournament metrics are sent to W&B."
            )
        if args.ckpt_interval != v0_monitor.TOURNAMENT_INTERVAL:
            raise ValueError(
                "TaurosV1B E/F requires ckpt_interval="
                f"{v0_monitor.TOURNAMENT_INTERVAL} so the periodic TaurosV0 "
                "tournament runs every five epochs."
            )
        if TaurosV1BOnlineExperiment.save_checkpoint is not v0_monitor._patched_save_checkpoint:
            raise RuntimeError("TaurosV0 tournament checkpoint hook was not installed.")

    if args.ckpt_interval <= 0:
        raise ValueError("--ckpt_interval must be positive")
    if args.full_state_ckpt_interval > 0 and args.full_state_ckpt_interval % args.ckpt_interval:
        raise ValueError("--full_state_ckpt_interval must be a multiple of --ckpt_interval")
    if args.from_scratch and (args.prev_run_dir is not None or args.base_weights is not None):
        raise ValueError("--from_scratch cannot be combined with continuation/base weights")
    if args.resume_training_state and (args.prev_run_dir is not None or args.base_weights is not None):
        raise ValueError("full-state resume cannot be combined with --prev_run_dir/--base_weights")
    if args.validation_once and args.mode != "validate":
        raise ValueError("--validation_once requires --mode validate")
    if args.validation_epoch is not None and not args.validation_once:
        raise ValueError("--validation_epoch requires --validation_once")
    if args.validation_epoch is not None and args.validation_epoch < 0:
        raise ValueError("--validation_epoch must be non-negative")
    if args.validation_once and args.val_timesteps <= 0:
        raise ValueError("--validation_once requires positive --val_timesteps")

    metamon.print_banner()
    os.environ.setdefault("METAMON_SAVE_DIR", os.path.abspath(args.save_dir))
    os.environ.setdefault("TAUROSV1B_RUN_NAME", args.run_name)

    pretrained = get_pretrained_model(args.base_model)
    train_gin_config_path = legacy._resolve_train_gin_path(pretrained, args.train_gin_config)
    reward_function = (
        get_reward_function(args.reward_function)
        if args.reward_function is not None
        else pretrained.reward_function
    )
    dataset_config_path = legacy._resolve_dataset_config_path(args.dataset_config)
    dataset_config = load_dataset_config(dataset_config_path)
    battle_format = (
        args.battle_format
        or (dataset_config.formats[0] if dataset_config.formats else None)
        or legacy.DEFAULT_BATTLE_FORMAT
    )
    formats = dataset_config.formats or [battle_format]
    val_opponent_kwargs = legacy._resolve_val_opponent_config(
        val_pool_path=args.val_pool,
        val_opponent=args.val_opponent,
        base_model=args.base_model,
        battle_format=battle_format,
    )

    if args.mode == "collect":
        fifo_root = os.path.abspath(args.buffer_dir)
        os.makedirs(os.path.join(fifo_root, battle_format), exist_ok=True)
        fifo_metamon = MetamonDataset(
            dset_root=fifo_root,
            observation_space=pretrained.observation_space,
            action_space=pretrained.action_space,
            reward_function=reward_function,
            formats=formats,
            shuffle=True,
            verbose=False,
            write_index_cache=False,
        )
        amago_dataset = MetamonFIFODataset(
            parsed_replay_dset=fifo_metamon,
            dset_max_size=args.dset_max_size,
            dset_min_size=args.dset_min_size,
            dset_name="Online FIFO Buffer",
        )
    elif args.mode == "validate":
        amago_dataset = amago.loading.DoNothingDataset()
    else:
        # IMPORTANT: pass zero through literally.  The legacy main used ``x or 170``.
        amago_dataset = build_ready_aware_online_dataset(
            pretrained=pretrained,
            buffer_dir=args.buffer_dir,
            dataset_config_path=dataset_config_path,
            online_weight=args.online_weight,
            dset_max_size=args.dset_max_size,
            dset_min_size=args.dset_min_size,
            online_anneal_epochs=args.online_anneal_epochs,
            battle_format=battle_format,
            reward_function=reward_function,
            stats_dropout_prob=args.stats_dropout_prob,
            initial_online_weight=args.initial_online_weight,
            online_anneal_start_epoch=args.online_anneal_start_epoch,
            online_anneal_end_epoch=args.online_anneal_end_epoch,
        )

    config_save_path = os.path.join(args.save_dir, args.run_name, "dataset_config.yaml")
    save_dataset_config(flatten_config(dataset_config), config_save_path)

    experiment = _create_experiment_with_safe_class(
        mixed_precision=args.mixed_precision,
        full_state_interval=args.full_state_ckpt_interval,
        lr_warmup_start_lr=args.lr_warmup_start_lr,
        mode=args.mode,
        run_name=args.run_name,
        save_dir=args.save_dir,
        pretrained=pretrained,
        train_gin_config_path=train_gin_config_path,
        amago_dataset=amago_dataset,
        battle_format=battle_format,
        reward_function=reward_function,
        opponent_config_path=args.train_pool,
        val_opponent_kwargs=val_opponent_kwargs,
        buffer_dir=args.buffer_dir,
        save_results_to=args.save_results_to,
        lanes=args.lanes,
        n_workers=args.n_workers,
        train_team_set=args.train_team_set,
        val_team_set=args.val_team_set,
        temp_low=args.temp_low,
        temp_high=args.temp_high,
        epochs=args.epochs,
        train_timesteps_per_epoch=args.train_timesteps_per_epoch,
        steps_per_epoch=args.steps_per_epoch,
        batch_size_per_gpu=args.batch_size_per_gpu,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_warmup_epochs=args.lr_warmup_epochs,
        seq_floor_warmup_epochs=args.seq_floor_warmup_epochs,
        val_timesteps=args.val_timesteps,
        val_interval=args.val_interval,
        ckpt_interval=args.ckpt_interval,
        dloader_workers=args.dloader_workers,
        seed=args.seed,
        log=args.log,
    )
    experiment.start()

    if args.resume_training_state:
        resume_epoch = (
            args.resume_epoch
            if args.resume_epoch is not None
            else legacy._latest_training_state_epoch(experiment.ckpt_dir, args.run_name)
        )
        experiment.load_checkpoint(resume_epoch, resume_training_state=True)
        # A saved epoch is complete.  Continue with the next epoch rather than
        # replaying its data/update/checkpoint schedule.
        experiment.epoch = resume_epoch + 1
        if hasattr(experiment.dataset, "update_dset_weights"):
            experiment.dataset.update_dset_weights(experiment.epoch)
        print(f"Resumed completed epoch {resume_epoch}; next epoch is {experiment.epoch}.")
    elif args.base_weights is not None:
        if not os.path.exists(args.base_weights):
            raise FileNotFoundError(args.base_weights)
        experiment.load_checkpoint_from_path(args.base_weights, is_accelerate_state=False)
        print(f"Loaded distilled V1B bootstrap: {args.base_weights}")
    elif args.from_scratch:
        print("From scratch: leaving random initialization in place.")
    else:
        ckpt_path = _explicit_checkpoint_path(args, pretrained)
        experiment.load_checkpoint_from_path(ckpt_path, is_accelerate_state=False)
        print(f"Loaded initial weights: {ckpt_path}")

    if args.mode == "collect" and os.environ.get(
        "METAMON_ACTOR_ONLY_COLLECTION", "1"
    ) == "1":
        from metamon.env.vectorized.amago_policy import optimize_policy_for_collection

        optimize_policy_for_collection(experiment.policy, experiment.DEVICE)
        # optimize_policy_for_collection detaches training-only children from
        # the module, but torch optimizers retain their own strong references to
        # every original parameter. Collector processes never step an optimizer;
        # prune those stale references so removed critic/training tensors can
        # actually leave the GPU and parked collectors can be made lightweight.
        active_param_ids = {
            id(parameter) for parameter in experiment.policy.parameters()
        }
        optimizer = experiment.optimizer
        dropped_optimizer_params = 0
        for group in optimizer.param_groups:
            original = group["params"]
            retained = [
                parameter
                for parameter in original
                if id(parameter) in active_param_ids
            ]
            dropped_optimizer_params += len(original) - len(retained)
            group["params"] = retained
        for parameter in list(optimizer.state):
            if id(parameter) not in active_param_ids:
                del optimizer.state[parameter]
        removed = getattr(experiment.policy, "_metamon_removed_children", ())
        dtype = getattr(experiment.policy, "_metamon_collection_dtype", torch.float32)
        cache_len = getattr(
            experiment.policy, "_metamon_collection_cache_len", "unchanged"
        )
        print(
            "Collector policy optimized: "
            f"dtype={dtype}, kv_cache_len={cache_len}, "
            f"removed={','.join(removed) or 'none'}, "
            f"optimizer_params_released={dropped_optimizer_params}",
            flush=True,
        )

    if args.validation_once:
        if args.validation_epoch is not None:
            experiment.epoch = args.validation_epoch
        policy_step = getattr(experiment.policy, "_ataraxos_step", None)
        if policy_step is not None:
            experiment.grad_update_counter = int(policy_step.item())
        experiment.policy_aclr.eval()
        experiment.evaluate_val()
        if args.log:
            wandb.finish()
        return

    experiment.learn()
    if args.log:
        wandb.finish()


if __name__ == "__main__":
    main()
