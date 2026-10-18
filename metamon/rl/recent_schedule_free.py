"""Recent-weighted Schedule-Free AdamW for the Ataraxos continuation.

This keeps AdamW's first and second moments and the caller-managed learning
rate, while adding the Schedule-Free x/y/z iterates.  A high polynomial
weighting power deliberately makes the average responsive enough for a
non-stationary self-play objective.

The update follows ``facebookresearch/schedule_free``'s
``AdamWScheduleFree`` implementation (Apache-2.0), including its inner-momentum
extension, with log-space averaging weights for long-running jobs.
"""

# Schedule-Free update derived from facebookresearch/schedule_free.
# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed Apache-2.0.

from __future__ import annotations

import math
import os
from collections.abc import Iterable

import torch


class RecentScheduleFreeAdamW(torch.optim.Optimizer):
    """AdamW moments plus recent-weighted Schedule-Free averaging.

    ``beta`` is the x/z interpolation used to form the gradient-query point y;
    ``inner_momentum`` is ordinary AdamW first-moment momentum.  Learning rates
    remain entirely controlled through each parameter group's ``lr`` field.
    """

    def __init__(
        self,
        params: Iterable,
        *,
        lr: float,
        beta: float = 0.8,
        beta2: float = 0.999,
        inner_momentum: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        r: float = 99.0,
        weight_lr_power: float = 0.0,
    ):
        if not 0.0 < beta < 1.0:
            raise ValueError("Schedule-Free beta must be in (0, 1)")
        if not 0.0 <= inner_momentum < 1.0:
            raise ValueError("inner_momentum must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if r < 0.0:
            raise ValueError("Schedule-Free r must be non-negative")
        defaults = dict(
            lr=float(lr),
            beta=float(beta),
            beta2=float(beta2),
            inner_momentum=float(inner_momentum),
            eps=float(eps),
            weight_decay=float(weight_decay),
            r=float(r),
            weight_lr_power=float(weight_lr_power),
            k=0,
            train_mode=False,
            lr_max=0.0,
            log_weight_sum=-math.inf,
            sf_ckp1=1.0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self) -> None:
        """Install the averaged evaluation iterate x in model parameters."""
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta = group["beta"]
            for parameter in group["params"]:
                state = self.state[parameter]
                if "z" in state:
                    parameter.lerp_(state["z"], weight=1.0 - 1.0 / beta)
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        """Install the gradient-query iterate y in model parameters."""
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta = group["beta"]
            for parameter in group["params"]:
                state = self.state[parameter]
                if "z" in state:
                    parameter.lerp_(state["z"], weight=1.0 - beta)
            group["train_mode"] = True

    @staticmethod
    def _average_coefficient(group: dict) -> float:
        """Return w_t / sum_i w_i without overflowing for large ``r``."""
        step = int(group["k"]) + 1
        lr = float(group["lr"])
        group["lr_max"] = max(float(group["lr_max"]), lr)
        log_weight = float(group["r"]) * math.log(step)
        lr_power = float(group["weight_lr_power"])
        if lr_power != 0.0:
            if group["lr_max"] <= 0.0:
                return 0.0
            log_weight += lr_power * math.log(group["lr_max"])
        previous = float(group["log_weight_sum"])
        if previous == -math.inf:
            log_sum = log_weight
        else:
            max_log = max(previous, log_weight)
            log_sum = max_log + math.log(
                math.exp(previous - max_log) + math.exp(log_weight - max_log)
            )
        group["log_weight_sum"] = log_sum
        coefficient = math.exp(log_weight - log_sum)
        group["sf_ckp1"] = coefficient
        return coefficient

    @torch.no_grad()
    def step(self, closure=None):
        if any(not group["train_mode"] for group in self.param_groups):
            raise RuntimeError(
                "RecentScheduleFreeAdamW.step() requires optimizer.train()"
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta = float(group["beta"])
            beta2 = float(group["beta2"])
            inner = float(group["inner_momentum"])
            eps = float(group["eps"])
            decay = float(group["weight_decay"])
            lr = float(group["lr"])
            step = int(group["k"]) + 1
            ckp1 = self._average_coefficient(group)
            correction1 = 1.0 - inner**step
            correction2 = 1.0 - beta2**step

            parameters = [
                parameter
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            for parameter in parameters:
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError(
                        "Schedule-Free AdamW does not support sparse gradients"
                    )
                state = self.state[parameter]
                if "z" not in state:
                    state["z"] = parameter.detach().clone(
                        memory_format=torch.preserve_format
                    )
                    state["exp_avg"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )

            # The model contains hundreds of parameter tensors. Issuing the
            # Adam/Schedule-Free algebra as individual CUDA kernels makes the
            # optimizer launch-bound, so batch it through PyTorch's multi-tensor
            # implementation. Keep the scalar loop for CPU/debug portability.
            use_foreach = (
                bool(parameters)
                and parameters[0].is_cuda
                and os.environ.get("METAMON_DISABLE_FOREACH_OPTIMIZER") != "1"
            )
            if use_foreach:
                gradients = [parameter.grad for parameter in parameters]
                states = [self.state[parameter] for parameter in parameters]
                z_values = [state["z"] for state in states]
                exp_avgs = [state["exp_avg"] for state in states]
                exp_avg_sqs = [state["exp_avg_sq"] for state in states]

                torch._foreach_mul_(exp_avgs, inner)
                torch._foreach_add_(exp_avgs, gradients, alpha=1.0 - inner)
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(
                    exp_avg_sqs, gradients, gradients, value=1.0 - beta2
                )
                directions = torch._foreach_div(exp_avgs, correction1)
                denominators = torch._foreach_div(exp_avg_sqs, correction2)
                torch._foreach_sqrt_(denominators)
                torch._foreach_add_(denominators, eps)
                torch._foreach_div_(directions, denominators)
                if decay != 0.0:
                    torch._foreach_add_(directions, parameters, alpha=decay)
                torch._foreach_lerp_(parameters, z_values, ckp1)
                torch._foreach_add_(
                    parameters,
                    directions,
                    alpha=lr * (beta * (1.0 - ckp1) - 1.0),
                )
                torch._foreach_add_(z_values, directions, alpha=-lr)
                group["k"] = step
                continue

            for parameter in parameters:
                gradient = parameter.grad
                state = self.state[parameter]

                z = state["z"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(inner).add_(gradient, alpha=1.0 - inner)
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient, gradient, value=1.0 - beta2
                )
                direction = exp_avg.div(correction1).div(
                    exp_avg_sq.div(correction2).sqrt().add_(eps)
                )
                if decay != 0.0:
                    direction.add_(parameter, alpha=decay)

                # p contains y. This fused algebra updates x and immediately
                # installs the next y without materializing a third full buffer.
                parameter.lerp_(z, weight=ckp1)
                parameter.add_(direction, alpha=lr * (beta * (1.0 - ckp1) - 1.0))
                z.add_(direction, alpha=-lr)

            group["k"] = step
        return loss
