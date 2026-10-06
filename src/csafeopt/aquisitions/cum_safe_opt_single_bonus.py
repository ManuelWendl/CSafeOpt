import math
from typing import Optional

import torch
from torch import Tensor

from csafeopt.aquisitions.cum_safe_opt import CSafeOpt


class CSafeOptSimple(CSafeOpt):
    def __init__(self, *args, delta_f: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_f = delta_f

    def uncertainty_coefficient(self, ut: Tensor, safe: Tensor) -> float:
        if self.delta_f is not None:
            delta_f_hat = self.delta_f
        else:
            delta_f_hat = (ut[safe].max() - ut[safe].min()).item() if safe.any() else 0.0
        beta_t = self.growing_beta()
        tau_t = self.threshold()
        return (delta_f_hat + self.epsilon + self.zeta) * math.sqrt(beta_t) / tau_t

    def evaluate(self, x: Tensor, step: int = 0) -> Tensor:  # noqa: ARG002
        posterior = self.model_posterior(x)
        l, u = self.get_confidence_interval(posterior)  # noqa: E741
        mean = posterior.mean.reshape(-1, self.dim_obs)
        std = torch.sqrt(posterior.variance.reshape(-1, self.dim_obs).clamp_min(0.0))[:, 0]

        safe = torch.all(l[:, 1:] > self.fmin[1:], axis=1)  # type: ignore

        slack = l - self.fmin
        penalty = self.soft_penalty(slack)
        ut = torch.where(safe, u[:, 0] + penalty, torch.full_like(u[:, 0], -1e10))

        lambda_t = self.uncertainty_coefficient(ut, safe)

        score = mean[:, 0] + penalty + lambda_t * std
        return torch.where(safe, score, torch.full_like(score, -1e10))
