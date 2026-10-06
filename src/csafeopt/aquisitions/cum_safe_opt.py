import math
from typing import Optional, Tuple  # noqa: UP035

import torch
from botorch.models.pairwise_gp import GPyTorchPosterior
from torch import Tensor

from csafeopt.aquisitions.base_aquisition import BaseAquisition


class CSafeOpt(BaseAquisition):
    def __init__(
        self,
        dim_obs: int,
        scale_beta: float,
        beta: float,
        epsilon: float = 1.0,
        alpha: float = 0.6,
        zeta: float = 0.1,
        rkhs_bound: float = 2.0,
        noise_proxy: float = 0.1,
        delta: float = 0.1,
        lipschitz: Optional[float] = None,
        context: Optional[Tensor] = None,
    ):
        super().__init__(dim_obs, scale_beta, beta, context=context, n_steps=1)
        self.epsilon = epsilon
        self.alpha = alpha
        self.zeta = zeta
        self.rkhs_bound = rkhs_bound
        self.noise_proxy = noise_proxy
        self.delta = delta
        self.lipschitz = lipschitz
        self.t = 1
        self.last_gate_max = 0.0
        self.last_gate_at_chosen = 0.0

    def is_internal_step(self, step: int = 0) -> bool:  # noqa: ARG002
        return False

    def after_optimization(self) -> None:
        self.t += 1

    def information_gain(self) -> float:
        reward_model = self.model.models[0]
        train_x = reward_model.train_inputs[0]
        noise = reward_model.likelihood.noise.mean().item()
        with torch.no_grad():
            k = reward_model.forward(train_x).covariance_matrix
        n = k.shape[-1]
        gram = torch.eye(n, dtype=k.dtype, device=k.device) + k / noise
        return 0.5 * torch.linalg.slogdet(gram)[1].item()

    def beta_t_coefficient(self) -> float:
        gamma = self.information_gain()
        return self.rkhs_bound + self.noise_proxy * math.sqrt(2.0 * (gamma + 1.0 + math.log(1.0 / self.delta)))

    def growing_beta(self) -> float:
        return self.beta_t_coefficient() ** 2

    def threshold(self) -> float:
        return self.epsilon / self.growing_beta() ** self.alpha

    def get_confidence_interval(self, posterior: GPyTorchPosterior) -> Tuple[Tensor, Tensor]:
        mean = posterior.mean.reshape(-1, self.dim_obs)
        var = posterior.variance.reshape(-1, self.dim_obs)
        std = torch.sqrt(var.clamp_min(0.0))
        half_width = self.scale_beta * self.beta_t_coefficient() * std
        return mean - half_width, mean + half_width

    def acquisition_terms(self, x: Tensor) -> dict:
        posterior = self.model_posterior(x)
        l, u = self.get_confidence_interval(posterior)  # noqa: E741
        constraint_std = torch.sqrt(posterior.variance.reshape(-1, self.dim_obs)[:, 1:].clamp_min(0.0)).amin(dim=1)

        safe = torch.all(l[:, 1:] > self.fmin[1:], axis=1)  # type: ignore
        slack = l - self.fmin
        ut = torch.where(safe, u[:, 0] + self.soft_penalty(slack), torch.full_like(u[:, 0], -1e10))

        tau_t = self.threshold()
        expander = safe & (constraint_std > tau_t)
        if self.lipschitz is not None:
            expander &= self.safeopt_expanders(x, u, safe)

        gate = torch.clamp(constraint_std - tau_t, min=0.0)
        gate[~expander] = 0.0

        gate_max = gate.max()
        normalized_gate = gate / gate_max if gate_max > 0 else torch.zeros_like(gate)

        kappa = (ut[safe].max() - ut[safe].min() + self.zeta) if safe.any() else self.zeta

        return {
            "ut": ut, "kappa": kappa, "normalized_gate": normalized_gate, "gate_max": gate_max,
            "safe": safe, "expander": expander, "lower": l, "upper": u, "tau": tau_t,
        }

    def safeopt_expanders(self, x: Tensor, upper: Tensor, safe: Tensor, chunk: int = 2048) -> Tensor:
        result = torch.zeros_like(safe)
        if not safe.any() or safe.all():
            return result

        reach = (upper[:, 1:] - self.fmin[1:]).amin(dim=1) / self.lipschitz
        unsafe_x = x[~safe]
        safe_index = torch.nonzero(safe).squeeze(1)
        for start in range(0, len(safe_index), chunk):
            index = safe_index[start : start + chunk]
            nearest = torch.cdist(x[index], unsafe_x).amin(dim=1)
            result[index] = nearest <= reach[index]
        return result

    def evaluate(self, x: Tensor, step: int = 0) -> Tensor:  # noqa: ARG002
        terms = self.acquisition_terms(x)
        self.last_gate_max = float(terms["gate_max"].item())
        scores = terms["ut"] + terms["kappa"] * terms["normalized_gate"]
        self.last_gate_at_chosen = float(terms["normalized_gate"][torch.argmax(scores)].item())
        return scores

    def reset(self):
        self.t = 1
        self.last_gate_max = 0.0
        self.last_gate_at_chosen = 0.0
