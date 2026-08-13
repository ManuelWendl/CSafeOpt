import math
from typing import Optional, Tuple

import torch
from botorch.models.pairwise_gp import GPyTorchPosterior
from torch import Tensor

from gosafeopt.aquisitions.base_aquisition import BaseAquisition


class CSafeOpt(BaseAquisition):
    """Cumulative Safe Opt: safe GP-UCB gated by a shrinking expansion bonus.

    Implements the single scalar acquisition A_t(x) = u_t(x) + kappa_t *
    qbar_t(x) from "Cumulative Safe Opt" (Wendl, 2026). The gate qbar_t is
    the normalized, thresholded posterior standard deviation of safe,
    *expander* points; its support is exactly the safe points whose
    constraint uncertainty AND reward uncertainty both still exceed the same
    shrinking threshold tau_t = epsilon / beta_t ** alpha (eq. 29, cf.
    GOOSE's W_t^eps in goose.py). Restricting to expanders keeps the gate
    from firing on a safe point whose constraint is already tau_t-accurately
    known but whose reward happens to still be uncertain -- such a point
    can't move the safe/unsafe boundary, so exploring it for reward alone
    isn't the kind of expansion the gate is meant to prioritize. Whenever an
    active gate exists it strictly dominates every zero-gate point (Lemma
    2), so the planner always prefers an "expander" candidate over plain
    GP-UCB; once no expander remains sufficiently reward-uncertain the gate
    vanishes exactly and the acquisition reduces to safe GP-UCB. alpha
    controls how fast the threshold shrinks over rounds (Section 6.5):
    alpha=0 is a fixed raw threshold, alpha=1/2 is confidence-width matched,
    alpha>1/2 asymptotically reaches the full safely-reachable comparator.

    beta_t follows Chowdhury & Gopalan (2017), "On Kernelized Multi-armed
    Bandits", Theorem 2 -- the confidence sequence for a continuous (RKHS)
    domain, as opposed to Srinivas et al.'s finite-domain bound. The same
    beta_t governs both the confidence interval (Assumption 1, eq. 1-2) and
    the gate threshold tau_t (eq. 29), matching the paper's construction.
    gamma_{t-1} (the maximum information gain) is computed from the reward
    model's own kernel over the points observed so far, not approximated.
    """

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
        context: Optional[Tensor] = None,
    ):
        # `beta` is accepted for interface compatibility with BaseAquisition
        # but is not used directly here: the confidence width comes from the
        # Chowdhury-Gopalan schedule (rkhs_bound/noise_proxy/delta) below.
        super().__init__(dim_obs, scale_beta, beta, context=context, n_steps=1)
        self.epsilon = epsilon
        self.alpha = alpha
        self.zeta = zeta
        self.rkhs_bound = rkhs_bound
        self.noise_proxy = noise_proxy
        self.delta = delta
        self.t = 1

    def is_internal_step(self, step: int = 0) -> bool:  # noqa: ARG002
        return False

    def after_optimization(self) -> None:
        self.t += 1

    def information_gain(self) -> float:
        # gamma_{t-1} = 0.5 * log det(I + K/lambda) (Lemma 5), from the reward
        # model's own kernel matrix over the points observed so far.
        reward_model = self.model.models[0]
        train_x = reward_model.train_inputs[0]
        noise = reward_model.likelihood.noise.mean().item()
        with torch.no_grad():
            k = reward_model.forward(train_x).covariance_matrix
        n = k.shape[-1]
        gram = torch.eye(n, dtype=k.dtype, device=k.device) + k / noise
        return 0.5 * torch.linalg.slogdet(gram)[1].item()

    def beta_t_coefficient(self) -> float:
        # Chowdhury & Gopalan (2017), Thm 2: w.p. >= 1-delta, for all t, x,
        # |f(x) - mu_{t-1}(x)| <= beta_t_coefficient() * sigma_{t-1}(x).
        gamma = self.information_gain()
        return self.rkhs_bound + self.noise_proxy * math.sqrt(2.0 * (gamma + 1.0 + math.log(1.0 / self.delta)))

    def growing_beta(self) -> float:
        # Qt(x) (eq. 2) is defined with sqrt(beta_t); beta_t_coefficient() IS
        # that sqrt(beta_t) directly, so square it to keep tau_t governed by
        # the same beta_t sequence used for the confidence interval below.
        return self.beta_t_coefficient() ** 2

    def threshold(self) -> float:
        return self.epsilon / self.growing_beta() ** self.alpha

    def get_confidence_interval(self, posterior: GPyTorchPosterior) -> Tuple[Tensor, Tensor]:
        mean = posterior.mean.reshape(-1, self.dim_obs)
        var = posterior.variance.reshape(-1, self.dim_obs)
        std = torch.sqrt(var.clamp_min(0.0))
        half_width = self.scale_beta * self.beta_t_coefficient() * std
        return mean - half_width, mean + half_width

    def evaluate(self, x: Tensor, step: int = 0) -> Tensor:  # noqa: ARG002
        posterior = self.model_posterior(x)
        l, u = self.get_confidence_interval(posterior)  # noqa: E741
        std = torch.sqrt(posterior.variance.reshape(-1, self.dim_obs))[:, 0]

        safe = torch.all(l[:, 1:] > self.fmin[1:], axis=1)  # type: ignore

        slack = l - self.fmin
        ut = u[:, 0] + self.soft_penalty(slack)

        # Expander indicator (cf. GOOSE's W_t^eps, goose.py): safe points whose
        # constraint confidence interval is still wider than tau_t = epsilon /
        # beta_t^alpha (eq. 29) -- the same shrinking threshold the reward
        # gate uses below, not a fixed raw epsilon -- on the tightest
        # constraint. A fixed epsilon here collapsed the expander pool to
        # empty within a couple of rounds whenever the constraint model's
        # fitted uncertainty starts out small relative to a benchmark's raw
        # epsilon, permanently disabling the gate since nothing could ever
        # become an expander again. tau_t starts generous (beta_t small
        # early on) and only tightens as the model matures, keeping a single
        # coherent schedule for both reward- and constraint-uncertainty
        # gating instead of two thresholds tuned on different scales.
        tau_t = self.threshold()
        constraint_width = (u[:, 1:] - l[:, 1:]).amin(dim=1)
        expander = safe & (constraint_width > tau_t)

        # q_t^(alpha) (eq. 30): support is G_t^(alpha), the safe, expander,
        # and sufficiently (reward-)uncertain points.
        gate = torch.clamp(std - tau_t, min=0.0)
        gate[~expander] = 0.0

        gate_max = gate.max()
        normalized_gate = gate / gate_max if gate_max > 0 else torch.zeros_like(gate)

        # kappa_t > osc_{S_t}(u_t) guarantees exact expansion priority (Lemma 2 / eq. 146).
        kappa = (ut[safe].max() - ut[safe].min() + self.zeta) if safe.any() else self.zeta

        return ut + kappa * normalized_gate

    def reset(self):
        self.t = 1
