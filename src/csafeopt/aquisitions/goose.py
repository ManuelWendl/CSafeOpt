from typing import Optional

import torch
from torch import Tensor

from csafeopt.aquisitions.base_aquisition import BaseAquisition


class Goose(BaseAquisition):
    def __init__(
        self,
        dim_obs: int,
        scale_beta: float,
        beta: float,
        lipschitz: float,
        epsilon: float = 0.1,
        context: Optional[Tensor] = None,
    ):
        super().__init__(dim_obs, scale_beta, beta, context=context, n_steps=1)
        self.lipschitz = lipschitz
        self.epsilon = epsilon

    def is_internal_step(self, step: int = 0) -> bool:  # noqa: ARG002
        return False

    def evaluate(self, x: Tensor, step: int = 0) -> Tensor:  # noqa: ARG002
        posterior = self.model_posterior(x)
        l, u = self.get_confidence_interval(posterior)  # noqa: E741

        safe = torch.all(l[:, 1:] > self.fmin[1:], axis=1)  # type: ignore  # S_t^p
        optimistic_safe = torch.all(u[:, 1:] > self.fmin[1:] + self.epsilon, axis=1)  # type: ignore

        slack = l - self.fmin
        ut = torch.where(safe, u[:, 0] + self.soft_penalty(slack), torch.full_like(u[:, 0], -1e10))

        if optimistic_safe.any():
            oracle_scores = torch.where(optimistic_safe, u[:, 0], torch.full_like(u[:, 0], -1e10))
        else:
            oracle_scores = u[:, 0]
        oracle_idx = int(torch.argmax(oracle_scores).item())

        if safe[oracle_idx]:
            return ut

        return self._safe_expansion_scores(x, l, u, ut, safe, oracle_idx)

    def _safe_expansion_scores(
        self, x: Tensor, l: Tensor, u: Tensor, ut: Tensor, safe: Tensor, oracle_idx: int  # noqa: E741
    ) -> Tensor:
        width = (u[:, 1:] - l[:, 1:]).amin(dim=1)
        expander_pool = safe & (width > self.epsilon)

        if not expander_pool.any():
            # Nothing left to usefully learn from; fall back to plain safe-UCB.
            return ut

        d = torch.cdist(x, x)  # pairwise decision-space distance

        margin = u[:, 1:].unsqueeze(1) - self.lipschitz * d.unsqueeze(-1) - self.fmin[1:]
        margin = margin.amin(dim=-1)  # (n, n)

        not_yet_safe = ~safe

        margin_from_pool = margin.masked_fill(~expander_pool.unsqueeze(1), float("-inf"))
        reachable_target = not_yet_safe & ((margin_from_pool + self.epsilon) >= 0.0).any(dim=0)

        if not reachable_target.any():
            # No not-yet-safe candidate is even optimistically reachable yet.
            return ut

        priority = torch.where(reachable_target, d[:, oracle_idx], torch.full_like(d[:, oracle_idx], float("inf")))
        target_idx = int(torch.argmin(priority).item())

        can_certify = ((margin[:, target_idx] + self.epsilon) >= 0.0) & expander_pool

        if not can_certify.any():
            # Should not happen given reachable_target above, but degrade gracefully.
            return ut

        gate_score = (u[:, 1:] - l[:, 1:]).amin(dim=1)
        return torch.where(can_certify, gate_score, torch.full_like(gate_score, -1e10))
