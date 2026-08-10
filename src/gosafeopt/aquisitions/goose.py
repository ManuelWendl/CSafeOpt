from typing import Optional

import torch
from torch import Tensor

from gosafeopt.aquisitions.base_aquisition import BaseAquisition


class Goose(BaseAquisition):
    """GOOSE (Turchetta, Berkenkamp & Krause, NeurIPS 2019): goal-oriented safe exploration wrapped around a GP-UCB oracle.

    Each round, an *oracle* (plain GP-UCB, eq. u_t(x)) proposes a candidate
    x*_i by maximizing u_t within the optimistic safe set S_t^{o,eps}. If
    x*_i is already pessimistically safe (x*_i in S_t^p, i.e. l_t(x*_i) >
    fmin), it is evaluated directly -- no expansion happens. Otherwise, Safe
    Expansion (Algorithm 2) is triggered: instead of reducing uncertainty
    anywhere in the safe set (as SafeOpt's expanders do), GOOSE only samples
    points that are *informative about x*_i specifically* -- safe-but-
    uncertain points x for which the Lipschitz-discounted upper bound
    u_t(x) - L*d(x, x*_i) could certify x*_i as safe. If no such direct
    "bridge" to x*_i exists yet, the closest not-yet-safe candidate that
    *does* have a bridge is targeted instead (a stepping stone toward x*_i),
    matching the priority-by-reachability ordering of eq. (25)-(26). This is
    the defining GOOSE behaviour: "it only expands if it is required to
    certify a UCB suggestion."

    Two simplifications relative to the paper, both consistent with how the
    other acquisitions in this library already treat the graph-free,
    single-rollout BO setting (Section 4.1 of the paper: "In the BO setting
    the graph is fully connected"):
      - Safe-set membership (S^p_t, S^{o,eps}_t) is checked directly via the
        GP's own confidence bounds rather than by iterating the pessimistic/
        optimistic set-expansion operators (p_t, o^eps_t) to a fixed point;
        the kernel's own smoothness already plays the role the explicit
        Lipschitz propagation plays in the paper's more general graph
        setting. The Lipschitz constant `lipschitz` is only used for the
        one-hop reachability checks inside Safe Expansion itself, which is
        GOOSE's actual distinguishing mechanism.
      - Every round still evaluates the full (reward, constraint) tuple via
        one environment rollout (like every other acquisition here); GOOSE's
        "expansion" step chooses *where* to sample based on the goal-directed
        logic above rather than sampling the constraint q in isolation.
    """

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
        # one-hop S_t^{o,eps} (eq. 2): needs the same epsilon slack the expansion
        # certifier below uses, or points can be "optimistically safe" by less
        # than the accuracy GOOSE is willing to certify to.
        optimistic_safe = torch.all(u[:, 1:] > self.fmin[1:] + self.epsilon, axis=1)  # type: ignore

        slack = l - self.fmin
        # ut is only ever handed to the outer optimizer's argmax as a fallback:
        # line 10's "evaluate x*_i" once it's *already confirmed* pessimistically
        # safe, or graceful degradation in _safe_expansion_scores below when
        # nothing is left to expand toward. soft_penalty alone doesn't guarantee
        # it dominates a barely-unsafe point with a high reward UCB, so mask it
        # to -1e10 outside the safe set -- matching how oracle_scores below is
        # already masked to optimistic_safe -- to keep every fallback confined
        # to points already certified safe.
        ut = torch.where(safe, u[:, 0] + self.soft_penalty(slack), torch.full_like(u[:, 0], -1e10))

        # Oracle (line 4): plain GP-UCB on the objective, ignorant of safety -- the
        # paper's "unsafe IML algorithm" -- restricted only to the optimistic safe
        # set S_t^{o,eps} it is allowed to search over. No soft_penalty here: that
        # term reflects GOOSE's own safety awareness, not the oracle's.
        if optimistic_safe.any():
            oracle_scores = torch.where(optimistic_safe, u[:, 0], torch.full_like(u[:, 0], -1e10))
        else:
            oracle_scores = u[:, 0]
        oracle_idx = int(torch.argmax(oracle_scores).item())

        if safe[oracle_idx]:
            # x*_i in S_t^p already: evaluate it directly, no expansion needed (line 10).
            return ut

        return self._safe_expansion_scores(x, l, u, ut, safe, oracle_idx)

    def _safe_expansion_scores(
        self, x: Tensor, l: Tensor, u: Tensor, ut: Tensor, safe: Tensor, oracle_idx: int  # noqa: E741
    ) -> Tensor:
        # W_t^eps (eq. 24): safe points whose constraint confidence interval
        # is still wider than epsilon on the tightest constraint.
        width = (u[:, 1:] - l[:, 1:]).amin(dim=1)
        expander_pool = safe & (width > self.epsilon)

        if not expander_pool.any():
            # Nothing left to usefully learn from; fall back to plain safe-UCB.
            return ut

        d = torch.cdist(x, x)  # pairwise decision-space distance

        # margin[i, j] = how far expander candidate i's own upper bound, discounted
        # by the Lipschitz cost of reaching j, clears every constraint threshold.
        margin = u[:, 1:].unsqueeze(1) - self.lipschitz * d.unsqueeze(-1) - self.fmin[1:]
        margin = margin.amin(dim=-1)  # (n, n)

        not_yet_safe = ~safe

        # A_t (eq. 25 pool): not-yet-safe candidates reachable within the
        # optimistic (eps-slack) envelope of *some* currently safe point.
        margin_from_safe = margin.masked_fill(~safe.unsqueeze(1), float("-inf"))
        reachable_target = not_yet_safe & ((margin_from_safe + self.epsilon) >= 0.0).any(dim=0)

        if not reachable_target.any():
            # No not-yet-safe candidate is even optimistically reachable yet.
            return ut

        # Priority h(x) = distance to the oracle's own suggestion x*_i (eq. 25):
        # among reachable targets, prefer the one closest to what the oracle wants.
        priority = torch.where(reachable_target, d[:, oracle_idx], torch.full_like(d[:, oracle_idx], float("inf")))
        target_idx = int(torch.argmin(priority).item())

        # G_t^eps(alpha*) (eq. 26): expanders that directly certify the target. Uses
        # the same eps-slack as the reachability check above -- target_idx was only
        # ever reachable within that slack (the one-hop simplification collapses the
        # paper's separate S^{o,eps} construction and no-slack certifier check into a
        # single consistent tolerance), so requiring an unslacked margin here would
        # make every reachable target systematically uncertifiable.
        can_certify = ((margin[:, target_idx] + self.epsilon) >= 0.0) & expander_pool

        if not can_certify.any():
            # Should not happen given reachable_target above, but degrade gracefully.
            return ut

        # w_t(x) = u_t(x) - l_t(x) on the constraint channel (eq. "most uncertain expander").
        gate_score = (u[:, 1:] - l[:, 1:]).amin(dim=1)
        return torch.where(can_certify, gate_score, torch.full_like(gate_score, -1e10))
