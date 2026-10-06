import copy
import math
from typing import Optional

import torch
from torch import Tensor

from csafeopt.aquisitions.base_aquisition import BaseAquisition
from csafeopt.tools.data import Data

# Constants of the entropy approximation (Bottero et al., eq. 12-13).
C1 = 1.0 / (math.log(2.0) * math.pi)
C2 = 2.0 * C1 - 1.0
LN2 = math.log(2.0)


class ISEBO(BaseAquisition):
    def __init__(
        self,
        dim_obs: int,
        scale_beta: float,
        beta: float,
        n_max_value_samples: int = 10,
        n_max_value_candidates: int = 500,
        max_z: int = 4096,
        z_chunk: int = 512,
        x_chunk: int = 2048,
        data: Optional[Data] = None,
        context: Optional[Tensor] = None,
    ):
        super().__init__(dim_obs, scale_beta, beta, context=context, data=data, n_steps=1)
        if dim_obs != 2:
            raise ValueError("ISEBO is defined for one objective and one safety constraint (dim_obs == 2)")
        self.n_max_value_samples = n_max_value_samples
        self.n_max_value_candidates = n_max_value_candidates
        self.max_z = max_z
        self.z_chunk = z_chunk
        self.x_chunk = x_chunk
        # Which component won the argmax in the last evaluate(): "ISE" or "MES".
        self.last_choice = ""

    def is_internal_step(self, step: int = 0) -> bool:  # noqa: ARG002
        return False

    def evaluate(self, x: Tensor, step: int = 0) -> Tensor:  # noqa: ARG002
        posterior = self.model_posterior(x)
        l, u = self.get_confidence_interval(posterior)  # noqa: E741
        safe = torch.all(l[:, 1:] > self.fmin[1:], dim=1)
        scores = torch.full((x.shape[0],), -torch.inf, dtype=l.dtype, device=l.device)
        if not safe.any():
            return scores

        x_safe = x[safe]
        mean = posterior.mean.reshape(-1, self.dim_obs)
        std = torch.sqrt(posterior.variance.reshape(-1, self.dim_obs).clamp_min(1e-20))

        mes = self.max_value_entropy(x_safe, mean[safe, 0], std[safe, 0], u[safe, 0])
        ise = self.safe_exploration(x_safe, x, floor=mes).to(mes)

        scores[safe] = torch.maximum(ise, mes)
        best = int(torch.argmax(scores[safe]))
        self.last_choice = "ISE" if ise[best] >= mes[best] else "MES"
        return scores

    def max_value_samples(self, x_safe: Tensor, ucb: Tensor) -> Tensor:
        top = torch.topk(ucb, min(self.n_max_value_candidates, len(ucb))).indices
        with torch.no_grad():
            samples = self.model.models[0].posterior(x_safe[top]).rsample(torch.Size([self.n_max_value_samples]))
        f_star = samples.reshape(self.n_max_value_samples, -1).max(dim=1).values
        # f* can't be below the best safe value already observed (Wang & Jegelka, 2017).
        if self.data is not None and self.data.train_y is not None:
            observed = self.data.train_y.to(f_star)
            safe_obs = torch.all(observed[:, 1:] >= self.fmin[1:].to(f_star), dim=1)
            if safe_obs.any():
                f_star = torch.clamp(f_star, min=float(observed[safe_obs, 0].max()))
        return f_star

    def max_value_entropy(self, x_safe: Tensor, mean: Tensor, std: Tensor, ucb: Tensor) -> Tensor:
        f_star = self.max_value_samples(x_safe, ucb)
        gamma = (f_star.unsqueeze(1) - mean.unsqueeze(0)) / std.clamp_min(1e-10).unsqueeze(0)
        log_cdf = torch.special.log_ndtr(gamma)
        log_pdf = -0.5 * gamma**2 - 0.5 * math.log(2.0 * math.pi)
        values = 0.5 * gamma * torch.exp(log_pdf - log_cdf) - log_cdf
        return values.mean(dim=0).clamp_min(0.0)

    def constraint_factor(self) -> dict:
        model = self.model.models[1]
        train_x = model.train_inputs[0].double()
        noise = float(model.likelihood.noise.detach().mean())
        kernel = copy.deepcopy(model.covar_module).double()  # a copy: .double() converts in place
        with torch.no_grad():
            gram = kernel(train_x).to_dense() + noise * torch.eye(len(train_x), dtype=torch.float64)
            chol, info = torch.linalg.cholesky_ex(gram)
            jitter = 1e-10
            while info != 0:
                chol, info = torch.linalg.cholesky_ex(gram + jitter * torch.eye(len(train_x), dtype=torch.float64))
                jitter *= 10
        transform = getattr(model, "outcome_transform", None)
        scale = float(transform.stdvs.squeeze() ** 2) if transform is not None and hasattr(transform, "stdvs") else 1.0
        input_transform = getattr(model, "input_transform", None)
        return {
            "kernel": kernel, "train_x": train_x, "chol": chol, "scale": scale, "noise": scale * noise,
            "transform_inputs": (lambda x: input_transform(x)) if input_transform is not None else (lambda x: x),
        }

    @staticmethod
    def _whitened(factor: dict, x: Tensor) -> tuple:
        """(transformed inputs, L^{-1} k(X, x), posterior variance at x) for the constraint GP."""
        xt = factor["transform_inputs"](x).double()
        with torch.no_grad():
            k_train = factor["kernel"](factor["train_x"], xt).to_dense()
            prior_var = factor["kernel"](xt, diag=True)
        v = torch.linalg.solve_triangular(factor["chol"], k_train, upper=False)
        var = factor["scale"] * (prior_var - (v**2).sum(dim=0))
        return xt, v, var.clamp_min(1e-20)

    def safe_exploration(self, x_safe: Tensor, z_pool: Tensor, floor: Tensor) -> Tensor:
        """alpha_ISE(x) = max_z I_n({x, y}; 1[s(z) >= 0]) for every x in x_safe, z over z_pool (see class docstring)."""
        model = self.model.models[1]
        factor = self.constraint_factor()
        threshold = float(self.fmin[1])

        # Current entropy H_n(z) (eq. 12), an upper bound of I(x, z) for every x. Exact posterior (no LOVE),
        # so the bound matches the values computed below.
        with torch.no_grad():
            z_margin = model.posterior(z_pool).mean.reshape(-1).double() - threshold
        zt, v_z, z_var = self._whitened(factor, z_pool)
        entropy = LN2 * torch.exp(-C1 * z_margin**2 / z_var)
        order = torch.argsort(entropy, descending=True)[: self.max_z]
        xt, v_x, var_x = self._whitened(factor, x_safe)

        ise = torch.zeros(len(x_safe), dtype=torch.float64)
        floor = floor.double()
        for start in range(0, len(order), self.z_chunk):
            z_idx = order[start : start + self.z_chunk]
            # Every remaining z has I(x, z) <= H_n(z) <= H_n(z_idx[0]): stop once none can change the argmax.
            if entropy[z_idx[0]] <= torch.maximum(ise, floor).max():
                break
            ratio_sq = (z_margin[z_idx] ** 2 / z_var[z_idx]).unsqueeze(0)
            prior_h = torch.exp(-C1 * ratio_sq)
            for x_start in range(0, len(x_safe), self.x_chunk):
                chunk = slice(x_start, x_start + self.x_chunk)
                with torch.no_grad():
                    k_xz = factor["kernel"](xt[chunk], zt[z_idx]).to_dense()
                cross = factor["scale"] * (k_xz - v_x[:, chunk].T @ v_z[:, z_idx])
                # r = rho^2 sigma_x^2 / (sigma_x^2 + sigma_nu^2): the fraction of var(s(z)) an observation at x removes.
                r = (cross**2 / (z_var[z_idx].unsqueeze(0) * (var_x[chunk].unsqueeze(1) + factor["noise"])))
                r = r.clamp(0.0, 1.0 - 1e-12)
                # E_y[H_{n+1}(z) | {x, y}] (eq. 13), divided by ln 2.
                posterior_h = torch.sqrt((1.0 - r) / (1.0 + C2 * r)) * torch.exp(-C1 * ratio_sq / (1.0 + C2 * r))
                values = LN2 * (prior_h - posterior_h)  # eq. 14
                ise[chunk] = torch.maximum(ise[chunk], values.max(dim=1).values)
        return ise.clamp_min(0.0)
