"""A safe-exploration benchmark on real Mars terrain, reproducing the data
source of the GOOSE paper's (Turchetta, Berkenkamp & Krause, NeurIPS 2019,
Sec. 4) Mars rover experiment: a rover lands in a known-safe spot and must
reach a distant, scientifically interesting outcrop without traversing
terrain steeper than its rated safety limit.

The elevation grid in examples/data/mars_dtm.npy is a real 120x70 (~1.01
m/pixel) crop of a HiRISE Digital Terrain Model (McEwen et al., 2007 -- the
exact citation GOOSE itself uses for its Mars terrain), extracted at the same
pixel offsets as the Mars example in the directly related, publicly released
Turchetta et al. (2016) SafeMDP codebase (github.com/befelix/SafeMDP), which
the GOOSE Mars experiment builds on. GOOSE's paper does not name the specific
16 locations it evaluated on, so this is the closest reproducible stand-in:
same instrument, same data source, same extraction recipe, from the authors'
own earlier public code. See examples/data/README.md for exact provenance.

The safety constraint is the real local terrain slope (rise/run of a smooth
spline fit through the DTM), thresholded at tan(20 deg) -- slightly more
conservative than the 25 deg margin GOOSE itself uses (Sec. 4: "we set
conservatively the safety constraint to be ... 25 deg", vs. the Mars Science
Laboratory rover's actual 30 deg rating), chosen here because it is the
threshold at which real terrain near SafeMDP's own start node happens to
carve out a genuine detour requirement rather than a single-step
peek-through. The starting point matches SafeMDP's own start node; the
target is a real, breadth-first-search-verified safe location whose only
safe access requires a real detour (path length ~1.9x the straight-line
distance) around genuinely unsafe intermediate terrain -- not a hand-placed
synthetic bottleneck. The reward ("scientific interest") field is still
synthetic, as neither GOOSE nor SafeMDP define one: their Mars experiments
optimize a shortest *safe path*, not a reward, whereas this benchmark (like
the rest of this file's siblings) is safe Bayesian
optimization. Reuses the paper-style plotting and Chowdhury-Gopalan
infrastructure from bottleneck_demo.py rather than duplicating it.
"""

import random
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
from botorch.models.gp_regression import SingleTaskGP
from tueplots import figsizes
from bottleneck_demo import (
    MARKERS,
    PALETTE,
    GrowingGoose,
    GrowingGoSafeOpt,
    GrowingSafeOpt,
    GrowingSafeUCB,
    InstrumentedCumSafeOpt,
    _apply_theme,
    _legend,
    _plot_series,
    _style_axis,
    reset_global_state,
)
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.interpolate import RectBivariateSpline
from torch import Tensor

import gosafeopt
from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.experiments.environment import Environment
from gosafeopt.experiments.experiment import Experiment
from gosafeopt.models.model import ModelGenerator
from gosafeopt.optim.grid_opt import GridOpt
from gosafeopt.tools.data import Data
from gosafeopt.tools.logger import Logger
from gosafeopt.trainer import Trainer

# GridOpt builds domain-bound tensors on the CPU and never moves them onto
# gosafeopt.device; force CPU here so this stays consistent (see examples/compare.py).
gosafeopt.device = torch.device("cpu")


app = typer.Typer()

# --- real Mars DTM (see examples/data/README.md for provenance) --------
DATA_PATH = Path(__file__).parent / "data" / "mars_dtm.npy"
PIXEL_STEP = 1.01  # meters/pixel, HiRISE DTM native resolution
_dtm = np.load(DATA_PATH)
_rows, _cols = _dtm.shape
_xs_grid = np.arange(_rows) * PIXEL_STEP
_ys_grid = np.arange(_cols) * PIXEL_STEP
_elevation_spline = RectBivariateSpline(_xs_grid, _ys_grid, _dtm)

# The optimization domain is intentionally tighter than the full DTM extent:
# a real BFS-verified safe path between SEED and TARGET (see below) exists
# entirely within pixel rows [29,61] / cols [18,60], so restricting candidate
# sampling to that corridor (with margin) multiplies effective candidate
# density (~2.8x) for every algorithm -- unlike raising set_size, this is
# free and doesn't blow up GOOSE's O(n^2) reachability cost. The elevation
# spline itself is still fit over the full 120x70 grid (_xs_grid/_ys_grid).
DOMAIN_X = (20.0 * PIXEL_STEP, 70.0 * PIXEL_STEP)
DOMAIN_Y = (8.0 * PIXEL_STEP, 68.0 * PIXEL_STEP)
SEED = (65 * PIXEL_STEP, 60 * PIXEL_STEP)  # 1px from SafeMDP's own start node (comfortable margin at 20deg)
# Real, BFS-verified safe outcrop ~50m away whose *straight-line* path from
# SEED is firmly blocked (constraint margin -0.44) by real terrain, while a
# real detour (~1.4x the direct distance) exists -- reaching it requires
# genuinely routing around the unsafe terrain, at the 20 deg threshold below.
TARGET = (35 * PIXEL_STEP, 35 * PIXEL_STEP)

SLOPE_LIMIT = float(np.tan(np.deg2rad(27.0)))  # more conservative than GOOSE's 25 deg -- see SEED/TARGET comment


def _spline_eval(x: np.ndarray, y: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    shape = np.broadcast(x, y).shape
    xf = np.broadcast_to(x, shape).ravel()
    yf = np.broadcast_to(y, shape).ravel()
    return _elevation_spline.ev(xf, yf, dx=dx, dy=dy).reshape(shape)


def elevation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _spline_eval(x, y)


def slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dhdx = _spline_eval(x, y, dx=1, dy=0)
    dhdy = _spline_eval(x, y, dx=0, dy=1)
    return np.sqrt(dhdx**2 + dhdy**2)


def constraint_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return SLOPE_LIMIT - slope(x, y)


def reward_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # "Scientific interest": synthetic (neither GOOSE nor SafeMDP define one
    # for this terrain), independent of elevation -- climbing isn't itself
    # rewarding, only reaching the target outcrop is. Wide enough to give
    # every algorithm a directional gradient toward the target (a fully flat
    # reward field just turns this into an undirected full-domain search,
    # which defeats *every* algorithm equally and stops being a meaningful
    # comparison). What actually separates SafeUCB from the others is that
    # the direct route the gradient points along is blocked by a firm real
    # safety violation (see TARGET comment): SafeUCB has no mechanism to
    # deliberately resolve uncertainty at that boundary and reduce to
    # tolerating the violation via soft_penalty, while SafeOpt/CumSafeOpt/
    # GoSafeOpt/GOOSE actively learn about the safe detour around it.
    return (
        1.0 * np.exp(-(((x - SEED[0]) ** 2 + (y - SEED[1]) ** 2)) / (2 * 8.0**2))
        + 3.5 * np.exp(-(((x - TARGET[0]) ** 2 + (y - TARGET[1]) ** 2)) / (2 * 10.0**2))
        - 0.3
    )


class MarsEnv(Environment):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(None, render_mode)

    def reset(self, *, seed=None, options=None):
        return np.zeros(2), {}

    def step(self, k):
        x, y = float(k[0]), float(k[1])
        # Experiment.rollout divides by len(trajectory) == 2 for a single-step
        # episode; double the raw values so data.train_y matches reward_fn/constraint_fn.
        reward = np.array([2 * float(reward_fn(x, y)), 2 * float(constraint_fn(x, y))])
        return np.array([x, y]), reward, True, False, {}


def _fit_constraint_kernel(n_samples: int = 1000, noise_std: float = 0.1, seed: int = 0):
    """MLE-fit a Matern kernel's lengthscale/noise once, offline, from dense samples.

    Sampled uniformly across the map -- exactly GOOSE's own Sec. 4 recipe
    ("we take 1000 noisy measurements at random locations from each map,
    which in reality, could come from satellite images, to find a maximum a
    posteriori estimator of the hyperparameters to fine tune our prior to
    each site"). This replaces guessing a lengthscale by hand: the online run
    never refits (refit_interval=0), it just starts from hyperparameters
    already calibrated to this terrain's true roughness, which is what
    online GP-UCB/expander acquisitions implicitly assume is correct.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(DOMAIN_X[0]+30, DOMAIN_X[1], n_samples)
    ys = rng.uniform(DOMAIN_Y[0]+20, DOMAIN_Y[1], n_samples)
    values = constraint_fn(xs, ys) + rng.normal(0.0, noise_std, n_samples)

    # Fit on inputs normalized to [0, 1] per axis -- botorch's L-BFGS-B fit
    # is numerically unstable on raw ~20-70m-scale coordinates (confirmed:
    # it reliably threw ModelFittingError without this). The domain is fixed
    # here (unlike the online run, which the *paper's own* 25 deg threshold
    # discussion shows may legitimately vary per site), so lengthscale can be
    # rescaled back to raw meters afterward without losing anything.
    width_x, width_y = DOMAIN_X[1] - DOMAIN_X[0], DOMAIN_Y[1] - DOMAIN_Y[0]
    xs_norm = (xs - DOMAIN_X[0]) / width_x
    ys_norm = (ys - DOMAIN_Y[0]) / width_y

    train_x = torch.tensor(np.stack([xs_norm, ys_norm], axis=1))
    train_y = torch.tensor(values)

    covar_module = ScaleKernel(MaternKernel(ard_num_dims=2))
    likelihood = GaussianLikelihood()
    likelihood.noise = noise_std**2
    # No Standardize() here: constraint_fn's values are already O(1)-scaled.
    model = SingleTaskGP(train_x, train_y.unsqueeze(-1), likelihood=likelihood, covar_module=covar_module)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # A manual Adam loop rather than fit_gpytorch_mll's multi-restart L-BFGS-B:
    # the latter reliably threw ModelFittingError ("scipy_minimize ... ABNORMAL")
    # on this exact data whenever gosafeopt's own float64 default dtype (set at
    # import time) was active -- reproduced deterministically across many restart
    # seeds and both float32/float64 explicit casts, so it's specific to botorch's
    # scipy bridge here, not this data being unfittable. Adam converges cleanly.
    model.train()
    model.likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(300):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    model.likelihood.eval()

    lengthscale_norm = model.covar_module.base_kernel.lengthscale.detach().reshape(-1).tolist()
    lengthscale_raw = [lengthscale_norm[0] * width_x, lengthscale_norm[1] * width_y]
    fitted_noise = float(model.likelihood.noise.detach().mean())
    print(
        f"Fitted constraint kernel from {n_samples} dense samples: "
        f"lengthscale(normalized)={lengthscale_norm} (raw meters={lengthscale_raw}) noise={fitted_noise:.2e}"
    )
    # CONFIG["model"]["normalize_input"] is True, so ModelGenerator's own
    # Normalize(domain_start, domain_end) transform maps inputs into this same
    # [0, 1] space -- the kernel needs its lengthscale in those units too.
    return lengthscale_norm, fitted_noise


_FITTED_LENGTHSCALE, _FITTED_NOISE = _fit_constraint_kernel()

CONFIG = {
    "log_video": False,
    "log_plots": False,
    "dim_obs": 2,
    "dim_params": 2,
    "dim_context": 0,
    "dim_model": 2,
    "domain_start": [DOMAIN_X[0], DOMAIN_Y[0]],
    "domain_end": [DOMAIN_X[1], DOMAIN_Y[1]],
    "model": {
        # normalize_input=True -> ModelGenerator's Normalize(domain_start,
        # domain_end) maps inputs to [0, 1], so lengthscale must be in that
        # same normalized space (see _fit_constraint_kernel's return value).
        # Fitted once offline from dense samples of the real terrain rather
        # than hand-picked or refit online from the handful of sparse points
        # an actual BO run collects.
        "lenghtscale": _FITTED_LENGTHSCALE,
        "normalize_input": True,
        "normalize_output": True,
        "likelihood_noise": _FITTED_NOISE,
    },
    "Optimization": {
        "set_size": 8000,
        "set_init": "random",
        "max_global_steps_without_progress_tolerance": 0.9,
        "max_global_steps_without_progress": 10_000,  # effectively disabled
    },
    # GOOSE's reachability check is O(set_size^2) (a full pairwise distance
    # matrix each round), so it keeps the base set_size above. The other four
    # acquisitions are O(set_size) and benefit a lot from denser candidate
    # coverage of the (already-tightened) safe corridor -- see SET_SIZE_OVERRIDES.
    "SET_SIZE_OVERRIDES": {"SafeOpt": 24000, "SafeUCB": 24000, "CumSafeOpt": 24000, "GoSafeOpt": 24000},
    "SafeOpt": {"scale_beta": 1.0, "beta": 9},
    "SafeUCB": {"scale_beta": 1.0, "beta": 9},
    # "CumSafeOpt": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.2, "alpha": 0.7, "zeta": 0.001},
    "CumSafeOpt": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.15, "alpha": 0.5, "zeta": 0.001},
    "GoSafeOpt": {"scale_beta": 1.0, "beta": 9, "n_max_local": 5, "n_max_global": 3},
    "Goose": {"scale_beta": 1.0, "beta": 9, "lipschitz": 1.0, "epsilon": 0.2},
}


class GateTrackingCumSafeOpt(InstrumentedCumSafeOpt):
    """Records whether the *actual chosen point* each round came from the gate or from plain UCB.

    Some candidate clearing `std > tau_t` doesn't necessarily mean the point
    CumSafeOpt actually picks needed the gate: what matters is whether the
    *argmax* of the full acquisition `ut + kappa*normalized_gate` has a
    nonzero gate term. Lemma 2's dominance property (see cum_safe_opt.py)
    means the point achieving normalized_gate=1 (the single most-uncertain
    gate-active point) is *guaranteed* to win over every plain-UCB point --
    but a weaker gate-active point (0 < normalized_gate < 1) is not
    guaranteed to win over a plain-UCB point with a high enough ut of its
    own, so which one actually wins is not fully determined by the count of
    gate-active candidates alone. This checks the winner directly rather
    than relying on that argument: it replicates CumSafeOpt.evaluate()'s
    computation locally (once, not calling it twice), finds the argmax
    itself, and records normalized_gate at that specific point -- exactly 0
    when plain UCB determined the choice, and how close to 1 the winning
    point's own gate strength was otherwise. Does not change acquisition
    behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_history: list[tuple[int, float]] = []  # (round t, normalized_gate at the chosen point)

    def evaluate(self, x: Tensor, step: int = 0) -> Tensor:  # noqa: ARG002
        posterior = self.model_posterior(x)
        l, u = self.get_confidence_interval(posterior)  # noqa: E741
        std = torch.sqrt(posterior.variance.reshape(-1, self.dim_obs))[:, 0]

        safe = torch.all(l[:, 1:] > self.fmin[1:], axis=1)  # type: ignore

        slack = l - self.fmin
        ut = u[:, 0] + self.soft_penalty(slack)

        gate = torch.clamp(std - self.threshold(), min=0.0)
        gate[~safe] = 0.0

        gate_max = gate.max()
        normalized_gate = gate / gate_max if gate_max > 0 else torch.zeros_like(gate)

        kappa = (ut[safe].max() - ut[safe].min() + self.zeta) if safe.any() else self.zeta

        scores = ut + kappa * normalized_gate
        chosen_idx = int(torch.argmax(scores).item())
        self.gate_history.append((self.t, float(normalized_gate[chosen_idx].item())))
        return scores


def _gate_spans(gate_history: list) -> list:
    """Collapse a per-round (t, gate_strength) history into contiguous (gate_used, start, end) spans."""
    spans = []
    state, start, prev = None, None, None
    for t, strength in gate_history:
        used = strength > 0.0
        if used != state:
            if state is not None:
                spans.append((state, start, prev))
            state, start = used, t
        prev = t
    if state is not None:
        spans.append((state, start, prev))
    return spans


def _shade_gate_intensity(ax, gate_history: list) -> None:
    """Shade the background by normalized_gate at the chosen point (already in [0, 1]).

    0 means plain UCB picked this round's point; values approaching 1 mean
    the winning point was itself the (or close to the) most gate-active
    candidate that round.
    """
    ts = np.array([t for t, _ in gate_history], dtype=float)
    intensity = np.array([s for _, s in gate_history], dtype=float)

    ylim = ax.get_ylim()
    extent = [ts.min() - 0.5, ts.max() + 0.5, ylim[0], ylim[1]]
    ax.imshow(
        intensity[np.newaxis, :],
        aspect="auto",
        extent=extent,
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        alpha=0.45,
        zorder=0,
        origin="lower",
    )
    ax.set_ylim(ylim)


def build_aquisition(name: str, dim_obs: int, data: Data, alpha: Optional[float] = None) -> BaseAquisition:
    if name == "SafeOpt":
        return GrowingSafeOpt(**CONFIG["SafeOpt"], dim_obs=dim_obs)
    elif name == "SafeUCB":
        return GrowingSafeUCB(**CONFIG["SafeUCB"], dim_obs=dim_obs)
    elif name == "CumSafeOpt":
        kwargs = dict(CONFIG["CumSafeOpt"])
        if alpha is not None:
            kwargs["alpha"] = alpha
        return InstrumentedCumSafeOpt(**kwargs, dim_obs=dim_obs)
    elif name == "GoSafeOpt":
        return GrowingGoSafeOpt(**CONFIG["GoSafeOpt"], dim_obs=dim_obs, data=data)
    elif name == "Goose":
        return GrowingGoose(**CONFIG["Goose"], dim_obs=dim_obs)
    else:
        raise ValueError(f"Unknown aquisition {name}")


def run(
    name: str,
    seed: int,
    n_opt_samples: int,
    alpha: Optional[float] = None,
    aquisition_override: Optional[BaseAquisition] = None,
) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    reset_global_state()

    data = Data()
    x_safe = torch.tensor([[SEED[0], SEED[1]]])

    environment = MarsEnv(render_mode=None)
    experiment = Experiment(CONFIG, environment, data=data, backup=None)

    trainer = Trainer(
        dim_params=CONFIG["dim_params"],
        dim_obs=CONFIG["dim_obs"],
        n_opt_samples=n_opt_samples,
        show_progress=False,
        refit_interval=0,  # kernel is fit once offline (_fit_constraint_kernel), not online
        data=data,
    )

    # aquisition_override lets callers (e.g. gate_activity) swap in an
    # instrumented subclass while reusing this function's model/optimizer/
    # set_size-override setup as-is.
    aquisition = aquisition_override if aquisition_override is not None else build_aquisition(name, CONFIG["dim_obs"], data, alpha=alpha)

    model = ModelGenerator(
        **CONFIG["model"],
        domain_start=Tensor(CONFIG["domain_start"]),
        domain_end=Tensor(CONFIG["domain_end"]),
        dim_obs=CONFIG["dim_obs"],
        dim_model=CONFIG["dim_model"],
    )

    optimization_config = dict(CONFIG["Optimization"])
    if name in CONFIG["SET_SIZE_OVERRIDES"]:
        optimization_config["set_size"] = CONFIG["SET_SIZE_OVERRIDES"][name]

    optimizer = GridOpt(
        aquisition,
        **optimization_config,
        domain_start=Tensor(CONFIG["domain_start"]),
        domain_end=Tensor(CONFIG["domain_end"]),
        dim_params=CONFIG["dim_params"],
        dim_context=CONFIG["dim_context"],
        data=data,
        context=None,
    )

    Logger.info(f"=== Running {name} ===")
    trainer.train(experiment, model, optimizer, aquisition, x_safe)
    return data, aquisition


def true_optimum(resolution: int = 400) -> float:
    xs = np.linspace(*DOMAIN_X, resolution)
    ys = np.linspace(*DOMAIN_Y, resolution)
    X, Y = np.meshgrid(xs, ys)
    return float(reward_fn(X, Y).max())


def plot_mars(results: dict, out_path: str):
    _apply_theme()

    names = list(results.keys())
    style = {n: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, n in enumerate(names)}
    j_star = true_optimum()

    fig, ((ax_terrain, ax_trace), (ax_regret, ax_threshold)) = plt.subplots(2, 2)

    # --- terrain (elevation background + unsafe / too-steep region) --------
    xs = np.linspace(*DOMAIN_X, _rows * 2)
    ys = np.linspace(*DOMAIN_Y, _cols * 2)
    X, Y = np.meshgrid(xs, ys)
    Z = elevation(X, Y)
    C = constraint_fn(X, Y)

    for ax in (ax_terrain, ax_trace):
        ax.contourf(X, Y, Z, levels=20, cmap="Greys", alpha=0.5)
        ax.contourf(X, Y, C, levels=[-100.0, 0.0], colors=["#CC503E"], alpha=0.35)
        ax.set_xlim(*DOMAIN_X)
        ax.set_ylim(*DOMAIN_Y)
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")

    ax_terrain.scatter(*SEED, marker="*", s=90, color="black", zorder=5, label="landing site")
    ax_terrain.scatter(*TARGET, marker="P", s=90, color="black", zorder=5, label="target outcrop")
    ax_terrain.set_title("Real Mars terrain: elevation + slope $>20°$ (unsafe)")
    ax_terrain.legend(loc="upper left", fontsize=6, frameon=False)

    # --- trace (sampled points overlaid on the same terrain) ---------------
    for n in names:
        data, _aq = results[n]
        xs_n = data.train_x[:, 0].numpy()
        ys_n = data.train_x[:, 1].numpy()
        color, marker = style[n]
        ax_trace.scatter(
            xs_n, ys_n, s=10, color=color, marker=marker, label=n, alpha=0.85, linewidths=0.3, edgecolors="white"
        )
    ax_trace.set_title("Points evaluated (all rounds)")
    _legend(ax_trace, names, style, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=6)

    # --- cumulative regret ---------------------------------------------------
    regret_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "round": np.arange(data.train_y.shape[0]),
                    "cumulative_regret": np.cumsum(j_star - data.train_y[:, 0].numpy()),
                    "name": n,
                }
            )
            for n, (data, _aq) in results.items()
        ],
        ignore_index=True,
    )
    _plot_series(ax_regret, regret_df, "round", "cumulative_regret", names, style)
    ax_regret.set_xlabel("round")
    ax_regret.set_ylabel(r"cumulative regret $R_N$")
    ax_regret.set_title(rf"$R_N = \sum_t (J^\star - f(x_t))$, $J^\star = {j_star:.3f}$")
    _legend(ax_regret, names, style)

    # --- gate threshold (eta_t) -------------------------------------------
    threshold_frames = []
    for name, (data, aquisition) in results.items():
        history = getattr(aquisition, "threshold_history", None)
        if not history:
            continue
        rounds_h, _tau_h, eta_h = zip(*history)
        threshold_frames.append(pd.DataFrame({"round": rounds_h, "value": eta_h, "name": name}))

    if threshold_frames:
        threshold_df = pd.concat(threshold_frames, ignore_index=True)
        gate_names = [n for n in names if n in threshold_df["name"].unique()]
        _plot_series(ax_threshold, threshold_df, "round", "value", gate_names, style, pointsize=3.5)
        _legend(ax_threshold, gate_names, style)

    ax_threshold.set_xlabel("round")
    ax_threshold.set_ylabel(r"$\eta_t$")
    ax_threshold.set_title(r"$\eta_t = 2\varepsilon\,\beta_t^{1/2-\alpha}$")

    for ax in fig.get_axes():
        _style_axis(ax)

    fig.savefig(out_path)
    pdf_path = str(Path(out_path).with_suffix(".pdf"))
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved Mars benchmark plot to {out_path} (and {pdf_path})")


def _print_summary(name: str, data, j_star: float):
    xs = data.train_x[:, 0].numpy()
    ys = data.train_x[:, 1].numpy()
    reward = data.train_y[:, 0].numpy()
    dist_to_target = np.sqrt((xs - TARGET[0]) ** 2 + (ys - TARGET[1]) ** 2)
    reached = (dist_to_target < 10.0).any()
    cumulative_regret = (j_star - reward).sum()
    print(
        f"{name}: reached target vicinity: {reached}, closest approach: {dist_to_target.min():.1f}m, "
        f"best reward found: {reward.max():.3f}, cumulative regret after {len(reward)} rounds: {cumulative_regret:.2f}"
    )


@app.command()
def mars(
    n_opt_samples: int = typer.Option(350, help="Number of BO rounds for every run"),
    seed: int = typer.Option(42, help="RNG seed shared by every run"),
    algorithms: List[str] = typer.Option(
        ["SafeOpt", "SafeUCB", "CumSafeOpt", "GoSafeOpt", "Goose"], help="Which acquisitions to run"
        # ["CumSafeOpt"], help="Which acquisitions to run"
    ),
    out: str = f"{Path().absolute()}/examples/mars.png",
):
    Logger.set_verbosity(2)
    j_star = true_optimum()
    print(f"true optimum J*={j_star:.3f}, safety threshold tan(20deg)={SLOPE_LIMIT:.3f}")

    results = {}
    for name in algorithms:
        data, aquisition = run(name, seed, n_opt_samples)
        results[name] = (data, aquisition)
        _print_summary(name, data, j_star)

    plot_mars(results, out)


@app.command()
def gate_activity(
    n_opt_samples: int = typer.Option(350, help="Number of BO rounds"),
    seed: int = typer.Option(42, help="RNG seed"),
    out: str = f"{Path().absolute()}/examples/mars_gate_activity.png",
):
    """Run CumSafeOpt alone and mark whether each round's chosen point came from the gate or plain UCB.

    Separate from the main `mars` comparison, matching bottleneck_demo.py's
    alpha_ablation. Some candidate having std above tau_t (eq. 29) doesn't
    guarantee the *winning* point actually needed the gate -- see
    GateTrackingCumSafeOpt's docstring. This command checks the argmax
    directly every round and shades the cumulative regret and eta_t plots by
    normalized_gate at the chosen point: 0 (no shading) when plain UCB
    determined the choice, darker green the more the winning point's own
    choice was driven by the gate.
    """
    Logger.set_verbosity(2)
    j_star = true_optimum()

    aquisition = GateTrackingCumSafeOpt(**CONFIG["CumSafeOpt"], dim_obs=CONFIG["dim_obs"])
    data, aquisition = run("CumSafeOpt", seed, n_opt_samples, aquisition_override=aquisition)

    gate_history = aquisition.gate_history
    n_gate_used = sum(1 for _, strength in gate_history if strength > 0.0)
    spans = _gate_spans(gate_history)
    n_closures = sum(1 for used, _, _ in spans if not used)
    n_used_spans = sum(1 for used, _, _ in spans if used)
    # The very first "gate used" span is an initial opening, not a reopening --
    # regardless of whether it happened to start that way, only the 2nd, 3rd,
    # ... such spans represent the gate losing and then winning the argmax again.
    n_reopenings = max(n_used_spans - 1, 0)
    strengths = [s for _, s in gate_history]
    print(f"gate determined the chosen point in {n_gate_used}/{len(gate_history)} rounds, across {len(spans)} spans")
    print(f"plain UCB determined the choice {n_closures} time(s); gate regained the argmax {n_reopenings} time(s)")
    print(f"normalized_gate at chosen point: min={min(strengths):.3f} max={max(strengths):.3f}")
    for used, start, end in spans:
        print(f"  {'GATE  ' if used else 'UCB   '} rounds {start}-{end} ({end - start + 1} rounds)")

    _apply_theme()
    # _apply_theme sizes figures for bottleneck_demo.py's 2x2 grids; this is a
    # 1x2 row, so it needs its own (wider, shorter) size rather than inheriting
    # a height-to-width ratio meant for a 4-panel figure.
    plt.rcParams.update(figsizes.iclr2023(nrows=1, ncols=2))
    name = "CumSafeOpt"
    style = {name: (PALETTE[0], MARKERS[0])}

    fig, (ax_regret, ax_threshold) = plt.subplots(1, 2)

    regret_df = pd.DataFrame(
        {
            "round": np.arange(data.train_y.shape[0]),
            "cumulative_regret": np.cumsum(j_star - data.train_y[:, 0].numpy()),
            "name": name,
        }
    )
    _plot_series(ax_regret, regret_df, "round", "cumulative_regret", [name], style)
    _shade_gate_intensity(ax_regret, gate_history)
    ax_regret.set_xlabel("round")
    ax_regret.set_ylabel(r"cumulative regret $R_N$")
    ax_regret.set_title(rf"$R_N$, $J^\star={j_star:.3f}$" + "\n(darker green = chosen point gate-driven)")

    rounds_h, _tau_h, eta_h = zip(*aquisition.threshold_history)
    threshold_df = pd.DataFrame({"round": rounds_h, "value": eta_h, "name": name})
    _plot_series(ax_threshold, threshold_df, "round", "value", [name], style, pointsize=3.5)
    _shade_gate_intensity(ax_threshold, gate_history)
    ax_threshold.set_xlabel("round")
    ax_threshold.set_ylabel(r"$\eta_t$")
    ax_threshold.set_title(
        r"$\eta_t = 2\varepsilon\,\beta_t^{1/2-\alpha}$" + "\n(darker green = chosen point gate-driven)"
    )

    for ax in fig.get_axes():
        _style_axis(ax)

    fig.savefig(out)
    pdf_path = str(Path(out).with_suffix(".pdf"))
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved gate activity plot to {out} (and {pdf_path})")


if __name__ == "__main__":
    app()
