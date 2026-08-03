"""A 2D Mars-terrain safe-exploration benchmark, in the spirit of the GOOSE
paper's (Turchetta, Berkenkamp & Krause, NeurIPS 2019) Mars rover experiment:
a rover lands in a flat, safe basin and must reach scientifically interesting
terrain across a mountain ridge whose slope is only traversable through one
narrow, gentle pass.

Adapted to this repo's fully-connected BO setting (see goose.py's docstring)
rather than the paper's literal grid-graph traversal: the domain is a
continuous 2D patch of terrain, the safety constraint is the local slope
(steepness) of a synthetic elevation field, and the reward is a separate
"scientific interest" field, independent of elevation -- matching the paper's
framing that a rover's objective isn't simply "go uphill". Reuses the
paper-style plotting and Chowdhury-Gopalan infrastructure from
bottleneck_demo.py rather than duplicating it.
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

DOMAIN_X = (1.0, 11.0)
DOMAIN_Y = (1.0, 7.0)
SEED = (1.5, 4.0)  # landing site: flat, safe, scientifically mediocre
TARGET = (10.0, 4.0)  # distant outcrop: the real payoff

RIDGE_X = 6.0
PASS_Y = 4.0
RIDGE_WIDTH_BASE = 0.5  # steep ridge (narrow Gaussian -> high slope) away from the pass
RIDGE_WIDTH_PASS = 1.8  # gentle ridge (wide Gaussian -> low slope) right at the pass
PASS_SPAN = 0.5  # how localized (in y) the gentle pass is
# Kept deliberately small (O(1) output scale, matching bottleneck_demo.py /
# double_bottleneck_demo.py) so epsilon/lipschitz calibrated for those
# benchmarks stay meaningful here too.
RIDGE_AMPLITUDE = 1.0
SLOPE_LIMIT = 0.65  # rover's max traversable slope


def ridge_width(y: np.ndarray) -> np.ndarray:
    return RIDGE_WIDTH_BASE + (RIDGE_WIDTH_PASS - RIDGE_WIDTH_BASE) * np.exp(
        -((y - PASS_Y) ** 2) / (2 * PASS_SPAN**2)
    )


def elevation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = ridge_width(y)
    return RIDGE_AMPLITUDE * np.exp(-((x - RIDGE_X) ** 2) / (2 * w**2))


def slope(x: np.ndarray, y: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    # Numerical gradient magnitude of the elevation field -- the terrain's
    # steepness, and hence the rover's actual safety constraint.
    dhdx = (elevation(x + eps, y) - elevation(x - eps, y)) / (2 * eps)
    dhdy = (elevation(x, y + eps) - elevation(x, y - eps)) / (2 * eps)
    return np.sqrt(dhdx**2 + dhdy**2)


def constraint_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return SLOPE_LIMIT - slope(x, y)


def reward_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # "Scientific interest": independent of elevation, so climbing isn't
    # itself rewarding -- only reaching the target outcrop is.
    return (
        1.0 * np.exp(-(((x - SEED[0]) ** 2 + (y - SEED[1]) ** 2)) / (2 * 0.7**2))
        + 3.5 * np.exp(-(((x - TARGET[0]) ** 2 + (y - TARGET[1]) ** 2)) / (2 * 1.3**2))
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
        reward = np.array([2 * reward_fn(x, y), 2 * constraint_fn(x, y)])
        return np.array([x, y]), reward, True, False, {}


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
        # Normalize() maps each axis independently onto [0, 1], so a lengthscale
        # here of l corresponds to l * domain_width raw units on that axis. The
        # narrowest true feature (the ridge flank / pass transition) is ~0.5-0.55
        # raw units wide, so pick lengthscales well below that once rescaled:
        # 0.035 * 12 = 0.42 raw x-units, 0.05 * 8 = 0.4 raw y-units.
        "lenghtscale": [0.42, 0.4],
        "normalize_input": False,
        "normalize_output": True,
        "likelihood_noise": 1e-4,
    },
    "Optimization": {
        "set_size": 6000,
        "set_init": "random",
        "max_global_steps_without_progress_tolerance": 0.9,
        "max_global_steps_without_progress": 10_000,  # effectively disabled
    },
    "SafeOpt": {"scale_beta": 1.0, "beta": 9},
    "SafeUCB": {"scale_beta": 1.0, "beta": 9},
    "CumSafeOpt": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.1, "alpha": 1, "zeta": 0.1},
    "GoSafeOpt": {"scale_beta": 1.0, "beta": 9, "n_max_local": 5, "n_max_global": 3},
    "Goose": {"scale_beta": 1.0, "beta": 9, "lipschitz": 2.0, "epsilon": 0.1},
}


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


def run(name: str, seed: int, n_opt_samples: int, alpha: Optional[float] = None) -> tuple:
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
        refit_interval=0,
        data=data,
    )

    aquisition = build_aquisition(name, CONFIG["dim_obs"], data, alpha=alpha)

    model = ModelGenerator(
        **CONFIG["model"],
        domain_start=Tensor(CONFIG["domain_start"]),
        domain_end=Tensor(CONFIG["domain_end"]),
        dim_obs=CONFIG["dim_obs"],
        dim_model=CONFIG["dim_model"],
    )

    optimizer = GridOpt(
        aquisition,
        **CONFIG["Optimization"],
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


def pass_margin(resolution: int = 4000) -> float:
    """m* of the ridge pass: the worst-case constraint value while crossing
    the ridge in x along the safest row (y = PASS_Y) -- the margin any
    algorithm must certify safe before it can reach the target side.
    """
    xs = np.linspace(RIDGE_X - 3, RIDGE_X + 3, resolution)
    ys = np.full_like(xs, PASS_Y)
    return float(constraint_fn(xs, ys).min())


def plot_mars(results: dict, out_path: str):
    _apply_theme()

    names = list(results.keys())
    style = {n: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, n in enumerate(names)}
    j_star = true_optimum()

    fig, ((ax_terrain, ax_trace), (ax_regret, ax_threshold)) = plt.subplots(2, 2)

    # --- terrain (elevation background + unsafe / too-steep region) --------
    xs = np.linspace(*DOMAIN_X, 300)
    ys = np.linspace(*DOMAIN_Y, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = elevation(X, Y)
    C = constraint_fn(X, Y)

    for ax in (ax_terrain, ax_trace):
        ax.contourf(X, Y, Z, levels=20, cmap="Greys", alpha=0.5)
        ax.contourf(X, Y, C, levels=[-100.0, 0.0], colors=["#CC503E"], alpha=0.35)
        ax.set_xlim(*DOMAIN_X)
        ax.set_ylim(*DOMAIN_Y)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

    ax_terrain.scatter(*SEED, marker="*", s=90, color="black", zorder=5, label="landing site")
    ax_terrain.scatter(*TARGET, marker="P", s=90, color="black", zorder=5, label="target outcrop")
    ax_terrain.set_title("Terrain: elevation + unsafe (too-steep) region")
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
    _legend(ax_trace, names, style, loc="upper left", fontsize=6)

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
    reward = data.train_y[:, 0].numpy()
    crossed = (xs > RIDGE_X + 1.0).any()  # meaningfully past the ridge, on the target side
    cumulative_regret = (j_star - reward).sum()
    print(
        f"{name}: crossed ridge: {crossed}, best reward found: {reward.max():.3f}, "
        f"cumulative regret after {len(reward)} rounds: {cumulative_regret:.2f}"
    )


@app.command()
def mars(
    n_opt_samples: int = typer.Option(120, help="Number of BO rounds for every run"),
    seed: int = typer.Option(42, help="RNG seed shared by every run"),
    algorithms: List[str] = typer.Option(
        ["SafeOpt", "SafeUCB", "CumSafeOpt", "GoSafeOpt", "Goose"], help="Which acquisitions to run"
    ),
    out: str = f"{Path().absolute()}/examples/mars.png",
):
    Logger.set_verbosity(2)
    j_star = true_optimum()
    m_star = pass_margin()
    print(f"true optimum J*={j_star:.3f}, ridge-pass margin m*={m_star:.3f}")

    results = {}
    for name in algorithms:
        data, aquisition = run(name, seed, n_opt_samples)
        results[name] = (data, aquisition)
        _print_summary(name, data, j_star)

    plot_mars(results, out)


if __name__ == "__main__":
    app()
