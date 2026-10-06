import json
import math
import random
import secrets
from collections import deque
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
from mars_column import Terrain, plot_mars_column, plot_mars_column_stats
from matplotlib.animation import FuncAnimation, PillowWriter
from botorch.models.gp_regression import SingleTaskGP
from tueplots import figsizes
from bottleneck_demo import (
    MARKERS,
    PALETTE,
    GrowingGoose,
    GrowingISEBO,
    GrowingGoSafeOpt,
    GrowingSafeOpt,
    GrowingSafeUCB,
    StageOpt,
    InstrumentedCSafeOpt,
    InstrumentedCSafeOptSimple,
    _apply_theme,
    _legend,
    _legend_order,
    _plot_series,
    _style_axis,
    reset_global_state,
)
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.interpolate import RectBivariateSpline
from torch import Tensor

import csafeopt
from csafeopt.aquisitions.base_aquisition import BaseAquisition
from csafeopt.experiments.environment import Environment
from csafeopt.experiments.experiment import Experiment
from csafeopt.models.model import ModelGenerator
from csafeopt.optim.grid_opt import GridOpt
from csafeopt.tools.data import Data
from csafeopt.tools.logger import Logger
from csafeopt.trainer import Trainer

csafeopt.device = torch.device("cpu")


app = typer.Typer()

DATA_PATH = Path(__file__).parent / "data" / "mars_dtm.npy"
PIXEL_STEP = 1.01
_dtm = np.load(DATA_PATH)
_rows, _cols = _dtm.shape
_xs_grid = np.arange(_rows) * PIXEL_STEP
_ys_grid = np.arange(_cols) * PIXEL_STEP
_elevation_spline = RectBivariateSpline(_xs_grid, _ys_grid, _dtm)

DOMAIN_X = (35.0 * PIXEL_STEP, 67.0 * PIXEL_STEP)
DOMAIN_Y = (35.0 * PIXEL_STEP, 67.0 * PIXEL_STEP)
SEED = (40 * PIXEL_STEP, 40 * PIXEL_STEP)
TARGET = (65 * PIXEL_STEP, 60 * PIXEL_STEP)

SLOPE_LIMIT = float(np.tan(np.deg2rad(27.0)))


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
    return (
        1.0 * np.exp(-(((x - SEED[0]) ** 2 + (y - SEED[1]) ** 2)) / (2 * 8.0**2))
        + 3.5 * np.exp(-(((x - TARGET[0]) ** 2 + (y - TARGET[1]) ** 2)) / (2 * 10.0**2))
        - 0.3
    )


class HistoryAugmentedGridOpt(GridOpt):

    def __init__(
        self,
        *args,
        local_jitter_std: float = 0.5,
        local_jitter_per_point: int = 40,
        local_jitter_anchors: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.local_jitter_std = local_jitter_std
        self.local_jitter_per_point = local_jitter_per_point
        self.local_jitter_anchors = local_jitter_anchors

    def optimize(self, step: int = 0):
        x = self.get_initial_params(self.set_init)
        observed = self.data.train_x
        if observed is not None and observed.shape[0] > 0:
            observed = observed.to(dtype=x.dtype, device=x.device)
            anchors = observed[-self.local_jitter_anchors :]
            jitter = anchors.repeat_interleave(self.local_jitter_per_point, dim=0)
            jitter = jitter + torch.randn_like(jitter) * self.local_jitter_std
            jitter = torch.clamp(jitter, min=self.domain_start, max=self.domain_end)
            x = torch.vstack([x, observed, jitter])
        loss = self.aquisition.evaluate(x, step)
        return [x, loss]


class MarsEnv(Environment):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(None, render_mode)

    def reset(self, *, seed=None, options=None):
        return np.zeros(2), {}

    def step(self, k):
        x, y = float(k[0]), float(k[1])
        reward = np.array([2 * float(reward_fn(x, y)), 2 * float(constraint_fn(x, y))])
        return np.array([x, y]), reward, True, False, {}


def _fit_kernel(value_fn, name: str, n_samples: int = 500, noise_std: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(DOMAIN_X[0], DOMAIN_X[1], n_samples)
    ys = rng.uniform(DOMAIN_Y[0], DOMAIN_Y[1], n_samples)
    values = value_fn(xs, ys) + rng.normal(0.0, noise_std, n_samples)

    width_x, width_y = DOMAIN_X[1] - DOMAIN_X[0], DOMAIN_Y[1] - DOMAIN_Y[0]
    xs_norm = (xs - DOMAIN_X[0]) / width_x
    ys_norm = (ys - DOMAIN_Y[0]) / width_y

    train_x = torch.tensor(np.stack([xs_norm, ys_norm], axis=1))
    train_y = torch.tensor(values)

    covar_module = ScaleKernel(MaternKernel(ard_num_dims=2))
    likelihood = GaussianLikelihood()
    likelihood.noise = noise_std**2
    model = SingleTaskGP(train_x, train_y.unsqueeze(-1), likelihood=likelihood, covar_module=covar_module)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

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
        f"Fitted {name} kernel from {n_samples} dense samples: "
        f"lengthscale(normalized)={lengthscale_norm} (raw meters={lengthscale_raw}) noise={fitted_noise:.2e}"
    )
    return lengthscale_norm, fitted_noise


_FITTED_CONSTRAINT_LENGTHSCALE, _FITTED_CONSTRAINT_NOISE = _fit_kernel(constraint_fn, "constraint")
_FITTED_REWARD_LENGTHSCALE, _FITTED_REWARD_NOISE = _fit_kernel(reward_fn, "reward")

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
        "lenghtscale": _FITTED_REWARD_LENGTHSCALE,
        "constraint_lengthscale": _FITTED_CONSTRAINT_LENGTHSCALE,
        "normalize_input": True,
        "normalize_output": True,
        "likelihood_noise": 1e-4,
    },
    "Optimization": {
        "set_size": 8000,
        "set_init": "random",
        "max_global_steps_without_progress_tolerance": 0.9,
        "max_global_steps_without_progress": 10_000,
    },
    "SET_SIZE_OVERRIDES": {
        "SafeOpt": 24000, "StageOpt": 24000,
        "SafeUCB": 24000,
        "CSafeOpt": 24000,
        "CSafeOptSimple": 24000,
        "GoSafeOpt": 24000,
    },
    "SafeOpt": {"scale_beta": 1.0, "beta": 9},
    "SafeUCB": {"scale_beta": 1.0, "beta": 9},
    "CSafeOpt": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.02, "alpha": 1.75, "zeta": 0.01},
    "CSafeOptSimple": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.01, "alpha": 0.75, "zeta": 0.01},
    "GoSafeOpt":{"scale_beta": 1.0, "beta": 9, "n_max_local": 5, "n_max_global": 3},
    "GoOSE": {"scale_beta": 1.0, "beta": 9, "lipschitz": 1.0, "epsilon": 0.01},
    "StageOpt": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.01},
    "ISE-BO": {"scale_beta": 1.0, "beta": 9},
}


class GateTrackingCSafeOpt(InstrumentedCSafeOpt):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_history: list[tuple[int, float, float]] = []

    def evaluate(self, x: Tensor, step: int = 0) -> Tensor:
        scores = super().evaluate(x, step)
        self.gate_history.append((self.t, self.last_gate_max, self.last_gate_at_chosen))
        return scores


def _gate_spans(gate_history: list) -> list:
    spans = []
    state, start, prev = None, None, None
    for t, _gate_max, gate_at_chosen in gate_history:
        used = gate_at_chosen > 0.0
        if used != state:
            if state is not None:
                spans.append((state, start, prev))
            state, start = used, t
        prev = t
    if state is not None:
        spans.append((state, start, prev))
    return spans


def _shade_gate_intensity(ax, gate_history: list) -> None:
    ts = np.array([t for t, _gate_max, _gate_at_chosen in gate_history], dtype=float)
    intensity = np.array([s for _t, _gate_max, s in gate_history], dtype=float)

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


def build_aquisition(name: str, dim_obs: int, data: Data, alpha: Optional[float] = None,
                     epsilon: Optional[float] = None) -> BaseAquisition:
    if name == "SafeOpt":
        return GrowingSafeOpt(**CONFIG["SafeOpt"], dim_obs=dim_obs)
    elif name == "SafeUCB":
        return GrowingSafeUCB(**CONFIG["SafeUCB"], dim_obs=dim_obs)
    elif name == "CSafeOpt":
        kwargs = dict(CONFIG["CSafeOpt"])
        if alpha is not None:
            kwargs["alpha"] = alpha
        if epsilon is not None:
            kwargs["epsilon"] = epsilon
        return InstrumentedCSafeOpt(**kwargs, dim_obs=dim_obs)
    elif name == "CSafeOptSimple":
        kwargs = dict(CONFIG["CSafeOptSimple"])
        if alpha is not None:
            kwargs["alpha"] = alpha
        return InstrumentedCSafeOptSimple(**kwargs, dim_obs=dim_obs)
    elif name == "GoSafeOpt":
        return GrowingGoSafeOpt(**CONFIG["GoSafeOpt"], dim_obs=dim_obs, data=data)
    elif name == "GoOSE":
        return GrowingGoose(**CONFIG["GoOSE"], dim_obs=dim_obs)
    elif name == "StageOpt":
        kwargs = dict(CONFIG["StageOpt"])
        eta = kwargs.pop("epsilon")
        return StageOpt(**kwargs, expansion_eta=eta, dim_obs=dim_obs)
    elif name == "ISE-BO":
        return GrowingISEBO(**CONFIG["ISE-BO"], dim_obs=dim_obs, data=data)
    else:
        raise ValueError(f"Unknown aquisition {name}")


def run(
    name: str,
    seed: int,
    n_opt_samples: int,
    alpha: Optional[float] = None,
    aquisition_override: Optional[BaseAquisition] = None,
    initial_point: Optional[tuple] = None,
    epsilon: Optional[float] = None,
) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    reset_global_state()

    data = Data()
    landing = np.asarray(SEED if initial_point is None else initial_point, dtype=float)
    if (
        landing.shape != (2,) or not np.isfinite(landing).all()
        or not (DOMAIN_X[0] <= landing[0] <= DOMAIN_X[1] and DOMAIN_Y[0] <= landing[1] <= DOMAIN_Y[1])
        or float(constraint_fn(*landing)) < 0
    ):
        raise ValueError("Initial landing point must be finite, inside the domain, and safe")
    x_safe = torch.tensor(landing.reshape(1, 2))

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

    aquisition = aquisition_override if aquisition_override is not None else build_aquisition(name, CONFIG["dim_obs"], data, alpha=alpha, epsilon=epsilon)

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

    optimizer = HistoryAugmentedGridOpt(
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


_MARS_Z_ORDER = ["SafeOpt", "CSafeOpt", "GoOSE", "SafeUCB"]
_REGRET_ONLY = ("GoOSE", "ISE-BO")


def _z_order(names: list) -> list:
    return sorted(names, key=lambda n: _MARS_Z_ORDER.index(n) if n in _MARS_Z_ORDER else len(_MARS_Z_ORDER))


def plot_mars(results: dict, out_path: str, one_column: bool = False, map_layout: str = "grid"):
    _apply_theme()

    names = _legend_order(list(results.keys()))
    style = {n: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, n in enumerate(names)}
    j_star = true_optimum()

    if one_column:
        terrain = Terrain(
            DOMAIN_X, DOMAIN_Y, elevation, constraint_fn, SEED, TARGET,
            float(np.rad2deg(np.arctan(SLOPE_LIMIT))), j_star,
        )
        map_names = [n for n in names if n not in _REGRET_ONLY]
        plot_mars_column(results, out_path, terrain, _z_order(names), names, style, layout=map_layout, map_names=map_names)
        return

    fig, ((ax_terrain, ax_trace), (ax_regret, ax_threshold)) = plt.subplots(2, 2)

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

    ax_terrain.scatter(*SEED, marker=".", s=90, color="black", zorder=5, label="landing site")
    ax_terrain.scatter(*TARGET, marker="*", s=90, color="black", zorder=5, label="target outcrop")
    slope_limit_deg = np.rad2deg(np.arctan(SLOPE_LIMIT))
    ax_terrain.set_title(rf"Real Mars terrain: elevation + slope $>{slope_limit_deg:.0f}°$ (unsafe)")
    ax_terrain.legend(loc="upper left", fontsize=6, frameon=False)

    for n in _z_order(names):
        data, _aq = results[n]
        xs_n = data.train_x[:, 0].numpy()
        ys_n = data.train_x[:, 1].numpy()
        color, marker = style[n]
        ax_trace.scatter(
            xs_n, ys_n, s=10, color=color, marker=marker, label=n, alpha=0.85, linewidths=0.3, edgecolors="white"
        )
    ax_trace.set_title("Points evaluated (all rounds)")
    _legend(ax_trace, names, style, loc="upper left", ncol=1, fontsize=6)

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
    _plot_series(ax_regret, regret_df, "round", "cumulative_regret", _z_order(names), style)
    ax_regret.set_xlabel("round")
    ax_regret.set_ylabel(r"cumulative regret $R_N$")
    ax_regret.set_title(rf"$R_N = \sum_t (J^\star - f(x_t))$, $J^\star = {j_star:.3f}$")
    _legend(ax_regret, names, style)

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
        _plot_series(ax_threshold, threshold_df, "round", "value", _z_order(gate_names), style, pointsize=3.5)
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


def animate_mars(results: dict, out_path: str, n_frames: int = 60, fps: int = 8, dpi: int = 200):
    _apply_theme()

    names = _legend_order(list(results.keys()))
    style = {n: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, n in enumerate(names)}
    j_star = true_optimum()

    fig, ((ax_terrain, ax_trace), (ax_regret, ax_threshold)) = plt.subplots(2, 2)

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
    slope_limit_deg = np.rad2deg(np.arctan(SLOPE_LIMIT))
    ax_terrain.set_title(rf"Real Mars terrain: elevation + slope $>{slope_limit_deg:.0f}°$ (unsafe)")
    ax_terrain.legend(loc="upper left", fontsize=6, frameon=False)

    ax_trace.set_title("Points evaluated")
    _legend(ax_trace, names, style, loc="upper left", ncol=1, fontsize=6)

    regret_curves = {n: np.cumsum(j_star - data.train_y[:, 0].numpy()) for n, (data, _aq) in results.items()}

    threshold_curves = {}
    for n, (_data, aquisition) in results.items():
        history = getattr(aquisition, "threshold_history", None)
        if history:
            rounds_h, _tau_h, eta_h = zip(*history)
            threshold_curves[n] = (np.array(rounds_h), np.array(eta_h))

    max_round = max(data.train_x.shape[0] for data, _aq in results.values())
    frame_rounds = sorted(set(np.linspace(1, max_round, min(n_frames, max_round)).astype(int)))

    trace_artists = {
        n: ax_trace.scatter([], [], s=10, color=style[n][0], marker=style[n][1], alpha=0.85, linewidths=0.3, edgecolors="white")
        for n in _z_order(names)
    }
    regret_lines = {n: ax_regret.plot([], [], color=style[n][0], linewidth=1.8)[0] for n in _z_order(names)}
    threshold_lines = {
        n: ax_threshold.plot([], [], color=style[n][0], linewidth=1.8)[0]
        for n in _z_order([n for n in names if n in threshold_curves])
    }

    ax_regret.set_xlim(0, max_round)
    ax_regret.set_ylim(0, max(curve.max() for curve in regret_curves.values()) * 1.05)
    ax_regret.set_xlabel("round")
    ax_regret.set_ylabel(r"cumulative regret $R_N$")
    ax_regret.set_title(rf"$R_N = \sum_t (J^\star - f(x_t))$, $J^\star = {j_star:.3f}$")
    _legend(ax_regret, names, style)

    ax_threshold.set_xlabel("round")
    ax_threshold.set_ylabel(r"$\eta_t$")
    ax_threshold.set_title(r"$\eta_t = 2\varepsilon\,\beta_t^{1/2-\alpha}$")
    if threshold_curves:
        all_rounds = np.concatenate([r for r, _e in threshold_curves.values()])
        all_eta = np.concatenate([e for _r, e in threshold_curves.values()])
        ax_threshold.set_xlim(all_rounds.min(), max_round)
        pad = 0.05 * (all_eta.max() - all_eta.min() + 1e-12)
        ax_threshold.set_ylim(all_eta.min() - pad, all_eta.max() + pad)
        _legend(ax_threshold, _legend_order([n for n in names if n in threshold_curves]), style)

    for ax in fig.get_axes():
        _style_axis(ax)

    round_text = fig.suptitle("")

    def update(frame_idx):
        r = frame_rounds[frame_idx]
        artists = []
        for n in names:
            data, _aq = results[n]
            k = min(r, data.train_x.shape[0])
            trace_artists[n].set_offsets(data.train_x[:k, :2].numpy())
            artists.append(trace_artists[n])

            k_r = min(r, len(regret_curves[n]))
            regret_lines[n].set_data(np.arange(k_r), regret_curves[n][:k_r])
            artists.append(regret_lines[n])

            if n in threshold_curves:
                rounds_h, eta_h = threshold_curves[n]
                mask = rounds_h <= r
                threshold_lines[n].set_data(rounds_h[mask], eta_h[mask])
                artists.append(threshold_lines[n])

        round_text.set_text(f"round {r}/{max_round}")
        artists.append(round_text)
        return artists

    ani = FuncAnimation(fig, update, frames=len(frame_rounds), blit=False)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved Mars animation to {out_path}")


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
    n_opt_samples: int = typer.Option(400, help="Number of BO rounds for every run"),
    seed: int = typer.Option(42, help="RNG seed shared by every run"),
    algorithms: List[str] = typer.Option(
        ["SafeOpt", "SafeUCB", "CSafeOpt", "GoOSE"], help="Which acquisitions to run"
    ),
    out: str = f"{Path().absolute()}/examples/mars.png",
    gif: bool = typer.Option(False, help="Also render an animated GIF showing samples appearing round by round"),
    gif_frames: int = typer.Option(60, help="Number of frames in the GIF (rounds are subsampled to this count)"),
    gif_fps: int = typer.Option(8, help="Playback speed of the GIF"),
    gif_dpi: int = typer.Option(200, help="Resolution of the GIF (figure is a fixed 5.5x3.4in, so this sets pixel size)"),
    one_column: bool = typer.Option(False, help="Draw the compact one-column figure (maps + regret + eta)"),
    map_layout: str = typer.Option("grid", help="With --one-column: 'grid' (one small map per algorithm) or 'single'"),
    reachable_episodes: List[int] = typer.Option(
        [1, 20, 100], help="Episodes at which to plot CSafeOpt's eta_t-reachable safe set (needs CSafeOpt in --algorithms)"
    ),
    epsilon: Optional[float] = typer.Option(None, min=1e-12, help="Override CSafeOpt's epsilon (sets the eta_t scale)"),
    alpha: Optional[float] = typer.Option(None, help="Override CSafeOpt's alpha (sets how fast eta_t shrinks)"),
):
    Logger.set_verbosity(2)
    j_star = true_optimum()
    slope_limit_deg = np.rad2deg(np.arctan(SLOPE_LIMIT))
    print(f"true optimum J*={j_star:.3f}, safety threshold tan({slope_limit_deg:.0f}deg)={SLOPE_LIMIT:.3f}")

    results = {}
    for name in algorithms:
        data, aquisition = run(name, seed, n_opt_samples, alpha=alpha, epsilon=epsilon)
        results[name] = (data, aquisition)
        _print_summary(name, data, j_star)

    plot_mars(results, out, one_column=one_column, map_layout=map_layout)

    if "CSafeOpt" in results and reachable_episodes:
        reach_out = str(Path(out).with_name(Path(out).stem + "_reachable_sets.png"))
        plot_reachable_sets(results["CSafeOpt"][1], reach_out, episodes=tuple(reachable_episodes))

    if gif:
        gif_out = str(Path(out).with_suffix(".gif"))
        animate_mars(results, gif_out, n_frames=gif_frames, fps=gif_fps, dpi=gif_dpi)


def _reachable_mask(margin: float = 0.0, resolution: int = 201, start: Optional[tuple] = None) -> tuple:
    if resolution < 3:
        raise ValueError("Connectivity resolution must be at least 3")
    start = tuple(SEED) if start is None else tuple(start)
    if not (DOMAIN_X[0] <= start[0] <= DOMAIN_X[1] and DOMAIN_Y[0] <= start[1] <= DOMAIN_Y[1]):
        raise ValueError("Reference landing site is outside the domain")
    xs = np.unique(np.r_[np.linspace(*DOMAIN_X, resolution), start[0]])
    ys = np.unique(np.r_[np.linspace(*DOMAIN_Y, resolution), start[1]])
    X, Y = np.meshgrid(xs, ys)
    safe = constraint_fn(X, Y) > margin
    horizontal = safe[:, :-1] & safe[:, 1:]
    vertical = safe[:-1, :] & safe[1:, :]
    for fraction in (0.25, 0.5, 0.75):
        horizontal &= constraint_fn(X[:, :-1] + fraction * np.diff(xs)[None, :], Y[:, :-1]) > margin
        vertical &= constraint_fn(X[:-1, :], Y[:-1, :] + fraction * np.diff(ys)[:, None]) > margin
    start = (int(np.searchsorted(ys, start[1])), int(np.searchsorted(xs, start[0])))
    visited = np.zeros_like(safe)
    if not safe[start]:
        return xs, ys, visited
    visited[start] = True
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        neighbors = []
        if col + 1 < len(xs) and horizontal[row, col]:
            neighbors.append((row, col + 1))
        if col > 0 and horizontal[row, col - 1]:
            neighbors.append((row, col - 1))
        if row + 1 < len(ys) and vertical[row, col]:
            neighbors.append((row + 1, col))
        if row > 0 and vertical[row - 1, col]:
            neighbors.append((row - 1, col))
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    return xs, ys, visited


def connected_safe_component(resolution: int = 201) -> np.ndarray:
    if float(constraint_fn(*SEED)) <= 0:
        raise ValueError("Reference landing site must have a positive safety margin")
    xs, ys, visited = _reachable_mask(0.0, resolution)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack((X[visited], Y[visited]))


def _draw_reachable_panels(panels: list, out_path: str, resolution: int = 201,
                           reach_label: str = r"$\overline{R}_{\eta_t^{(\alpha)}}$") -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mars_column import COLUMN_FIGSIZE

    _apply_theme()
    width = COLUMN_FIGSIZE[0]
    fig, axes = plt.subplots(1, len(panels), sharex=True, sharey=True, squeeze=False,
                             figsize=(width, 1.75), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02)
    axes = axes[0]

    xs_bg = np.linspace(*DOMAIN_X, _rows * 2)
    ys_bg = np.linspace(*DOMAIN_Y, _cols * 2)
    X_bg, Y_bg = np.meshgrid(xs_bg, ys_bg)
    Z = elevation(X_bg, Y_bg)
    C = constraint_fn(X_bg, Y_bg)
    xs0, ys0, full = _reachable_mask(0.0, resolution)

    for ax, (title, eta) in zip(axes, panels):
        xs, ys, reach = _reachable_mask(eta, resolution)
        ax.contourf(X_bg, Y_bg, Z, levels=20, cmap="Greys", alpha=0.5)
        ax.contourf(X_bg, Y_bg, C, levels=[-100.0, 0.0], colors=["#CC503E"], alpha=0.35)
        if reach.any():
            ax.contourf(xs, ys, reach.astype(float), levels=[0.5, 1.5], colors=["#2E9E44"], alpha=0.45)
        ax.contour(xs0, ys0, full.astype(float), levels=[0.5], colors="#1B5E20", linewidths=0.6, linestyles="--")
        ax.scatter(*SEED, marker=".", s=25, color="black", zorder=5)
        ax.scatter(*TARGET, marker="*", s=25, color="black", zorder=5)
        ax.set_xlim(*DOMAIN_X)
        ax.set_ylim(*DOMAIN_Y)
        ax.set_aspect("equal")
        ax.set_xticks([40, 50, 60])
        ax.set_yticks([40, 50, 60])
        coverage = reach.sum() / max(full.sum(), 1)
        ax.set_title(title, fontsize=7, pad=2)
        print(f"{title!r}: eta={eta:.4g}, reachable set covers {100 * coverage:.1f}% of the eta=0 component")
        _style_axis(ax)
        ax.grid(False)
        ax.tick_params(labelsize=6, pad=1)
    axes[0].set_ylabel(r"$y$ [m]", fontsize=7, labelpad=1)
    axes[len(axes) // 2].set_xlabel(r"$x$ [m]", fontsize=7, labelpad=1)
    handles = [
        Patch(facecolor="#2E9E44", alpha=0.45, label=reach_label),
        Line2D([], [], color="#1B5E20", linewidth=0.6, linestyle="--", label=r"$\overline{R}_{0^+}$"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7,
               handlelength=1.5, columnspacing=1.2, borderpad=0.1)

    fig.savefig(out_path)
    pdf_path = str(Path(out_path).with_suffix(".pdf"))
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved reachable-set plot to {out_path} (and {pdf_path})")


def plot_reachable_sets(aquisition, out_path: str, episodes: tuple = (1, 20, 100), resolution: int = 201):
    history = getattr(aquisition, "threshold_history", None)
    if not history:
        raise ValueError("Aquisition has no threshold_history (needs an instrumented CSafeOpt)")
    eta_by_round = {t: eta for t, _tau, eta in history}
    skipped = [e for e in episodes if e not in eta_by_round]
    if skipped:
        print(f"Skipping episodes {skipped}: eta_t only recorded for rounds {min(eta_by_round)}-{max(eta_by_round)}")
    episodes = [e for e in episodes if e in eta_by_round]
    if not episodes:
        raise ValueError(f"None of the requested episodes were run (last round: {max(eta_by_round)})")
    panels = [(rf"$t={e}$" + "\n" + rf"$\eta_t={eta_by_round[e]:.3g}$", eta_by_round[e]) for e in episodes]
    _draw_reachable_panels(panels, out_path, resolution)


@app.command()
def reachable_comparison(
    rounds: int = typer.Option(400, min=2, help="BO rounds of each CSafeOpt run"),
    stuck_x: float = typer.Option(58.0, help="x of the second landing site"),
    stuck_y: float = typer.Option(37.0, help="y of the second landing site"),
    seed: int = typer.Option(42, help="RNG seed of both runs"),
    resolution: int = typer.Option(201, min=3, help="Grid resolution for the flood fill and the GP safe set"),
    out: str = f"{Path().absolute()}/examples/mars_reachable_comparison.png",
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mars_column import COLUMN_FIGSIZE

    Logger.set_verbosity(2)
    stuck = (stuck_x, stuck_y)
    data_nominal, aq_nominal = run("CSafeOpt", seed, rounds)
    _print_summary("usual landing", data_nominal, true_optimum())
    t_last, _tau, eta_last = aq_nominal.threshold_history[-1]

    _apply_theme()
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(COLUMN_FIGSIZE[0], 2.25), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02)
    xs_bg = np.linspace(*DOMAIN_X, _rows * 2)
    ys_bg = np.linspace(*DOMAIN_Y, _cols * 2)
    X_bg, Y_bg = np.meshgrid(xs_bg, ys_bg)
    Z, C = elevation(X_bg, Y_bg), constraint_fn(X_bg, Y_bg)

    reach_color, full_color = "#2E9E44", "#1B5E20"
    panels = [
        (axes[0], tuple(SEED), rf"$t={t_last}$, $\eta_t={eta_last:.3g}$"),
        (axes[1], stuck, rf"landing $({stuck_x:g}, {stuck_y:g})$, $\eta_t={eta_last:.3g}$"),
    ]
    for ax, start, title in panels:
        ax.contourf(X_bg, Y_bg, Z, levels=20, cmap="Greys", alpha=0.5)
        ax.contourf(X_bg, Y_bg, C, levels=[-100.0, 0.0], colors=["#CC503E"], alpha=0.35)
        xs0, ys0, full = _reachable_mask(0.0, resolution, start=start)
        xs, ys, fill = _reachable_mask(eta_last, resolution, start=start)
        if fill.any():
            ax.contourf(xs, ys, fill.astype(float), levels=[0.5, 1.5], colors=[reach_color], alpha=0.45)
        ax.contour(xs0, ys0, full.astype(float), levels=[0.5], colors=full_color, linewidths=0.6, linestyles="--")
        ax.scatter(*start, marker=".", s=25, color="black", zorder=5)
        ax.scatter(*TARGET, marker="*", s=25, color="black", zorder=5)
        ax.set_xlim(*DOMAIN_X)
        ax.set_ylim(*DOMAIN_Y)
        ax.set_aspect("equal")
        ax.set_xticks([40, 50, 60])
        ax.set_yticks([40, 50, 60])
        ax.set_title(title, fontsize=7, pad=2)
        _style_axis(ax)
        ax.grid(False)
        ax.tick_params(labelsize=6, pad=1)
        ax.set_xlabel(r"$x$ [m]", fontsize=7, labelpad=1)
        covered = (fill & full).sum() / max(full.sum(), 1)
        print(f"{title!r}: shaded set covers {100 * covered:.1f}% of the landing site's safe component")
    axes[0].set_ylabel(r"$y$ [m]", fontsize=7, labelpad=1)
    handles = [
        Patch(facecolor=reach_color, alpha=0.45, label=r"$\overline{R}_{\eta_t^{(\alpha)}}$"),
        Line2D([], [], color=full_color, linewidth=0.6, linestyle="--", label=r"$\overline{R}_{0^+}$"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7,
               handlelength=1.5, columnspacing=1.2, borderpad=0.1)
    fig.savefig(out)
    fig.savefig(str(Path(out).with_suffix(".pdf")))
    plt.close(fig)
    print(f"Saved {out} (and {Path(out).with_suffix('.pdf')})")


@app.command()
def reachable_sweep(
    etas: List[float] = typer.Option([0.3, 0.2, 0.05], min=0.0, help="Repeat --etas for each panel"),
    resolution: int = typer.Option(201, min=3, help="Grid resolution for the flood fill"),
    out: str = f"{Path().absolute()}/examples/mars_reachable_sweep.png",
):
    panels = [(rf"$\eta={eta:g}$", eta) for eta in etas]
    _draw_reachable_panels(panels, out, resolution, reach_label=r"$\overline{R}_{\eta}$")


def sample_connected_landings(seed: int, count: int = 10, resolution: int = 201,
                              landing_radius: float = 2.0) -> np.ndarray:
    if not np.isfinite(landing_radius) or landing_radius <= 0:
        raise ValueError("Landing radius must be finite and positive")
    component = connected_safe_component(resolution)
    component = component[np.linalg.norm(component - np.asarray(SEED), axis=1) <= landing_radius]
    if len(component) < count:
        raise ValueError(f"Safe component within {landing_radius:g} m contains only {len(component)} grid points; need {count}")
    indices = np.random.default_rng(seed).choice(len(component), size=count, replace=False)
    return component[indices]


@app.command()
def random_landings(
    n_opt_samples: int = typer.Option(400, min=1, help="Number of BO rounds per landing and algorithm"),
    seed: Optional[int] = typer.Option(None, min=0, max=2**32 - 1, help="Random seed; omitted means generate and save one"),
    algorithms: List[str] = typer.Option(
        ["SafeOpt", "SafeUCB", "CSafeOpt", "GoOSE"], help="Algorithms evaluated at the same ten landing sites"
    ),
    connectivity_resolution: int = typer.Option(201, min=3, help="Grid resolution for the original landing site's safe component"),
    landing_radius: float = typer.Option(2.0, min=1e-12, help="Maximum distance in metres from the nominal landing site for random starts"),
    out: str = f"{Path().absolute()}/examples/mars_random_landings.png",
):
    Logger.set_verbosity(2)
    seed = secrets.randbits(32) if seed is None else seed
    points = sample_connected_landings(seed, resolution=connectivity_resolution, landing_radius=landing_radius)
    trial_seeds = np.random.default_rng(seed).integers(0, 2**32, size=10).tolist()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    j_star = true_optimum()
    names = _legend_order(list(dict.fromkeys(algorithms)))
    starts = pd.DataFrame(points, columns=["landing_x", "landing_y"])
    starts.insert(0, "trial", np.arange(1, 11))
    starts["distance_from_nominal"] = np.linalg.norm(points - np.asarray(SEED), axis=1)
    starts["run_seed"] = trial_seeds
    starts["constraint"] = constraint_fn(points[:, 0], points[:, 1])
    starts.to_csv(path.with_name(path.stem + "_landings.csv"), index=False)
    path.with_suffix(".json").write_text(json.dumps({
        "seed": seed, "rounds": n_opt_samples, "algorithms": names,
        "connectivity_resolution": connectivity_resolution, "landing_radius": landing_radius,
        "component_anchor": SEED, "domain_x": DOMAIN_X, "domain_y": DOMAIN_Y,
        "slope_limit": SLOPE_LIMIT, "j_star": j_star,
        "band": "one population standard deviation across ten starts",
        "config": CONFIG,
    }, indent=2) + "\n")
    print(f"Landing-selection seed: {seed}; ten starts within {landing_radius:g} m of {SEED} in the grid-connected safe component", flush=True)
    frames = []
    for index, (point, run_seed) in enumerate(zip(points, trial_seeds), start=1):
        for name in names:
            print(f"Landing {index}/10 {tuple(point)}: {name}, seed={run_seed}", flush=True)
            data, _ = run(name, run_seed, n_opt_samples, initial_point=tuple(point))
            values = data.train_y.numpy()
            locations = data.train_x.numpy()
            frames.append(pd.DataFrame({
                "algorithm": name, "trial": index, "run_seed": run_seed,
                "landing_x": point[0], "landing_y": point[1],
                "round": np.arange(len(values)), "x": locations[:, 0], "y": locations[:, 1],
                "reward": values[:, 0], "constraint": values[:, 1],
                "cumulative_regret": np.cumsum(j_star - values[:, 0]),
            }))
            pd.concat(frames, ignore_index=True).to_csv(path.with_suffix(".csv"), index=False)
    _plot_random_landings(pd.concat(frames, ignore_index=True), names, path)


def _plot_random_landings(results: pd.DataFrame, names: list, path: Path) -> None:
    _apply_theme()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    style = {name: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, name in enumerate(names)}
    aggregates = []
    for name in _z_order(names):
        curves = results[results.algorithm == name].pivot(index="round", columns="trial", values="cumulative_regret")
        mean = curves.mean(axis=1).to_numpy()
        std = curves.std(axis=1, ddof=0).to_numpy()
        rounds = curves.index.to_numpy()
        color, marker = style[name]
        ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(rounds, mean, color=color, marker=marker, markevery=max(1, len(rounds) // 15),
                markersize=3, linewidth=1.8)
        aggregates.append(pd.DataFrame({"algorithm": name, "round": rounds, "mean": mean, "std": std}))
    ax.set_xlabel("round")
    ax.set_ylabel(r"cumulative regret $R_N$")
    ax.set_title("10 random safe landing sites: mean $\\pm$ 1 SD")
    _legend(ax, names, style)
    _style_axis(ax)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    pd.concat(aggregates, ignore_index=True).to_csv(path.with_name(path.stem + "_summary.csv"), index=False)
    print(f"Saved cumulative-regret plot to {path} and {path.with_suffix('.pdf')}")


def _apply_overrides(overrides: list) -> None:
    for section, key, value in overrides:
        CONFIG[section][key] = value


def _landing_run(task: dict) -> pd.DataFrame:
    Logger.set_verbosity(0)
    torch.set_num_threads(task["threads"])
    _apply_overrides(task["overrides"])
    point = (task["landing_x"], task["landing_y"])
    data, _ = run(task["algorithm"], task["run_seed"], task["rounds"], initial_point=point)
    values = data.train_y.numpy()
    locations = data.train_x.numpy()
    return pd.DataFrame({
        "algorithm": task["algorithm"], "trial": task["trial"], "run_seed": task["run_seed"],
        "landing_x": point[0], "landing_y": point[1],
        "round": np.arange(len(values)), "x": locations[:, 0], "y": locations[:, 1],
        "reward": values[:, 0], "constraint": values[:, 1],
        "cumulative_regret": np.cumsum(task["j_star"] - values[:, 0]),
    })


@app.command()
def random_landings_extend(
    algorithms: List[str] = typer.Option(..., help="Algorithms to (re)run at the saved landing sites"),
    out: str = f"{Path().absolute()}/examples/mars_random_landings.png",
    source: Optional[str] = typer.Option(
        None, help="Experiment whose landing sites and seeds to reuse; default is `out` itself. A different `out` gets its own files"
    ),
    override: List[str] = typer.Option(
        [], help="Change a CONFIG entry for these runs only, e.g. model.normalize_output=false (repeatable)"
    ),
    jobs: int = typer.Option(1, min=1, help="Runs in parallel (each needs up to about 9 GB of memory)"),
    threads: int = typer.Option(4, min=1, help="Torch threads per run"),
):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    from multistart import parse_overrides

    Logger.set_verbosity(2)
    path = Path(out)
    source_path = Path(source) if source is not None else path
    overrides = parse_overrides(override)
    _apply_overrides(overrides)
    meta = json.loads(source_path.with_suffix(".json").read_text())
    j_star = true_optimum()
    if not math.isclose(j_star, meta["j_star"], rel_tol=1e-9):
        raise typer.BadParameter(f"J* changed since the experiment was run ({meta['j_star']} -> {j_star})")
    landings = pd.read_csv(source_path.with_name(source_path.stem + "_landings.csv"))
    new = list(dict.fromkeys(algorithms))

    separate = path.resolve() != source_path.resolve()
    if separate:
        landings.to_csv(path.with_name(path.stem + "_landings.csv"), index=False)
        previous = pd.read_csv(path.with_suffix(".csv")) if path.with_suffix(".csv").exists() else pd.DataFrame()
        meta = {**meta, "algorithms": [], "source": str(source_path), "config": CONFIG}
        meta.pop("extended_config", None)
        done = set()
        if len(previous):
            counts = previous.groupby(["algorithm", "trial"]).size()
            done = {key for key, n in counts.items() if n == meta["rounds"]}
            previous = previous[[(a, t) in done for a, t in zip(previous.algorithm, previous.trial)]]
    else:
        previous = pd.read_csv(path.with_suffix(".csv"))
        previous = previous[~previous.algorithm.isin(new)]
        done = set()
    if overrides:
        meta["overrides"] = [f"{s}.{k}={v}" for s, k, v in overrides]

    tasks = [
        {
            "algorithm": name, "trial": int(landing.trial), "run_seed": int(landing.run_seed),
            "landing_x": float(landing.landing_x), "landing_y": float(landing.landing_y),
            "rounds": meta["rounds"], "j_star": j_star, "overrides": overrides, "threads": threads,
        }
        for landing in landings.itertuples()
        for name in new
        if (name, int(landing.trial)) not in done
    ]
    frames = [previous] if len(previous) else []
    print(f"{len(tasks)} runs to do, {len(done)} already finished in {path.with_suffix('.csv').name}", flush=True)
    with ProcessPoolExecutor(jobs, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(_landing_run, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), start=1):
            frame = future.result()
            frames.append(frame)
            pd.concat(frames, ignore_index=True).to_csv(path.with_suffix(".csv"), index=False)
            print(
                f"[{count}/{len(tasks)}] landing {frame.trial.iloc[0]} {frame.algorithm.iloc[0]}: "
                f"R_N={frame.cumulative_regret.iloc[-1]:.3f}",
                flush=True,
            )
    results = pd.concat(frames, ignore_index=True)
    names = _legend_order(list(dict.fromkeys(meta["algorithms"] + new)))
    meta["algorithms"] = names
    if not separate:
        meta.setdefault("extended_config", {}).update({name: CONFIG.get(name) for name in new})
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    _plot_random_landings(results, names, path)


@app.command()
def gate_activity(
    n_opt_samples: int = typer.Option(400, help="Number of BO rounds"),
    seed: int = typer.Option(42, help="RNG seed"),
    out: str = f"{Path().absolute()}/examples/mars_gate_activity.png",
):
    Logger.set_verbosity(2)
    j_star = true_optimum()

    aquisition = GateTrackingCSafeOpt(**CONFIG["CSafeOpt"], dim_obs=CONFIG["dim_obs"])
    data, aquisition = run("CSafeOpt", seed, n_opt_samples, aquisition_override=aquisition)

    gate_history = aquisition.gate_history
    n_gate_used = sum(1 for _, _gate_max, gate_at_chosen in gate_history if gate_at_chosen > 0.0)
    spans = _gate_spans(gate_history)
    n_closures = sum(1 for used, _, _ in spans if not used)
    n_used_spans = sum(1 for used, _, _ in spans if used)
    n_reopenings = max(n_used_spans - 1, 0)
    gate_maxes = [g for _, g, _ in gate_history]
    at_chosen = [c for _, _, c in gate_history]
    print(f"gate determined the chosen point in {n_gate_used}/{len(gate_history)} rounds, across {len(spans)} spans")
    print(f"plain UCB determined the choice {n_closures} time(s); gate regained the argmax {n_reopenings} time(s)")
    print(f"gate_max: min={min(gate_maxes):.4f} max={max(gate_maxes):.4f}")
    print(f"gate(x_t)/gate_max at the sampled point: min={min(at_chosen):.4f} max={max(at_chosen):.4f}")
    for used, start, end in spans:
        print(f"  {'GATE  ' if used else 'UCB   '} rounds {start}-{end} ({end - start + 1} rounds)")

    _apply_theme()
    plt.rcParams.update(figsizes.iclr2023(nrows=1, ncols=2))
    name = "CSafeOpt"
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
    ax_regret.set_title(rf"$R_N$, $J^\star={j_star:.3f}$" + "\n(dark green = expansion-driven)")

    rounds_h, _tau_h, eta_h = zip(*aquisition.threshold_history)
    threshold_df = pd.DataFrame({"round": rounds_h, "value": eta_h, "name": name})
    _plot_series(ax_threshold, threshold_df, "round", "value", [name], style, pointsize=3.5)
    _shade_gate_intensity(ax_threshold, gate_history)
    ax_threshold.set_xlabel("round")
    ax_threshold.set_ylabel(r"$\eta_t$")
    ax_threshold.set_title(
        r"Expansion confidence threshold" + "\n(dark green = expansion-driven)"
    )

    for ax in fig.get_axes():
        _style_axis(ax)

    fig.savefig(out)
    pdf_path = str(Path(out).with_suffix(".pdf"))
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved gate activity plot to {out} (and {pdf_path})")


@app.command()
def gate_activity_comparison(
    n_opt_samples: int = typer.Option(400, min=2, help="Total evaluations, including the initial landing at round zero"),
    seed: int = typer.Option(42, help="RNG seed shared by both runs"),
    expansion_rounds: int = typer.Option(50, min=1, help="SafeOpt queries before switching StageOpt to SafeUCB"),
    expansion_eta: Optional[float] = typer.Option(None, min=1e-12, help="Stop expansion when maximum safe-candidate constraint confidence width <= eta; overrides expansion-rounds"),
    out: str = f"{Path().absolute()}/examples/mars_gate_activity_comparison.png",
):
    Logger.set_verbosity(2)
    j_star = true_optimum()
    csafe = GateTrackingCSafeOpt(**CONFIG["CSafeOpt"], dim_obs=CONFIG["dim_obs"])
    stage = StageOpt(**CONFIG["SafeOpt"], dim_obs=CONFIG["dim_obs"], expansion_rounds=expansion_rounds, expansion_eta=expansion_eta)
    print("Running CSafeOpt", flush=True)
    csafe_data, _ = run("CSafeOpt", seed, n_opt_samples, aquisition_override=csafe)
    stopping_rule = (f"constraint width <= {expansion_eta}" if expansion_eta is not None
                     else f"{expansion_rounds} SafeOpt queries")
    print(f"Running StageOpt: {stopping_rule}, then SafeUCB", flush=True)
    stage_data, _ = run("SafeOpt", seed, n_opt_samples, aquisition_override=stage)
    _apply_theme()
    plt.rcParams.update(figsizes.iclr2023(nrows=1, ncols=2))
    fig, axes = plt.subplots(1, 2, sharey=True)
    frames = []
    entries = [
        ("CSafeOpt", csafe_data, csafe.gate_history, "green = gate activity"),
        ("StageOpt", stage_data, stage.activity_history, (rf"green = SafeOpt phase ($\eta={expansion_eta:g}$)" if expansion_eta is not None
         else f"green = SafeOpt phase (queries 1–{expansion_rounds})")),
    ]
    ymax = max(np.cumsum(j_star - data.train_y[:, 0].numpy()).max()
               for data in (csafe_data, stage_data))
    for index, (ax, (name, data, history, subtitle)) in enumerate(zip(axes, entries)):
        values = data.train_y.numpy()
        activity = {t: chosen for t, _, chosen in history}
        frame = pd.DataFrame({
            "name": name, "round": np.arange(len(values)),
            "x": data.train_x[:, 0].numpy(), "y": data.train_x[:, 1].numpy(),
            "reward": values[:, 0], "constraint": values[:, 1],
            "cumulative_regret": np.cumsum(j_star - values[:, 0]),
            "activity": [activity.get(t, 0.0) for t in range(len(values))],
        })
        frame["phase"] = [
            "initial" if t == 0 else
            ("SafeOpt" if activity.get(t, 0.0) > 0 else "SafeUCB") if name == "StageOpt" else
            ("gate" if activity.get(t, 0.0) > 0 else "UCB")
            for t in range(len(values))
        ]
        widths = dict(stage.width_history) if name == "StageOpt" else {}
        frame["max_safe_constraint_width"] = [widths.get(t, float("nan")) for t in range(len(values))]
        frames.append(frame)
        _plot_series(ax, frame, "round", "cumulative_regret", [name],
                     {name: (PALETTE[index], MARKERS[index])})
        ax.set_ylim(0, max(1e-6, ymax * 1.05))
        _shade_gate_intensity(ax, history)
        ax.set_xlim(0, n_opt_samples - 1)
        ax.set_xlabel("round")
        ax.set_title(f"{name}\n{subtitle}")
        if name == "StageOpt" and stage.switch_after_query is not None and n_opt_samples > stage.switch_after_query + 1:
            ax.axvline(stage.switch_after_query + 0.5, color="0.4", linestyle="--", linewidth=0.8)
        _style_axis(ax)
    axes[0].set_ylabel(r"cumulative regret $R_N$")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    pd.concat(frames, ignore_index=True).to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".json").write_text(json.dumps({
        "seed": seed, "n_opt_samples": n_opt_samples, "safeopt_queries": int(sum(h[2] for h in stage.activity_history)),
        "expansion_rounds": expansion_rounds, "expansion_eta": expansion_eta,
        "switch_after_query": stage.switch_after_query,
        "eta_scope": "full constraint confidence width over all evaluated certified-safe candidates",
        "j_star": j_star, "config": CONFIG,
        "activity": {"CSafeOpt": "normalized gate at chosen point", "StageOpt": "SafeOpt phase indicator"},
    }, indent=2) + "\n")
    if expansion_eta is not None and stage.expanding:
        print("Evaluation budget reached while expansion criterion remains unmet; no SafeUCB switch.", flush=True)
    print(f"Saved gate activity comparison to {path} and {path.with_suffix('.pdf')}")


@app.command()
def gate_activity_eta_comparison(
    n_opt_samples: int = typer.Option(400, min=2),
    seed: int = typer.Option(42),
    etas: List[float] = typer.Option([0.03812, 0.03813], min=1e-12, help="Repeat --etas for each StageOpt threshold"),
    reference: Optional[Path] = typer.Option(None, help="Reuse a matching comparison CSV and its JSON for CSafeOpt"),
    out: str = f"{Path().absolute()}/examples/mars_gate_activity_eta_comparison.png",
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple

    Logger.set_verbosity(2)
    etas = list(dict.fromkeys(etas))
    if not etas or any(not np.isfinite(eta) or eta <= 0 for eta in etas):
        raise ValueError("Provide finite, positive eta values")
    j_star = true_optimum()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)

    def make_frame(name, data, history, widths=()):
        values = data.train_y.numpy()
        activity = dict((t, chosen) for t, _, chosen in history)
        widths = dict(widths)
        return pd.DataFrame({
            "name": name, "round": np.arange(len(values)),
            "x": data.train_x[:, 0].numpy(), "y": data.train_x[:, 1].numpy(),
            "reward": values[:, 0], "constraint": values[:, 1],
            "cumulative_regret": np.cumsum(j_star - values[:, 0]),
            "activity": [activity.get(t, 0.) for t in range(len(values))],
            "max_safe_constraint_width": [widths.get(t, float("nan")) for t in range(len(values))],
        })

    if reference is not None:
        saved = json.loads(reference.with_suffix(".json").read_text())
        if (saved["seed"] != seed or saved["n_opt_samples"] != n_opt_samples
                or saved["config"] != json.loads(json.dumps(CONFIG))
                or not np.isclose(saved["j_star"], j_star, rtol=0, atol=1e-12)):
            raise ValueError("CSafeOpt reference settings do not match this experiment")
        frame = pd.read_csv(reference)
        frame = frame[frame.name == "CSafeOpt"].sort_values("round").copy()
        if not np.array_equal(frame["round"], np.arange(n_opt_samples)):
            raise ValueError("CSafeOpt reference must contain every round exactly once")
        print(f"Reusing CSafeOpt reference: {reference}", flush=True)
    else:
        csafe = GateTrackingCSafeOpt(**CONFIG["CSafeOpt"], dim_obs=CONFIG["dim_obs"])
        data, _ = run("CSafeOpt", seed, n_opt_samples, aquisition_override=csafe)
        frame = make_frame("CSafeOpt", data, csafe.gate_history)
    frame["eta"] = np.nan
    frames = [frame]
    metadata = {"seed": seed, "n_opt_samples": n_opt_samples, "etas": etas,
                "j_star": j_star, "config": CONFIG,
                "csafeopt_reference": str(reference) if reference is not None else None,
                "eta_scope": "full constraint confidence width over all evaluated certified-safe candidates",
                "runs": []}
    for eta in etas:
        print(f"Running StageOpt eta={eta:g}", flush=True)
        stage = StageOpt(**CONFIG["SafeOpt"], dim_obs=CONFIG["dim_obs"], expansion_eta=eta)
        data, _ = run("SafeOpt", seed, n_opt_samples, aquisition_override=stage)
        result = make_frame("StageOpt", data, stage.activity_history, stage.width_history)
        result["eta"] = eta
        result["phase"] = np.where(result["round"] == 0, "initial",
                                   np.where(result.activity > 0, "SafeOpt", "SafeUCB"))
        frames.append(result)
        record = {"eta": eta, "switch_after_query": stage.switch_after_query,
                  "expansion_queries": int(result.activity.sum()),
                  "final_regret": float(result.cumulative_regret.iloc[-1]),
                  "final_reward": float(result.reward.iloc[-1]),
                  "unsafe_count": int((result.constraint < 0).sum())}
        metadata["runs"].append(record)
        print(json.dumps(record), flush=True)
        pd.concat(frames, ignore_index=True).to_csv(path.with_suffix(".csv"), index=False)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

    _apply_theme()
    plt.rcParams.update(figsizes.iclr2023(nrows=1, ncols=2))
    fig, axes = plt.subplots(1, 2, sharey=True)
    ymax = max(f.cumulative_regret.max() for f in frames)
    for ax in axes:
        ax.set_ylim(0, max(1e-6, 1.05 * ymax))
        ax.set_xlim(0, n_opt_samples - 1)
        ax.set_xlabel("round")
        _style_axis(ax)
    left, right = axes
    left.plot(frame["round"], frame.cumulative_regret, color=PALETTE[0], label="CSafeOpt", linewidth=1.8)
    _shade_gate_intensity(left, [(int(r["round"]), float("nan"), float(r.activity))
                               for _, r in frame[frame["round"] > 0].iterrows()])
    left.set_title("CSafeOpt\ngreen = expansion-driven")
    left.set_ylabel(r"cumulative regret $R_N$")
    left.legend()
    handles, labels = [], []
    for i, (eta, result) in enumerate(zip(etas, frames[1:])):
        color = PALETTE[(i + 1) % len(PALETTE)]
        linestyle = ["-", "--", ":", "-."][i % 4]
        opacity = 0.16 + 0.16 * i / max(1, len(etas) - 1)
        right.plot(result["round"], result.cumulative_regret, color=color, linestyle=linestyle,
                   linewidth=1.8, marker=MARKERS[(i+1) % len(MARKERS)],
                   markevery=max(1, n_opt_samples // 15), markersize=3)
        active = result.loc[result.activity > 0, "round"]
        if len(active):
            right.axvspan(active.min() - .5, max(active.max() + .5, 20),
                          color="forestgreen", alpha=opacity, linewidth=0, zorder=0)
        handles.append((Line2D([], [], color=color, linestyle=linestyle, linewidth=1.8),
                        Patch(facecolor="forestgreen", alpha=opacity)))
        labels.append(rf"StageOpt $\eta={eta:g}$")
    right.set_title("StageOpt\ngreen = expansion phase")
    right.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)},
                 fontsize=7, handlelength=3.5, loc="best")
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved eta overlay to {path}", flush=True)


@app.command()
def gate_activity_panel(
    n_opt_samples: int = typer.Option(400, min=2, help="Total evaluations, including the initial landing at round zero"),
    seed: int = typer.Option(42, help="RNG seed"),
    out: str = f"{Path().absolute()}/examples/mars_gate_activity_panel.png",
):
    Logger.set_verbosity(2)
    j_star = true_optimum()
    aquisition = GateTrackingCSafeOpt(**CONFIG["CSafeOpt"], dim_obs=CONFIG["dim_obs"])
    data, aquisition = run("CSafeOpt", seed, n_opt_samples, aquisition_override=aquisition)
    regret = np.cumsum(j_star - data.train_y[:, 0].numpy())
    print(f"CSafeOpt {CONFIG['CSafeOpt']}: cumulative regret after {len(regret)} rounds: {regret[-1]:.2f}")

    _apply_theme()
    fig, ax = plt.subplots(figsize=(3.25, 1.85))
    ax.set_ylim(0, max(1e-6, 1.05 * regret.max()))
    ax.set_xlim(0, n_opt_samples - 1)
    ax.plot(np.arange(len(regret)), regret, color=PALETTE[0], label="CSafeOpt", linewidth=1.8)
    _shade_gate_intensity(ax, aquisition.gate_history)
    ax.set_title("CSafeOpt\ngreen = expansion-driven")
    ax.set_xlabel("round")
    ax.set_ylabel(r"cumulative regret $R_N$")
    ax.legend()
    _style_axis(ax)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(Path(out).with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved gate activity panel to {out} (and {Path(out).with_suffix('.pdf')})")


@app.command()
def random_landings_column(
    source: str = typer.Option(
        f"{Path().absolute()}/examples/mars_random_landings_goose_matched_all.csv",
        help="Per-round results of random_landings / random_landings_extend (its .json sits next to it)",
    ),
    map_trial: int = typer.Option(7, min=1, help="Landing trial whose samples the maps show"),
    maps: List[str] = typer.Option(
        ["CSafeOpt", "SafeOpt", "SafeUCB"], help="Algorithms drawn on the maps (a row of three, or a 2 x 2 grid)"
    ),
    layout: str = typer.Option("row", help="'row': square maps in one row; 'grid': 2 x 2 maps"),
    exclude: List[str] = typer.Option(["CSafeOptSimple"], help="Algorithms left out of the figure entirely"),
    extra_map: Optional[str] = typer.Option(None, help="Algorithm whose map goes beside a half-width regret panel"),
    out: str = f"{Path().absolute()}/examples/mars_random_landings_column.png",
):
    _apply_theme()
    results = pd.read_csv(source)
    j_star = json.loads(Path(source).with_suffix(".json").read_text())["j_star"]
    present = list(dict.fromkeys(results.algorithm))
    order = ["CSafeOpt", "SafeOpt", "SafeUCB", "GoOSE", "CSafeOptSimple", "ISE-BO", "StageOpt"]
    names = [n for n in order if n in present] + [n for n in present if n not in order]
    style = {n: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, n in enumerate(names)}
    names = [n for n in names if n not in exclude]
    results = results[results.algorithm.isin(names)]

    trial = results[results.trial == map_trial]
    if trial.empty:
        raise ValueError(f"no trial {map_trial} in {source}")
    maps = {n: trial[trial.algorithm == n].sort_values("round")[["x", "y"]].to_numpy()
            for n in [*maps, *([extra_map] if extra_map else [])]}
    landing = (float(trial.landing_x.iloc[0]), float(trial.landing_y.iloc[0]))
    terrain = Terrain(
        DOMAIN_X, DOMAIN_Y, elevation, constraint_fn, landing, TARGET, float(np.rad2deg(np.arctan(SLOPE_LIMIT))), j_star,
    )
    plot_mars_column_stats(maps, results, out, terrain, names, style, [n for n in maps if n in names], extra_map,
                           layout=layout)


if __name__ == "__main__":
    app()
