import math
import random
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
import torch
import typer
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
    _draw_order,
    _legend,
    _legend_order,
    _plot_series,
    _style_axis,
    reset_global_state,
)
from confidence_bounds import SCHEMATIC_FIGSIZE, plot_confidence_schematic
from torch import Tensor
from trajectories_3d import COLUMN_FIGSIZE, Landscape, plot_trajectories

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

SEED_X = 0.5
MID_PEAK_X = 5.5
FINAL_PEAK_X = 10.5
BOTTLENECK_1 = (1.5, 4.0)
BOTTLENECK_2 = (7.3, 9.0)
DOMAIN = (0.0, 12.0)


def reward_fn(x: np.ndarray) -> np.ndarray:
    return (
        1.2 * np.exp(-((x - SEED_X) ** 2) / (2 * 0.35**2))
        + 2.1 * np.exp(-((x - MID_PEAK_X) ** 2) / (2 * 0.8**2))
        + 3.9 * np.exp(-((x - FINAL_PEAK_X) ** 2) / (2 * 0.9**2))
        - 0.3
    )


BOTTLENECK_2_DIP_CENTER = 8.87
BOTTLENECK_2_DIP_WIDTH = 1
BOTTLENECK_2_DIP_AMPLITUDE = 0.1


def constraint_fn(x: np.ndarray) -> np.ndarray:
    return (
        0.9 * np.exp(-((x - SEED_X) ** 2) / (2 * 1.0**2))
        + 0.9 * np.exp(-((x - MID_PEAK_X) ** 2) / (2 * 1.0**2))
        + 0.9 * np.exp(-((x - FINAL_PEAK_X) ** 2) / (2 * 0.45**2))
        + 0.12
        - BOTTLENECK_2_DIP_AMPLITUDE
        * np.exp(-((x - BOTTLENECK_2_DIP_CENTER) ** 2) / (2 * BOTTLENECK_2_DIP_WIDTH**2))
    )


class DoubleBottleneckEnv(Environment):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(None, render_mode)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1), {}

    def step(self, k):
        x = float(k[0])
        reward = np.array([2 * reward_fn(x), 2 * constraint_fn(x)])
        return np.array([x]), reward, True, False, {}


CONFIG = {
    "log_video": False,
    "log_plots": False,
    "dim_obs": 2,
    "dim_params": 1,
    "dim_context": 0,
    "dim_model": 1,
    "domain_start": [DOMAIN[0]],
    "domain_end": [DOMAIN[1]],
    "model": {
        "lenghtscale": [0.4],
        "normalize_input": True,
        "normalize_output": True,
        "likelihood_noise": 1e-4,
    },
    "Optimization": {
        "set_size": 6000,
        "set_init": "random",
        "max_global_steps_without_progress_tolerance": 0.9,
        "max_global_steps_without_progress": 10_000,
    },
    "SafeOpt": {"scale_beta": 1.0, "beta": 9},
    "SafeUCB": {"scale_beta": 1.0, "beta": 9},
    "CSafeOpt": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.1 / 3, "alpha": 1, "zeta": 0.01, "lipschitz": 0.5},
    "CSafeOptSimple": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.3, "alpha": 0.51, "zeta": 0.0, "delta_f": 4.0},
    "GoSafeOpt": {"scale_beta": 1.0, "beta": 9, "n_max_local": 5, "n_max_global": 3},
    "GoOSE": {"scale_beta": 1.0, "beta": 9, "lipschitz": 0.25, "epsilon": 0.1 / 3},
    "StageOpt": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.1 / 3},
    "ISE-BO": {"scale_beta": 1.0, "beta": 9},
}


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
        if epsilon is not None:
            kwargs["epsilon"] = epsilon
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
    name: str, seed: int, n_opt_samples: int, alpha: Optional[float] = None, initial_point: Optional[tuple] = None,
    epsilon: Optional[float] = None,
) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    reset_global_state()

    data = Data()
    x_safe = torch.tensor([[SEED_X if initial_point is None else initial_point[0]]])

    environment = DoubleBottleneckEnv(render_mode=None)
    experiment = Experiment(CONFIG, environment, data=data, backup=None)

    trainer = Trainer(
        dim_params=CONFIG["dim_params"],
        dim_obs=CONFIG["dim_obs"],
        n_opt_samples=n_opt_samples,
        show_progress=False,
        refit_interval=0,
        data=data,
    )

    aquisition = build_aquisition(name, CONFIG["dim_obs"], data, alpha=alpha, epsilon=epsilon)

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


def true_optimum(resolution: int = 200_000) -> float:
    xs = np.linspace(DOMAIN[0], DOMAIN[1], resolution)
    return float(reward_fn(xs).max())


def bottleneck_margin(bottleneck: tuple, resolution: int = 20_000) -> float:
    xs = np.linspace(bottleneck[0], bottleneck[1], resolution)
    return float(constraint_fn(xs).min())


def _greedy_gamma_sequence(covar: gpytorch.kernels.Kernel, lam: float, grid_size: int, n_direct: int) -> torch.Tensor:
    xs = torch.linspace(0.0, 1.0, grid_size, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        k = covar(xs).evaluate().double()
    gram_cols = torch.zeros(grid_size, n_direct, dtype=torch.float64)
    var = torch.diag(k).clone()
    log_det_sum = 0.0
    gamma_seq = torch.zeros(n_direct, dtype=torch.float64)
    for n in range(n_direct):
        i = int(torch.argmax(var).item())
        d = var[i] + lam
        if n == 0:
            g = k[:, i] / torch.sqrt(d)
        else:
            g = (k[:, i] - gram_cols[:, :n] @ gram_cols[i, :n]) / torch.sqrt(d)
        gram_cols[:, n] = g
        var = torch.clamp(var - g**2, min=0.0)
        log_det_sum += torch.log(d / lam).item()
        gamma_seq[n] = 0.5 * log_det_sum
    return gamma_seq


def _fit_gamma_extrapolation(gamma_seq: torch.Tensor):
    n_total = len(gamma_seq)
    fit_from = max(n_total // 2, 10)
    ns = np.arange(fit_from, n_total + 1)
    gs = gamma_seq[fit_from - 1 : n_total].numpy()
    feat = (ns ** (1 / 6)) * (np.log(ns) ** (5 / 6))
    design = np.vstack([np.ones_like(ns, dtype=float), feat]).T
    (a, b), *_ = np.linalg.lstsq(design, gs, rcond=None)
    residual = float(np.abs(gs - (a + b * feat)).max())

    def gamma_fn(n: int) -> float:
        if n <= n_total:
            return gamma_seq[n - 1].item()
        return a + b * (n ** (1 / 6)) * (math.log(n) ** (5 / 6))

    return gamma_fn, residual


def _solve_nbar(gamma_fn, beta_fn, C_lambda: float, m_star: float, hi_cap: float = 1e12):

    def rhs(n: float) -> float:
        gamma = gamma_fn(max(int(round(n)), 1))
        beta = beta_fn(gamma)
        return 4.0 * C_lambda * beta * gamma / (m_star**2)

    n = 1.0
    while rhs(n) > n:
        n *= 2.0
        if n > hi_cap:
            return None, rhs(n)
    lo, hi = n / 2.0, n
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if rhs(mid) > mid:
            lo = mid
        else:
            hi = mid
    return hi, rhs(hi)


def _plot_landscape_and_trace(ax_landscape, ax_trace, results: dict, names: list, style: dict) -> None:
    xs = np.linspace(DOMAIN[0], DOMAIN[1], 800)
    landscape_df = pd.DataFrame(
        {
            "x": np.concatenate([xs, xs]),
            "value": np.concatenate([reward_fn(xs), constraint_fn(xs)]),
            "curve": [r"reward$(x)$"] * len(xs) + [r"constraint$(x)$"] * len(xs),
        }
    )
    curve_names = [r"reward$(x)$", r"constraint$(x)$"]
    so.Plot(landscape_df, x="x", y="value", color="curve").add(so.Line(linewidth=2.0), legend=False).scale(
        color=so.Nominal(values=[PALETTE[0], PALETTE[7]], order=curve_names)
    ).on(ax_landscape).plot()
    ax_landscape.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    for bottleneck in (BOTTLENECK_1, BOTTLENECK_2):
        ax_landscape.axvspan(*bottleneck, color="gainsboro", alpha=0.6, zorder=0)
    ax_landscape.axvline(SEED_X, color="black", linestyle=":", linewidth=1.2)
    ax_landscape.set_xlim(*DOMAIN)

    ax_landscape.set_xlabel(r"$x$")
    ax_landscape.set_ylabel("value")
    ax_landscape.set_title("True reward / constraint landscape")
    _legend(ax_landscape, curve_names, {curve_names[0]: (PALETTE[0], None), curve_names[1]: (PALETTE[7], None)})

    trace_df = pd.concat(
        [
            pd.DataFrame({"round": np.arange(data.train_x.shape[0]), "chosen_x": data.train_x[:, 0].numpy(), "name": n})
            for n, (data, _aq) in results.items()
        ],
        ignore_index=True,
    )
    _plot_series(ax_trace, trace_df, "round", "chosen_x", _draw_order(names), style)
    for bottleneck in (BOTTLENECK_1, BOTTLENECK_2):
        ax_trace.axhspan(*bottleneck, color="gainsboro", alpha=0.6, zorder=0)
    ax_trace.set_xlabel("round")
    ax_trace.set_ylabel(r"chosen $x$")
    ax_trace.set_title("Point evaluated per round")
    _legend(ax_trace, names, style)


def _landscape(j_star: float) -> Landscape:
    return Landscape(
        domain=DOMAIN,
        reward=reward_fn,
        constraint=constraint_fn,
        j_star=j_star,
        crossings=(BOTTLENECK_1[1], BOTTLENECK_2[1]),
        strips=((*BOTTLENECK_1, "gainsboro", "bottleneck"), (*BOTTLENECK_2, "gainsboro", "bottleneck")),
        z_ticks=(0, 2),
    )


def _plot_regret_and_threshold(ax_regret, ax_threshold, results: dict, names: list, style: dict, j_star: float,
                               narrow: bool = False, show_legend: bool = True, show_margins: bool = False,
                               threshold_legend: bool = True) -> None:
    n_rounds = max(data.train_y.shape[0] for data, _aq in results.values())
    series_style = {"marker_step": max(5, n_rounds // 20), "pointsize": 3.0} if narrow else {}

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
    _plot_series(ax_regret, regret_df, "round", "cumulative_regret", _draw_order(names), style, **series_style)
    ax_regret.set_xlabel("round")
    ax_regret.set_ylabel(r"cumulative regret $R_N$")
    ax_regret.set_title(
        rf"$J^\star = {j_star:.3f}$" if narrow
        else rf"$R_N = \sum_t (J^\star - f(x_t))$, $J^\star = {j_star:.3f}$"
    )
    if show_legend:
        _legend(ax_regret, names, style)

    threshold_frames = []
    crossing_lines = []
    for name, (data, aquisition) in results.items():
        history = getattr(aquisition, "threshold_history", None)
        if not history:
            continue
        rounds_h, _tau_h, eta_h = zip(*history)
        threshold_frames.append(pd.DataFrame({"round": rounds_h, "value": eta_h, "name": name}))

        crossed_mask = data.train_x[:, 0].numpy() > BOTTLENECK_2[1]
        if crossed_mask.any():
            crossing_lines.append((name, int(np.argmax(crossed_mask))))

    if threshold_frames:
        threshold_df = pd.concat(threshold_frames, ignore_index=True)
        gate_names = [n for n in names if n in threshold_df["name"].unique()]
        _plot_series(
            ax_threshold, threshold_df, "round", "value", _draw_order(gate_names), style,
            **{"pointsize": 3.5, **series_style},
        )

        for name, first_cross_round in crossing_lines:
            ax_threshold.axvline(first_cross_round, color=style[name][0], linestyle="-.", linewidth=1.2, alpha=0.8)

        if show_legend:
            _legend(ax_threshold, gate_names, style)

    if show_margins:
        for label, bottleneck in [(r"$m_1$", BOTTLENECK_1), (r"$m_2$", BOTTLENECK_2)]:
            margin = bottleneck_margin(bottleneck)
            ax_threshold.axhline(margin, color="black", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
            ax_threshold.annotate(
                label, (1.0, margin), xycoords=("axes fraction", "data"), xytext=(-3, 2), textcoords="offset points",
                ha="right", va="bottom", fontsize="small",
            )
        top = max(ax_threshold.get_ylim()[1], bottleneck_margin(BOTTLENECK_1))
        ax_threshold.set_ylim(0.0, 1.15 * top)
    legend = ax_threshold.get_legend()
    if not threshold_legend and legend is not None:
        legend.remove()

    ax_threshold.set_xlabel("round")
    ax_threshold.set_ylabel(r"$\eta_t$")
    ax_threshold.set_title(r"$\eta_t = 2\varepsilon\,\beta_t^{1/2-\alpha}$")


def plot_double_bottleneck(
    results: dict, out_path: str, trajectories_3d: bool = False, narrow: bool = False,
    confidence_schematic: bool = False, squeezed: bool = False, regret_threshold_only: bool = False,
    show_margins: bool = False,
):
    narrow = narrow or confidence_schematic
    _apply_theme()

    names = _legend_order(list(results.keys()))
    style = {n: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, n in enumerate(names)}
    j_star = true_optimum()

    if regret_threshold_only:
        width, height = plt.rcParams["figure.figsize"]
        fig, (ax_regret, ax_threshold) = plt.subplots(1, 2, figsize=(width, 0.5 * height))
    elif confidence_schematic:
        n_rounds = max(data.train_x.shape[0] for data, _aq in results.values())
        fig = plt.figure(figsize=SCHEMATIC_FIGSIZE)
        top, bottom = fig.subfigures(2, 1, height_ratios=[2.1, 1])
        plot_confidence_schematic(
            top, CONFIG, build_aquisition, results, names, style, _landscape(j_star), (5, n_rounds // 2, n_rounds)
        )
        ax_regret, ax_threshold = bottom.subplots(1, 2)
    elif trajectories_3d and squeezed and not narrow:
        width, height = plt.rcParams["figure.figsize"]
        fig = plt.figure(figsize=(width, 0.9 * height))
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.38, 1])
        plot_trajectories(top, results, names, style, _landscape(j_star), panel_span=(0.09, 0.74), zoom=1.06,
                          sparse=True)
        ax_regret, ax_threshold = bottom.subplots(1, 2)
    elif trajectories_3d:
        width, height = plt.rcParams["figure.figsize"]
        fig = plt.figure(figsize=COLUMN_FIGSIZE if narrow else (width, 1.11 * height))
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.8, 1] if narrow else [1.65, 1])
        plot_trajectories(top, results, names, style, _landscape(j_star), compact=narrow)
        ax_regret, ax_threshold = bottom.subplots(1, 2)
    else:
        fig, ((ax_landscape, ax_trace), (ax_regret, ax_threshold)) = plt.subplots(2, 2)
        _plot_landscape_and_trace(ax_landscape, ax_trace, results, names, style)

    _plot_regret_and_threshold(ax_regret, ax_threshold, results, names, style, j_star, narrow=narrow,
                               show_legend=regret_threshold_only or not (trajectories_3d or confidence_schematic),
                               show_margins=show_margins,
                               threshold_legend=not regret_threshold_only)

    for ax in [ax_regret, ax_threshold] if confidence_schematic else fig.get_axes():
        if ax.name != "3d":
            _style_axis(ax)

    fig.savefig(out_path)
    pdf_path = str(Path(out_path).with_suffix(".pdf"))
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved double-bottleneck plot to {out_path} (and {pdf_path})")


def plot_epsilon_alpha_grid(blocks: list, out_path: str) -> None:
    _apply_theme()
    j_star = true_optimum()
    width, height = plt.rcParams["figure.figsize"]
    legend_height = 0.12
    block_height = 0.82 * height
    fig = plt.figure(figsize=(width, block_height * len(blocks) + legend_height * height))
    ratios = [block_height + legend_height * height] + [block_height] * (len(blocks) - 1)
    for index, (block, (label, results)) in enumerate(zip(fig.subfigures(len(blocks), 1, height_ratios=ratios),
                                                            blocks)):
        names = list(results.keys())
        style = {n: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, n in enumerate(names)}
        first = index == 0
        top, bottom = block.subfigures(2, 1, height_ratios=[1.2 + (0.25 if first else 0.0), 1])
        span = (0.1, 0.76) if first else (0.1, 0.92)
        plot_trajectories(top, results, names, style, _landscape(j_star), panel_span=span, zoom=1.06,
                          sparse=True, legend=first)
        ax_regret, ax_threshold = bottom.subplots(1, 2)
        _plot_regret_and_threshold(ax_regret, ax_threshold, results, names, style, j_star, show_legend=False)
        for ax in (ax_regret, ax_threshold):
            _style_axis(ax)
        top.text(0.004, span[0] + span[1] / 2, label, rotation=90, ha="left",
                 va="center", fontsize=10)
    fig.savefig(out_path)
    pdf_path = str(Path(out_path).with_suffix(".pdf"))
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved epsilon x alpha ablation to {out_path} (and {pdf_path})")


def _print_summary(name: str, data, j_star: float):
    chosen_x = data.train_x[:, 0].numpy()
    reward = data.train_y[:, 0].numpy()
    crossed_1 = (chosen_x > BOTTLENECK_1[1]).any()
    crossed_2 = (chosen_x > BOTTLENECK_2[1]).any()
    cumulative_regret = (j_star - reward).sum()
    print(
        f"{name}: crossed bottleneck 1: {crossed_1}, crossed bottleneck 2: {crossed_2}, "
        f"furthest x reached: {chosen_x.max():.2f}, "
        f"cumulative regret after {len(reward)} rounds: {cumulative_regret:.2f}"
    )


@app.command()
def bottleneck(
    n_opt_samples: int = typer.Option(100, help="Number of BO rounds for every run"),
    seed: int = typer.Option(42, help="RNG seed shared by every run"),
    algorithms: List[str] = typer.Option(
        ["SafeOpt", "SafeUCB", "CSafeOpt", "GoOSE"], help="Which acquisitions to run"
    ),
    out: str = f"{Path().absolute()}/examples/double_bottleneck.png",
    trajectories_3d: bool = typer.Option(False, help="Replace the landscape/trace panels by one 3D panel per run"),
    narrow: bool = typer.Option(False, help="Size the figure for one column of a two-column page"),
    confidence_schematic: bool = typer.Option(
        False, help="Replace the landscape/trace panels by schematic confidence bounds (half-page format)"
    ),
):
    Logger.set_verbosity(2)
    j_star = true_optimum()

    results = {}
    for name in algorithms:
        data, aquisition = run(name, seed, n_opt_samples)
        results[name] = (data, aquisition)
        _print_summary(name, data, j_star)

    plot_double_bottleneck(
        results, out, trajectories_3d=trajectories_3d, narrow=narrow, confidence_schematic=confidence_schematic
    )


def _epsilon_for_eta0(algorithm: str, seed: int, alpha: float, eta0: float, tol: float = 1e-4) -> float:
    epsilon = CONFIG[algorithm]["epsilon"]
    for _ in range(10):
        _data, aquisition = run(algorithm, seed, 3, alpha=alpha, epsilon=epsilon)
        eta1 = aquisition.threshold_history[0][2]
        if abs(eta1 / eta0 - 1.0) < tol:
            break
        epsilon *= eta0 / eta1
    print(f"alpha={alpha:g}: epsilon={epsilon:.6g} gives eta_1={eta1:.6g} (target {eta0:g})", flush=True)
    return epsilon


@app.command()
def alpha_ablation(
    n_opt_samples: int = typer.Option(100, help="Number of BO rounds for every run"),
    seed: int = typer.Option(42, help="RNG seed shared by every run"),
    alphas: List[float] = typer.Option([0.5, 0.75, 1], help="alpha values to compare"),
    algorithm: str = typer.Option("CSafeOpt", help="Which acquisition to ablate over alpha (CSafeOpt or CSafeOptSimple)"),
    out: str = f"{Path().absolute()}/examples/double_bottleneck_alpha_ablation.png",
    eta0: Optional[float] = typer.Option(
        None, min=1e-12, help="Choose epsilon per alpha so every run starts at the same threshold eta_1 = eta0"
    ),
    regret_threshold_only: bool = typer.Option(False, help="Only the regret and eta_t panels (no samples)"),
    show_margins: bool = typer.Option(False, help="With --regret-threshold-only: bottleneck margins on eta_t"),
):
    Logger.set_verbosity(2)
    j_star = true_optimum()

    results = {}
    for a in alphas:
        plain_name = f"alpha={a:g}"
        name = rf"$\alpha={a:g}$"
        epsilon = _epsilon_for_eta0(algorithm, seed, a, eta0) if eta0 is not None else None
        data, aquisition = run(algorithm, seed, n_opt_samples, alpha=a, epsilon=epsilon)
        results[name] = (data, aquisition)
        _print_summary(plain_name, data, j_star)

    if regret_threshold_only:
        plot_double_bottleneck(results, out, regret_threshold_only=True, show_margins=show_margins)
    else:
        plot_double_bottleneck(results, out, trajectories_3d=True, squeezed=True)


@app.command()
def epsilon_ablation(
    n_opt_samples: int = typer.Option(100, help="Number of BO rounds for every run"),
    seed: int = typer.Option(42, help="RNG seed shared by every run"),
    alphas: List[float] = typer.Option([0.75, 1, 1.25], help="alpha values, one stacked block each (top to bottom)"),
    epsilons: List[float] = typer.Option([0.15, 0.1, 0.04], help="epsilon values, one column (and colour) each"),
    out: str = f"{Path().absolute()}/examples/double_bottleneck_epsilon_ablation_3d.png",
):
    Logger.set_verbosity(2)
    j_star = true_optimum()
    original = dict(CONFIG["CSafeOpt"])
    CONFIG["CSafeOpt"].pop("lipschitz", None)
    try:
        blocks = []
        for a in alphas:
            results = {}
            for epsilon in epsilons:
                data, aquisition = run("CSafeOpt", seed, n_opt_samples, alpha=a, epsilon=epsilon)
                results[rf"$\varepsilon={epsilon:g}$"] = (data, aquisition)
                _print_summary(f"alpha={a:g}, epsilon={epsilon:g}", data, j_star)
            blocks.append((rf"$\alpha={a:g}$", results))
    finally:
        CONFIG["CSafeOpt"].clear()
        CONFIG["CSafeOpt"].update(original)
    plot_epsilon_alpha_grid(blocks, out)


@app.command()
def nbar(
    grid_size: int = typer.Option(4000, help="Grid resolution (in normalized [0,1] space) for the gamma_n estimate"),
    n_direct: int = typer.Option(8000, help="Compute gamma_n exactly (via greedy selection) up to this many points"),
):
    Logger.set_verbosity(1)

    lam = CONFIG["model"]["likelihood_noise"]
    covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=1))
    covar.base_kernel.lengthscale = torch.tensor(CONFIG["model"]["lenghtscale"])
    C_lambda = 2.0 / math.log(1.0 + 1.0 / lam)

    probe = build_aquisition("CSafeOpt", CONFIG["dim_obs"], Data())
    rkhs_bound, noise_proxy, delta = probe.rkhs_bound, probe.noise_proxy, probe.delta

    def beta_fn(gamma: float) -> float:
        coef = rkhs_bound + noise_proxy * math.sqrt(2.0 * (gamma + 1.0 + math.log(1.0 / delta)))
        return coef**2

    print(f"C_lambda={C_lambda:.4f}  (lambda={lam}, lengthscale={CONFIG['model']['lenghtscale']})")
    print(f"rkhs_bound={rkhs_bound}, noise_proxy={noise_proxy}, delta={delta}")

    print(f"Computing gamma_n via greedy selection up to n={n_direct} on a {grid_size}-point grid...")
    gamma_seq = _greedy_gamma_sequence(covar, lam, grid_size, n_direct)
    gamma_fn, residual = _fit_gamma_extrapolation(gamma_seq)
    print(f"gamma_{n_direct} (directly computed) = {gamma_seq[-1].item():.4f}")
    print(f"extrapolation fit residual (max abs, back half of computed range): {residual:.4f}")

    for label, bottleneck in [("bottleneck 1", BOTTLENECK_1), ("bottleneck 2", BOTTLENECK_2)]:
        m_star = bottleneck_margin(bottleneck)
        n_bar, rhs_at_nbar = _solve_nbar(gamma_fn, beta_fn, C_lambda, m_star)
        extrapolated = n_bar is not None and n_bar > n_direct
        if n_bar is None:
            print(f"{label}: m*={m_star:.5f}  Nbar > cap, did not converge")
        else:
            note = " (extrapolated beyond n_direct)" if extrapolated else " (within directly-computed range)"
            print(f"{label}: m*={m_star:.5f}  Nbar={n_bar:,.0f}{note}")


if __name__ == "__main__":
    app()
