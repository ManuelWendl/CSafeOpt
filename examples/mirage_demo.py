import random
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
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
from torch import Tensor
from trajectories_3d import COLUMN_FIGSIZE, Landscape, plot_trajectories

import gosafeopt
from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.experiments.environment import Environment
from gosafeopt.experiments.experiment import Experiment
from gosafeopt.models.model import ModelGenerator
from gosafeopt.optim.grid_opt import GridOpt
from gosafeopt.tools.data import Data
from gosafeopt.tools.logger import Logger
from gosafeopt.trainer import Trainer

gosafeopt.device = torch.device("cpu")

app = typer.Typer()

DOMAIN = (0.0, 13.0)
SEED_X = 6.5

LURE_X = 1.5
LURE_AMPLITUDE = 5.0
LURE_WIDTH = 2.6
WALL_X = 5.1
WALL_SCALE = 0.4
WALL_DEPTH = 1.6
WALL_BOUNDARY_X = 4.9994

BOTTLENECK = (7.75, 10.75)
BOTTLENECK_CENTER = 9.25
BOTTLENECK_WIDTH = 0.5
BOTTLENECK_DEPTH = 0.83
LOCAL_PEAK_X = 11.5
LOCAL_AMPLITUDE = 3.0
LOCAL_WIDTH = 0.6

SEED_BUMP_AMPLITUDE = 0.6
SEED_BUMP_WIDTH = 0.4


def reward_fn(x: np.ndarray) -> np.ndarray:
    return (
        SEED_BUMP_AMPLITUDE * np.exp(-((x - SEED_X) ** 2) / (2 * SEED_BUMP_WIDTH**2))
        + LURE_AMPLITUDE * np.exp(-((x - LURE_X) ** 2) / (2 * LURE_WIDTH**2))
        + LOCAL_AMPLITUDE * np.exp(-((x - LOCAL_PEAK_X) ** 2) / (2 * LOCAL_WIDTH**2))
        - 0.3
    )


def constraint_fn(x: np.ndarray) -> np.ndarray:
    wall = WALL_DEPTH / (1.0 + np.exp(-(WALL_X - x) / WALL_SCALE))
    bottleneck = BOTTLENECK_DEPTH * np.exp(-((x - BOTTLENECK_CENTER) ** 2) / (2 * BOTTLENECK_WIDTH**2))
    return 0.9 - wall - bottleneck


class MirageEnv(Environment):
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
        "lenghtscale": [0.08],
        "normalize_input": True,
        "normalize_output": False,
        "outputscale": 0.6931471805599453,
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
    "CSafeOpt": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.1, "alpha": 0.75, "zeta": 0.01, "lipschitz": 1.5},
    "CSafeOptSimple": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.3, "alpha": 0.75, "zeta": 0.01},
    "GoSafeOpt": {"scale_beta": 1.0, "beta": 9, "n_max_local": 5, "n_max_global": 3},
    "GoOSE": {"scale_beta": 1.0, "beta": 9, "lipschitz": 1.5, "epsilon": 0.1},
    "StageOpt": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.1},
    "ISE-BO": {"scale_beta": 1.0, "beta": 9},
}


def build_aquisition(name: str, dim_obs: int, data: Data, alpha: Optional[float] = None) -> BaseAquisition:
    if name == "SafeOpt":
        return GrowingSafeOpt(**CONFIG["SafeOpt"], dim_obs=dim_obs)
    elif name == "SafeUCB":
        return GrowingSafeUCB(**CONFIG["SafeUCB"], dim_obs=dim_obs)
    elif name == "CSafeOpt":
        kwargs = dict(CONFIG["CSafeOpt"])
        if alpha is not None:
            kwargs["alpha"] = alpha
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
    name: str, seed: int, n_opt_samples: int, alpha: Optional[float] = None, initial_point: Optional[tuple] = None,
    aquisition_override: Optional[BaseAquisition] = None, data: Optional[Data] = None,
) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    reset_global_state()

    data = Data() if data is None else data
    x_safe = torch.tensor([[SEED_X if initial_point is None else initial_point[0]]])

    environment = MirageEnv(render_mode=None)
    experiment = Experiment(CONFIG, environment, data=data, backup=None)

    trainer = Trainer(
        dim_params=CONFIG["dim_params"],
        dim_obs=CONFIG["dim_obs"],
        n_opt_samples=n_opt_samples,
        show_progress=False,
        refit_interval=0,
        data=data,
    )

    aquisition = aquisition_override if aquisition_override is not None else build_aquisition(
        name, CONFIG["dim_obs"], data, alpha=alpha
    )

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


def unreachable_optimum(resolution: int = 200_000) -> float:
    xs = np.linspace(DOMAIN[0], DOMAIN[1], resolution)
    return float(reward_fn(xs).max())


def true_optimum(resolution: int = 200_000) -> float:
    xs = np.linspace(DOMAIN[0], DOMAIN[1], resolution)
    safe = constraint_fn(xs) > 0
    return float(reward_fn(xs)[safe].max())


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
    ax_landscape.axvspan(DOMAIN[0], WALL_BOUNDARY_X, color="lightcoral", alpha=0.35, zorder=0)
    ax_landscape.axvspan(*BOTTLENECK, color="gainsboro", alpha=0.6, zorder=0)
    ax_landscape.axvline(SEED_X, color="black", linestyle=":", linewidth=1.2)
    ax_landscape.set_xlim(*DOMAIN)

    ax_landscape.set_xlabel(r"$x$")
    ax_landscape.set_ylabel("value")
    ax_landscape.set_title("Mirage (red, unsafe) vs. honest bottleneck (gray, safe)")
    _legend(ax_landscape, curve_names, {curve_names[0]: (PALETTE[0], None), curve_names[1]: (PALETTE[7], None)})

    trace_df = pd.concat(
        [
            pd.DataFrame({"round": np.arange(data.train_x.shape[0]), "chosen_x": data.train_x[:, 0].numpy(), "name": n})
            for n, (data, _aq) in results.items()
        ],
        ignore_index=True,
    )
    _plot_series(ax_trace, trace_df, "round", "chosen_x", _draw_order(names), style)
    ax_trace.axhspan(DOMAIN[0], WALL_BOUNDARY_X, color="lightcoral", alpha=0.35, zorder=0)
    ax_trace.axhspan(*BOTTLENECK, color="gainsboro", alpha=0.6, zorder=0)
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
        crossings=(BOTTLENECK[1],),
        strips=((DOMAIN[0], WALL_BOUNDARY_X, "lightcoral", "unsafe (mirage)"), (*BOTTLENECK, "gainsboro", "bottleneck")),
        z_ticks=(0, 2, 4),
    )


def plot_mirage(results: dict, out_path: str, trajectories_3d: bool = False, narrow: bool = False):
    _apply_theme()

    names = _legend_order(list(results.keys()))
    style = {n: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, n in enumerate(names)}
    j_star = true_optimum()
    j_mirage = unreachable_optimum()

    if trajectories_3d:
        width, height = plt.rcParams["figure.figsize"]
        fig = plt.figure(figsize=COLUMN_FIGSIZE if narrow else (width, 1.11 * height))
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.8, 1] if narrow else [1.65, 1])
        plot_trajectories(top, results, names, style, _landscape(j_star), compact=narrow)
        ax_regret, ax_threshold = bottom.subplots(1, 2)
    else:
        fig, ((ax_landscape, ax_trace), (ax_regret, ax_threshold)) = plt.subplots(2, 2)
        _plot_landscape_and_trace(ax_landscape, ax_trace, results, names, style)

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
    if narrow:
        ax_regret.set_title(rf"$J^\star_{{\mathrm{{safe}}}} = {j_star:.3f}$")
    elif trajectories_3d:
        ax_regret.set_title(rf"$J^\star_{{\mathrm{{safe}}}} = {j_star:.3f}$ (mirage $= {j_mirage:.3f}$)")
    else:
        ax_regret.set_title(
            rf"$R_N = \sum_t (J^\star_{{\mathrm{{safe}}}} - f(x_t))$, $J^\star_{{\mathrm{{safe}}}} = {j_star:.3f}$"
            rf" (mirage $= {j_mirage:.3f}$)"
        )
    if not trajectories_3d:
        _legend(ax_regret, names, style)

    threshold_frames = []
    crossing_lines = []
    for name, (data, aquisition) in results.items():
        history = getattr(aquisition, "threshold_history", None)
        if not history:
            continue
        rounds_h, _tau_h, eta_h = zip(*history)
        threshold_frames.append(pd.DataFrame({"round": rounds_h, "value": eta_h, "name": name}))

        chosen_x = data.train_x[:, 0].numpy()
        crossed_mask = chosen_x > BOTTLENECK[1]
        if crossed_mask.any():
            first_cross_episode = int(np.argmax(crossed_mask))
            if first_cross_episode >= 1:
                crossing_lines.append((name, first_cross_episode, history[first_cross_episode - 1][2]))

    if threshold_frames:
        threshold_df = pd.concat(threshold_frames, ignore_index=True)
        gate_names = [n for n in names if n in threshold_df["name"].unique()]
        _plot_series(
            ax_threshold, threshold_df, "round", "value", _draw_order(gate_names), style,
            **{"pointsize": 3.5, **series_style},
        )

        for name, _first_cross_episode, eta_at_crossing in crossing_lines:
            ax_threshold.axhline(eta_at_crossing, color=style[name][0], linestyle="-.", linewidth=1.6, alpha=0.7)

        if not trajectories_3d:
            _legend(ax_threshold, gate_names, style)

    ax_threshold.set_xlabel("round")
    ax_threshold.set_ylabel(r"$\eta_t$")
    ax_threshold.set_title(r"$\eta_t = 2\varepsilon\,\beta_t^{1/2-\alpha}$")

    for ax in fig.get_axes():
        if ax.name != "3d":
            _style_axis(ax)

    fig.savefig(out_path)
    pdf_path = str(Path(out_path).with_suffix(".pdf"))
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved mirage plot to {out_path} (and {pdf_path})")


def _print_summary(name: str, data, j_star: float):
    chosen_x = data.train_x[:, 0].numpy()
    reward = data.train_y[:, 0].numpy()
    crossed_bottleneck = (chosen_x > BOTTLENECK[1]).any()
    violated_wall = (chosen_x < WALL_BOUNDARY_X).any()
    cumulative_regret = (j_star - reward).sum()
    print(
        f"{name}: crossed bottleneck: {crossed_bottleneck}, "
        f"furthest left reached: {chosen_x.min():.2f} (wall at {WALL_BOUNDARY_X:.2f}, violated: {violated_wall}), "
        f"furthest right reached: {chosen_x.max():.2f}, "
        f"cumulative regret after {len(reward)} rounds: {cumulative_regret:.2f}"
    )


class GateSnapshotCSafeOpt(InstrumentedCSafeOpt):

    def __init__(self, *args, snapshot_rounds=(), data: Optional[Data] = None, resolution: int = 2000, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_rounds = set(snapshot_rounds)
        self.data = data
        self.grid = torch.linspace(DOMAIN[0], DOMAIN[1], resolution, dtype=torch.float64).reshape(-1, 1)
        self.snapshots: dict = {}

    def evaluate(self, x: Tensor, step: int = 0) -> Tensor:
        scores = super().evaluate(x, step)
        if self.t in self.snapshot_rounds:
            with torch.no_grad():
                terms = self.acquisition_terms(self.grid)
            self.snapshots[self.t] = {
                "x": self.grid[:, 0].numpy(),
                "safe": terms["safe"].numpy(),
                "expander": terms["expander"].numpy(),
                "ut": terms["ut"].numpy(),
                "gate": (terms["kappa"] * terms["normalized_gate"]).numpy(),
                "kappa": float(terms["kappa"]),
                "tau": float(terms["tau"]),
                "lower": terms["lower"].numpy(),
                "upper": terms["upper"].numpy(),
                "observed": self.data.train_x[:, 0].numpy().copy() if self.data is not None else None,
                "observed_y": self.data.train_y.numpy().copy() if self.data is not None else None,
            }
        return scores


def plot_gate_acquisition(snapshots: dict, out_path: str, x_max: Optional[float] = None) -> None:
    from matplotlib.patches import Patch

    ucb_color, gate_color, reward_color, constraint_color = PALETTE[2], PALETTE[0], "0.35", PALETTE[7]
    uncertified_color = "0.93"
    truth_style = {"linewidth": 0.9, "linestyle": (0, (3, 2)), "alpha": 0.8, "zorder": 2}
    label_box = {"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}
    _apply_theme()
    rounds = sorted(snapshots)
    fig, axes = plt.subplots(len(rounds), 1, sharex=True, squeeze=False,
                             figsize=(COLUMN_FIGSIZE[0], 0.5 + 1.25 * len(rounds)), constrained_layout=True)
    axes = axes[:, 0]

    certified = np.concatenate([snapshots[t]["x"][snapshots[t]["safe"]] for t in rounds])
    x_lo = max(DOMAIN[0], certified.min() - 0.7)
    x_hi = min(DOMAIN[1], max(certified.max() + 1.4, LOCAL_PEAK_X + 0.35))
    if x_max is not None:
        x_hi = min(x_hi, x_max)
    xs = np.linspace(x_lo, x_hi, 800)

    for ax, t in zip(axes, rounds):
        snap = snapshots[t]
        x, safe, lower, upper = snap["x"], snap["safe"], snap["lower"], snap["upper"]
        ut = np.where(safe, snap["ut"], np.nan)
        total = ut + np.where(safe, snap["gate"], np.nan)
        y_min = min(constraint_fn(xs).min(), -0.5) - 0.25
        y_max = max(np.nanmax(total), reward_fn(np.array(LOCAL_PEAK_X))) + 0.7

        edges = np.flatnonzero(np.diff(np.r_[0, (~safe).astype(int), 0]))
        dx = x[1] - x[0]
        spans = [(x[start] - dx / 2, x[stop - 1] + dx / 2) for start, stop in zip(edges[::2], edges[1::2])]
        for lo, hi in spans:
            ax.axvspan(lo, hi, color=uncertified_color, linewidth=0, zorder=0)

        ax.fill_between(x, lower[:, 1], upper[:, 1], color=constraint_color, alpha=0.18, linewidth=0, zorder=1)
        ax.axhline(0.0, color="black", linewidth=0.6, zorder=2)
        ax.plot(xs, constraint_fn(xs), color=constraint_color, **truth_style)

        ax.fill_between(x, lower[:, 0], upper[:, 0], color=reward_color, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(xs, reward_fn(xs), color=reward_color, **truth_style)

        ax.fill_between(x, ut, total, where=safe & (snap["gate"] > 0), color=gate_color, alpha=0.45,
                        linewidth=0, zorder=3)
        ax.plot(x, total, color=gate_color, linewidth=1.0, zorder=3)
        ax.fill_between(x, 0.0, 0.035, where=safe & (snap["gate"] > 0), color=gate_color, linewidth=0,
                        transform=ax.get_xaxis_transform(), zorder=5)
        ax.plot(x, ut, color=ucb_color, linewidth=1.8, zorder=4)
        i_ucb, i_total = np.nanargmax(ut), np.nanargmax(total)
        ax.plot(x[i_ucb], ut[i_ucb], "o", markersize=5, markerfacecolor="white", markeredgecolor=ucb_color,
                markeredgewidth=1.2, zorder=6)
        ax.plot(x[i_total], total[i_total], "*", markersize=8, color=gate_color, markeredgecolor="black",
                markeredgewidth=0.5, zorder=6)
        i_min = np.nanargmin(ut)
        width = x_hi - x_lo
        x_gate_arrow, x_range_arrow = x[i_total] + 0.02 * width, x[i_total] + 0.22 * width
        range_style = {"color": "0.5", "linewidth": 0.4, "linestyle": (0, (2, 1.5)), "zorder": 5}
        ax.hlines(ut[i_ucb], x[i_ucb], x_range_arrow, **range_style)
        ax.hlines(ut[i_min], x[i_min], x_range_arrow, **range_style)
        double_arrow = {"arrowstyle": "<->", "linewidth": 0.8, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 6}
        ax.annotate("", (x_gate_arrow, total[i_total]), xytext=(x_gate_arrow, ut[i_total]),
                    arrowprops={**double_arrow, "color": gate_color}, zorder=7)
        ax.annotate("", (x_range_arrow, ut[i_ucb]), xytext=(x_range_arrow, ut[i_min]),
                    arrowprops={**double_arrow, "color": ucb_color}, zorder=7)
        pieces = [TextArea(text, textprops={"fontsize": 7, "color": color})
                  for text, color in ((r"$\kappa_t$", gate_color), (r"$>$", "0.2"), (r"$\Delta u_t$", ucb_color))]
        ax.add_artist(AnnotationBbox(
            HPacker(children=pieces, align="baseline", sep=2.5, pad=0),
            (0.5 * (x_gate_arrow + x_range_arrow), 0.5 * (ut[i_ucb] + ut[i_min])),
            frameon=True, pad=0.25, zorder=7,
            bboxprops={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        ))

        arrow = {"arrowstyle": "-", "color": "0.3", "linewidth": 0.5, "shrinkA": 0, "shrinkB": 3}
        ax.annotate("UCB", (x[i_ucb], ut[i_ucb]), xytext=(8, 10), textcoords="offset points",
                    ha="left", fontsize=7, color=ucb_color, arrowprops=arrow, bbox=label_box, zorder=7)
        ax.annotate(r"$A_t^{(\alpha)}(x)$", (x[i_total], total[i_total]), xytext=(-10, 6),
                    textcoords="offset points", ha="right", va="bottom", fontsize=7,
                    color=gate_color, arrowprops=arrow, bbox=label_box, zorder=7)

        if snap["observed"] is not None:
            for i, color in ((0, reward_color), (1, constraint_color)):
                ax.plot(snap["observed"], snap["observed_y"][:, i], "o", markersize=3, color=color,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=5)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("value")
        if len(rounds) > 1:
            ax.set_title(rf"Round $t={t}$")
        _style_axis(ax)
        ax.grid(False)
        print(f"round {t}: kappa={snap['kappa']:.3f}, tau={snap['tau']:.4f}, "
              f"S_t covers [{x[safe].min():.2f}, {x[safe].max():.2f}], "
              f"UCB argmax x={x[i_ucb]:.2f}, gated argmax x={x[i_total]:.2f}")
    axes[-1].set_xlabel(r"$x$")

    handles = [
        Line2D([], [], color=reward_color, **{**truth_style, "zorder": None}, label=r"$f(x)$"),
        Patch(facecolor=reward_color, alpha=0.35, label="reward GP"),
        Line2D([], [], color=constraint_color, **{**truth_style, "zorder": None}, label=r"$c(x)$"),
        Patch(facecolor=constraint_color, alpha=0.35, label="constraint GP"),
        Line2D([], [], color=ucb_color, linewidth=1.8, label=r"$u_t(x)$"),
        Patch(facecolor=gate_color, alpha=0.6, label=r"$\kappa_t\,\bar q_t^{(\alpha)}(x)$"),
        Line2D([], [], color=gate_color, linewidth=3.0, solid_capstyle="butt", label=r"$q_t^{(\alpha)}(x) > 0$"),
        Patch(facecolor=uncertified_color, edgecolor="0.7", linewidth=0.4, label="not certified"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, fontsize=7,
               handlelength=1.3, columnspacing=0.9, handletextpad=0.4, labelspacing=0.3)

    fig.savefig(out_path)
    pdf_path = str(Path(out_path).with_suffix(".pdf"))
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved gated-acquisition plot to {out_path} (and {pdf_path})")


@app.command()
def gate_acquisition(
    rounds: List[int] = typer.Option([10], min=1, help="Repeat --rounds for each panel"),
    seed: int = typer.Option(42, help="RNG seed"),
    out: str = f"{Path().absolute()}/examples/mirage_gate_acquisition.png",
    lengthscale: Optional[float] = typer.Option(None, help="Override the (reward) GP lengthscale, normalized"),
    constraint_lengthscale: Optional[float] = typer.Option(None, help="Separate constraint GP lengthscale, normalized"),
    alpha: Optional[float] = typer.Option(None, help="Override CSafeOpt's alpha"),
    epsilon: Optional[float] = typer.Option(None, help="Override CSafeOpt's epsilon"),
    x_max: Optional[float] = typer.Option(None, help="Right edge of the x-axis (default: just past the optimum)"),
):
    Logger.set_verbosity(2)
    if lengthscale is not None:
        CONFIG["model"]["lenghtscale"] = [lengthscale]
    if constraint_lengthscale is not None:
        CONFIG["model"]["constraint_lengthscale"] = [constraint_lengthscale]
    kwargs = dict(CONFIG["CSafeOpt"])
    if alpha is not None:
        kwargs["alpha"] = alpha
    if epsilon is not None:
        kwargs["epsilon"] = epsilon
    print(f"model: {CONFIG['model']}, CSafeOpt: {kwargs}")
    data = Data()
    aquisition = GateSnapshotCSafeOpt(**kwargs, dim_obs=CONFIG["dim_obs"], snapshot_rounds=rounds, data=data)
    run("CSafeOpt", seed, max(rounds) + 1, aquisition_override=aquisition, data=data)
    plot_gate_acquisition(aquisition.snapshots, out, x_max=x_max)


@app.command()
def mirage(
    n_opt_samples: int = typer.Option(200, help="Number of BO rounds for every run"),
    seed: int = typer.Option(42, help="RNG seed shared by every run"),
    algorithms: List[str] = typer.Option(
        ["SafeOpt", "SafeUCB", "CSafeOpt", "GoOSE"], help="Which acquisitions to run"
    ),
    out: str = f"{Path().absolute()}/examples/mirage.png",
    trajectories_3d: bool = typer.Option(False, help="Replace the landscape/trace panels by one 3D panel per run"),
    narrow: bool = typer.Option(False, help="Size the figure for one column of a two-column page"),
):
    Logger.set_verbosity(2)
    j_star = true_optimum()

    results = {}
    for name in algorithms:
        data, aquisition = run(name, seed, n_opt_samples)
        results[name] = (data, aquisition)
        _print_summary(name, data, j_star)

    plot_mirage(results, out, trajectories_3d=trajectories_3d, narrow=narrow)


if __name__ == "__main__":
    app()
