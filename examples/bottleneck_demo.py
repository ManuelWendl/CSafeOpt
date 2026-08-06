"""A hand-crafted 1D safe-optimization benchmark with a genuine bottleneck.

The safe seed already sits at a mediocre local optimum. Reaching the much
better optimum requires crossing a corridor that is low-reward (so plain
reward-greedy UCB has no incentive to go there) but only barely safe (so it
cannot be crossed without deliberately sampling near the safety boundary to
shrink the confidence interval there first). This is the same structure as
the "three-point safe chain" in Appendix A of the CSafeOpt write-up.
"""

import math
import random
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn.objects as so
import torch
import typer
from matplotlib.lines import Line2D
from seaborn import axes_style
from tueplots import bundles, figsizes

import gosafeopt
from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.aquisitions.cum_safe_opt import CSafeOpt
from gosafeopt.aquisitions.go_safe_opt import GoSafeOpt
from gosafeopt.aquisitions.goose import Goose
from gosafeopt.aquisitions.safe_opt import SafeOpt
from gosafeopt.aquisitions.safe_ucb import SafeUCB
from gosafeopt.experiments.environment import Environment
from gosafeopt.experiments.experiment import Experiment
from gosafeopt.models.model import ModelGenerator
from gosafeopt.optim.grid_opt import GridOpt
from gosafeopt.optim.safe_set import SafeSet
from gosafeopt.tools.data import Data
from gosafeopt.tools.logger import Logger
from gosafeopt.trainer import Trainer
from torch import Tensor

# GridOpt/SwarmOpt build domain-bound tensors on the CPU and never move them onto
# gosafeopt.device; force CPU here so this stays consistent (see examples/compare.py).
gosafeopt.device = torch.device("cpu")

app = typer.Typer()

SEED_X = 0.5
VALLEY = (1.5, 4.0)
DOMAIN = (0.0, 6.0)

# Paper-style plotting: tueplots (ICLR bundle) + seaborn.objects, matching the
# house style used for the ss2r WandB plots (fixed categorical palette, small
# markers on thin lines, LaTeX serif text, light grid, heavier spines).
PALETTE = [
    "#5F4690",
    "#1D6996",
    "#38A6A5",
    "#0F8554",
    "#73AF48",
    "#EDAD08",
    "#E17C05",
    "#CC503E",
    "#94346E",
    "#6F4070",
    "#994E95",
    "#666666",
]
MARKERS = ["o", "x", "^", "s", "*", "D", "v", "P", "<", ">"]

_theme_applied = False


def _apply_theme():
    global _theme_applied
    if _theme_applied:
        return
    theme = bundles.iclr2023()
    so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
    plt.rcParams.update(theme)
    plt.rcParams.update(figsizes.iclr2023(nrows=2, ncols=2))
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    _theme_applied = True


def _style_axis(ax):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _legend(ax, names, style, **kwargs):
    handles = [
        Line2D([0], [0], color=style[n][0], marker=style[n][1], markersize=5, linewidth=1.8) for n in names
    ]
    ax.legend(handles, names, **kwargs)


def _plot_series(ax, df, x, y, names, style, linewidth=1.8, pointsize=4.0, edgewidth=0.5):
    """Draw one continuous line per series (df's "name" column), with point
    markers shown only every 5th round once a series has more than 50 rounds
    -- denser marker sets get visually noisy without adding information the
    line doesn't already show.
    """
    so.Plot(df, x=x, y=y, color="name").add(so.Line(linewidth=linewidth), legend=False).scale(
        color=so.Nominal(values=[style[n][0] for n in names], order=names)
    ).on(ax).plot()

    marker_frames = []
    for n in names:
        sub = df[df["name"] == n]
        step = 5 if len(sub) > 50 else 1
        marker_frames.append(sub.iloc[::step])
    marker_df = pd.concat(marker_frames, ignore_index=True)

    so.Plot(marker_df, x=x, y=y, color="name", marker="name").add(
        so.Dot(pointsize=pointsize, edgewidth=edgewidth), legend=False
    ).scale(
        color=so.Nominal(values=[style[n][0] for n in names], order=names),
        marker=so.Nominal(values=[style[n][1] for n in names], order=names),
    ).on(ax).plot()


def reward_fn(x: np.ndarray) -> np.ndarray:
    return (
        1.2 * np.exp(-((x - SEED_X) ** 2) / (2 * 0.35**2))
        + 3.0 * np.exp(-((x - 5.0) ** 2) / (2 * 1.0**2))
        - 0.3
    )


def constraint_fn(x: np.ndarray) -> np.ndarray:
    return (
        0.9 * np.exp(-((x - SEED_X) ** 2) / (2 * 1.0**2))
        + 0.9 * np.exp(-((x - 5.0) ** 2) / (2 * 1.0**2))
        + 0.15
    )


class BottleneckEnv(Environment):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(None, render_mode)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1), {}

    def step(self, k):
        x = float(k[0])
        # Experiment.rollout divides by len(trajectory), which is 2 for a
        # single-step episode (the reset observation plus this one); double
        # the raw values here so data.train_y matches reward_fn/constraint_fn.
        reward = np.array([2 * reward_fn(x), 2 * constraint_fn(x)])
        return np.array([x]), reward, True, False, {}


class ChowdhuryGopalanBeta:
    rkhs_bound: float = 2.0
    noise_proxy: float = 0.1
    delta: float = 0.1
    noise_variance: float = 1e-4

    def information_gain(self) -> float:
        reward_model = self.model.models[0]
        train_x = reward_model.train_inputs[0]
        with torch.no_grad():
            K = reward_model.forward(train_x).covariance_matrix
        n = K.shape[-1]
        gram = torch.eye(n, dtype=K.dtype) + K / self.noise_variance
        return 0.5 * torch.linalg.slogdet(gram)[1].item()

    def beta_t_coefficient(self) -> float:
        gamma = self.information_gain()
        return self.rkhs_bound + self.noise_proxy * math.sqrt(2.0 * (gamma + 1.0 + math.log(1.0 / self.delta)))

    def get_confidence_interval(self, posterior):
        mean = posterior.mean.reshape(-1, self.dim_obs)
        var = posterior.variance.reshape(-1, self.dim_obs)
        std = torch.sqrt(var.clamp_min(0.0))
        half_width = self.scale_beta * self.beta_t_coefficient() * std
        return mean - half_width, mean + half_width


class GrowingSafeOpt(ChowdhuryGopalanBeta, SafeOpt):
    pass


class GrowingSafeUCB(ChowdhuryGopalanBeta, SafeUCB):
    pass


class GrowingGoSafeOpt(ChowdhuryGopalanBeta, GoSafeOpt):
    pass


class GrowingGoose(ChowdhuryGopalanBeta, Goose):
    pass


class InstrumentedCSafeOpt(CSafeOpt):
    """Records tau_t and eta_t each round for plotting.

    tau_t = epsilon / beta_t^alpha (eq. 29) is the raw threshold the gate
    applies to sigma(x). eta_t = 2*sqrt(beta_t)*tau_t (eq. 43) is the
    confidence-width tolerance it corresponds to -- the quantity Lemma 3
    connects to the comparator class Pi_eta_t actually being targeted.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_history: list[tuple[int, float, float]] = []

    def after_optimization(self) -> None:
        beta_t = self.growing_beta()
        tau_t = self.threshold()
        eta_t = 2.0 * math.sqrt(beta_t) * tau_t
        self.threshold_history.append((self.t, tau_t, eta_t))
        super().after_optimization()


def reset_global_state():
    SafeSet.safe_sets = []
    SafeSet.current_safe_set = 0
    SafeSet.best_sage_set = 0
    SafeSet.y_min = -1e10
    SafeSet.global_y_min = -1e10
    SafeSet.i = 0
    SafeOpt.best_lcb = -1e10
    GoSafeOpt.n = 0


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
        "lenghtscale": [0.6],
        "normalize_input": True,
        "normalize_output": True,
        "likelihood_noise": 1e-4,
    },
    "Optimization": {
        # "safe" mode proposes candidates via a hardcoded 1e-3 covariance jitter
        # around the safe set's mean (base_optimizer.py), tuned for the pendulum
        # example's much smaller parameter scale. On this domain it never
        # reaches the bottleneck at all, so sample the full domain uniformly at
        # random every round instead and let the acquisition decide.
        "set_size": 2000,
        "set_init": "random",
        "max_global_steps_without_progress_tolerance": 0.9,
        "max_global_steps_without_progress": 10_000,  # effectively disabled
    },
    "SafeOpt": {"scale_beta": 1.0, "beta": 9},
    "SafeUCB": {"scale_beta": 1.0, "beta": 9},
    "CSafeOpt": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.146, "alpha": 0.55, "zeta": 0.0},
    "GoSafeOpt": {"scale_beta": 1.0, "beta": 9, "n_max_local": 5, "n_max_global": 3},
    "Goose": {"scale_beta": 1.0, "beta": 9, "lipschitz": 1.0, "epsilon": 0.15},
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
    x_safe = torch.tensor([[SEED_X]])

    environment = BottleneckEnv(render_mode=None)
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


def true_optimum(resolution: int = 200_000) -> float:
    xs = np.linspace(DOMAIN[0], DOMAIN[1], resolution)
    return float(reward_fn(xs).max())


def plot_bottleneck(results: dict, out_path: str):
    _apply_theme()

    names = list(results.keys())
    style = {n: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, n in enumerate(names)}
    j_star = true_optimum()

    fig, ((ax_landscape, ax_trace), (ax_regret, ax_threshold)) = plt.subplots(2, 2)

    # --- landscape ---------------------------------------------------------
    xs = np.linspace(DOMAIN[0], DOMAIN[1], 400)
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
    ax_landscape.axvspan(*VALLEY, color="gainsboro", alpha=0.6, zorder=0)
    ax_landscape.axvline(SEED_X, color="black", linestyle=":", linewidth=1.2)
    ax_landscape.set_xlabel(r"$x$")
    ax_landscape.set_ylabel("value")
    ax_landscape.set_title("True reward / constraint landscape")
    _legend(ax_landscape, curve_names, {curve_names[0]: (PALETTE[0], None), curve_names[1]: (PALETTE[7], None)})

    # --- trace ---------------------------------------------------------------
    trace_df = pd.concat(
        [
            pd.DataFrame({"round": np.arange(data.train_x.shape[0]), "chosen_x": data.train_x[:, 0].numpy(), "name": n})
            for n, (data, _aq) in results.items()
        ],
        ignore_index=True,
    )
    _plot_series(ax_trace, trace_df, "round", "chosen_x", names, style)
    ax_trace.axhspan(*VALLEY, color="gainsboro", alpha=0.6, zorder=0)
    ax_trace.set_xlabel("round")
    ax_trace.set_ylabel(r"chosen $x$")
    ax_trace.set_title("Point evaluated per round")
    _legend(ax_trace, names, style)

    # --- cumulative regret -----------------------------------------------
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
    # Only CSafeOpt has a gate threshold; SafeOpt/SafeUCB/GoSafeOpt have no eta_t.
    threshold_frames = []
    crossing_lines = []
    for name, (data, aquisition) in results.items():
        history = getattr(aquisition, "threshold_history", None)
        if not history:
            continue
        rounds_h, _tau_h, eta_h = zip(*history)
        threshold_frames.append(pd.DataFrame({"round": rounds_h, "value": eta_h, "name": name}))

        # threshold_history[i] holds (round i+1, tau, eta); data.train_x[episode] is the point
        # chosen using that round's threshold (episode 0 is the seed, not chosen via the gate at
        # all, so episode e >= 1 was picked using threshold_history[e - 1]).
        chosen_x = data.train_x[:, 0].numpy()
        crossed_mask = chosen_x > VALLEY[1]
        if crossed_mask.any():
            first_cross_episode = int(np.argmax(crossed_mask))
            if first_cross_episode >= 1:
                crossing_lines.append((name, first_cross_episode, history[first_cross_episode - 1][2]))

    if threshold_frames:
        threshold_df = pd.concat(threshold_frames, ignore_index=True)
        gate_names = [n for n in names if n in threshold_df["name"].unique()]
        _plot_series(ax_threshold, threshold_df, "round", "value", gate_names, style, pointsize=3.5)

        for name, first_cross_episode, eta_at_crossing in crossing_lines:
            ax_threshold.axhline(eta_at_crossing, color=style[name][0], linestyle="-.", linewidth=1.6, alpha=0.7)

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
    print(f"Saved bottleneck plot to {out_path} (and {pdf_path})")


@app.command()
def bottleneck(
    n_opt_samples: int = typer.Option(40, help="Number of BO rounds for every run"),
    seed: int = typer.Option(42, help="RNG seed shared by every run"),
    algorithms: List[str] = typer.Option(
        ["SafeOpt", "SafeUCB", "CSafeOpt", "GoSafeOpt", "Goose"], help="Which acquisitions to run"
    ),
    out: str = f"{Path().absolute()}/examples/bottleneck.png",
):
    Logger.set_verbosity(2)

    j_star = true_optimum()

    results = {}
    for name in algorithms:
        data, aquisition = run(name, seed, n_opt_samples)
        results[name] = (data, aquisition)

        chosen_x = data.train_x[:, 0].numpy()
        reward = data.train_y[:, 0].numpy()
        crossed = (chosen_x > VALLEY[1]).any()
        cumulative_regret = (j_star - reward).sum()
        print(
            f"{name}: crossed the bottleneck: {crossed}, furthest x reached: {chosen_x.max():.2f}, "
            f"cumulative regret after {len(reward)} rounds: {cumulative_regret:.2f}"
        )

    plot_bottleneck(results, out)


@app.command()
def alpha_ablation(
    n_opt_samples: int = typer.Option(60, help="Number of BO rounds for every run"),
    seed: int = typer.Option(42, help="RNG seed shared by every run"),
    alphas: List[float] = typer.Option([0.0, 0.5, 0.75, 1.0, 4.0], help="alpha values to compare for CSafeOpt"),
    out: str = f"{Path().absolute()}/examples/bottleneck_alpha_ablation.png",
):
    """Same benchmark, same figure, but comparing CSafeOpt at different alpha instead of different algorithms."""
    Logger.set_verbosity(2)

    j_star = true_optimum()

    results = {}
    for a in alphas:
        plain_name = f"alpha={a:g}"
        name = rf"$\alpha={a:g}$"
        data, aquisition = run("CSafeOpt", seed, n_opt_samples, alpha=a)
        results[name] = (data, aquisition)

        chosen_x = data.train_x[:, 0].numpy()
        reward = data.train_y[:, 0].numpy()
        crossed = (chosen_x > VALLEY[1]).any()
        cumulative_regret = (j_star - reward).sum()
        print(
            f"{plain_name}: crossed the bottleneck: {crossed}, furthest x reached: {chosen_x.max():.2f}, "
            f"cumulative regret after {len(reward)} rounds: {cumulative_regret:.2f}"
        )

    plot_bottleneck(results, out)


if __name__ == "__main__":
    app()
