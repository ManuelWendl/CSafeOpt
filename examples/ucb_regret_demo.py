"""Sublinear-regret demo: plain GP-UCB, no safety constraint at all, on a
synthetic multi-modal 1D objective -- animated round by round to make the
textbook GP-UCB regret guarantee (Srinivas et al. 2010; using the tighter
Chowdhury & Gopalan 2017 beta_t sequence this repo's own Growing*
acquisitions already rely on) visible rather than just asserted: cumulative
regret R_N = sum_t (f* - f(x_t)) bends below the straight line a *constant*
per-round regret would trace, and average regret R_N/N trends toward 0 --
literally the definition of sublinear regret.

Deliberately standalone and unconstrained: no safe set, no constraint
channel, no wall/bottleneck -- just plain UCB (mean + confidence half-width,
nothing else) wrapped in bottleneck_demo.py's ChowdhuryGopalanBeta
growing-beta mixin, reusing that file's paper-style plotting the same way
every other demo in this suite does.

gosafeopt.aquisitions.ucb.UCB itself is not used directly: its evaluate()
is declared as evaluate(self, x) with no `step` argument, but both
GridOpt.optimize and SwarmOpt.optimize call `aquisition.evaluate(x, step)` --
a TypeError as soon as it's actually driven through either optimizer. It also
never overrides is_internal_step(), so it inherits BaseAquisition's default
(True at step=0), which makes BaseOptimizer.optimize_steps() discard every
candidate and leave x=None, crashing next_params() next. Both are
pre-existing bugs in ucb.py, unrelated to this script; rather than patching
that file, GrowingUCB below duplicates its one-line scoring locally with a
correct signature (same fix SafeUCB/Goose already apply to the same two
methods).

Usage:
    python examples/ucb_regret_demo.py
"""

import random
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from bottleneck_demo import ChowdhuryGopalanBeta, MARKERS, PALETTE, _apply_theme, _style_axis, reset_global_state
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from torch import Tensor
from tueplots import figsizes

import gosafeopt
from gosafeopt.aquisitions.ucb import UCB
from gosafeopt.experiments.environment import Environment
from gosafeopt.experiments.experiment import Experiment
from gosafeopt.models.model import ModelGenerator
from gosafeopt.optim.grid_opt import GridOpt
from gosafeopt.tools.data import Data
from gosafeopt.tools.logger import Logger
from gosafeopt.trainer import Trainer

gosafeopt.device = torch.device("cpu")

app = typer.Typer()

DOMAIN = (0.0, 10.0)
SEED_X = 0.2  # arbitrary initial sample -- unconstrained, so no "safety" meaning here, just a starting point

# Three well-separated bumps of different heights: a couple of decoys plus a
# clear single global maximum, so GP-UCB actually has some exploring to do
# before it can settle into (mostly) exploiting the true optimum.
def objective_fn(x: np.ndarray) -> np.ndarray:
    return (
        1.0 * np.exp(-((x - 1.5) ** 2) / (2 * 0.6**2))
        + 1.3 * np.exp(-((x - 4.0) ** 2) / (2 * 0.5**2))
        + 1.8 * np.exp(-((x - 7.0) ** 2) / (2 * 0.8**2))
        - 0.3
    )


class UnconstrainedEnv(Environment):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(None, render_mode)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1), {}

    def step(self, k):
        x = float(k[0])
        # Experiment.rollout divides by len(trajectory) == 2 for a single-step
        # episode; double the raw value so data.train_y matches objective_fn.
        reward = np.array([2 * objective_fn(x)])
        return np.array([x]), reward, True, False, {}


class GrowingUCB(ChowdhuryGopalanBeta, UCB):
    """Plain GP-UCB (mean + growing-beta confidence half-width, no safety
    machinery at all) -- see module docstring for why this overrides
    UCB.evaluate()/is_internal_step() rather than inheriting them as-is.
    """

    def is_internal_step(self, step: int = 0) -> bool:  # noqa: ARG002
        return False

    def evaluate(self, x: Tensor, step: int = 0) -> Tensor:  # noqa: ARG002
        posterior = self.model_posterior(x)
        _, u = self.get_confidence_interval(posterior)  # noqa: E741
        return u[:, 0]


CONFIG = {
    "log_video": False,
    "log_plots": False,
    "dim_obs": 1,
    "dim_params": 1,
    "dim_context": 0,
    "dim_model": 1,
    "domain_start": [DOMAIN[0]],
    "domain_end": [DOMAIN[1]],
    "model": {
        "lenghtscale": [0.15],
        "normalize_input": True,
        "normalize_output": False,
        "likelihood_noise": 1e-4,
    },
    "Optimization": {
        "set_size": 4000,
        "set_init": "random",
        "max_global_steps_without_progress_tolerance": 0.9,
        "max_global_steps_without_progress": 10_000,  # effectively disabled
    },
    "UCB": {"scale_beta": 1.0, "beta": 9},
}


def _fresh_model_generator() -> ModelGenerator:
    return ModelGenerator(
        **CONFIG["model"],
        domain_start=Tensor(CONFIG["domain_start"]),
        domain_end=Tensor(CONFIG["domain_end"]),
        dim_obs=CONFIG["dim_obs"],
        dim_model=CONFIG["dim_model"],
    )


def run(seed: int, n_opt_samples: int) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    reset_global_state()

    data = Data()
    x_safe = torch.tensor([[SEED_X]])

    environment = UnconstrainedEnv(render_mode=None)
    experiment = Experiment(CONFIG, environment, data=data, backup=None)

    trainer = Trainer(
        dim_params=CONFIG["dim_params"],
        dim_obs=CONFIG["dim_obs"],
        n_opt_samples=n_opt_samples,
        show_progress=False,
        refit_interval=0,
        data=data,
    )

    aquisition = GrowingUCB(**CONFIG["UCB"], dim_obs=CONFIG["dim_obs"])

    model = _fresh_model_generator()

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

    Logger.info("=== Running GP-UCB (unconstrained) ===")
    trainer.train(experiment, model, optimizer, aquisition, x_safe)
    return data, aquisition


def true_optimum(resolution: int = 200_000) -> float:
    xs = np.linspace(*DOMAIN, resolution)
    return float(objective_fn(xs).max())


def _precompute_frames(data, aquisition, xs_t: Tensor, frame_rounds: list) -> list:
    model_generator = _fresh_model_generator()
    frames = []
    for r in frame_rounds:
        k = min(r, data.train_x.shape[0])
        sub_data = Data(train_x=data.train_x[:k], train_y=data.train_y[:k])
        model = model_generator.generate(sub_data)
        aquisition.update_model(model)
        posterior = aquisition.model_posterior(xs_t)
        l, u = aquisition.get_confidence_interval(posterior)  # noqa: E741
        mean = posterior.mean.reshape(-1, aquisition.dim_obs)
        frames.append(
            {
                "k": k,
                "mean": mean[:, 0].detach().numpy(),
                "lo": l[:, 0].detach().numpy(),
                "hi": u[:, 0].detach().numpy(),
            }
        )
    return frames


def animate_regret(
    data,
    aquisition,
    j_star: float,
    out_path: str,
    n_frames: int = 60,
    fps: int = 5,
    dpi: int = 300,
    grid_size: int = 500,
    height_scale: float = 1.0,
):
    """Render a 1x2 GIF: objective+GP+samples (left), cumulative regret
    against a linear-growth reference (right) -- R_N visibly bending below
    that reference and flattening out is what makes sublinearity visible.
    """
    _apply_theme()
    figsize_rc = figsizes.iclr2023(nrows=1, ncols=2)
    width, height = figsize_rc["figure.figsize"]
    figsize_rc["figure.figsize"] = (width, height * height_scale)
    plt.rcParams.update(figsize_rc)

    obj_color = PALETTE[0]

    fig, (ax_obj, ax_regret) = plt.subplots(1, 2)

    xs = np.linspace(*DOMAIN, grid_size)
    xs_t = torch.tensor(xs, dtype=data.train_x.dtype).reshape(-1, 1)
    true_obj = objective_fn(xs)

    max_round = data.train_x.shape[0]
    frame_rounds = sorted(set(np.linspace(1, max_round, min(n_frames, max_round)).astype(int)))
    frames = _precompute_frames(data, aquisition, xs_t, frame_rounds)

    # --- objective panel: static background ---------------------------------
    ax_obj.plot(xs, true_obj, color=obj_color, linewidth=1.2, linestyle="--", zorder=2)
    ax_obj.axvline(SEED_X, color="black", linestyle=":", linewidth=1.0, zorder=1)
    ax_obj.set_xlim(*DOMAIN)
    ax_obj.set_xlabel(r"$x$")
    ax_obj.set_ylabel(r"$f(x)$")
    ax_obj.set_title(r"GP-UCB: objective, GP $\pm\,\beta_t^{1/2}\sigma$")

    all_bounds = np.concatenate([f["lo"] for f in frames] + [f["hi"] for f in frames] + [true_obj])
    pad = 0.05 * (all_bounds.max() - all_bounds.min())
    ax_obj.set_ylim(-1, all_bounds.max() + pad)

    (mean_line,) = ax_obj.plot([], [], color=obj_color, linewidth=2.0, zorder=4)
    scatter = ax_obj.scatter(
        [], [], s=16, color=obj_color, marker=MARKERS[0], zorder=5, linewidths=0.4, edgecolors="white"
    )
    band = {"patch": None}

    legend_handles = [
        Line2D([0], [0], color=obj_color, linestyle="--", linewidth=1.2, label=r"true $f(x)$"),
        Line2D([0], [0], color=obj_color, linewidth=2.0, label="GP mean"),
    ]
    ax_obj.legend(handles=legend_handles, loc="upper left", fontsize=6, frameon=False)

    # --- cumulative regret vs. a linear-growth reference --------------------
    regret_curve = np.cumsum(j_star - data.train_y[:, 0].numpy())
    (regret_line,) = ax_regret.plot([], [], color=obj_color, linewidth=1.8, zorder=2, label=r"$R_N$")
    regret_point = ax_regret.scatter([], [], s=18, color=obj_color, marker=MARKERS[0], zorder=3)
    ax_regret.set_xlim(0, max_round)
    ax_regret.set_ylim(0, max(regret_curve.max(), 1e-6) * 1.05)
    ax_regret.set_xlabel("round")
    ax_regret.set_ylabel(r"cumulative regret $R_N$")
    ax_regret.set_title(r"$R_N = \sum_t (f^\star - f(x_t))$")
    ax_regret.legend(loc="lower right", fontsize=6, frameon=False)

    # Average regret R_N/N -- no dedicated panel, just reported in the
    # per-frame title text (R_N/N -> 0 is literally what "sublinear regret"
    # means).
    avg_regret_curve = regret_curve / np.arange(1, max_round + 1)

    for ax in fig.get_axes():
        _style_axis(ax)

    round_text = fig.suptitle("")

    def update(frame_idx):
        frame = frames[frame_idx]
        k = frame["k"]

        mean_line.set_data(xs, frame["mean"])

        if band["patch"] is not None:
            band["patch"].remove()
        band["patch"] = ax_obj.fill_between(xs, frame["lo"], frame["hi"], color=obj_color, alpha=0.25, linewidth=0, zorder=3)

        scatter.set_offsets(np.column_stack([data.train_x[:k, 0].numpy(), data.train_y[:k, 0].numpy()]))

        regret_line.set_data(np.arange(k), regret_curve[:k])
        regret_point.set_offsets(np.array([[k - 1, regret_curve[k - 1]]]))

        round_text.set_text(f"round {k}/{max_round}")
        return [mean_line, band["patch"], scatter, regret_line, regret_point, round_text]

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved GP-UCB regret GIF to {out_path}")


@app.command()
def regret(
    n_opt_samples: int = typer.Option(20, help="Number of BO rounds"),
    seed: int = typer.Option(42, help="RNG seed"),
    out: str = f"{Path().absolute()}/examples/ucb_regret.gif",
    n_frames: int = typer.Option(60, help="Number of animation frames (rounds are subsampled to this count)"),
    fps: int = typer.Option(1, help="Playback speed of the GIF"),
    dpi: int = typer.Option(300, help="Resolution of the GIF"),
    grid_size: int = typer.Option(500, help="Number of x points used to draw the GP mean/confidence band"),
    height_scale: float = typer.Option(1.0, help="Multiplier on the figure height"),
):
    Logger.set_verbosity(2)
    j_star = true_optimum()

    data, aquisition = run(seed, n_opt_samples)

    reward = data.train_y[:, 0].numpy()
    cumulative_regret = float((j_star - reward).sum())
    print(f"true optimum f*={j_star:.3f}")
    print(f"best found: {reward.max():.3f}, cumulative regret after {len(reward)} rounds: {cumulative_regret:.2f}")
    print(f"average regret R_N/N at final round: {cumulative_regret / len(reward):.4f}")

    animate_regret(
        data,
        aquisition,
        j_star,
        out,
        n_frames=n_frames,
        fps=fps,
        dpi=dpi,
        grid_size=grid_size,
        height_scale=height_scale,
    )


if __name__ == "__main__":
    app()
