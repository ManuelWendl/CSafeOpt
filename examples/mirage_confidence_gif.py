"""Per-round animated GIFs for the mirage benchmark (see mirage_demo.py),
one algorithm at a time: left panel shows the objective/constraint landscape
together with the GP's growing-beta confidence band (Chowdhury-Gopalan,
via bottleneck_demo.py's ChowdhuryGopalanBeta mixin -- the same one every
Growing* acquisition in this file's siblings already uses) and the samples
taken so far; right panel shows cumulative regret against the safe-reachable
optimum, both advancing round by round in lockstep.

Deliberately a standalone script: reuses mirage_demo.run()/CONFIG/reward_fn/
constraint_fn and bottleneck_demo's plotting helpers as-is rather than
touching either file, so mirage_demo.py's own `mirage` command and its output
(mirage.png/.pdf) are unaffected.

"GP-UCB" here is mirage_demo.py's "SafeUCB" acquisition -- the plain
GP-UCB-with-safety-soft-penalty baseline, exactly what both mirage_demo.py's
own module docstring and goose.py's docstring call GOOSE's internal oracle
("plain GP-UCB restricted to the optimistic safe set"), just standing alone
as a full run here rather than wrapped inside GOOSE's expansion logic.

Usage:
    python examples/mirage_confidence_gif.py
    python examples/mirage_confidence_gif.py --algorithms SafeUCB
"""

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from bottleneck_demo import MARKERS, PALETTE, _apply_theme, _style_axis
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from mirage_demo import (
    BOTTLENECK,
    CONFIG,
    DOMAIN,
    SEED_X,
    WALL_BOUNDARY_X,
    constraint_fn,
    reward_fn,
    run,
    true_optimum,
)
from torch import Tensor
from tueplots import figsizes

import gosafeopt
from gosafeopt.models.model import ModelGenerator
from gosafeopt.tools.data import Data
from gosafeopt.tools.logger import Logger

gosafeopt.device = torch.device("cpu")

app = typer.Typer()

# mirage_demo.py's own algorithm key -> the label this script renders it
# under (see module docstring for why "SafeUCB" is displayed as "GP-UCB").
DISPLAY_NAME = {"SafeUCB": "GP-UCB"}

# Low-opacity background wash for the GP's *currently* pessimistically-safe
# region (l_t(x) > fmin on the constraint channel) -- animated, since that
# region grows/shifts round by round as more data comes in.
SAFE_COLOR = PALETTE[3]


def _fresh_model_generator() -> ModelGenerator:
    """A ModelGenerator matching mirage_demo.run()'s own construction exactly
    (same CONFIG["model"]/domain), used to regenerate the model from a
    data *prefix* for each animation frame. Safe to reconstruct per frame
    (no fitting happens: mirage's Trainer runs with refit_interval=0, so the
    kernel hyperparameters are fixed throughout the actual run too -- this
    just replays that same fixed-hyperparameter construction on however many
    rounds a given frame should show).
    """
    return ModelGenerator(
        **CONFIG["model"],
        domain_start=Tensor(CONFIG["domain_start"]),
        domain_end=Tensor(CONFIG["domain_end"]),
        dim_obs=CONFIG["dim_obs"],
        dim_model=CONFIG["dim_model"],
    )


def _precompute_frames(data, aquisition, xs_t: Tensor, frame_rounds: list) -> list:
    """For each requested round count, regenerate the model from that many
    rounds of data and record the GP mean/confidence band over xs_t. Done
    once up front (like mars_demo.py's animate_mars) so the animation itself
    is just cheap indexing and axis limits can be fixed from the full sweep.
    """
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
                "reward_mean": mean[:, 0].detach().numpy(),
                "constraint_mean": mean[:, 1].detach().numpy(),
                "reward_lo": l[:, 0].detach().numpy(),
                "reward_hi": u[:, 0].detach().numpy(),
                "constraint_lo": l[:, 1].detach().numpy(),
                "constraint_hi": u[:, 1].detach().numpy(),
            }
        )
    return frames


def animate_mirage_single(
    name: str,
    data,
    aquisition,
    j_star: float,
    out_path: str,
    n_frames: int = 60,
    fps: int = 1,
    dpi: int = 300,
    grid_size: int = 500,
    x_min: float = 3.0,
    x_max: float = 13.0,
    height_scale: float = 2.0,
):
    """Render `name`'s own run as a 1x2 GIF: landscape+GP+samples (left),
    cumulative regret (right). One algorithm per GIF, on purpose -- see
    module docstring.

    The landscape panel is windowed to [x_min, x_max] rather than the full
    mirage DOMAIN=(0, 13): this crops out the lure peak (~x=1.5, far left)
    and the honest local peak (~x=11.5, far right), zooming in on the wall
    (~x=5.1) and seed/bottleneck region (6.5-10.75) where the actual GP/
    sample action happens. The GP mean/band are only computed over this
    window too, not just visually clipped, so the y-axis isn't stretched by
    the (now offscreen) lure peak's much larger reward value.

    The green background wash marks the *currently* pessimistically-safe
    region (l_t(x) > fmin on the constraint channel, i.e. what the algorithm
    itself would call safe this round -- BaseAquisition.safe_set's own
    check), not the true safe region -- it grows/shifts round by round as
    frames["constraint_lo"] does, unlike the always-unsafe wall shading.
    """
    _apply_theme()
    # _apply_theme sizes figures for bottleneck_demo.py's 2x2 grids; this is
    # a 1x2 row, so it needs its own figsize (same override mars_demo.py's
    # gate_activity command uses for its own 1x2 layout) -- iclr2023's own
    # 1x2 preset is quite flat (5.5in x 1.7in), so height_scale stretches
    # just the height for a less wide-and-short look.
    figsize_rc = figsizes.iclr2023(nrows=1, ncols=2)
    width, height = figsize_rc["figure.figsize"]
    figsize_rc["figure.figsize"] = (width, height * height_scale)
    plt.rcParams.update(figsize_rc)

    display_name = DISPLAY_NAME.get(name, name)
    reward_color, constraint_color = PALETTE[0], PALETTE[7]

    fig, (ax_landscape, ax_regret) = plt.subplots(1, 2)

    xs = np.linspace(x_min, x_max, grid_size)
    xs_t = torch.tensor(xs, dtype=data.train_x.dtype).reshape(-1, 1)
    true_reward = reward_fn(xs)
    true_constraint = constraint_fn(xs)

    max_round = data.train_x.shape[0]
    frame_rounds = sorted(set(np.linspace(1, max_round, min(n_frames, max_round)).astype(int)))
    frames = _precompute_frames(data, aquisition, xs_t, frame_rounds)

    # --- landscape: static background (true curves, safety shading) --------
    ax_landscape.plot(xs, true_reward, color=reward_color, linewidth=1.2, linestyle="--", zorder=2)
    ax_landscape.plot(xs, true_constraint, color=constraint_color, linewidth=1.2, linestyle="--", zorder=2)
    ax_landscape.axhline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax_landscape.axvspan(DOMAIN[0], WALL_BOUNDARY_X, color="lightcoral", alpha=0.35, zorder=0)
    ax_landscape.axvline(SEED_X, color="black", linestyle=":", linewidth=1.0, zorder=1)
    ax_landscape.set_xlim(x_min, x_max)
    ax_landscape.set_xlabel(r"$x$")
    ax_landscape.set_ylabel("value")
    ax_landscape.set_title(rf"{display_name}: objective/constraint, GP $\pm\,\beta_t^{{1/2}}\sigma$, samples")

    all_bounds = np.concatenate(
        [f["reward_lo"] for f in frames]
        + [f["reward_hi"] for f in frames]
        + [f["constraint_lo"] for f in frames]
        + [f["constraint_hi"] for f in frames]
        + [true_reward, true_constraint]
    )
    pad = 0.05 * (all_bounds.max() - all_bounds.min())
    ax_landscape.set_ylim(all_bounds.min() - pad, all_bounds.max() + pad)

    # --- landscape: animated artists (GP mean lines, bands, samples) -------
    (reward_mean_line,) = ax_landscape.plot([], [], color=reward_color, linewidth=2.0, zorder=4)
    (constraint_mean_line,) = ax_landscape.plot([], [], color=constraint_color, linewidth=2.0, zorder=4)
    reward_scatter = ax_landscape.scatter(
        [], [], s=16, color=reward_color, marker=MARKERS[0], zorder=5, linewidths=0.4, edgecolors="white"
    )
    constraint_scatter = ax_landscape.scatter(
        [], [], s=16, color=constraint_color, marker=MARKERS[0], zorder=5, linewidths=0.4, edgecolors="white"
    )
    bands = {"reward": None, "constraint": None}
    safe_region = {"patch": None}

    # Deliberately just the two solid GP-mean lines -- true curves, bands,
    # and sample markers all already share these same two colors, so this
    # is a compact color legend rather than an entry per artist.
    legend_handles = [
        Line2D([0], [0], color=reward_color, linewidth=2.0, label="objective"),
        Line2D([0], [0], color=constraint_color, linewidth=2.0, label="constraint"),
    ]
    ax_landscape.legend(handles=legend_handles, loc="upper right", fontsize=6, frameon=False)

    # --- cumulative regret ---------------------------------------------------
    regret_curve = np.cumsum(j_star - data.train_y[:, 0].numpy())
    (regret_line,) = ax_regret.plot([], [], color=reward_color, linewidth=1.8, zorder=2)
    regret_point = ax_regret.scatter([], [], s=18, color=reward_color, marker=MARKERS[0], zorder=3)
    ax_regret.set_xlim(0, max_round)
    ax_regret.set_ylim(0, max(regret_curve.max(), 1e-6) * 1.05)
    ax_regret.set_xlabel("round")
    ax_regret.set_ylabel(r"cumulative regret $R_N$")
    ax_regret.set_title(
        rf"$R_N = \sum_t (J^\star_{{\mathrm{{safe}}}} - f(x_t))$, $J^\star_{{\mathrm{{safe}}}} = {j_star:.3f}$"
    )

    for ax in fig.get_axes():
        _style_axis(ax)

    round_text = fig.suptitle("")

    def update(frame_idx):
        frame = frames[frame_idx]
        k = frame["k"]

        reward_mean_line.set_data(xs, frame["reward_mean"])
        constraint_mean_line.set_data(xs, frame["constraint_mean"])

        if bands["reward"] is not None:
            bands["reward"].remove()
            bands["constraint"].remove()
        bands["reward"] = ax_landscape.fill_between(
            xs, frame["reward_lo"], frame["reward_hi"], color=reward_color, alpha=0.32, linewidth=0, zorder=3
        )
        bands["constraint"] = ax_landscape.fill_between(
            xs, frame["constraint_lo"], frame["constraint_hi"], color=constraint_color, alpha=0.32, linewidth=0, zorder=3
        )

        # Currently pessimistically-safe region this round -- l_t(x) > fmin
        # on the constraint channel, i.e. frame["constraint_lo"] > 0 -- redrawn
        # each frame since it grows/shifts as data accumulates (unlike the
        # static, always-unsafe wall shading).
        if safe_region["patch"] is not None:
            safe_region["patch"].remove()
        safe_region["patch"] = ax_landscape.fill_between(
            xs,
            0,
            1,
            where=frame["constraint_lo"] > 0.0,
            transform=ax_landscape.get_xaxis_transform(),
            color=SAFE_COLOR,
            alpha=0.12,
            zorder=0,
        )

        reward_scatter.set_offsets(np.column_stack([data.train_x[:k, 0].numpy(), data.train_y[:k, 0].numpy()]))
        constraint_scatter.set_offsets(np.column_stack([data.train_x[:k, 0].numpy(), data.train_y[:k, 1].numpy()]))

        regret_line.set_data(np.arange(k), regret_curve[:k])
        regret_point.set_offsets(np.array([[k - 1, regret_curve[k - 1]]]))

        round_text.set_text(f"{display_name} -- round {k}/{max_round}")
        return [
            reward_mean_line,
            constraint_mean_line,
            bands["reward"],
            bands["constraint"],
            safe_region["patch"],
            reward_scatter,
            constraint_scatter,
            regret_line,
            regret_point,
            round_text,
        ]

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved {display_name} GIF to {out_path}")


@app.command()
def gifs(
    n_opt_samples: int = typer.Option(40, help="Number of BO rounds for every run"),
    seed: int = typer.Option(42, help="RNG seed shared by every run"),
    algorithms: List[str] = typer.Option(
        ["SafeUCB", "GoOSE"], help="Which mirage_demo.py acquisitions to render (one GIF each)"
    ),
    out_dir: str = f"{Path().absolute()}/examples",
    n_frames: int = typer.Option(60, help="Number of animation frames (rounds are subsampled to this count)"),
    fps: int = typer.Option(1, help="Playback speed of each GIF"),
    dpi: int = typer.Option(300, help="Resolution of each GIF (figure is a fixed 1x2 iclr2023 size)"),
    grid_size: int = typer.Option(500, help="Number of x points used to draw the GP mean/confidence band"),
    x_min: float = typer.Option(3.0, help="Left edge of the landscape subplot's x-window"),
    x_max: float = typer.Option(13.0, help="Right edge of the landscape subplot's x-window"),
    height_scale: float = typer.Option(2.0, help="Multiplier on the figure height (taller = larger height:width ratio)"),
):
    """Run each requested algorithm on the mirage benchmark and render its
    own landscape+regret GIF -- one GIF per algorithm, never combined, so
    GP-UCB's mirage chase and GoOSE's mirage chase can each be watched on
    their own.
    """
    Logger.set_verbosity(2)
    j_star = true_optimum()

    for name in algorithms:
        data, aquisition = run(name, seed, n_opt_samples)
        out_path = str(Path(out_dir) / f"mirage_{name.lower()}.gif")
        animate_mirage_single(
            name,
            data,
            aquisition,
            j_star,
            out_path,
            n_frames=n_frames,
            fps=fps,
            dpi=dpi,
            grid_size=grid_size,
            x_min=x_min,
            x_max=x_max,
            height_scale=height_scale,
        )


if __name__ == "__main__":
    app()
