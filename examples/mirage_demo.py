"""A "mirage" benchmark: the global reward maximum sits just past a genuine,
permanent safety wall immediately to the left of the safe seed, reached by a
real, continuously-rising reward gradient the whole way there; a lower (but
real, safely reachable) local maximum sits behind an ordinary tight-but-safe
bottleneck to the right, at the far end of a boring valley with no gradient
pointing toward it at all.

This isolates the mirror-image failure mode to reward_desert_demo.py's. There,
GOOSE degraded to plain safe-UCB because nothing was ever unsafe, so its
expansion trigger never fired. Here, something is *always* unsafe -- but it's
the wrong thing, and it's the thing every round of evidence keeps pointing
toward. GOOSE's oracle (src/gosafeopt/aquisitions/goose.py's plain GP-UCB
restricted to the optimistic safe set) has no notion of "this target is a
lost cause": every round it re-maximizes reward over whatever still *looks*
optimistically safe, and as the safe frontier creeps left toward the wall it
keeps observing genuinely rising reward right up to the boundary -- strong,
well-founded evidence to keep pushing that way, extrapolating into exactly
the direction that's truly closed off. Safe Expansion (Algorithm 2) then
spends its budget building Lipschitz-certified bridges toward that single
goal-directed target one cautious step at a time, and never even considers
the bottleneck on the right: GOOSE's `priority` (goose.py's
`_safe_expansion_scores`, eq. 25) always ranks candidates purely by distance
to the oracle's current suggestion, which stays pinned to the mirage for as
long as it remains un-disproven at the model's own kernel lengthscale.

SafeOpt/CSafeOpt/GoSafeOpt don't share this blind spot: their own expander
sets (safe_opt.py's `expanders`, cum_safe_opt.py's reward-std gate) are keyed
to *any* safe point with unresolved constraint or reward uncertainty, not to
a single reward-maximizing target. The bottleneck on the right is just as
good an expander candidate as the wall on the left, so they keep sampling
both frontiers and settle on whichever safely certified point actually pays
off best -- the honest local peak, once it's found -- while GOOSE's oracle
never lets it look right in the first place.

Reuses the paper-style plotting and Chowdhury-Gopalan infrastructure from
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
import seaborn.objects as so
import torch
import typer
from bottleneck_demo import (
    MARKERS,
    PALETTE,
    GrowingGoose,
    GrowingGoSafeOpt,
    GrowingSafeOpt,
    GrowingSafeUCB,
    InstrumentedCSafeOpt,
    _apply_theme,
    _draw_order,
    _legend,
    _legend_order,
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

gosafeopt.device = torch.device("cpu")

app = typer.Typer()

DOMAIN = (0.0, 13.0)
SEED_X = 6.5

# --- the mirage: a real, higher peak sitting behind a real, permanent wall,
# with reward rising continuously (no dip) the entire way from the seed. ---
LURE_X = 1.5  # the unreachable global reward maximum
LURE_AMPLITUDE = 5.0
LURE_WIDTH = 2.6  # wide enough that its rising slope reaches all the way to the seed, no flat/dip gap
WALL_X = 5.1  # sigmoid transition center of the safety cliff
WALL_SCALE = 0.4
WALL_DEPTH = 1.6
# True (numerically verified) safe/unsafe crossing of the wall, for plotting
# and for confirming the safe-reachable ceiling below.
WALL_BOUNDARY_X = 4.9994

# --- the honest route: an ordinary valley, a tight-but-safe bottleneck, and
# a real (lower) local peak, the same shape bottleneck_demo.py uses. ---
BOTTLENECK = (7.75, 10.75)
BOTTLENECK_CENTER = 9.25
BOTTLENECK_WIDTH = 0.5
BOTTLENECK_DEPTH = 0.83  # min constraint margin at the center ~= 0.07
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
    # A one-sided cliff (not a symmetric dip): everything comfortably left of
    # WALL_X collapses to a large, permanent constraint violation and stays
    # there all the way to the domain edge -- x=0 is exactly as unsafe as
    # x=LURE_X, so there is no safe island beyond the wall to "discover" by
    # luck. The bottleneck is a separate, ordinary symmetric dip: safe
    # throughout, just barely so at its center.
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
        # Experiment.rollout divides by len(trajectory) == 2 for a single-step
        # episode; double the raw values so data.train_y matches reward_fn/constraint_fn.
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
        # NB: Normalize() maps x into [0,1] before the kernel sees it, so this
        # lengthscale is a *fraction of the domain width*, not raw x-units --
        # here 0.08 * 13 =~ 1.04 raw units. Two separate reasons to keep it
        # small, not one:
        #  (a) Kept below the ~3.6-unit gap from the seed to the wall so a
        #      low sample taken while crossing the honest bottleneck can't
        #      smear the model's belief all the way over to the lure's side
        #      (too large washes out the local rising-gradient structure this
        #      benchmark needs).
        #  (b) Kept close to WALL_SCALE=0.4 (the true cliff's own transition
        #      width). A lengthscale much wider than the feature it's
        #      modeling makes the GP under-estimate how fast the constraint
        #      actually drops near the boundary -- it extrapolates the first
        #      confidently-safe sample almost linearly forward instead of
        #      anticipating the cliff, which is exactly what produces
        #      large, dangerous constraint violations (order -0.5 to -0.7)
        #      in the first handful of rounds. Below ~1.3 raw units those
        #      violations vanish; what's left is sub-0.05 boundary noise
        #      right at the true crossing (WALL_BOUNDARY_X) as the pessimistic
        #      bound tracks it tightly -- expected, not a bug, and present
        #      in bottleneck_demo.py/double_bottleneck_demo.py too.
        "lenghtscale": [0.08],
        "normalize_input": True,
        "normalize_output": False,
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
    "CSafeOpt": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.1, "alpha": 0.55, "zeta": 0.01},
    "GoSafeOpt": {"scale_beta": 1.0, "beta": 9, "n_max_local": 5, "n_max_global": 3},
    # lipschitz=1.5 safely bounds constraint_fn's true max slope (~1.01, from
    # the bottleneck dip; the wall's own max slope is ~0.8).
    "GoOSE": {"scale_beta": 1.0, "beta": 9, "lipschitz": 1.5, "epsilon": 0.05},
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
    elif name == "GoOSE":
        return GrowingGoose(**CONFIG["GoOSE"], dim_obs=dim_obs)
    else:
        raise ValueError(f"Unknown aquisition {name}")


def run(name: str, seed: int, n_opt_samples: int, alpha: Optional[float] = None) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    reset_global_state()

    data = Data()
    x_safe = torch.tensor([[SEED_X]])

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


def unreachable_optimum(resolution: int = 200_000) -> float:
    """The global max of reward_fn, ignoring safety entirely -- the mirage itself."""
    xs = np.linspace(DOMAIN[0], DOMAIN[1], resolution)
    return float(reward_fn(xs).max())


def true_optimum(resolution: int = 200_000) -> float:
    """The best reward actually achievable without ever leaving the safe set --
    i.e. the ceiling any *correct* safe algorithm can be judged against. Unlike
    every other demo in this suite, this is deliberately *not* the unconstrained
    max of reward_fn: that point (the mirage) is genuinely unsafe, so no safe
    algorithm should ever reach it, and scoring regret against it would make
    every algorithm look equally "bad" instead of isolating GOOSE specifically.
    """
    xs = np.linspace(DOMAIN[0], DOMAIN[1], resolution)
    safe = constraint_fn(xs) > 0
    return float(reward_fn(xs)[safe].max())


def plot_mirage(results: dict, out_path: str):
    _apply_theme()

    names = _legend_order(list(results.keys()))
    style = {n: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, n in enumerate(names)}
    j_star = true_optimum()
    j_mirage = unreachable_optimum()

    fig, ((ax_landscape, ax_trace), (ax_regret, ax_threshold)) = plt.subplots(2, 2)

    # --- landscape -----------------------------------------------------
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

    # --- trace -----------------------------------------------------------
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

    # --- cumulative regret (against the safe-reachable optimum) ------------
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
    _plot_series(ax_regret, regret_df, "round", "cumulative_regret", _draw_order(names), style)
    ax_regret.set_xlabel("round")
    ax_regret.set_ylabel(r"cumulative regret $R_N$")
    ax_regret.set_title(
        rf"$R_N = \sum_t (J^\star_{{\mathrm{{safe}}}} - f(x_t))$, $J^\star_{{\mathrm{{safe}}}} = {j_star:.3f}$"
        rf" (mirage $= {j_mirage:.3f}$)"
    )
    _legend(ax_regret, names, style)

    # --- gate threshold (eta_t) -------------------------------------------
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
        crossed_mask = chosen_x > BOTTLENECK[1]
        if crossed_mask.any():
            first_cross_episode = int(np.argmax(crossed_mask))
            if first_cross_episode >= 1:
                crossing_lines.append((name, first_cross_episode, history[first_cross_episode - 1][2]))

    if threshold_frames:
        threshold_df = pd.concat(threshold_frames, ignore_index=True)
        gate_names = [n for n in names if n in threshold_df["name"].unique()]
        _plot_series(ax_threshold, threshold_df, "round", "value", _draw_order(gate_names), style, pointsize=3.5)

        for name, _first_cross_episode, eta_at_crossing in crossing_lines:
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


@app.command()
def mirage(
    n_opt_samples: int = typer.Option(200, help="Number of BO rounds for every run"),
    seed: int = typer.Option(42, help="RNG seed shared by every run"),
    algorithms: List[str] = typer.Option(
        ["SafeOpt", "SafeUCB", "CSafeOpt", "GoOSE"], help="Which acquisitions to run"
    ),
    out: str = f"{Path().absolute()}/examples/mirage.png",
):
    Logger.set_verbosity(2)
    j_star = true_optimum()

    results = {}
    for name in algorithms:
        data, aquisition = run(name, seed, n_opt_samples)
        results[name] = (data, aquisition)
        _print_summary(name, data, j_star)

    plot_mirage(results, out)


if __name__ == "__main__":
    app()
