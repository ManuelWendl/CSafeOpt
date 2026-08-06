"""A 1D safe-optimization benchmark with two sequential bottlenecks.

Same idea as bottleneck_demo.py (a mediocre local optimum at the seed,
reached-by-crossing-a-low-reward-but-safe-corridor), but chained twice: seed
-> bottleneck 1 -> a medium peak -> bottleneck 2 -> the best peak. Bottleneck
2 is deliberately narrower and has a tighter safety margin than bottleneck 1,
so crossing it requires the gate threshold to have shrunk further -- a
sterner test of whether a given alpha schedule shrinks fast enough to reach
the full comparator class in a fixed round budget (Section 6.5 of the
write-up). Reuses the landscape-agnostic infrastructure (paper-style
plotting, the Chowdhury-Gopalan confidence sequence, the instrumented
CSafeOpt) directly from bottleneck_demo.py rather than duplicating it.
"""

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
    GrowingGoSafeOpt,
    GrowingSafeOpt,
    GrowingSafeUCB,
    InstrumentedCSafeOpt,
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

gosafeopt.device = torch.device("cpu")

app = typer.Typer()

SEED_X = 0.5
MID_PEAK_X = 5.5
FINAL_PEAK_X = 10.5
BOTTLENECK_1 = (1.5, 4.0)  # same width/shape as the single-bottleneck benchmark
BOTTLENECK_2 = (7.3, 9.0)  # same width as bottleneck 1; only the safety margin is tighter
DOMAIN = (0.0, 12.0)


def reward_fn(x: np.ndarray) -> np.ndarray:
    return (
        1.2 * np.exp(-((x - SEED_X) ** 2) / (2 * 0.35**2))
        + 2.1 * np.exp(-((x - MID_PEAK_X) ** 2) / (2 * 0.8**2))
        + 3.9 * np.exp(-((x - FINAL_PEAK_X) ** 2) / (2 * 0.9**2))
        - 0.3
    )


BOTTLENECK_2_DIP_CENTER = 8.87  # bottleneck 2's original minimizer
BOTTLENECK_2_DIP_WIDTH = 0.5
BOTTLENECK_2_DIP_AMPLITUDE = 0.09328  # halves bottleneck 2's margin again (0.062 -> 0.031); negligible at bottleneck 1


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
        "lenghtscale": [0.4],  # must not exceed the narrowest true feature (the seed bump, width 0.35)
        "normalize_input": True,
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
    "CSafeOpt": {"scale_beta": 1.0, "beta": 9, "epsilon": 0.06, "alpha": 0.55, "zeta": 0.01},
    "GoSafeOpt": {"scale_beta": 1.0, "beta": 9, "n_max_local": 5, "n_max_global": 3},
    # lipschitz=2.0 safely bounds constraint_fn's true max slope (~1.2, from the
    # narrower width-0.45 bump at the final peak plus the dip term).
    "Goose": {"scale_beta": 1.0, "beta": 9, "lipschitz": 5, "epsilon": 0.031},
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


def bottleneck_margin(bottleneck: tuple, resolution: int = 20_000) -> float:
    xs = np.linspace(bottleneck[0], bottleneck[1], resolution)
    return float(constraint_fn(xs).min())


def _greedy_gamma_sequence(covar: gpytorch.kernels.Kernel, lam: float, grid_size: int, n_direct: int) -> torch.Tensor:
    """gamma_n for n=1..n_direct via greedy (submodular, (1-1/e)-optimal) variance
    maximization -- the standard tractable proxy for the true (NP-hard) maximum
    information gain gamma_N := max_{|A|<=N} I(y_A; f_A) that Lemma 5 is stated for.
    Operates on the normalized [0,1] input space, matching the model's own
    Normalize() input_transform, so this uses exactly the kernel a real run sees.
    """
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
    """Fit gamma_n ~= a + b * n^(1/6) * log(n)^(5/6), the known asymptotic rate for
    a Matern-5/2 kernel on a 1D domain (Srinivas et al. 2012 / Vakili et al.), on the
    back half of the directly-computed sequence, so gamma (and hence Nbar) can be
    evaluated for n far beyond what's tractable to compute exactly.
    """
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
    """Smallest n >= 1 with n >= 4*C_lambda*beta_n*gamma_n / m_star^2 (Lemma 5's
    budget argument): beyond this n, some round t<=n is guaranteed to have already
    certified a point within margin m_star, regardless of how the points were chosen.
    """

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


def plot_double_bottleneck(results: dict, out_path: str):
    _apply_theme()

    names = list(results.keys())
    style = {n: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, n in enumerate(names)}
    j_star = true_optimum()

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
    for bottleneck in (BOTTLENECK_1, BOTTLENECK_2):
        ax_landscape.axvspan(*bottleneck, color="gainsboro", alpha=0.6, zorder=0)
    ax_landscape.axvline(SEED_X, color="black", linestyle=":", linewidth=1.2)
    ax_landscape.set_xlim(*DOMAIN)

    ax_landscape.set_xlabel(r"$x$")
    ax_landscape.set_ylabel("value")
    ax_landscape.set_title("True reward / constraint landscape")
    _legend(ax_landscape, curve_names, {curve_names[0]: (PALETTE[0], None), curve_names[1]: (PALETTE[7], None)})

    # --- trace -----------------------------------------------------------
    trace_df = pd.concat(
        [
            pd.DataFrame({"round": np.arange(data.train_x.shape[0]), "chosen_x": data.train_x[:, 0].numpy(), "name": n})
            for n, (data, _aq) in results.items()
        ],
        ignore_index=True,
    )
    _plot_series(ax_trace, trace_df, "round", "chosen_x", names, style)
    for bottleneck in (BOTTLENECK_1, BOTTLENECK_2):
        ax_trace.axhspan(*bottleneck, color="gainsboro", alpha=0.6, zorder=0)
    ax_trace.set_xlabel("round")
    ax_trace.set_ylabel(r"chosen $x$")
    ax_trace.set_title("Point evaluated per round")
    _legend(ax_trace, names, style)

    # --- cumulative regret -------------------------------------------------
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
    crossing_lines = []
    for name, (data, aquisition) in results.items():
        history = getattr(aquisition, "threshold_history", None)
        if not history:
            continue
        rounds_h, _tau_h, eta_h = zip(*history)
        threshold_frames.append(pd.DataFrame({"round": rounds_h, "value": eta_h, "name": name}))

        chosen_x = data.train_x[:, 0].numpy()
        for bottleneck in (BOTTLENECK_1, BOTTLENECK_2):
            crossed_mask = chosen_x > bottleneck[1]
            if crossed_mask.any():
                first_cross_episode = int(np.argmax(crossed_mask))
                if first_cross_episode >= 1:
                    crossing_lines.append((name, first_cross_episode, history[first_cross_episode - 1][2]))

    if threshold_frames:
        threshold_df = pd.concat(threshold_frames, ignore_index=True)
        gate_names = [n for n in names if n in threshold_df["name"].unique()]
        _plot_series(ax_threshold, threshold_df, "round", "value", gate_names, style, pointsize=3.5)

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
    print(f"Saved double-bottleneck plot to {out_path} (and {pdf_path})")


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
        ["SafeOpt", "SafeUCB", "CSafeOpt", "GoSafeOpt", "Goose"], help="Which acquisitions to run"
    ),
    out: str = f"{Path().absolute()}/examples/double_bottleneck.png",
):
    Logger.set_verbosity(2)
    j_star = true_optimum()

    results = {}
    for name in algorithms:
        data, aquisition = run(name, seed, n_opt_samples)
        results[name] = (data, aquisition)
        _print_summary(name, data, j_star)

    plot_double_bottleneck(results, out)


@app.command()
def alpha_ablation(
    n_opt_samples: int = typer.Option(100, help="Number of BO rounds for every run"),
    seed: int = typer.Option(42, help="RNG seed shared by every run"),
    alphas: List[float] = typer.Option([0.5, 0.6, 0.7, 0.75, 0.8, 1], help="alpha values to compare for CSafeOpt"),
    out: str = f"{Path().absolute()}/examples/double_bottleneck_alpha_ablation.png",
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
        _print_summary(plain_name, data, j_star)

    plot_double_bottleneck(results, out)


@app.command()
def nbar(
    grid_size: int = typer.Option(4000, help="Grid resolution (in normalized [0,1] space) for the gamma_n estimate"),
    n_direct: int = typer.Option(8000, help="Compute gamma_n exactly (via greedy selection) up to this many points"),
):
    """Nbar = min{n >= 1 : n >= 4*C_lambda*beta_n*gamma_n / (m*)^2} for both bottlenecks
    (Lemma 5's information-gain budget): the worst-case number of rounds guaranteed to
    contain a round that already certified a point within that bottleneck's margin,
    regardless of which algorithm or points were actually chosen. Uses the exact same
    kernel/lengthscale/noise CONFIG uses, and CSafeOpt's own rkhs_bound/noise_proxy/delta.
    gamma_n is computed exactly up to n_direct via greedy (near-optimal) selection, then
    extrapolated with the Matern-5/2-in-1D asymptotic rate for any larger n the solve needs.
    """
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
