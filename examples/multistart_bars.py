"""Bar plots of the last cumulative regret and the best simple regret of every algorithm, per benchmark, from
`multistart.py`'s text files.

Styled after the grouped bar comparisons of the other papers' figures: NeurIPS 2024 tueplots theme, contiguous
dodged bars in the shared palette (the highlighted method solid, the others faded, all with a same-colour edge),
bold panel titles, a light y-grid and one shared legend under the panels. Each panel is one benchmark; each bar
is one algorithm's mean over the initial configurations, with a whisker of +/- one standard error (sample SD / sqrt(n))
. Benchmarks without results yet (no rows in their file) are skipped, so the
plot fills in as `multistart.py` finishes more of them. The Mars random-landing runs of `mars_demo.py
random_landings` (per-round CSVs next to this script) are added as further panels when present. One call writes
both figures: the cumulative-regret bars and, next to them with a `_simple` suffix, the simple-regret bars.

    python examples/multistart_bars.py                 # reads examples/multistart/*.txt
    python examples/multistart_bars.py --replace double_bottleneck=examples/multistart_tuned/double_bottleneck.txt
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer
from bottleneck_demo import PALETTE, _legend_order
from matplotlib.patches import Patch
from seaborn import axes_style
from tueplots import bundles, figsizes

app = typer.Typer()

RESULTS_DIR = Path(__file__).parent / "multistart"
TITLES = {
    "double_bottleneck": "Double bottleneck",
    "mirage": "Mirage",
    "mountaincar": "Mountain car",
    "pendulum": "Pendulum",
    "harvester": "Harvester",
}
# Mars random-landing experiments from `mars_demo.py random_landings`: panel name -> (per-round CSV, title).
MARS_RUNS = {
    "mars_near_nominal": ("mars_random_landings_near_nominal.csv", "Mars"),
}
_ROW = re.compile(r"^\s*(\d+)\s+\(([^)]*)\)\s+(\d+)\s+([\w-]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")


ALGORITHM_ORDER = ["CSafeOpt", "SafeOpt", "SafeUCB", "GoOSE", "CSafeOptSimple", "ISE-BO", "StageOpt"]
METRICS = ("cumulative", "simple")
Y_LABELS = {"cumulative": r"cumulative regret $R_N$", "simple": r"simple regret $r_N$"}


def read_results(path: Path) -> Tuple[int, Dict[str, Dict[str, List[float]]]]:
    """(rounds per run, {metric: {algorithm: regret per trial}}) from one multistart text file."""
    rounds, values = 0, {metric: {} for metric in METRICS}
    for line in path.read_text().splitlines():
        if line.startswith("rounds per run:"):
            rounds = int(line.split()[3])
        match = _ROW.match(line)
        if match:
            values["cumulative"].setdefault(match.group(4), []).append(float(match.group(5)))
            values["simple"].setdefault(match.group(4), []).append(float(match.group(6)))
    return rounds, values


def read_mars_results(path: Path, j_star: float) -> Tuple[int, Dict[str, Dict[str, List[float]]]]:
    """(rounds per run, {metric: {algorithm: regret per trial}}) from a Mars random-landings CSV.

    The simple regret is J* minus the best reward among the evaluated points that were actually safe, the
    same definition `multistart.py` uses.
    """
    runs: Dict[Tuple[str, str], List[Tuple[int, float, float, float]]] = {}
    for row in csv.DictReader(path.open()):
        runs.setdefault((row["algorithm"], row["trial"]), []).append(
            (int(row["round"]), float(row["cumulative_regret"]), float(row["reward"]), float(row["constraint"]))
        )
    values = {metric: {} for metric in METRICS}
    rounds = 0
    for (algorithm, _trial), rows in runs.items():
        rows.sort()
        rounds = max(rounds, rows[-1][0] + 1)
        values["cumulative"].setdefault(algorithm, []).append(rows[-1][1])
        values["simple"].setdefault(algorithm, []).append(j_star - max(r for _, _, r, c in rows if c >= 0.0))
    return rounds, values


def _apply_style() -> None:
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(bundles.neurips2024())
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    plt.rcParams.update({"legend.frameon": False})
    plt.rcParams["figure.constrained_layout.use"] = False  # tight_layout below places the shared legend


def plot_bars(
    results: Dict[str, Tuple[int, Dict[str, List[float]], str]], out_path: Path, highlight: str = "CSafeOpt",
    y_label: str = Y_LABELS["cumulative"],
) -> None:
    _apply_style()
    # A fixed order, so an algorithm keeps its colour whichever results files a figure is built from.
    present = {a for _r, per_alg, _n in results.values() for a in per_alg}
    algorithms = [a for a in ALGORITHM_ORDER if a in present] + sorted(present - set(ALGORITHM_ORDER))
    algorithms = _legend_order(algorithms, first=highlight)
    colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(algorithms)}
    # Variants of the highlighted method (e.g. CSafeOptSimple) sit right next to it; colours stay as assigned above.
    variants = [a for a in algorithms if a != highlight and a.startswith(highlight)]
    rest = [a for a in algorithms if a != highlight and a not in variants]
    algorithms = ([highlight] if highlight in algorithms else []) + variants + rest

    n_panels = len(results)
    n_cols = n_panels  # all panels in one row
    n_rows = -(-n_panels // n_cols)
    # Enough height for the panels, which set_box_aspect below makes square; the spare height is cropped on save.
    height_to_width = 1.6 if n_cols >= 4 else 0.75
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsizes.neurips2024(nrows=n_rows, ncols=n_cols, height_to_width_ratio=height_to_width)["figure.figsize"],
        squeeze=False,
    )
    panels = axes.ravel()

    n_algorithms = len(algorithms)
    bar_width = 0.8 / n_algorithms  # the dodged bars fill a group of total width 0.8
    for idx, (ax, (name, (rounds, per_alg, note))) in enumerate(zip(panels, results.items())):
        tops, means = [], []
        for position, algorithm in enumerate(algorithms):
            trials = np.asarray(per_alg.get(algorithm, []), dtype=float)
            if not len(trials):
                continue
            offset = (position - n_algorithms / 2 + 0.5) * bar_width
            color = colors[algorithm]
            mean = trials.mean()
            sem = trials.std(ddof=1) / np.sqrt(len(trials)) if len(trials) > 1 else 0.0
            ax.bar(
                offset, mean, width=bar_width, color=color, edgecolor=color, linewidth=1.0,
                alpha=0.9 if algorithm == highlight else 0.4, zorder=2,
            )
            ax.errorbar(offset, mean, yerr=sem, color="0.25", linewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)
            means.append(mean)
            tops.append(mean + sem)

        row, col = divmod(idx, n_cols)
        if col == 0:
            ax.set_ylabel(y_label, fontsize=9)
        # Simple regret can be slightly negative (J* is a grid estimate for some benchmarks); keep 0 as the floor.
        ax.set_ylim(min(0.0, 1.1 * min(means)), 1.1 * max(tops) if max(tops) > 0 else 1.0)
        ax.set_xlim(-0.5, 0.5)
        ax.set_box_aspect(1)  # square panels
        ax.set_xticks([])
        ax.set_xlabel(f"{rounds} rounds" + (f"\n{note}" if note else ""), fontsize=7, labelpad=3)
        ax.grid(True, linewidth=0.5, color="gainsboro", alpha=0.5, axis="y")
        ax.set_axisbelow(True)
        title = TITLES.get(name) or MARS_RUNS.get(name, (None, None))[1] or name.replace("_", " ").title()
        if n_cols >= 4 and len(title) > 12:  # narrow panels: break long names over two lines
            title = title.replace(" ", "\n", 1)
        ax.set_title(title, weight="bold", size=9 if n_cols >= 4 else 10, pad=8)
        # A faint reference line at the lowest mean regret, the best result in this benchmark.
        ax.axhline(min(means), color="gray", linestyle="--", linewidth=0.8, alpha=0.3, zorder=0)
    for ax in panels[n_panels:]:
        ax.axis("off")

    handles = [Patch(facecolor=colors[a], edgecolor=colors[a], alpha=0.9 if a == highlight else 0.4) for a in algorithms]
    labels = list(algorithms)
    legend_columns = len(labels) if n_cols >= 4 else 3  # one row under a wide figure
    legend_rows = -(-len(labels) // legend_columns)
    bottom = 0.17 * legend_rows / fig.get_size_inches()[1]  # room for the legend, as a figure fraction
    plt.tight_layout(rect=[0, bottom, 1, 0.98])

    # The legend hangs just below the lowest x label, rather than at a fixed figure fraction.
    fig.canvas.draw()
    labels_bottom = min(ax.get_tightbbox().y0 for ax in panels[:n_panels]) / fig.bbox.height
    # The legend ends exactly at the right edge of the last panel's frame (read after drawing, since the square
    # box aspect moves the frames); no padding, so only the vertical gap below the x labels is kept explicitly.
    fontsize = 8 if n_cols >= 4 else 9
    right_edge = panels[n_panels - 1].get_position().x1
    gap = 0.6 * fontsize / 72 / fig.get_size_inches()[1]
    legend = fig.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(right_edge, labels_bottom - gap), ncol=legend_columns,
        frameon=False, borderaxespad=0.0, borderpad=0.0, fontsize=fontsize, columnspacing=0.55, handletextpad=0.35,
        handlelength=1.6 if n_cols >= 4 else 2.0,
    )
    # Measure the drawn legend against the frame and close the remaining offset, so the two edges coincide.
    fig.canvas.draw()
    overshoot = (legend.get_window_extent().x1 - panels[n_panels - 1].get_window_extent().x1) / fig.bbox.width
    legend.set_bbox_to_anchor((right_edge - overshoot, labels_bottom - gap), transform=fig.transFigure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_path} and {out_path.with_suffix('.pdf')}")


@app.command()
def bars(
    results_dir: Path = typer.Option(RESULTS_DIR, help="Directory with the per-benchmark text files"),
    out: Path = typer.Option(
        Path(__file__).parent / "multistart_regret_bars.png",
        help="Output image of the cumulative regret; the simple regret goes next to it with a _simple suffix",
    ),
    replace: List[str] = typer.Option(
        [], help="Use another results file for one benchmark, as name=path (repeatable), e.g. a tuned run; "
        "for Mars, mars_near_nominal=<random-landings CSV>"
    ),
):
    replacements = {}
    for spec in replace:
        name, _, path = spec.partition("=")
        if name not in {**TITLES, **MARS_RUNS} or not path:
            raise typer.BadParameter(f"--replace expects name=path with name in {list(TITLES) + list(MARS_RUNS)}, got {spec!r}")
        replacements[name] = Path(path)
    results = {metric: {} for metric in METRICS}
    for name in TITLES:
        path = replacements.get(name, results_dir / f"{name}.txt")
        if not path.exists():
            continue
        rounds, values = read_results(path)
        if values["cumulative"]:
            for metric in METRICS:
                results[metric][name] = (rounds, values[metric], "")
        else:
            print(f"Skipping {name}: no finished runs in {path.name}")
    mars_dir = Path(__file__).parent
    for name, (filename, _title) in MARS_RUNS.items():
        path = replacements.get(name, mars_dir / filename)
        if not path.exists():
            continue
        j_star = json.loads(path.with_suffix(".json").read_text())["j_star"]
        rounds, values = read_mars_results(path, j_star)
        if values["cumulative"]:
            for metric in METRICS:
                results[metric][name] = (rounds, values[metric], "")
    if not results["cumulative"]:
        raise typer.BadParameter(f"no results found in {results_dir}")
    plot_bars(results["cumulative"], out, y_label=Y_LABELS["cumulative"])
    plot_bars(results["simple"], out.with_name(out.stem + "_simple" + out.suffix), y_label=Y_LABELS["simple"])


if __name__ == "__main__":
    app()
