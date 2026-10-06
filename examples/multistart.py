import importlib
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import typer

app = typer.Typer()

OUT_DIR = Path(__file__).parent / "multistart"
DEFAULT_ALGORITHMS = ["SafeOpt", "SafeUCB", "CSafeOpt", "GoOSE", "ISE-BO"]


@dataclass(frozen=True)
class Benchmark:
    name: str
    module: str
    default_rounds: int  # the demo's own default number of BO rounds
    nominal: Callable[[object], Tuple[float, ...]]  # the demo's nominal safe seed
    domain: Callable[[object], Tuple[Tuple[float, float], ...]]  # (low, high) per parameter
    constraint: Callable[[object, np.ndarray], float]  # true constraint value at a point; safe iff >= 0
    # Optional benchmark-specific start sampler (module, count, radius, seed, region) -> starts; replaces the
    # generic rejection sampler, e.g. where safety is a connected terrain component rather than a formula.
    sampler: Optional[Callable] = None


def _pendulum_constraint(module, point: np.ndarray) -> float:
    import torch

    if not hasattr(module, "_multistart_experiment"):
        module._multistart_experiment = module.make_experiment()
    return float(module._multistart_experiment.rollout(torch.tensor(point, dtype=torch.float32), 0)[0][1])


def _mars_starts(module, count: int, radius: float, seed: int, region: str) -> np.ndarray:
    nominal, target = np.array(module.SEED), np.array(module.TARGET)
    sites = module.connected_safe_component(201)
    radius_m = radius * (module.DOMAIN_X[1] - module.DOMAIN_X[0])
    sites = sites[np.linalg.norm(sites - nominal, axis=1) <= radius_m]
    if region == "blocked":
        line = np.linspace(0.0, 1.0, 40)[None, :, None]
        route = sites[:, None, :] + line * (target - sites)[:, None, :]
        sites = sites[(module.constraint_fn(route[..., 0], route[..., 1]) < 0).any(axis=1)]
    if len(sites) < count:
        raise ValueError(f"mars: only {len(sites)} {region} landing sites within {radius_m:g} m; need {count}")
    return sites[np.random.default_rng(seed).choice(len(sites), size=count, replace=False)]


BENCHMARKS: Dict[str, Benchmark] = {
    b.name: b
    for b in [
        Benchmark(
            "double_bottleneck", "double_bottleneck_demo", 100,
            lambda m: (m.SEED_X,), lambda m: (m.DOMAIN,), lambda m, p: float(m.constraint_fn(np.array(p[0]))),
        ),
        Benchmark(
            "mirage", "mirage_demo", 200,
            lambda m: (m.SEED_X,), lambda m: (m.DOMAIN,), lambda m, p: float(m.constraint_fn(np.array(p[0]))),
        ),
        Benchmark(
            "mountaincar", "mountaincar_demo", 200,
            lambda m: (m.SEED_R,), lambda m: (m.DOMAIN,), lambda m, p: float(m.constraint_fn(p[0])),
        ),
        Benchmark(
            "pendulum", "pendulum_demo", 800,
            lambda m: tuple(m.SAFE_SEED), lambda m: (m.DOMAIN_KP, m.DOMAIN_KD), _pendulum_constraint,
        ),
        Benchmark(
            "harvester", "harvester_demo", 800,
            lambda m: (m.SEED_W, m.SEED_A), lambda m: (m.DOMAIN_W, m.DOMAIN_A),
            lambda m, p: float(m.constraint_fn(p[0], p[1])),
        ),
        Benchmark(
            "mars", "mars_demo", 400,
            lambda m: tuple(m.SEED), lambda m: (m.DOMAIN_X, m.DOMAIN_Y),
            lambda m, p: float(m.constraint_fn(p[0], p[1])), sampler=_mars_starts,
        ),
    ]
}


def sample_starts(
    benchmark: Benchmark, count: int, radius: float, seed: int, min_margin: float = 0.5, path_margin: float = 0.25,
    region: str = "all",
) -> np.ndarray:
    module = importlib.import_module(benchmark.module)
    if benchmark.sampler is not None:
        return benchmark.sampler(module, count, radius, seed, region)
    if region != "all":
        raise ValueError(f"{benchmark.name}: region {region!r} is only defined for benchmarks with a custom sampler")
    nominal = np.array(benchmark.nominal(module), dtype=float)
    low, high = (np.array(side, dtype=float) for side in zip(*benchmark.domain(module)))
    width = high - low
    nominal_margin = benchmark.constraint(module, nominal)
    if nominal_margin <= 0:
        raise ValueError(f"{benchmark.name}: the nominal seed is not safe (constraint {nominal_margin})")

    rng = np.random.default_rng(seed)
    starts: List[np.ndarray] = []
    for _ in range(100_000):
        if len(starts) == count:
            break
        direction = rng.normal(size=len(nominal))
        direction /= np.linalg.norm(direction)
        point = nominal + radius * rng.random() ** (1.0 / len(nominal)) * direction * width
        if np.any(point < low) or np.any(point > high):
            continue
        if any(np.allclose(point, other, atol=1e-9) for other in starts):
            continue
        if benchmark.constraint(module, point) < min_margin * nominal_margin:
            continue
        if any(
            benchmark.constraint(module, nominal + f * (point - nominal)) < path_margin * nominal_margin
            for f in (0.25, 0.5, 0.75)
        ):
            continue
        starts.append(point)
    if len(starts) < count:
        raise ValueError(f"{benchmark.name}: found only {len(starts)} safe starts within radius {radius}")
    return np.array(starts)


def parse_overrides(specs: List[str]) -> List[Tuple[str, str, object]]:
    parsed = []
    for spec in specs:
        target, _, raw = spec.partition("=")
        section, _, key = target.partition(".")
        if not (section and key and raw):
            raise typer.BadParameter(f"override {spec!r} must look like Section.key=value")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        parsed.append((section, key, value))
    return parsed


_SAVED_ROW = re.compile(r"^\s*(\d+)\s+\(([^)]*)\)\s+(\d+)\s+([\w-]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")


def load_saved_rows(path: Path, name: str, rounds: int, seed: int) -> List[dict]:
    """Finished runs saved in an earlier results file, or [] if there is none or it is for another setup."""
    if not path.exists():
        return []
    text = path.read_text().splitlines()
    if not any(line.startswith(f"rounds per run: {rounds} ") for line in text) or not any(
        f"selection seed: {seed}" in line for line in text
    ):
        return []
    rows = []
    for line in text:
        match = _SAVED_ROW.match(line)
        if match:
            rows.append({
                "benchmark": name, "trial": int(match.group(1)), "start": [float(v) for v in match.group(2).split(",")],
                "run_seed": int(match.group(3)), "algorithm": match.group(4),
                "last_cumulative_regret": float(match.group(5)), "best_simple_regret": float(match.group(6)),
                "seconds": 0.0,
            })
    return rows


def _worker_init(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(threads)
    import torch

    torch.set_num_threads(threads)


def _run_one(task: dict) -> dict:
    from gosafeopt.tools.logger import Logger

    Logger.set_verbosity(0)
    module = importlib.import_module(BENCHMARKS[task["benchmark"]].module)
    for section, key, value in task["overrides"]:
        module.CONFIG[section][key] = value
    started = time.time()
    data, _acquisition = module.run(
        task["algorithm"], task["run_seed"], task["rounds"], initial_point=tuple(task["start"])
    )
    values = data.train_y.numpy()
    reward, constraint = values[:, 0], values[:, 1]
    safe = constraint >= 0.0
    return {
        **task,
        "last_cumulative_regret": float(np.sum(task["j_star"] - reward)),
        "best_simple_regret": float(task["j_star"] - reward[safe].max()),
        "seconds": time.time() - started,
    }


def _stats(values: List[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), (float(array.std(ddof=1)) if len(array) > 1 else float("nan"))


def _table(rows: List[dict], algorithms: List[str]) -> List[str]:
    lines = [f"{'algorithm':<16}{'n':>3}  {'last cumulative regret':>26}  {'best simple regret':>26}"]
    lines.append(f"{'':<16}{'':>3}  {'mean':>12}{'std':>14}  {'mean':>12}{'std':>14}")
    for algorithm in algorithms:
        done = [r for r in rows if r["algorithm"] == algorithm]
        if not done:
            continue
        (m1, s1), (m2, s2) = (_stats([r[key] for r in done]) for key in ("last_cumulative_regret", "best_simple_regret"))
        lines.append(f"{algorithm:<16}{len(done):>3}  {m1:>12.4f}{s1:>14.4f}  {m2:>12.6f}{s2:>14.6f}")
    return lines


def write_reports(out_dir: Path, results: Dict[str, List[dict]], meta: Dict[str, dict], algorithms: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        "Regret statistics over initial configurations (mean and sample standard deviation, ddof=1, over the trials)",
        "last cumulative regret = sum_t (J* - f(x_t)) over all rounds incl. the initial sample",
        "best simple regret     = J* - max f(x_t) over the evaluated points with constraint >= 0",
        "J* is each demo's own comparator (a grid estimate for harvester and mountaincar), so a run can beat it",
        "slightly, which shows up as a small negative simple regret.",
        "",
    ]
    for name, info in meta.items():
        rows = sorted(results.get(name, []), key=lambda r: (r["trial"], algorithms.index(r["algorithm"])))
        header = [
            f"benchmark: {name}",
            f"rounds per run: {info['rounds']}   J*: {info['j_star']:.6f}   trials: {info['n_starts']}   "
            f"start radius: {info['radius']:g} of the domain width   selection seed: {info['seed']}",
            f"nominal safe seed: {tuple(round(v, 6) for v in info['nominal'])}",
        ]
        if info.get("region", "all") != "all":
            header.append(f"landing zone: {info['region']}-route sites only (straight route to the target crosses unsafe terrain)")
        if info["overrides"]:
            header.append("config overrides: " + ", ".join(info["overrides"]))
        lines = header + [""]
        lines.append(
            f"{'trial':>5}  {'start point':<26}{'run_seed':>11}  {'algorithm':<16}"
            f"{'last_cumulative_regret':>24}{'best_simple_regret':>21}"
        )
        for r in rows:
            start = "(" + ", ".join(f"{v:.5f}" for v in r["start"]) + ")"
            lines.append(
                f"{r['trial']:>5}  {start:<26}{r['run_seed']:>11}  {r['algorithm']:<16}"
                f"{r['last_cumulative_regret']:>24.6f}{r['best_simple_regret']:>21.8f}"
            )
        lines += ["", "mean and std across the trials:"] + _table(rows, algorithms)
        (out_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")

    # The summary is rebuilt from every benchmark file in the directory, so running a subset of the
    # benchmarks later extends it instead of replacing it.
    for name in BENCHMARKS:
        path = out_dir / f"{name}.txt"
        if not path.exists():
            continue
        text = path.read_text().splitlines()
        if "mean and std across the trials:" not in text:
            continue
        table = text[text.index("mean and std across the trials:") + 1:]
        summary += text[:2] + [line for line in text[:6] if line.startswith("config overrides:")] + table + [""]
    (out_dir / "summary.txt").write_text("\n".join(summary))


@app.command()
def run(
    benchmarks: List[str] = typer.Option(list(BENCHMARKS), help="Which benchmarks to run"),
    algorithms: List[str] = typer.Option(DEFAULT_ALGORITHMS, help="Which acquisitions to run"),
    n_starts: int = typer.Option(10, min=1, help="Number of different initial configurations per benchmark"),
    rounds: Optional[int] = typer.Option(None, min=2, help="BO rounds per run; default is each demo's own default"),
    radius: float = typer.Option(0.1, min=1e-9, help="Start radius as a fraction of the domain width"),
    seed: int = typer.Option(42, help="Seed for choosing the starts and the per-trial run seeds"),
    jobs: int = typer.Option(4, min=1, help="Runs executed in parallel (a mars run needs about 9 GB of memory)"),
    threads: int = typer.Option(5, min=1, help="Torch threads per run"),
    out_dir: Path = typer.Option(OUT_DIR, help="Directory for the text files"),
    resume: bool = typer.Option(False, help="Keep runs already saved in the results files (same rounds and seed) and only run the rest"),
    region: str = typer.Option("all", help='Landing zone for benchmarks with a custom sampler (mars): "all" or "blocked"'),
    override: List[str] = typer.Option(
        [], help="Change a demo CONFIG entry for this run only, e.g. CSafeOpt.epsilon=0.001 (repeatable)"
    ),
):
    overrides = parse_overrides(override)
    unknown = [b for b in benchmarks if b not in BENCHMARKS]
    if unknown:
        raise typer.BadParameter(f"unknown benchmark(s) {unknown}; choose from {list(BENCHMARKS)}")
    algorithms = list(dict.fromkeys(algorithms))

    tasks, meta = [], {}
    for name in dict.fromkeys(benchmarks):
        benchmark = BENCHMARKS[name]
        module = importlib.import_module(benchmark.module)
        starts = sample_starts(benchmark, n_starts, radius, seed, region=region)
        run_seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_starts).tolist()
        j_star = float(module.true_optimum())
        n_rounds = rounds or benchmark.default_rounds
        meta[name] = {
            "rounds": n_rounds, "j_star": j_star, "n_starts": n_starts, "radius": radius, "seed": seed,
            "nominal": benchmark.nominal(module), "overrides": [f"{a}.{b}={v}" for a, b, v in overrides],
            "region": region,
        }
        for trial, (start, run_seed) in enumerate(zip(starts, run_seeds), start=1):
            for algorithm in algorithms:
                tasks.append({
                    "benchmark": name, "trial": trial, "algorithm": algorithm, "rounds": n_rounds,
                    "run_seed": run_seed, "start": start.tolist(), "j_star": j_star, "overrides": overrides,
                })
        print(f"{name}: {n_starts} starts, {n_rounds} rounds, J*={j_star:.4f}", flush=True)

    results: Dict[str, List[dict]] = {name: [] for name in meta}
    if resume:
        for name, info in meta.items():
            saved = load_saved_rows(out_dir / f"{name}.txt", name, info["rounds"], seed)
            expected = {(t["trial"], t["algorithm"]): t["run_seed"] for t in tasks if t["benchmark"] == name}
            for row in saved:
                key = (row["trial"], row["algorithm"])
                if key in expected and expected[key] != row["run_seed"]:
                    raise typer.BadParameter(f"saved {name} row {key} has another run seed; the setup changed, run without --resume")
            saved = [r for r in saved if (r["trial"], r["algorithm"]) in expected]
            results[name] = saved
            print(f"{name}: resuming with {len(saved)} finished runs", flush=True)
        done = {(r["benchmark"], r["trial"], r["algorithm"]) for rows in results.values() for r in rows}
        tasks = [t for t in tasks if (t["benchmark"], t["trial"], t["algorithm"]) not in done]
        write_reports(out_dir, results, meta, algorithms)
    started = time.time()
    with ProcessPoolExecutor(jobs, mp_context=get_context("spawn"), initializer=_worker_init, initargs=(threads,)) as pool:
        futures = [pool.submit(_run_one, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results[row["benchmark"]].append(row)
            write_reports(out_dir, results, meta, algorithms)
            print(
                f"[{count}/{len(tasks)}] {row['benchmark']} trial {row['trial']} {row['algorithm']}: "
                f"R_N={row['last_cumulative_regret']:.3f} simple={row['best_simple_regret']:.5f} "
                f"({row['seconds']:.0f}s, elapsed {(time.time() - started) / 60:.1f} min)",
                flush=True,
            )
    print(f"Wrote {out_dir}/summary.txt and one file per benchmark")


if __name__ == "__main__":
    app()
