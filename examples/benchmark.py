"""
Benchmark script for PersistenceForest(point_cloud).

This script measures the construction time of PersistenceForest for
different point-cloud sampling schemes and point counts.

Output
------
A CSV file (default: `persforest_benchmark.csv`) with columns:

    sampler                     - name of the sampling method
    n_points                    - number of points in the point cloud
    run_index                   - index of the repetition (0..repeats-1)
    dim                         - ambient dimension of the point cloud
    reduce                      - 1 if reduction was used, 0 otherwise
    compute_barcode             - 1 if barcodes were computed, 0 otherwise
    time_point_cloud_s          - time to generate the point cloud
    time_persforest_s          - time to construct PersistenceForest
    time_total_s                - total time (generation + forest)
    seed                        - random seed used for this run

This format is suitable for direct use in Python/R/Julia for plotting
scaling curves in your paper.
"""

from __future__ import annotations


import csv
import os
import time
from typing import Callable, Dict, Iterable, Sequence, Optional
import statistics
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{sfmath}",
    "font.size": 6,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "figure.dpi": 300,
    "legend.handlelength": 0.5,
    "legend.frameon": False,
    "axes.grid": False,
})

import numpy as np

# Local imports
from persforest.PersistenceForest import PersistenceForest
from point_cloud_sampling import (
    sample_noisy_circle,
    sample_uniform_points,
    sample_points_without_balls,
    sample_noisy_sphere,
)

# Paths and file naming
BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "benchmarks")


def _benchmark_path(name: str, ext: str) -> str:
    """Return the path to a benchmark artifact inside ``benchmarks/``."""
    filename = name if name.endswith(f".{ext}") else f"{name}.{ext}"
    return os.path.join(BENCHMARK_DIR, filename)


def _resolve_csv_path(csv_path: Optional[str], benchmark_name: Optional[str]) -> str:
    """Prefer an explicit path, otherwise build one from ``benchmark_name``."""
    if csv_path and benchmark_name:
        raise ValueError("Provide only one of csv_path or benchmark_name")
    if csv_path:
        return csv_path
    if benchmark_name:
        return _benchmark_path(benchmark_name, "csv")
    raise ValueError("Provide csv_path or benchmark_name")


def _resolve_figure_path(save_path: Optional[str], artifact_name: Optional[str]) -> Optional[str]:
    """Prefer an explicit figure path, otherwise build one from the given name."""
    if save_path:
        return save_path
    if artifact_name:
        return _benchmark_path(artifact_name, "png")
    return None


# ---------------------------------------------------------------------------
# Sampling configuration
# ---------------------------------------------------------------------------

Sampler = Callable[[int, int], np.ndarray]


def _make_samplers() -> Dict[str, Sampler]:
    """Return a dictionary mapping method names to sampling callables.

    Each callable has the signature (n_points, seed) -> point_cloud (n,d).
    """
    return {
        "noisy_circle_noise-std-dot05": lambda n, seed: sample_noisy_circle(n, seed=seed, noise_std=0.05),
        "noisy_2sphere_noise-std-dot05": lambda n, seed: sample_noisy_sphere(n, dim =3, seed=seed, noise_std=0.05),
        "uniform_2D": lambda n, seed: sample_uniform_points(n,dim=2, seed=seed),
        "uniform_3D": lambda n, seed: sample_uniform_points(n,dim=3, seed=seed),
        "uniform_2D_with_30holes_radius-max-dot05": lambda n, seed: sample_points_without_balls(n=n,dim=2,seed=seed,radius_range=[0,0.5],num_discs=30),
        "uniform_3D_with_30holes_radius-max-dot05": lambda n, seed: sample_points_without_balls(n=n,dim=3,seed=seed,radius_range=[0,0.5],num_discs=30)
    }


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def make_sizes_linear(min_size: int, max_size: int, count: int) -> list[int]:
    """Create a linearly spaced list of sizes between ``min_size`` and ``max_size``."""
    if min_size <= 0 or max_size <= 0:
        raise ValueError("min_size and max_size must be positive")
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size")
    if count < 1:
        raise ValueError("count must be at least 1")

    if count == 1:
        return [int(min_size)]

    raw = np.linspace(min_size, max_size, count)
    sizes: list[int] = []
    for val in raw:
        size = int(round(val))
        if not sizes or size != sizes[-1]:
            sizes.append(size)
    sizes[0] = int(min_size)
    sizes[-1] = int(max_size)
    return sizes


def make_sizes_log(min_size: int, max_size: int, count: int, base: float = 10.0) -> list[int]:
    """Create a log-spaced list of sizes between ``min_size`` and ``max_size``."""
    if min_size <= 0 or max_size <= 0:
        raise ValueError("min_size and max_size must be positive for log spacing")
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size")
    if count < 1:
        raise ValueError("count must be at least 1")

    if count == 1:
        return [int(min_size)]

    raw = np.logspace(
        np.log(min_size) / np.log(base),
        np.log(max_size) / np.log(base),
        count,
        base=base,
    )
    sizes: list[int] = []
    for val in raw:
        size = int(round(val))
        if size < min_size:
            size = int(min_size)
        if size > max_size:
            size = int(max_size)
        if not sizes or size != sizes[-1]:
            sizes.append(size)
    sizes[0] = int(min_size)
    sizes[-1] = int(max_size)
    return sizes


def generate_point_cloud(
    sampler_name: str,
    n_points: int,
    seed: int,
    samplers: Dict[str, Sampler],
) -> np.ndarray:
    """Generate a point cloud using one of the predefined samplers."""
    try:
        sampler = samplers[sampler_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sampler '{sampler_name}'. "
            f"Available: {sorted(samplers.keys())}"
        ) from exc
    return sampler(n_points, seed)


def benchmark_single_run(
    sampler_name: str,
    n_points: int,
    seed: int,
    samplers: Dict[str, Sampler],
    reduce: bool = True,
    compute_barcode: bool = True,
) -> dict:
    """Benchmark a single construction of PersistenceForest.

    Returns a dict with timing information suitable for CSV output.
    """
    # Point cloud generation
    point_cloud = generate_point_cloud(sampler_name, n_points, seed, samplers)

    # PersistenceForest construction
    t_pf_start = time.perf_counter()
    forest = PersistenceForest(
        point_cloud,
        compute=True,
        reduce=reduce,
        compute_barcode=compute_barcode,
        print_info=False,
    )
    t_pf_end = time.perf_counter()

    dim = int(point_cloud.shape[1]) if forest.point_cloud.ndim == 2 else 1

    return {
        "sampler": sampler_name,
        "n_points": int(n_points),
        "run_index": 0,  # will be overwritten by caller
        "dim": dim,
        "reduce": int(bool(reduce)),
        "compute_barcode": int(bool(compute_barcode)),
        "time_persforest_s": t_pf_end - t_pf_start,
        "seed": int(seed),
    }


def benchmark_suite(
    samplers: Dict[str, Sampler],
    methods: Iterable[str],
    sizes: Iterable[int],
    n_repeats: int,
    base_seed: int,
    csv_path: Optional[str] = None,
    reduce: bool = True,
    compute_barcode: bool = True,
    benchmark_name: Optional[str] = None,
) -> None:
    """Run the benchmark and write results to a CSV file.

    Parameters
    ----------
    samplers:
        Dict mapping method name -> sampler callable.
    methods:
        Iterable of keys from ``samplers`` to benchmark.
    sizes:
        Iterable of point counts.
    n_repeats:
        Number of independent runs per (method, size) pair.
    base_seed:
        Base random seed; a different seed is derived for each run but
        is deterministic and reproducible.
    csv_path:
        Output path for the CSV file with all timing data. If omitted,
        provide ``benchmark_name`` instead to automatically save in
        ``benchmarks/{benchmark_name}.csv``.
    reduce, compute_barcode:
        Passed through to ``PersistenceForest``. You may want to set
        ``compute_barcode=False`` if you only care about the forest
        construction time.
    benchmark_name:
        Optional short name used to derive the CSV path inside the
        ``benchmarks`` directory.
    """
    methods = list(methods)
    sizes = [int(s) for s in sizes]

    csv_path = _resolve_csv_path(csv_path, benchmark_name)

    fieldnames = [
        "sampler",
        "n_points",
        "run_index",
        "dim",
        "reduce",
        "compute_barcode",
        "time_persforest_s",
        "seed",
    ]

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for method in methods:
            if method not in samplers:
                raise ValueError(
                    f"Unknown sampling method '{method}'. "
                    f"Available methods: {sorted(samplers.keys())}"
                )

        for method in methods:
            print(f"starting method {method}")
            for size in sizes:
                print(f"starting size {size}")
                for r in range(n_repeats):
                    # Derive a reproducible but distinct seed for each run
                    # (simple deterministic mixing of indexes)
                    seed = base_seed + 100000 * r + 1000 * size + hash(method) % 9973

                    result = benchmark_single_run(
                        sampler_name=method,
                        n_points=size,
                        seed=seed,
                        samplers=samplers,
                        reduce=reduce,
                        compute_barcode=compute_barcode,
                    )
                    # Logical run index (0..n_repeats-1)
                    result["run_index"] = r

                    writer.writerow(result)

def plot_runtimes_from_csv(
    csv_path: Optional[str],
    methods: Sequence[str],
    time_column: str = "time_persforest_s",
    ax = None,
    show_std: bool = True,
    save_dir: Optional[str]=None,
    label_dict: Optional[dict[str,str]] = None,
    color_dict: Optional[dict[str, str]] = None,
    linestyle_dict: Optional[dict[str, str]] = None,
    benchmark_name: Optional[str] = None,
    figure_name: Optional[str] = None,
    log_scale: bool = False,
    title: Optional[str] = None,
    show_axis_labels: bool = True,
) :
    """
    Plot runtimes vs. number of points for a list of methods.

    Parameters
    ----------
    csv_path:
        Path to the CSV produced by ``benchmark_suite``. If omitted,
        set ``benchmark_name`` to look inside ``benchmarks/`` using
        ``benchmarks/{benchmark_name}.csv``.
    methods:
        Iterable of sampler names (the `sampler` column in the CSV) to plot.
    time_column:
        Which time column to use, e.g. "time_persforest_s"
        or "time_total_s".
    ax:
        Optional existing matplotlib Axes to draw on. If None, a new
        figure and axes are created.
    show_std:
        If True, draw vertical error bars showing the standard deviation
        over repeated runs for each (method, n_points) pair.
    label_dict:
        Optional mapping from method name to display label.
    color_dict:
        Optional mapping from method name to color (hex string or Matplotlib color).
    linestyle_dict:
        Optional mapping from method name to line style (e.g., "-", "--").
    benchmark_name:
        Optional short name used to derive the CSV path inside the
        ``benchmarks`` directory. Also used as the default figure name
        when ``figure_name`` is not provided.
    figure_name:
        Optional name used for saving the figure into ``benchmarks``.
        Defaults to ``benchmark_name`` when set.
    log_scale:
        If True, use log-log axes for points and time.
    title:
        Optional title for the plot. Defaults to a generic runtime title
        when not provided.

    Returns
    -------
    ax:
        The matplotlib Axes containing the plot.
    """
    csv_path = _resolve_csv_path(csv_path, benchmark_name)

    # Read CSV
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data found in CSV {csv_path!r}")

    # Group times by (sampler, n_points)
    data: Dict[str, Dict[int, list[float]]] = {}
    for row in rows:
        sampler = row["sampler"]
        if sampler not in methods:
            continue

        try:
            n_points = int(row["n_points"])
        except (KeyError, ValueError):
            raise ValueError("CSV must contain an integer 'n_points' column")

        try:
            t = float(row[time_column])
        except KeyError:
            raise ValueError(
                f"CSV does not contain the requested time column {time_column!r}"
            )

        data.setdefault(sampler, {}).setdefault(n_points, []).append(t)

    if ax is None:
        fig, ax = plt.subplots()

    # Plot each method
    for method in methods:
        if method not in data:
            print(f"[plot_runtimes_from_csv] Warning: no data for method {method!r}")
            continue

        label = label_dict.get(method, method) if label_dict is not None else method

        color = color_dict.get(method) if color_dict is not None else None
        linestyle = linestyle_dict.get(method) if linestyle_dict is not None else "-"

        n_to_times = data[method]
        xs = sorted(n_to_times.keys())
        ys_mean = [statistics.mean(n_to_times[n]) for n in xs]

        if show_std:
            ys_std = [
                statistics.pstdev(n_to_times[n]) if len(n_to_times[n]) > 1 else 0.0
                for n in xs
            ]
            # ax.errorbar(
            ax.scatter(
                xs,
                ys_mean,
                # yerr=ys_std,
                marker="o",
                # capsize=3,
                edgecolors=None,
                s=1,
                label=label,
                # color=color,
                linestyle=linestyle,
            )
        else:
            ax.plot(xs, ys_mean, marker="o", label=label, color=color, linestyle=linestyle)

    if show_axis_labels:
        ax.set_xlabel("Number of points")
        ax.set_ylabel("Time [s]")
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_path = _resolve_figure_path(save_dir, figure_name or benchmark_name)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi =300)

    return ax

def plot_benchmark_grid(
    layout: Sequence[tuple[int, int, str, Sequence[str], bool]],
    label_dict: dict[str, str],
    column_titles: Sequence[str],
    row_titles: Sequence[str],
    figure_title: str,
    output_name: str,
    color_dict: Optional[dict[str, str]] = None,
    linestyle_dict: Optional[dict[str, str]] = None,
) -> None:
    """Render a 2x2 grid of benchmark plots with column/row headers."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={"width_ratios": [1, 1]})

    for row, col, bench_name, methods, log_scale in layout:
        plot_runtimes_from_csv(
            csv_path=_benchmark_path(bench_name, "csv"),
            methods=methods,
            time_column="time_persforest_s",
            label_dict=label_dict,
            color_dict=color_dict,
            linestyle_dict=linestyle_dict,
            ax=axes[row, col],
            log_scale=log_scale,
            title=None,
            show_axis_labels=False,
        )

    for col, title in enumerate(column_titles):
        bbox = axes[0, col].get_position()
        x_center = bbox.x0 + bbox.width / 2
        fig.text(x_center, bbox.y1 + 0.03, title, ha="center", va="bottom", fontsize=12, fontweight="bold")

    for row, title in enumerate(row_titles):
        bbox = axes[row, 0].get_position()
        y_center = bbox.y0 + bbox.height / 2
        fig.text(bbox.x0 - 0.05, y_center, title, ha="center", va="center", rotation="vertical", fontsize=11, fontweight="bold")

    #fig.suptitle(figure_title, fontsize=13, y=0.98)
    fig.supxlabel("Number of points", fontsize=12)
    fig.supylabel("Time [s]", fontsize=12)
    fig.savefig(_benchmark_path(output_name, "png"), dpi=300)
    plt.show()

def plot_benchmark_row(
    layout: Sequence[tuple[int, int, str, Sequence[str], bool]],
    label_dict: dict[str, str],
    column_titles: Sequence[str],
    figure_title: str,
    output_name: str,
    color_dict: Optional[dict[str, str]] = None,
    linestyle_dict: Optional[dict[str, str]] = None,
) -> None:
    """Render a 1x2 grid (e.g., only linear or only log plots)."""
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), gridspec_kw={"width_ratios": [1, 1]}, sharey="row", layout='constrained')

    for _, col, bench_name, methods, log_scale in layout:
        plot_runtimes_from_csv(
            csv_path=_benchmark_path(bench_name, "csv"),
            methods=methods,
            time_column="time_persforest_s",
            label_dict=label_dict,
            color_dict=color_dict,
            linestyle_dict=linestyle_dict,
            ax=axes[col],
            log_scale=log_scale,
            title=None,
            show_axis_labels=False,
        )

    for col, title in enumerate(column_titles):
        axes[col].grid(False)
        axes[col].set_title(title)

        # bbox = axes[col].get_position()
        # x_center = bbox.x0 + bbox.width / 2
        # fig.text(x_center, bbox.y1 + 0.03, title, ha="center", va="bottom", fontsize=12, fontweight="bold")

    #fig.suptitle(figure_title, fontsize=13, y=0.94)
    fig.supxlabel("number of points")
    fig.supylabel("time/s")
    fig.savefig(_benchmark_path(output_name, "pdf"), dpi=300, transparent=True)
    plt.show()

if __name__ == "__main__":

    samplers = _make_samplers()

    methods_2d = ["uniform_2D","noisy_circle_noise-std-dot05","uniform_2D_with_30holes_radius-max-dot05"]
    methods_3d = ["uniform_3D","noisy_2sphere_noise-std-dot05","uniform_3D_with_30holes_radius-max-dot05"]

    # Example size schedules
    max_size_2d = 200000
    max_size_3d = 50000

    sizes_linear_2d = make_sizes_linear(min_size=20000, max_size=max_size_2d, count=10)
    sizes_log_2d = make_sizes_log(min_size=100, max_size=max_size_2d, count=20)
    sizes_linear_3d = make_sizes_linear(min_size=5000, max_size=max_size_3d, count=10)
    sizes_log_3d = make_sizes_log(min_size=100, max_size=max_size_3d, count=20)

    

    label_dict = {"uniform_2D":"uniform 2D",
                  "noisy_circle_noise-std-dot05":"perturbed 1-sphere",
                  "uniform_2D_with_30holes_radius-max-dot05":"uniform 2D with 30 holes", 
                  "uniform_3D":"uniform 3D",
                  "noisy_2sphere_noise-std-dot05":"perturbed 2-sphere",
                  "uniform_3D_with_30holes_radius-max-dot05":"uniform 3D with 30holes"}

    # Styling: use three non-default colors (outside Matplotlib's first four).
    custom_colors = ["#8da0cb", "#fc8d62", "#66c2a5"]
    linestyle_cycle = ["-", "--", "-.", ":"]
    all_methods = methods_2d + methods_3d
    linestyle_dict = {m: linestyle_cycle[i % len(linestyle_cycle)] for i, m in enumerate(all_methods)}
    color_dict = {m: custom_colors[i % len(custom_colors)] for i, m in enumerate(all_methods)}

    name_2d = "benchmark_2d_lin_max-200000_10steps_10reps"
    name_3d = "benchmark_3d_lin_max-50000_10steps_10reps"
    name_2d_log = "benchmark_2d_log_max-200000_20steps_10reps"
    name_3d_log = "benchmark_3d_log_max-50000_20steps_10reps"

    n_repeats = 10


    if False:
        benchmark_suite(
            samplers=samplers,
            methods=methods_2d,
            sizes=sizes_linear_2d,
            n_repeats=n_repeats,
            base_seed=12345,
            benchmark_name=name_2d,
            reduce=True,
            compute_barcode=True, 
        )

    if False:
        benchmark_suite(
            samplers=samplers,
            methods=methods_3d,
            sizes=sizes_linear_3d,
            n_repeats=n_repeats,
            base_seed=12345,
            benchmark_name=name_3d,
            reduce=True,
            compute_barcode=True, 
        )


    if True:
        # grid_spec = [
        #     (0, 0, name_2d, methods_2d, False),
        #     (0, 1, name_3d, methods_3d, False),
        #     (1, 0, name_2d_log, methods_2d, True),
        #     (1, 1, name_3d_log, methods_3d, True),
        # ]

        # plot_benchmark_grid(
        #     layout=grid_spec,
        #     label_dict=label_dict,
        #     column_titles=["2D benchmarks", "3D benchmarks"],
        #     row_titles=["Linear scale", "Log scale"],
        #     figure_title="PersistenceForest runtime vs. number of points",
        #     output_name="benchmark_grid",
        #     color_dict=color_dict,
        # )

        # linear_spec = [
        #     (0, 0, name_2d, methods_2d, False),
        #     (0, 1, name_3d, methods_3d, False),
        # ]
        # plot_benchmark_row(
        #     layout=linear_spec,
        #     label_dict=label_dict,
        #     column_titles=["2D benchmarks", "3D benchmarks"],
        #     figure_title="PersistenceForest runtime (linear scale)",
        #     output_name="benchmark_linear_only",
        #     color_dict=color_dict,
        # )

        log_spec = [
            (0, 0, name_2d_log, methods_2d, True),
            (0, 1, name_3d_log, methods_3d, True),
        ]
        plot_benchmark_row(
            layout=log_spec,
            label_dict=label_dict,
            column_titles=["2D", "3D"],
            figure_title="PersistenceForest runtime (log-log scale)",
            output_name="benchmark_log_only",
            color_dict=color_dict,
        )
