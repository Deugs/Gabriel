"""Publication-Quality Plotting Utilities for C-RAN Thesis Evaluation."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.use("Agg")


def setup_matplotlib_style():
    """Configure Matplotlib for IEEE/Nature manuscript style figures."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )


def compute_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sample mean and 95% confidence interval bounds.

    Args:
        data (np.ndarray): Array of shape (n_seeds, n_steps) or (n_seeds,).
        confidence (float): Confidence level (default: 0.95).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Mean, lower bound, upper bound.
    """
    mean = np.mean(data, axis=0)
    if data.ndim == 1 or data.shape[0] <= 1:
        return mean, mean, mean

    n = data.shape[0]
    std_err = stats.sem(data, axis=0)
    h = std_err * stats.t.ppf((1.0 + confidence) / 2.0, n - 1)
    return mean, mean - h, mean + h


def plot_learning_curves(
    results_dict: Dict[str, np.ndarray],
    xlabel: str = "Episodes",
    ylabel: str = "Reward",
    title: str = "Learning Curve Comparison",
    save_path: Optional[str] = None,
):
    """Plot learning curves with 95% confidence interval shaded regions."""
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (algo_name, data) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        mean, lower, upper = compute_confidence_interval(data)
        x = np.arange(1, len(mean) + 1)

        ax.plot(x, mean, label=algo_name, color=color, linewidth=2.0)
        ax.fill_between(x, lower, upper, color=color, alpha=0.15)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    if save_path is not None:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file)
        plt.close(fig)
    else:
        plt.close(fig)


def plot_energy_efficiency_bar(
    metrics_dict: Dict[str, Dict[str, float]],
    ylabel: str = "Energy Efficiency (Mbit/Joule)",
    title: str = "Energy Efficiency Comparison",
    save_path: Optional[str] = None,
):
    """Plot bar chart comparing energy efficiency across algorithms."""
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    algos = list(metrics_dict.keys())
    means = [metrics_dict[a]["mean"] for a in algos]
    stds = [metrics_dict[a].get("std", 0.0) for a in algos]

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    bars = ax.bar(
        algos, means, yerr=stds, capsize=5, color=colors[: len(algos)], alpha=0.85
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    if save_path is not None:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file)
        plt.close(fig)
    else:
        plt.close(fig)


def plot_scalability_analysis(
    scalability_dict: Dict[str, Dict[str, Dict[str, float]]],
    save_path: Optional[str] = None,
):
    """Plot network scalability metrics (Execution Time & Total Power)."""
    setup_matplotlib_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    scales = list(
        scalability_dict.keys()
    )  # e.g. ["R=5, U=2", "R=12, U=10", "R=20, U=20", "R=35, U=25", "R=50, U=30 (stretch)"]

    for algo in scalability_dict[scales[0]].keys():
        powers = [scalability_dict[s][algo]["power"] for s in scales]
        times = [scalability_dict[s][algo]["time"] for s in scales]

        ax1.plot(scales, powers, marker="o", label=algo, linewidth=2.0)
        ax2.plot(scales, times, marker="s", label=algo, linewidth=2.0)

    ax1.set_ylabel("Mean Total Power (W)")
    ax1.set_title("Power Scaling vs Network Size")
    ax1.legend()

    ax2.set_ylabel("Step Execution Time (ms)")
    ax2.set_title("Computation Time vs Network Size")
    ax2.legend()

    fig.tight_layout()

    if save_path is not None:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file)
        plt.close(fig)
    else:
        plt.close(fig)


def plot_degradation_curve(
    curve_dict: Dict[str, Dict[float, float]],
    xlabel: str = "Perturbation level",
    ylabel: str = "Energy Efficiency (Mbit/Joule)",
    title: str = "Robustness Degradation Curve",
    save_path: Optional[str] = None,
):
    """Plot a metric-vs-perturbation-level degradation curve per method.

    Used for the CSI-robustness (Concept Note v3.0/v4.0 Section 12.5, S3) and
    cross-profile generalization (Section 12.3, A5) evaluations, both of which
    report a metric as a function of a single scalar perturbation/condition.

    Args:
        curve_dict: {method_name: {x_value: y_value, ...}, ...}.
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (method, points) in enumerate(curve_dict.items()):
        xs = sorted(points.keys())
        ys = [points[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=method, color=colors[idx % len(colors)], linewidth=2.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    if save_path is not None:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file)
        plt.close(fig)
    else:
        plt.close(fig)


def plot_ablation_comparison(
    ablation_dict: Dict[str, float],
    save_path: Optional[str] = None,
):
    """Plot ablation study metric comparison."""
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    variants = list(ablation_dict.keys())
    rewards = list(ablation_dict.values())

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.barh(variants, rewards, color=colors[: len(variants)], alpha=0.85)

    ax.set_xlabel("Mean Evaluation Reward")
    ax.set_title("Ablation Study: Architecture Component Contributions")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.2f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    if save_path is not None:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file)
    plt.close(fig)
    plt.close("all")
