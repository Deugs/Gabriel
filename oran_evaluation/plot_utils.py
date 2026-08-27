"""Plotting/statistics utilities for the O-RAN track's evaluation modules.

Deliberately duplicates evaluation/plot_utils.py's tiny statistical
helpers (compute_confidence_interval) rather than importing them, per this
package's decoupling guarantee -- see oran_env/README-equivalent
docstrings for the same rationale. Written correctly from the start: the
C-RAN version's compute_confidence_interval() had a bug (fixed in an
earlier round) where `data.ndim == 1` alone triggered the degenerate
zero-width-CI fallback regardless of actual seed count; this version's
guard is just `data.shape[0] <= 1`.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.use("Agg")


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
    if data.shape[0] <= 1:
        return mean, mean, mean

    n = data.shape[0]
    std_err = stats.sem(data, axis=0)
    h = std_err * stats.t.ppf((1.0 + confidence) / 2.0, n - 1)
    return mean, mean - h, mean + h


def plot_bar_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    ylabel: str = "Value",
    title: str = "Comparison",
    save_path: Optional[str] = None,
):
    """Plot a bar chart comparing a metric (mean +/- std) across methods."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    methods = list(metrics_dict.keys())
    means = [metrics_dict[m]["mean"] for m in methods]
    stds = [metrics_dict[m].get("std", 0.0) for m in methods]

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    bars = ax.bar(
        methods, means, yerr=stds, capsize=5, color=colors[: len(methods)], alpha=0.85
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
