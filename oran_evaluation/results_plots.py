"""Comparison bar charts for the O-RAN track's 4-method benchmark.

Generates one bar chart per headline metric (reward, power, QoS -- strict
and per-UE, switching frequency, throughput) plus a derived throughput-
per-watt efficiency chart, each showing every method found under
`results_dir` with mean +/- std error bars across seeds.

Reuses oran_evaluation.convergence.load_algo_seed_metrics() -- the same
per-seed parsing analyze_convergence() uses for the LaTeX summary table --
so the figures and the table are always derived from identical data, and
plot_utils.plot_bar_comparison for rendering consistent with this track's
other figures (e.g. latency_benchmark_oran.pdf).
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from oran_evaluation.convergence import PROPOSED_ALGO, load_algo_seed_metrics
from oran_evaluation.plot_utils import plot_bar_comparison

ALGO_LABELS: Dict[str, str] = {
    "BMPP_DQN": "BMPP-DQN",
    "dqn": "DQN",
    "ddpg": "DDPG",
    "mpdqn": "MP-DQN",
}

# (metric key in load_algo_seed_metrics()'s per-seed dict, y-axis label,
# output filename, whether to scale the mean/std by 100 for a percentage).
_FIGURES = [
    ("reward", "Mean Evaluation Reward", "reward_comparison_oran.pdf", False),
    ("power", "Mean Power (W)", "power_comparison_oran.pdf", False),
    (
        "qos",
        "QoS Rate -- All UEs Satisfied (%)",
        "qos_strict_comparison_oran.pdf",
        True,
    ),
    ("qos_per_ue", "Per-UE QoS Rate (%)", "qos_per_ue_comparison_oran.pdf", True),
    (
        "switching",
        "Switching Frequency (events/step)",
        "switching_comparison_oran.pdf",
        False,
    ),
    ("throughput", "Mean Throughput (Mbps)", "throughput_comparison_oran.pdf", False),
]


def label_for_algo(algo: str) -> str:
    return ALGO_LABELS.get(algo, algo)


def ordered_algos(metrics: Dict[str, Any]) -> List[str]:
    """Proposed method first, then baselines alphabetically -- matches the
    convergence table's own reading order."""
    return [a for a in metrics if a == PROPOSED_ALGO] + sorted(
        a for a in metrics if a != PROPOSED_ALGO
    )


def generate_comparison_plots(
    results_dir: str = "data/results_oran",
    save_dir: str = "thesis/figures_oran",
) -> Dict[str, Dict[str, float]]:
    """Generate one bar-chart figure per headline metric, all methods found
    under `results_dir`, plus a derived throughput-per-watt efficiency
    chart.

    Returns {metric: {method_label: mean}} for callers that want the
    plotted numbers without re-parsing the results directory.
    """
    fig_path = Path(save_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    metrics = load_algo_seed_metrics(results_dir)
    if not metrics:
        print(f"No summary.json files found under {results_dir} -- nothing to plot.")
        return {}

    algos = ordered_algos(metrics)
    summary: Dict[str, Dict[str, float]] = {}

    for field, ylabel, filename, as_percent in _FIGURES:
        series: Dict[str, Dict[str, float]] = {}
        for algo in algos:
            vals = [m[field] for m in metrics[algo].values()]
            if not vals:
                continue
            scale = 100.0 if as_percent else 1.0
            series[label_for_algo(algo)] = {
                "mean": float(np.mean(vals)) * scale,
                "std": (float(np.std(vals)) if len(vals) > 1 else 0.0) * scale,
            }
        plot_bar_comparison(
            series,
            ylabel=ylabel,
            title=f"O-RAN Track: {ylabel} by Method",
            save_path=str(fig_path / filename),
        )
        print(f"Saved {fig_path / filename}")
        summary[field] = {label: v["mean"] for label, v in series.items()}

    # Derived: throughput-per-watt, computed per seed (throughput/power for
    # that seed's own run) then averaged -- not throughput-of-means over
    # power-of-means, so a low-throughput/low-power outlier seed (e.g. a
    # collapsed run) contributes its own genuinely low ratio rather than
    # being smoothed into an aggregate. Seeds with ~0 power are skipped to
    # avoid a division blow-up rather than silently producing inf/nan.
    efficiency: Dict[str, Dict[str, float]] = {}
    for algo in algos:
        ratios = [
            m["throughput"] / m["power"]
            for m in metrics[algo].values()
            if m["power"] > 1e-6
        ]
        if not ratios:
            continue
        efficiency[label_for_algo(algo)] = {
            "mean": float(np.mean(ratios)),
            "std": float(np.std(ratios)) if len(ratios) > 1 else 0.0,
        }
    plot_bar_comparison(
        efficiency,
        ylabel="Throughput per Watt (Mbps/W)",
        title="O-RAN Track: Energy Efficiency (Throughput/Power) by Method",
        save_path=str(fig_path / "efficiency_comparison_oran.pdf"),
    )
    print(f"Saved {fig_path / 'efficiency_comparison_oran.pdf'}")
    summary["efficiency_mbps_per_w"] = {
        label: v["mean"] for label, v in efficiency.items()
    }

    return summary


def plot_pareto_scatter(
    results_dir: str = "data/results_oran",
    save_dir: str = "thesis/figures_oran",
    x_field: str = "power",
    y_field: str = "throughput",
) -> None:
    """Scatter of one seed-mean point per method (x=power, y=throughput by
    default), with error bars in both dimensions (std across seeds) and
    the Pareto-optimal frontier (methods no other method beats on *both*
    axes simultaneously) connected and highlighted.

    The energy-efficiency question this thesis asks -- "does the proposed
    method deliver more service per Watt?" -- is a two-objective trade-off
    (minimize power, maximize throughput), which a single bar chart per
    metric can't show jointly; this figure shows both at once and makes
    Pareto-dominance visually explicit rather than requiring the reader to
    cross-reference two separate bar charts.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = load_algo_seed_metrics(results_dir)
    if not metrics:
        print(f"No summary.json files found under {results_dir} -- nothing to plot.")
        return

    algos = ordered_algos(metrics)
    points = []
    for algo in algos:
        xs = [m[x_field] for m in metrics[algo].values()]
        ys = [m[y_field] for m in metrics[algo].values()]
        if not xs:
            continue
        points.append(
            {
                "algo": algo,
                "label": label_for_algo(algo),
                "x": float(np.mean(xs)),
                "x_std": float(np.std(xs)) if len(xs) > 1 else 0.0,
                "y": float(np.mean(ys)),
                "y_std": float(np.std(ys)) if len(ys) > 1 else 0.0,
            }
        )

    # Pareto-optimal (minimize x, maximize y): a point is dominated if some
    # other point has both x <= it and y >= it, with at least one strict.
    for p in points:
        p["pareto"] = not any(
            (q["x"] <= p["x"] and q["y"] >= p["y"])
            and (q["x"] < p["x"] or q["y"] > p["y"])
            for q in points
            if q is not p
        )

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, p in enumerate(points):
        marker = "*" if p["pareto"] else "o"
        size = 220 if p["pareto"] else 120
        ax.errorbar(
            p["x"], p["y"], xerr=p["x_std"], yerr=p["y_std"],
            fmt="none", ecolor=colors[i % len(colors)], alpha=0.5, capsize=4,
        )
        ax.scatter(
            p["x"], p["y"], s=size, marker=marker, color=colors[i % len(colors)],
            edgecolors="black", linewidths=0.8, zorder=3,
            label=f"{p['label']}" + (" (Pareto-optimal)" if p["pareto"] else ""),
        )
        ax.annotate(
            p["label"], (p["x"], p["y"]), textcoords="offset points",
            xytext=(8, 8), fontsize=9,
        )

    pareto_pts = sorted((p for p in points if p["pareto"]), key=lambda p: p["x"])
    if len(pareto_pts) > 1:
        ax.plot(
            [p["x"] for p in pareto_pts], [p["y"] for p in pareto_pts],
            linestyle="--", color="gray", alpha=0.6, zorder=1,
            label="Pareto frontier",
        )

    ax.set_xlabel("Mean Power (W) -- lower is better")
    ax.set_ylabel("Mean Throughput (Mbps) -- higher is better")
    ax.set_title("O-RAN Track: Power-Throughput Trade-off (Pareto Frontier)")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_file = Path(save_dir) / "pareto_power_throughput_oran.pdf"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {out_file} (+ .png)")


def plot_convergence_curve(
    results_dir: str = "data/results_oran",
    save_dir: str = "thesis/figures_oran",
) -> None:
    """Learning curve for the proposed method (BMPP-DQN): held-out eval
    reward vs. training episode, mean +/- 1 std across seeds (each seed's
    own summary.json "history.eval_history" checkpoints), with each
    baseline's final mean reward overlaid as a horizontal reference line.

    Not a true 4-method convergence-curve figure: baselines
    (oran_training/train_oran_baselines.py) record only each seed's final
    aggregate result, not a checkpointed trajectory, so there is no
    per-episode series to plot for them -- only where they ultimately
    landed, shown as a reference line against the proposed method's own
    training progress.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_path = Path(results_dir)
    seed_dirs = sorted(results_path.glob("bmpp_dqn_seed*"))
    if not seed_dirs:
        print(f"No bmpp_dqn_seed* directories found under {results_dir}.")
        return

    # episode -> list of eval_mean_reward across seeds that reached it.
    by_episode: Dict[int, List[float]] = {}
    for seed_dir in seed_dirs:
        summary_file = seed_dir / "summary.json"
        if not summary_file.exists():
            continue
        with open(summary_file) as f:
            data = json.load(f)
        for point in data.get("history", {}).get("eval_history", []):
            by_episode.setdefault(int(point["episode"]), []).append(
                float(point["eval_mean_reward"])
            )

    if not by_episode:
        print("No eval_history found in any bmpp_dqn_seed*/summary.json.")
        return

    episodes = sorted(by_episode)
    means = np.array([np.mean(by_episode[e]) for e in episodes])
    stds = np.array(
        [np.std(by_episode[e]) if len(by_episode[e]) > 1 else 0.0 for e in episodes]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episodes, means, color="#2ca02c", linewidth=2, label="BMPP-DQN (proposed)")
    ax.fill_between(
        episodes, means - stds, means + stds, color="#2ca02c", alpha=0.2,
        label="+/-1 std across seeds",
    )

    baseline_metrics = load_algo_seed_metrics(results_dir)
    ref_colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    for i, algo in enumerate(a for a in baseline_metrics if a != PROPOSED_ALGO):
        rewards = [m["reward"] for m in baseline_metrics[algo].values()]
        if not rewards:
            continue
        ax.axhline(
            float(np.mean(rewards)), linestyle="--", color=ref_colors[i % len(ref_colors)],
            alpha=0.8, label=f"{label_for_algo(algo)} (final mean)",
        )

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Held-out Evaluation Reward")
    ax.set_title(
        "O-RAN Track: BMPP-DQN Learning Curve vs. Baseline Final Performance"
    )
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_file = Path(save_dir) / "convergence_curve_oran.pdf"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {out_file} (+ .png)")


if __name__ == "__main__":
    generate_comparison_plots()
    plot_pareto_scatter()
    plot_convergence_curve()
