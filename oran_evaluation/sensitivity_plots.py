"""Power-model sensitivity sweep visualizations for the O-RAN track.

Concept Note ORAN_BMPP_DQN_Concept_Note_v1.md Section 6.9: each of the
RU/DU/CU/fronthaul power-model constant groups was independently scaled
10x and 0.1x (all others held at their Section 10.5 default), with the
full 4-method/3-seed/500-episode protocol re-run under each of the 8
resulting configurations, on top of the original default. This module
visualizes that sweep -- reuses oran_evaluation.convergence's loader and
oran_evaluation.multicriteria's compute_topsis() for scoring, rather than
re-deriving either.

The tabular companion to these figures is
thesis/tables_oran/sensitivity_summary_oran.{tex,csv}
(oran_evaluation.multicriteria's own docstring/table-export convention);
this module was added after that table shipped without a visual
counterpart -- a gap caught and fixed, not part of the original sweep.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from oran_evaluation.convergence import load_algo_seed_metrics
from oran_evaluation.multicriteria import compute_topsis
from oran_evaluation.results_plots import label_for_algo, ordered_algos

# (display name, results directory) for the default config plus all 8
# sensitivity perturbations, in the same order as the concept note's own
# Section 6.9 table.
SENSITIVITY_CONFIGS: List[Tuple[str, str]] = [
    ("Default", "data/results_oran"),
    ("RU x10", "data/results_oran_sensitivity/ru_10x"),
    ("RU x0.1", "data/results_oran_sensitivity/ru_0p1x"),
    ("DU x10", "data/results_oran_sensitivity/du_10x"),
    ("DU x0.1", "data/results_oran_sensitivity/du_0p1x"),
    ("CU x10", "data/results_oran_sensitivity/cu_10x"),
    ("CU x0.1", "data/results_oran_sensitivity/cu_0p1x"),
    ("Fronthaul x10", "data/results_oran_sensitivity/fronthaul_10x"),
    ("Fronthaul x0.1", "data/results_oran_sensitivity/fronthaul_0p1x"),
]


def plot_sensitivity_topsis(
    configs: Optional[List[Tuple[str, str]]] = None,
    save_path: str = "thesis/figures_oran/sensitivity_topsis_oran.pdf",
) -> None:
    """Bar chart of BMPP-DQN's TOPSIS composite score across every power-
    model sensitivity configuration, colored by whether it still ranks
    #1 of 4 that configuration, with its rank annotated on each bar --
    the direct visual companion to
    thesis/tables_oran/sensitivity_summary_oran.tex, showing at a glance
    that the score stays in a narrow band and the rank stays #1 in 8 of
    9 configurations, regardless of which power-model constant group is
    perturbed or in which direction.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    configs = configs or SENSITIVITY_CONFIGS
    names: List[str] = []
    scores: List[float] = []
    ranks: List[int] = []
    for name, path in configs:
        m = load_algo_seed_metrics(path)
        topsis = compute_topsis(m)
        names.append(name)
        scores.append(topsis["BMPP_DQN"]["topsis_score"])
        ranks.append(int(topsis["BMPP_DQN"]["rank"]))

    rank1_color, other_color = "#2ca02c", "#d62728"
    colors = [rank1_color if r == 1 else other_color for r in ranks]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        names, scores, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5
    )
    for bar, rank in zip(bars, ranks):
        ax.annotate(
            f"#{rank}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
        )
    ax.axhline(scores[0], linestyle="--", color="gray", alpha=0.6, zorder=1)

    ax.set_ylabel("BMPP-DQN TOPSIS Composite Score")
    ax.set_title(
        "O-RAN Track: Power-Model Sensitivity Sweep\n"
        "BMPP-DQN's Multi-Criteria Rank (of 4) Across Every Configuration"
    )
    ax.set_ylim(0, 1.0)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(
        handles=[
            Patch(color=rank1_color, label="TOPSIS rank #1 of 4"),
            Patch(color=other_color, label="TOPSIS rank #2+ of 4"),
        ],
        loc="lower right", fontsize=8,
    )

    fig.tight_layout()
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file.with_suffix(".pdf"))
    fig.savefig(out_file.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_sensitivity_reward_power(
    configs: Optional[List[Tuple[str, str]]] = None,
    save_path: str = "thesis/figures_oran/sensitivity_reward_power_oran.pdf",
) -> None:
    """Two-panel grouped bar chart (mean reward, mean power) for all 4
    methods across every power-model sensitivity configuration.

    Section 6.9's own prose reports two qualitative, cross-method
    findings ("MP-DQN has the best raw reward and BMPP-DQN never does",
    "BMPP-DQN's composite rank is driven by competitive power and the
    lowest switching") -- this figure is the visual evidence for the
    first half of that claim (reward, power) across every method, not
    just BMPP-DQN's own rank (see plot_sensitivity_topsis() for that).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = configs or SENSITIVITY_CONFIGS
    default_metrics = load_algo_seed_metrics(configs[0][1])
    algos = ordered_algos(default_metrics)
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

    reward_by_algo: Dict[str, List[float]] = {a: [] for a in algos}
    power_by_algo: Dict[str, List[float]] = {a: [] for a in algos}
    names = [name for name, _ in configs]
    for _, path in configs:
        m = load_algo_seed_metrics(path)
        for a in algos:
            vals = list(m[a].values())
            reward_by_algo[a].append(float(np.mean([v["reward"] for v in vals])))
            power_by_algo[a].append(float(np.mean([v["power"] for v in vals])))

    x = np.arange(len(configs))
    width = 0.8 / len(algos)

    fig, (ax_reward, ax_power) = plt.subplots(1, 2, figsize=(15, 5.5))
    for i, algo in enumerate(algos):
        offset = (i - (len(algos) - 1) / 2) * width
        color = colors[i % len(colors)]
        ax_reward.bar(
            x + offset, reward_by_algo[algo], width,
            label=label_for_algo(algo), color=color, alpha=0.85,
        )
        ax_power.bar(
            x + offset, power_by_algo[algo], width,
            label=label_for_algo(algo), color=color, alpha=0.85,
        )

    ax_reward.set_ylabel("Mean Evaluation Reward")
    ax_reward.set_title("Reward by Method Across Sensitivity Configurations")
    ax_power.set_ylabel("Mean Power (W)")
    ax_power.set_title("Power by Method Across Sensitivity Configurations")
    for ax in (ax_reward, ax_power):
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "O-RAN Track: Power-Model Sensitivity Sweep -- All 4 Methods",
        fontsize=12,
    )
    fig.tight_layout()
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file.with_suffix(".pdf"))
    fig.savefig(out_file.with_suffix(".png"), dpi=150)
    plt.close(fig)


def run_sensitivity_plots(
    configs: Optional[List[Tuple[str, str]]] = None,
    save_dir: str = "thesis/figures_oran",
) -> None:
    """End-to-end: generate both sensitivity-sweep figures."""
    configs = configs or SENSITIVITY_CONFIGS
    fig_path = Path(save_dir)
    plot_sensitivity_topsis(
        configs, save_path=str(fig_path / "sensitivity_topsis_oran.pdf")
    )
    print(f"Saved {fig_path / 'sensitivity_topsis_oran.pdf'} (+ .png)")
    plot_sensitivity_reward_power(
        configs, save_path=str(fig_path / "sensitivity_reward_power_oran.pdf")
    )
    print(f"Saved {fig_path / 'sensitivity_reward_power_oran.pdf'} (+ .png)")


if __name__ == "__main__":
    run_sensitivity_plots()
