"""RQ3 (multi-timescale design) ablation visualization for the O-RAN track.

Concept Note ORAN_BMPP_DQN_Concept_Note_v1.md Section 6.11: RQ3 asks how
the two-timescale design (separate upper/lower Q-networks, discrete
decisions held for `upper_level_period_steps` env steps) affects learning
convergence and energy efficiency. Isolated by collapsing
`upper_level_period_steps` from 10 to 1 (config/rq3_single_timescale.yaml)
-- degenerating the two-timescale separation to a no-op while keeping the
same architecture -- and re-running BMPP-DQN at the same 3-seed/
500-episode protocol (`data/results_oran_rq3_single_timescale/`).

Reuses oran_evaluation.convergence's loader rather than re-deriving it.
"""

from pathlib import Path

import numpy as np

from oran_evaluation.convergence import load_algo_seed_metrics

DEFAULT_TWO_TIMESCALE_DIR = "data/results_oran"
DEFAULT_SINGLE_TIMESCALE_DIR = "data/results_oran_rq3_single_timescale"


def plot_rq3_ablation(
    two_timescale_dir: str = DEFAULT_TWO_TIMESCALE_DIR,
    single_timescale_dir: str = DEFAULT_SINGLE_TIMESCALE_DIR,
    save_path: str = "thesis/figures_oran/rq3_ablation_oran.pdf",
) -> None:
    """Two-panel bar chart (reward, switching frequency) comparing
    BMPP-DQN's two-timescale (proposed) design against the single-
    timescale ablation, mean +/- std across the same 3 seeds. These are
    the two metrics Section 6.11's own analysis found move materially
    between the two designs (switching frequency: ~6x higher without the
    two-timescale separation, the largest effect size found; reward:
    no significant difference) -- power, QoS, and throughput are omitted
    here since none differ meaningfully (see the table this figure
    accompanies for the full metric set).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    two = load_algo_seed_metrics(two_timescale_dir)["BMPP_DQN"]
    one = load_algo_seed_metrics(single_timescale_dir)["BMPP_DQN"]

    labels = ["Two-Timescale\n(Proposed)", "Single-Timescale\n(Ablation)"]
    colors = ["#2ca02c", "#9467bd"]

    fig, (ax_reward, ax_switch) = plt.subplots(1, 2, figsize=(10, 5))

    for ax, key, ylabel in [
        (ax_reward, "reward", "Mean Evaluation Reward"),
        (ax_switch, "switching", "Mean Switching Events (per step)"),
    ]:
        two_vals = [v[key] for v in two.values()]
        one_vals = [v[key] for v in one.values()]
        means = [float(np.mean(two_vals)), float(np.mean(one_vals))]
        stds = [float(np.std(two_vals)), float(np.std(one_vals))]
        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
        for bar, mean in zip(bars, means):
            ax.annotate(
                f"{mean:.2f}" if key == "switching" else f"{mean:.0f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9,
            )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "O-RAN Track: RQ3 Ablation -- Effect of the Two-Timescale Design\n"
        "(upper_level_period_steps=10 vs. 1, BMPP-DQN, 3 seeds, 500 episodes)",
        fontsize=11,
    )
    fig.tight_layout()
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file.with_suffix(".pdf"))
    fig.savefig(out_file.with_suffix(".png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_rq3_ablation()
