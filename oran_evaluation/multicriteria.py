"""Multi-criteria composite ranking for the O-RAN track's 4-method benchmark.

Uses TOPSIS (Technique for Order Preference by Similarity to Ideal
Solution; Hwang & Yoon, 1981) -- a standard, citable multi-criteria
decision-making method -- to combine several metrics of different units
and scales (Watts, percentages, Mbps, events/step) into one bounded [0, 1]
composite score per method, without needing to hand-pick a common unit or
an ad-hoc weighted-sum formula.

Deliberately excludes "reward" from the criteria set: reward is already
itself a weighted combination of energy efficiency, QoS violation, and
switching cost (the environment's own alpha/beta/gamma-weighted reward
function, Concept Note Section 10.1-equivalent), baked in at *training*
time. Including it alongside its own components here would double-count
those factors. This module's composite score is instead an independent,
equal-weighted audit across the observable operational metrics --
a cross-check on the reward-based comparison in
convergence_summary_oran.tex, not a replacement for it.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from oran_evaluation.convergence import load_algo_seed_metrics
from oran_evaluation.results_plots import label_for_algo, ordered_algos

# Criteria used for the composite score, and whether higher is better
# (True, a "benefit" criterion) or lower is better (False, a "cost"
# criterion). power and switching are costs; qos, qos_per_ue, and
# throughput are benefits.
CRITERIA: List[str] = ["power", "qos", "qos_per_ue", "switching", "throughput"]
BENEFIT: Dict[str, bool] = {
    "power": False,
    "qos": True,
    "qos_per_ue": True,
    "switching": False,
    "throughput": True,
}
CRITERION_LABELS: Dict[str, str] = {
    "power": "Power (W)",
    "qos": "QoS Strict (%)",
    "qos_per_ue": "QoS Per-UE (%)",
    "switching": "Switching Freq.",
    "throughput": "Throughput (Mbps)",
}


def compute_topsis(
    metrics: Dict[str, Dict[int, Dict[str, float]]],
    criteria: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Rank methods by TOPSIS similarity to an ideal solution.

    Steps (standard TOPSIS):
      1. Decision matrix: one row per method, one column per criterion,
         each entry that method's own mean across its seeds.
      2. Vector-normalize each column (divide by its Euclidean norm) so
         criteria on very different raw scales (Watts vs. percentages vs.
         Mbps) contribute comparably -- no manual unit conversion needed.
      3. Apply criterion weights (equal by default: 1/len(criteria) each,
         since no criterion is given a privileged role here -- see this
         module's own docstring for why "reward" itself is excluded
         rather than weighted).
      4. Ideal-best/ideal-worst vectors: per criterion, the best value
         across methods for a benefit criterion (or the worst value for a
         cost criterion), and vice versa for ideal-worst.
      5. Score = distance-to-ideal-worst / (distance-to-ideal-best +
         distance-to-ideal-worst), in [0, 1]; higher is closer to ideal.

    Returns {algo: {"topsis_score", "rank", "mean_<criterion>": ...}}.
    """
    criteria = criteria or CRITERIA
    weights = weights or {c: 1.0 / len(criteria) for c in criteria}

    algos = [a for a in metrics if metrics[a]]
    if not algos:
        return {}

    matrix = np.array(
        [
            [np.mean([m[c] for m in metrics[algo].values()]) for c in criteria]
            for algo in algos
        ]
    )

    norms = np.sqrt((matrix**2).sum(axis=0))
    norms[norms == 0] = 1.0  # a constant-zero column would otherwise divide by zero
    normalized = matrix / norms

    w = np.array([weights[c] for c in criteria])
    weighted = normalized * w

    ideal_best = np.array(
        [
            weighted[:, i].max() if BENEFIT[c] else weighted[:, i].min()
            for i, c in enumerate(criteria)
        ]
    )
    ideal_worst = np.array(
        [
            weighted[:, i].min() if BENEFIT[c] else weighted[:, i].max()
            for i, c in enumerate(criteria)
        ]
    )

    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    denom = dist_best + dist_worst
    scores = np.where(denom > 0, dist_worst / np.where(denom == 0, 1.0, denom), 0.5)

    order = np.argsort(-scores)
    ranks = {algos[idx]: int(pos + 1) for pos, idx in enumerate(order)}

    result: Dict[str, Dict[str, float]] = {}
    for i, algo in enumerate(algos):
        entry: Dict[str, float] = {
            "topsis_score": float(scores[i]),
            "rank": float(ranks[algo]),
        }
        for j, c in enumerate(criteria):
            entry[f"mean_{c}"] = float(matrix[i, j])
        result[algo] = entry
    return result


def plot_composite_score(
    topsis_results: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    """Bar chart of each method's TOPSIS composite score, best (rank 1) first."""
    from oran_evaluation.plot_utils import plot_bar_comparison

    ordered = sorted(topsis_results.items(), key=lambda kv: kv[1]["rank"])
    series = {
        label_for_algo(algo): {"mean": r["topsis_score"]} for algo, r in ordered
    }
    plot_bar_comparison(
        series,
        ylabel="TOPSIS Composite Score (0-1, higher = closer to ideal)",
        title="O-RAN Track: Multi-Criteria Composite Ranking (power, QoS, "
        "switching, throughput)",
        save_path=save_path,
    )


def plot_radar(
    metrics: Dict[str, Dict[int, Dict[str, float]]],
    save_path: str,
    criteria: Optional[List[str]] = None,
) -> None:
    """Radar/spider chart comparing all methods across all criteria at once.

    Each axis is min-max normalized across methods to [0, 1] *oriented so
    that 1 is always better* (a cost criterion like power is flipped:
    1 = lowest power observed), purely for this visualization -- distinct
    from TOPSIS's own vector normalization used for scoring above. A
    method whose polygon is uniformly near the outer edge is good on
    every axis simultaneously; a spiky polygon reveals a specific
    trade-off (e.g. great QoS, poor power).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    criteria = criteria or CRITERIA
    algos = [a for a in metrics if metrics[a]]
    algos = ordered_algos({a: metrics[a] for a in algos})

    matrix = np.array(
        [
            [np.mean([m[c] for m in metrics[algo].values()]) for c in criteria]
            for algo in algos
        ]
    )
    norm = np.zeros_like(matrix)
    for j, c in enumerate(criteria):
        col = matrix[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            norm[:, j] = 0.5
        else:
            scaled = (col - lo) / (hi - lo)
            norm[:, j] = scaled if BENEFIT[c] else (1.0 - scaled)

    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    for i, algo in enumerate(algos):
        values = norm[i].tolist()
        values += values[:1]
        ax.plot(
            angles, values, color=colors[i % len(colors)], linewidth=2,
            label=label_for_algo(algo),
        )
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([CRITERION_LABELS.get(c, c) for c in criteria])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(
        "O-RAN Track: Normalized Multi-Criteria Profile\n"
        "(each axis independently scaled 0-1, always oriented so outward = better)",
        fontsize=10,
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file.with_suffix(".pdf"))
    fig.savefig(out_file.with_suffix(".png"), dpi=150)
    plt.close(fig)


def export_multicriteria_table(
    topsis_results: Dict[str, Dict[str, float]],
    table_save_dir: str,
    criteria: Optional[List[str]] = None,
) -> str:
    """Write a LaTeX table of the per-criterion means, TOPSIS score, and
    rank for every method, sorted best-to-worst."""
    criteria = criteria or CRITERIA
    table_path = Path(table_save_dir)
    table_path.mkdir(parents=True, exist_ok=True)

    ordered = sorted(topsis_results.items(), key=lambda kv: kv[1]["rank"])
    col_headers = " & ".join(CRITERION_LABELS.get(c, c) for c in criteria)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{O-RAN Track: Multi-Criteria Composite Ranking (TOPSIS). "
        "Excludes reward (already a weighted combination of these same "
        "operational metrics at training time -- see this module's own "
        "docstring); an independent, equal-weighted cross-check, not a "
        "replacement for the reward-based comparison.}",
        "\\begin{tabular}{l" + "c" * (len(criteria) + 2) + "}",
        "\\hline",
        f"Algorithm & {col_headers} & TOPSIS Score & Rank \\\\",
        "\\hline",
    ]
    # qos/qos_per_ue are stored as raw [0, 1] fractions in the per-seed
    # metrics dict (matching load_algo_seed_metrics()'s convention) but
    # this table's own column headers read "(%)" -- scale only those two
    # for display, not the whole matrix.
    percent_criteria = {"qos", "qos_per_ue"}

    def _fmt(c: str, value: float) -> str:
        return f"{value * 100:.1f}" if c in percent_criteria else f"{value:.2f}"

    for algo, r in ordered:
        crit_vals = " & ".join(_fmt(c, r[f"mean_{c}"]) for c in criteria)
        lines.append(
            f"{label_for_algo(algo)} & {crit_vals} & {r['topsis_score']:.3f} & "
            f"{int(r['rank'])} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]

    out_path = table_path / "multicriteria_summary_oran.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Exported multi-criteria summary LaTeX table to {out_path}")

    # A plain CSV alongside the LaTeX, for anyone who wants the raw numbers
    # without parsing LaTeX (e.g. to double-check the ranking by hand).
    csv_path = table_path / "multicriteria_summary_oran.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", *criteria, "topsis_score", "rank"])
        for algo, r in ordered:
            writer.writerow(
                [
                    label_for_algo(algo),
                    *[r[f"mean_{c}"] for c in criteria],
                    r["topsis_score"],
                    int(r["rank"]),
                ]
            )
    print(f"Exported multi-criteria summary CSV to {csv_path}")

    return str(out_path)


def run_multicriteria_analysis(
    results_dir: str = "data/results_oran",
    save_dir: str = "thesis/figures_oran",
    table_save_dir: str = "thesis/tables_oran",
) -> Dict[str, Dict[str, float]]:
    """End-to-end: load results, compute TOPSIS, plot the composite-score
    bar chart and the radar chart, export the LaTeX+CSV table.

    Returns the TOPSIS results dict (see compute_topsis()'s docstring).
    """
    metrics = load_algo_seed_metrics(results_dir)
    if not metrics:
        print(f"No summary.json files found under {results_dir} -- nothing to rank.")
        return {}

    topsis_results = compute_topsis(metrics)

    fig_path = Path(save_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    plot_composite_score(
        topsis_results, save_path=str(fig_path / "composite_score_oran.pdf")
    )
    print(f"Saved {fig_path / 'composite_score_oran.pdf'} (+ .png)")
    plot_radar(metrics, save_path=str(fig_path / "radar_comparison_oran.pdf"))
    print(f"Saved {fig_path / 'radar_comparison_oran.pdf'} (+ .png)")

    export_multicriteria_table(topsis_results, table_save_dir)

    ordered = sorted(topsis_results.items(), key=lambda kv: kv[1]["rank"])
    print("Composite ranking (best to worst):")
    for algo, r in ordered:
        print(f"  #{int(r['rank'])} {label_for_algo(algo):10s} TOPSIS={r['topsis_score']:.3f}")

    return topsis_results


if __name__ == "__main__":
    run_multicriteria_analysis()
