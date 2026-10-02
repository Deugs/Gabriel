"""Multi-criteria composite ranking for the O-RAN track's 4-method benchmark.

Uses TOPSIS (Technique for Order Preference by Similarity to Ideal
Solution; Hwang & Yoon, 1981) -- a standard, citable multi-criteria
decision-making method -- to combine several metrics of different units
and scales (Watts, percentages, Mbps, events/step) into one bounded [0, 1]
composite score per method, without needing to hand-pick a common unit or
an ad-hoc weighted-sum formula.

Also provides two cross-checks on that TOPSIS ranking, since a single
MCDA method's ranking can be an artifact of its own weighting/aggregation
choice rather than a robust conclusion:
  - compute_entropy_weights(): the Shannon entropy method (Zeleny, 1982)
    derives objective, data-driven criterion weights from how much each
    criterion actually discriminates between the methods being compared,
    rather than this module's own default equal weighting.
  - compute_vikor(): VIKOR (Opricovic & Tzeng, 2004) ranks methods by a
    different aggregation rule entirely (weighted-sum "group utility" S
    and worst-single-criterion "individual regret" R, compromised via
    Q = v*S + (1-v)*R) instead of TOPSIS's Euclidean distance to an
    ideal/anti-ideal point. Where VIKOR and TOPSIS agree, the ranking is
    robust to the choice of aggregation rule; where they disagree
    (typically on closely-scored methods), that disagreement is itself
    evidence worth reporting, not a bug to resolve by picking one.

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


def _latex_label(criterion: str) -> str:
    """CRITERION_LABELS as a LaTeX-safe string -- the dict itself is
    shared with matplotlib labels (plot_ranking_robustness(), etc.) where
    a literal '%' is correct and a '\\%' would show the backslash, so the
    escape happens only here, at the point of LaTeX table generation."""
    return CRITERION_LABELS.get(criterion, criterion).replace("%", "\\%")


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


def compute_entropy_weights(
    metrics: Dict[str, Dict[int, Dict[str, float]]],
    criteria: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Objective, data-driven criterion weights via the Shannon entropy
    method (Zeleny, 1982): a criterion whose values barely differ across
    methods carries little information for ranking them and gets a small
    weight; a criterion that sharply separates the methods gets a large
    one -- no researcher-chosen weight vector to defend.

    Steps:
      1. Decision matrix as in compute_topsis() (one row per method, mean
         across seeds).
      2. Orient every column so higher = better (min-max scaled, cost
         criteria flipped -- the same convention plot_radar() already
         uses), so entropy is computed on a consistent "goodness" scale
         rather than raw mixed-direction units.
      3. Column-normalize to a probability distribution p_ij (each column
         sums to 1); Shannon entropy e_j = -(1/ln n) * sum_i p_ij ln p_ij,
         in [0, 1] (1 = every method identical on this criterion, i.e. no
         discriminating power).
      4. Weight_j = (1 - e_j) / sum_j(1 - e_j).

    Returns {criterion: weight}, falling back to equal weights if every
    criterion is equally uninformative (e.g. all methods tied everywhere).
    """
    criteria = criteria or CRITERIA
    algos = [a for a in metrics if metrics[a]]
    if not algos:
        return {c: 1.0 / len(criteria) for c in criteria}

    matrix = np.array(
        [
            [np.mean([m[c] for m in metrics[algo].values()]) for c in criteria]
            for algo in algos
        ]
    )

    n = matrix.shape[0]
    oriented = np.zeros_like(matrix)
    for j, c in enumerate(criteria):
        col = matrix[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            oriented[:, j] = 1.0
        else:
            scaled = (col - lo) / (hi - lo)
            oriented[:, j] = scaled if BENEFIT[c] else (1.0 - scaled)

    col_sums = oriented.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    p = oriented / col_sums

    with np.errstate(divide="ignore", invalid="ignore"):
        p_ln_p = np.where(p > 0, p * np.log(p), 0.0)
    k = 1.0 / np.log(n) if n > 1 else 0.0
    entropy = -k * p_ln_p.sum(axis=0)
    diversification = 1.0 - entropy

    total_d = diversification.sum()
    if total_d <= 1e-12:
        return {c: 1.0 / len(criteria) for c in criteria}

    weights = diversification / total_d
    return {c: float(weights[j]) for j, c in enumerate(criteria)}


def compute_vikor(
    metrics: Dict[str, Dict[int, Dict[str, float]]],
    criteria: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    v: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Rank methods by VIKOR (Opricovic & Tzeng, 2004): a compromise-
    ranking MCDM method that cross-checks compute_topsis()'s ranking
    using a different aggregation rule than TOPSIS's own Euclidean
    distance to an ideal/anti-ideal point.

    Steps:
      1. Decision matrix as in compute_topsis().
      2. Per criterion, f* = ideal-best value across methods (max for a
         benefit criterion, min for a cost one), f- = ideal-worst value.
      3. Normalized distance-from-best d_ij = w_j * (f*_j - x_ij) /
         (f*_j - f-_j), in [0, 1] regardless of benefit/cost direction.
      4. S_i = sum_j d_ij ("group utility": weighted-sum distance from
         ideal across all criteria); R_i = max_j d_ij ("individual
         regret": the single worst criterion for that method).
      5. Q_i = v*(S_i - S*)/(S- - S*) + (1-v)*(R_i - R*)/(R- - R*), where
         S*/S- and R*/R- are the best/worst observed S and R; v=0.5 is
         VIKOR's standard "majority of criteria" compromise weight.

    Lower Q = better (opposite direction from compute_topsis()'s score).
    Returns {algo: {"vikor_q", "vikor_s", "vikor_r", "rank"}}.
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

    f_star = np.array(
        [
            matrix[:, j].max() if BENEFIT[c] else matrix[:, j].min()
            for j, c in enumerate(criteria)
        ]
    )
    f_worst = np.array(
        [
            matrix[:, j].min() if BENEFIT[c] else matrix[:, j].max()
            for j, c in enumerate(criteria)
        ]
    )
    spread = f_star - f_worst
    spread_safe = np.where(np.abs(spread) < 1e-12, 1.0, spread)

    w = np.array([weights[c] for c in criteria])
    d = w * (f_star - matrix) / spread_safe

    S = d.sum(axis=1)
    R = d.max(axis=1)

    s_denom = (S.max() - S.min()) or 1.0
    r_denom = (R.max() - R.min()) or 1.0
    Q = v * (S - S.min()) / s_denom + (1 - v) * (R - R.min()) / r_denom

    order = np.argsort(Q)
    ranks = {algos[idx]: int(pos + 1) for pos, idx in enumerate(order)}

    result: Dict[str, Dict[str, float]] = {}
    for i, algo in enumerate(algos):
        result[algo] = {
            "vikor_q": float(Q[i]),
            "vikor_s": float(S[i]),
            "vikor_r": float(R[i]),
            "rank": float(ranks[algo]),
        }
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
    col_headers = " & ".join(_latex_label(c) for c in criteria)

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


def export_mcda_robustness_table(
    topsis_equal: Dict[str, Dict[str, float]],
    topsis_entropy: Dict[str, Dict[str, float]],
    vikor_results: Dict[str, Dict[str, float]],
    entropy_weights: Dict[str, float],
    table_save_dir: str,
    criteria: Optional[List[str]] = None,
) -> str:
    """Write a LaTeX+CSV table comparing the rank each method gets under
    three MCDA schemes -- equal-weighted TOPSIS, entropy-weighted TOPSIS,
    and entropy-weighted VIKOR -- so agreement/disagreement between
    aggregation rules and weighting schemes is visible at a glance rather
    than asserting a single ranking as if it were the only possible one.

    Sorted by equal-weighted TOPSIS rank (this module's original/default
    scheme), so the two cross-check columns read as "does this hold up".
    """
    criteria = criteria or CRITERIA
    table_path = Path(table_save_dir)
    table_path.mkdir(parents=True, exist_ok=True)

    ordered = sorted(topsis_equal.items(), key=lambda kv: kv[1]["rank"])

    weight_str = ", ".join(
        f"{_latex_label(c)}={entropy_weights.get(c, 0.0):.2f}"
        for c in criteria
    )

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{O-RAN Track: MCDA Ranking Robustness. Compares the rank "
        "each method gets under three schemes -- equal-weighted TOPSIS "
        "(this chapter's default), entropy-weighted TOPSIS (objective, "
        "data-driven weights: " + weight_str + "), and entropy-weighted "
        "VIKOR (a different aggregation rule -- weighted-sum group "
        "utility $S$ and worst-single-criterion regret $R$, $v=0.5$, "
        "rather than TOPSIS's distance-to-ideal). Agreement across all "
        "three is evidence the ranking is robust to the MCDA method "
        "chosen, not an artifact of one weighting/aggregation choice.}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Algorithm & TOPSIS-Equal Rank & TOPSIS-Entropy Rank & VIKOR Rank "
        "& VIKOR $Q$ & Agreement \\\\",
        "\\hline",
    ]
    for algo, r_eq in ordered:
        r_ent = topsis_entropy.get(algo, {})
        r_vik = vikor_results.get(algo, {})
        ranks = [
            int(r_eq["rank"]),
            int(r_ent["rank"]) if r_ent else -1,
            int(r_vik["rank"]) if r_vik else -1,
        ]
        agree = "Yes" if len(set(ranks)) == 1 else "No"
        vikor_q = f"{r_vik['vikor_q']:.3f}" if r_vik else "N/A"
        lines.append(
            f"{label_for_algo(algo)} & {ranks[0]} & {ranks[1]} & {ranks[2]} "
            f"& {vikor_q} & {agree} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]

    out_path = table_path / "mcda_robustness_oran.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Exported MCDA robustness LaTeX table to {out_path}")

    csv_path = table_path / "mcda_robustness_oran.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "topsis_equal_rank",
                "topsis_entropy_rank",
                "vikor_rank",
                "vikor_q",
                "agreement",
            ]
        )
        for algo, r_eq in ordered:
            r_ent = topsis_entropy.get(algo, {})
            r_vik = vikor_results.get(algo, {})
            ranks = [
                int(r_eq["rank"]),
                int(r_ent["rank"]) if r_ent else -1,
                int(r_vik["rank"]) if r_vik else -1,
            ]
            writer.writerow(
                [
                    label_for_algo(algo),
                    ranks[0],
                    ranks[1],
                    ranks[2],
                    r_vik.get("vikor_q") if r_vik else "",
                    "yes" if len(set(ranks)) == 1 else "no",
                ]
            )
    print(f"Exported MCDA robustness CSV to {csv_path}")

    return str(out_path)


def plot_ranking_robustness(
    topsis_equal: Dict[str, Dict[str, float]],
    topsis_entropy: Dict[str, Dict[str, float]],
    vikor_results: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    """Bump chart: each method's rank (1 = best) under all three MCDA
    schemes side by side, so ranking agreement/disagreement across
    weighting and aggregation choices is visible at a glance -- a visual
    companion to export_mcda_robustness_table()'s own numbers. A flat
    (horizontal) line means that method's rank is robust to the MCDA
    method chosen; a line that crosses others means it isn't.

    Uses the same fixed per-algorithm color assignment as every other
    figure in this track (plot_radar, plot_pareto_scatter,
    plot_convergence_curve) so a method's color is identical across every
    figure in the thesis, not re-derived here.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos = ordered_algos({a: {0: {}} for a in topsis_equal})
    schemes = [
        ("TOPSIS\n(Equal Weights)", topsis_equal),
        ("TOPSIS\n(Entropy Weights)", topsis_entropy),
        ("VIKOR\n(Entropy Weights)", vikor_results),
    ]
    n_algos = len(algos)

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(8, 5.5))

    x = list(range(len(schemes)))
    for i, algo in enumerate(algos):
        ranks = [int(scheme_dict[algo]["rank"]) for _, scheme_dict in schemes]
        color = colors[i % len(colors)]
        ax.plot(
            x, ranks, marker="o", markersize=9, linewidth=2.5, color=color,
            zorder=3,
        )
        # Direct label at each end (only 3 x-positions -- both ends stay
        # readable, unlike labeling every point on a long series).
        ax.annotate(
            label_for_algo(algo), (x[0], ranks[0]), textcoords="offset points",
            xytext=(-14, 0), ha="right", va="center", fontsize=9,
            color=color, fontweight="bold",
        )
        ax.annotate(
            label_for_algo(algo), (x[-1], ranks[-1]), textcoords="offset points",
            xytext=(14, 0), ha="left", va="center", fontsize=9,
            color=color, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in schemes])
    ax.set_xlim(-0.6, len(schemes) - 0.4)
    ax.set_yticks(range(1, n_algos + 1))
    ax.set_ylim(n_algos + 0.6, 0.4)  # inverted: rank 1 (best) at the top
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title(
        "O-RAN Track: MCDA Ranking Robustness Across Weighting/Aggregation Schemes"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file.with_suffix(".pdf"))
    fig.savefig(out_file.with_suffix(".png"), dpi=150)
    plt.close(fig)


def run_multicriteria_analysis(
    results_dir: str = "data/results_oran",
    save_dir: str = "thesis/figures_oran",
    table_save_dir: str = "thesis/tables_oran",
) -> Dict[str, Dict[str, float]]:
    """End-to-end: load results, compute TOPSIS (equal weights, this
    module's default), plot the composite-score bar chart and the radar
    chart, export the LaTeX+CSV table -- then cross-check that ranking
    with entropy-weighted TOPSIS and entropy-weighted VIKOR, exporting a
    robustness table comparing all three.

    Returns the equal-weighted TOPSIS results dict (unchanged return
    shape/contract from before -- see compute_topsis()'s docstring), so
    existing callers keep working; the two cross-checks are exported as
    files and printed, not returned, since callers only ever consumed
    this one dict.
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
        print(
            f"  #{int(r['rank'])} {label_for_algo(algo):10s} "
            f"TOPSIS={r['topsis_score']:.3f}"
        )

    entropy_weights = compute_entropy_weights(metrics)
    topsis_entropy_results = compute_topsis(metrics, weights=entropy_weights)
    vikor_results = compute_vikor(metrics, weights=entropy_weights)

    export_mcda_robustness_table(
        topsis_results, topsis_entropy_results, vikor_results, entropy_weights,
        table_save_dir,
    )

    plot_ranking_robustness(
        topsis_results, topsis_entropy_results, vikor_results,
        save_path=str(fig_path / "mcda_robustness_oran.pdf"),
    )
    print(f"Saved {fig_path / 'mcda_robustness_oran.pdf'} (+ .png)")

    print("Entropy-derived criterion weights: " + ", ".join(
        f"{CRITERION_LABELS.get(c, c)}={w:.3f}" for c, w in entropy_weights.items()
    ))
    print("Robustness cross-check (rank under each scheme, best to worst by "
          "equal-weighted TOPSIS):")
    for algo, r_eq in ordered:
        r_ent = topsis_entropy_results.get(algo, {})
        r_vik = vikor_results.get(algo, {})
        q = r_vik.get("vikor_q", float("nan"))
        print(
            f"  {label_for_algo(algo):10s} TOPSIS-equal=#{int(r_eq['rank'])}  "
            f"TOPSIS-entropy=#{int(r_ent.get('rank', -1))}  "
            f"VIKOR=#{int(r_vik.get('rank', -1))} (Q={q:.3f})"
        )

    return topsis_results


if __name__ == "__main__":
    run_multicriteria_analysis()
