"""Statistical Convergence Analysis Module for the O-RAN track.

Mirrors evaluation/convergence.py's structure (seed-keyed pairing,
paired t-test, Cohen's d), but with `proposed_algo = "BMPP_DQN"` hardcoded
in the same style as the original -- deliberately not parameterized only
on this side, since importing the original directly would create a real
dependency on the C-RAN evaluation module (this package's decoupling
guarantee) and the two tracks' results must never be comparable/paired
against each other's proposed-method name by accident.

Written correctly from the start: aggregation keys by seed (not list
position), so it never has the seed-pairing bug an earlier round found
and fixed in evaluation/convergence.py.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy import stats

from oran_evaluation.plot_utils import compute_confidence_interval

PROPOSED_ALGO = "BMPP_DQN"


def perform_paired_ttest(
    proposed_scores: np.ndarray, baseline_scores: np.ndarray
) -> Tuple[float, float, bool]:
    """Perform paired t-test between proposed method and baseline scores across seeds."""
    if len(proposed_scores) != len(baseline_scores) or len(proposed_scores) < 2:
        return 0.0, 1.0, False
    t_stat, p_val = stats.ttest_rel(proposed_scores, baseline_scores)
    is_significant = float(p_val) < 0.05
    return float(t_stat), float(p_val), is_significant


def compute_cohens_d(proposed_scores: np.ndarray, baseline_scores: np.ndarray) -> float:
    """Cohen's d effect size for a paired comparison (mean diff / std diff)."""
    if len(proposed_scores) != len(baseline_scores) or len(proposed_scores) < 2:
        return 0.0
    diffs = np.asarray(proposed_scores) - np.asarray(baseline_scores)
    diff_std = float(np.std(diffs, ddof=1))
    if diff_std == 0.0:
        return 0.0
    return float(np.mean(diffs) / diff_std)


def analyze_convergence(
    results_dir: str = "data/results_oran",
    save_dir: str = "thesis/figures_oran",
    table_save_dir: str = "thesis/tables_oran",
) -> Dict[str, Any]:
    """Aggregate multi-seed O-RAN benchmark results, compute 95% CIs, run
    t-tests, and export a LaTeX summary table."""
    results_path = Path(results_dir)
    fig_path = Path(save_dir)
    table_path = Path(table_save_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    table_path.mkdir(parents=True, exist_ok=True)

    summary_files = list(results_path.rglob("summary.json"))
    print(f"Found {len(summary_files)} result summary files under {results_dir}")

    # Keyed by seed so paired comparisons genuinely pair the same seed's
    # runs against each other -- the proposed method saves one summary.json
    # per seed directory (oran_training/train_bmpp_dqn.py), baselines save
    # all seeds in one summary.json list (oran_training/train_oran_baselines.py).
    algo_scores: Dict[str, Dict[int, float]] = {}
    algo_powers: Dict[str, Dict[int, float]] = {}
    algo_qos: Dict[str, Dict[int, float]] = {}
    algo_switching: Dict[str, Dict[int, float]] = {}

    for s_file in summary_files:
        try:
            with open(s_file, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                algo = str(data.get("algorithm", "unknown"))
                seed = int(data.get("seed", -1))
                reward = float(data.get("final_eval_reward", 0.0))
                power = float(data.get("final_eval_power_w", 0.0))
                qos = float(data.get("final_qos_rate", 0.0))
                switching = float(data.get("final_switching_events", 0.0))

                algo_scores.setdefault(algo, {})[seed] = reward
                algo_powers.setdefault(algo, {})[seed] = power
                algo_qos.setdefault(algo, {})[seed] = qos
                algo_switching.setdefault(algo, {})[seed] = switching
            elif isinstance(data, list):
                for item in data:
                    algo = str(item.get("algorithm", "unknown"))
                    seed = int(item.get("seed", -1))
                    reward = float(item.get("mean_reward", 0.0))
                    power = float(item.get("mean_power_w", 0.0))
                    qos = float(item.get("qos_satisfaction_rate", 0.0))
                    switching = float(item.get("mean_switching_events", 0.0))

                    algo_scores.setdefault(algo, {})[seed] = reward
                    algo_powers.setdefault(algo, {})[seed] = power
                    algo_qos.setdefault(algo, {})[seed] = qos
                    algo_switching.setdefault(algo, {})[seed] = switching
        except Exception as e:
            print(f"Warning: Failed to parse {s_file}: {e}")

    analysis_report: Dict[str, Any] = {"algorithms": {}, "paired_ttests": {}}

    proposed_by_seed = algo_scores.get(PROPOSED_ALGO, {})

    for algo, scores_by_seed in algo_scores.items():
        arr = (
            np.array(list(scores_by_seed.values()))
            if scores_by_seed
            else np.array([0.0])
        )
        mean, lower, upper = compute_confidence_interval(arr)

        analysis_report["algorithms"][algo] = {
            "mean_reward": float(mean),
            "ci_95_lower": float(lower),
            "ci_95_upper": float(upper),
            "mean_power_w": float(
                np.mean(list(algo_powers.get(algo, {}).values()) or [0.0])
            ),
            "mean_qos_rate": float(
                np.mean(list(algo_qos.get(algo, {}).values()) or [0.0])
            ),
            "mean_switching_events": float(
                np.mean(list(algo_switching.get(algo, {}).values()) or [0.0])
            ),
        }

        if algo != PROPOSED_ALGO:
            common_seeds = sorted(set(proposed_by_seed) & set(scores_by_seed))
            if len(common_seeds) > 1:
                proposed_paired = np.array([proposed_by_seed[s] for s in common_seeds])
                baseline_paired = np.array([scores_by_seed[s] for s in common_seeds])
                t_stat, p_val, is_sig = perform_paired_ttest(
                    proposed_paired, baseline_paired
                )
                cohens_d = compute_cohens_d(proposed_paired, baseline_paired)
                analysis_report["paired_ttests"][algo] = {
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "statistically_significant_0_05": is_sig,
                    "cohens_d": cohens_d,
                    "n_paired_seeds": len(common_seeds),
                }

    latex_content = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{O-RAN Track: Performance Comparison and Statistical Significance.}\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\hline\n"
        "Algorithm & Mean Reward (95\\% CI) & Mean Power (W) & "
        "QoS Rate (\\%) & Switching Freq. & $p$-value (vs Proposed) & Cohen's $d$ \\\\\n"
        "\\hline\n"
    )

    for algo, m in analysis_report["algorithms"].items():
        if algo == PROPOSED_ALGO:
            p_val_str = "N/A (Proposed)"
            d_val_str = "N/A (Proposed)"
        else:
            ttest_info = analysis_report["paired_ttests"].get(algo, {})
            if ttest_info:
                p_val_str = f"{ttest_info['p_value']:.4f}"
                d_val_str = f"{ttest_info['cohens_d']:.3f}"
            else:
                p_val_str = "N/A (insufficient data)"
                d_val_str = "N/A (insufficient data)"
        qos_pct = m["mean_qos_rate"] * 100

        latex_content += (
            f"{algo} & {m['mean_reward']:.2f} [{m['ci_95_lower']:.2f}, {m['ci_95_upper']:.2f}] & "
            f"{m['mean_power_w']:.1f} & {qos_pct:.1f}\\% & "
            f"{m['mean_switching_events']:.2f} & {p_val_str} & {d_val_str} \\\\\n"
        )

    latex_content += "\\hline\n\\end{tabular}\n\\end{table}\n"

    with open(table_path / "convergence_summary_oran.tex", "w") as f:
        f.write(latex_content)

    print(
        f"Exported O-RAN convergence summary LaTeX table to "
        f"{table_path / 'convergence_summary_oran.tex'}"
    )
    return analysis_report


if __name__ == "__main__":
    analyze_convergence()
