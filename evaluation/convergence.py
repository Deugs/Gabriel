"""Statistical Convergence Analysis Module for Thesis Chapter 4."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

from evaluation.plot_utils import compute_confidence_interval


def perform_paired_ttest(
    proposed_scores: np.ndarray, baseline_scores: np.ndarray
) -> Tuple[float, float, bool]:
    """Perform paired t-test between proposed method and baseline scores across seeds.

    Args:
        proposed_scores (np.ndarray): Scores for proposed method across seeds (n_seeds,).
        baseline_scores (np.ndarray): Scores for baseline method across seeds (n_seeds,).

    Returns:
        Tuple[float, float, bool]: t-statistic, p-value, is_significant_at_0.05.
    """
    if len(proposed_scores) != len(baseline_scores) or len(proposed_scores) < 2:
        return 0.0, 1.0, False

    t_stat, p_val = stats.ttest_rel(proposed_scores, baseline_scores)
    is_significant = float(p_val) < 0.05
    return float(t_stat), float(p_val), is_significant


def compute_cohens_d(proposed_scores: np.ndarray, baseline_scores: np.ndarray) -> float:
    """Cohen's d effect size for a paired comparison (Concept Note v3.0/v4.0 Section 12.4, S4/G11).

    Reported alongside the t-test/p-value so a statistically significant but
    practically small difference is visible as such, given the modest 5%
    target margin over the DDQN baseline.

    Args:
        proposed_scores (np.ndarray): Scores for proposed method across seeds (n_seeds,).
        baseline_scores (np.ndarray): Scores for baseline method across seeds (n_seeds,).

    Returns:
        float: Cohen's d computed on the paired differences (mean diff / std diff); 0.0
            if fewer than 2 paired samples or the differences have zero variance.
    """
    if len(proposed_scores) != len(baseline_scores) or len(proposed_scores) < 2:
        return 0.0

    diffs = np.asarray(proposed_scores) - np.asarray(baseline_scores)
    diff_std = float(np.std(diffs, ddof=1))
    if diff_std == 0.0:
        return 0.0
    return float(np.mean(diffs) / diff_std)


def analyze_convergence(
    results_dir: str = "data/results",
    save_dir: str = "thesis/figures",
    table_save_dir: str = "thesis/tables",
) -> Dict[str, Any]:
    """Aggregate multi-seed benchmark results, compute 95% CIs, run t-tests, and plot."""
    results_path = Path(results_dir)
    fig_path = Path(save_dir)
    table_path = Path(table_save_dir)

    fig_path.mkdir(parents=True, exist_ok=True)
    table_path.mkdir(parents=True, exist_ok=True)

    summary_files = list(results_path.rglob("summary.json"))
    print(f"Found {len(summary_files)} result summary files under {results_dir}")

    # Parse and organize algorithm scores
    algo_scores: Dict[str, List[float]] = {}
    algo_powers: Dict[str, List[float]] = {}
    algo_qos: Dict[str, List[float]] = {}
    algo_switching: Dict[str, List[float]] = {}

    for s_file in summary_files:
        try:
            with open(s_file, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                algo = str(data.get("algorithm", "unknown"))
                reward = float(data.get("final_eval_reward", 0.0))
                power = float(data.get("final_eval_power_w", 0.0))
                qos = float(data.get("final_qos_rate", 0.0))
                switching = float(data.get("final_switching_events", 0.0))

                algo_scores.setdefault(algo, []).append(reward)
                algo_powers.setdefault(algo, []).append(power)
                algo_qos.setdefault(algo, []).append(qos)
                algo_switching.setdefault(algo, []).append(switching)
            elif isinstance(data, list):
                for item in data:
                    algo = str(item.get("algorithm", "unknown"))
                    reward = float(item.get("mean_reward", 0.0))
                    power = float(item.get("mean_power_w", 0.0))
                    qos = float(item.get("qos_satisfaction_rate", 0.0))
                    switching = float(item.get("mean_switching_events", 0.0))

                    algo_scores.setdefault(algo, []).append(reward)
                    algo_powers.setdefault(algo, []).append(power)
                    algo_qos.setdefault(algo, []).append(qos)
                    algo_switching.setdefault(algo, []).append(switching)
        except Exception as e:
            print(f"Warning: Failed to parse {s_file}: {e}")

    analysis_report: Dict[str, Any] = {
        "algorithms": {},
        "paired_ttests": {},
    }

    proposed_algo = "Branching_MP_DQN"
    proposed_arr = np.array(algo_scores.get(proposed_algo, [0.0]))

    for algo, scores in algo_scores.items():
        arr = np.array(scores)
        mean, lower, upper = compute_confidence_interval(arr)

        analysis_report["algorithms"][algo] = {
            "mean_reward": float(mean),
            "ci_95_lower": float(lower),
            "ci_95_upper": float(upper),
            "mean_power_w": float(np.mean(algo_powers.get(algo, [0.0]))),
            "mean_qos_rate": float(np.mean(algo_qos.get(algo, [0.0]))),
            "mean_switching_events": float(np.mean(algo_switching.get(algo, [0.0]))),
        }

        if (
            algo != proposed_algo
            and len(proposed_arr) > 1
            and len(arr) == len(proposed_arr)
        ):
            t_stat, p_val, is_sig = perform_paired_ttest(proposed_arr, arr)
            cohens_d = compute_cohens_d(proposed_arr, arr)
            analysis_report["paired_ttests"][algo] = {
                "t_statistic": t_stat,
                "p_value": p_val,
                "statistically_significant_0_05": is_sig,
                "cohens_d": cohens_d,
            }

    # Export LaTeX table
    latex_content = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Performance Comparison and Statistical Significance Analysis.}\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\hline\n"
        "Algorithm & Mean Reward (95\\% CI) & Mean Power (W) & "
        "QoS Rate (\\%) & Switching Freq. & $p$-value (vs Proposed) & Cohen's $d$ \\\\\n"
        "\\hline\n"
    )

    for algo, m in analysis_report["algorithms"].items():
        if algo == proposed_algo:
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

    with open(table_path / "convergence_summary.tex", "w") as f:
        f.write(latex_content)

    print(
        f"Exported convergence summary LaTeX table to {table_path / 'convergence_summary.tex'}"
    )
    return analysis_report


if __name__ == "__main__":
    analyze_convergence()
