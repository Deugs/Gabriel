"""Regenerate thesis/tables_oran/sensitivity_summary_oran.tex from the
n=10-revalidation-era data/results_oran_sensitivity/ sweep (8 configs x 4
methods x 3 seeds x 500 episodes), matching the table structure the
pre-fix version of this file used (archived at
thesis/tables_oran_archive/pre_env_bugfix_20261001/sensitivity_summary_oran.tex)
-- same columns, same 9 rows (Default + 8 perturbations), same %.3f TOPSIS
score precision, regenerated from the corrected environment's data instead
of hand-copied from that archived version.

Reuses oran_evaluation.sensitivity_plots.SENSITIVITY_CONFIGS for the
(label, results_dir) list, oran_evaluation.convergence.load_algo_seed_metrics
for parsing, and oran_evaluation.multicriteria.compute_topsis for scoring --
no re-derivation of any of the three.
"""

from pathlib import Path

from oran_evaluation.convergence import load_algo_seed_metrics
from oran_evaluation.multicriteria import compute_topsis
from oran_evaluation.sensitivity_plots import SENSITIVITY_CONFIGS

PROPOSED_ALGO = "BMPP_DQN"
CORE_ALGOS = {"BMPP_DQN", "dqn", "ddpg", "mpdqn"}


def main() -> None:
    rows = []
    for label, results_dir in SENSITIVITY_CONFIGS:
        metrics = load_algo_seed_metrics(results_dir)
        # data/results_oran ("Default") also holds the heuristic/oracle
        # calibration baselines (Section~sec:oran-calibration) -- restrict
        # to the 4 trained methods here so every row's rank is "of 4",
        # matching the 8 perturbation configs, which only ever have these
        # 4 (no calibration baselines were rerun per-scenario).
        metrics = {a: m for a, m in metrics.items() if a in CORE_ALGOS}
        if PROPOSED_ALGO not in metrics or not metrics[PROPOSED_ALGO]:
            print(f"SKIP {label}: no {PROPOSED_ALGO} data under {results_dir}")
            continue

        reward_ranking = sorted(
            metrics,
            key=lambda a: sum(
                m["reward"] for m in metrics[a].values()
            ) / len(metrics[a]),
            reverse=True,
        )
        reward_rank = reward_ranking.index(PROPOSED_ALGO) + 1

        topsis = compute_topsis(metrics)
        topsis_rank = int(topsis[PROPOSED_ALGO]["rank"])
        topsis_score = topsis[PROPOSED_ALGO]["topsis_score"]

        rows.append((label, reward_rank, topsis_rank, topsis_score, len(metrics)))
        print(
            f"{label:16s} reward_rank={reward_rank} topsis_rank={topsis_rank} "
            f"topsis_score={topsis_score:.3f} (n_methods={len(metrics)})"
        )

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{O-RAN Track: Power-Model Sensitivity Analysis "
        r"(Concept Note Section 6.9). Each row scales one RU/DU/CU/"
        r"fronthaul constant group by an order of magnitude (all others "
        r"at their Section 10.5 default), re-running the full "
        r"4-method/3-seed/500-episode protocol against the corrected "
        r"environment (Section~\ref{sec:oran-bugfixes}).}",
        r"\label{tab:oran-sensitivity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Configuration & BMPP-DQN Reward Rank (of 4) & "
        r"BMPP-DQN TOPSIS Rank (of 4) & TOPSIS Score \\",
        r"\hline",
    ]
    for label, reward_rank, topsis_rank, topsis_score, n_methods in rows:
        lines.append(
            f"{label} & {reward_rank} & {topsis_rank} & "
            f"{topsis_score:.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}%", "}", r"\end{table}", ""]

    out_path = Path("thesis/tables_oran/sensitivity_summary_oran.tex")
    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
