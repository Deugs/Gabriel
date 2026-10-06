"""Derive data/results_oran_core4/ from data/results_oran/.

thesis/tables_oran/ and thesis/figures_oran/ (the headline, original-scope
4-method comparison: BMPP-DQN/DQN/DDPG/MP-DQN) are generated from this
core4 subset, not from data/results_oran/ directly, so the heuristic/
oracle calibration baselines living alongside the 4 trained methods in
data/results_oran/ don't leak into the headline comparison
(thesis/tables_oran_calibration/ and thesis/figures_oran_calibration/ use
data/results_oran/ directly, 6 methods, for that separate comparison).

Not committed as its own data/ directory -- it is a pure, deterministic
filter of data/results_oran/ already in the repo, and keeping a second
copy of the same summary.json files around would duplicate data for no
reason. Run this to recreate it locally before re-running
oran_evaluation.analyze_convergence / run_multicriteria_analysis /
generate_comparison_plots / plot_pareto_scatter / plot_convergence_curve
against results_dir="data/results_oran_core4".
"""

import shutil
from pathlib import Path

SRC = Path("data/results_oran")
DST = Path("data/results_oran_core4")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    # Discover every bmpp_dqn_seed* dir rather than a hardcoded 3-seed
    # list -- data/results_oran now holds 10 canonical seeds (the n=10
    # statistical-power revalidation, Section~sec:oran-n10), and a
    # hardcoded list silently dropped the 7 new ones here before this fix.
    bmpp_dqn_dirs = sorted(SRC.glob("bmpp_dqn_seed*"))
    for seed_dir in bmpp_dqn_dirs:
        shutil.copytree(seed_dir, DST / seed_dir.name, dirs_exist_ok=True)
    shutil.copytree(
        SRC / "per_seed_baselines", DST / "per_seed_baselines", dirs_exist_ok=True
    )
    print(
        f"Copied {SRC} ({len(bmpp_dqn_dirs)} BMPP-DQN seeds + "
        f"per_seed_baselines only) -> {DST}"
    )


if __name__ == "__main__":
    main()
