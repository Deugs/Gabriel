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

BMPP_DQN_SEED_DIRS = ["bmpp_dqn_seed42", "bmpp_dqn_seed123", "bmpp_dqn_seed456"]


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name in BMPP_DQN_SEED_DIRS:
        shutil.copytree(SRC / name, DST / name, dirs_exist_ok=True)
    shutil.copytree(
        SRC / "per_seed_baselines", DST / "per_seed_baselines", dirs_exist_ok=True
    )
    print(f"Copied {SRC} (BMPP-DQN + per_seed_baselines only) -> {DST}")


if __name__ == "__main__":
    main()
