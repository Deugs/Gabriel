#!/usr/bin/env bash
# Run the full required experiment suite for the O-RAN / BMPP-DQN track
# (manuscript/ORAN_BMPP_DQN_Concept_Note_v1.md), per its Section 5.3 and
# docs/oran_thesis_guide.md's Key Figures/Tables list. See
# docs/oran_experiment_guide.md for the full step-by-step writeup this
# script automates.
#
# Setup (once): same as scripts/run_cran_experiments.sh -- see its header.
#
# Usage:
#   bash scripts/run_oran_experiments.sh
#
# Override any default via environment variables, e.g. a fast smoke test:
#   EPISODES=5 bash scripts/run_oran_experiments.sh

set -eo pipefail

CONFIG="${CONFIG:-config/oran_default.yaml}"
SEEDS="${SEEDS:-42 123 456}"    # Concept Note Section 5.3: 3 random seeds
EPISODES="${EPISODES:-500}"    # config/oran_default.yaml's algorithm.max_episodes cap
SAVE_DIR="${SAVE_DIR:-data/results_oran}"
FIGURES_DIR="${FIGURES_DIR:-thesis/figures_oran}"
TABLES_DIR="${TABLES_DIR:-thesis/tables_oran}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=================================================="
echo "O-RAN / BMPP-DQN full experiment suite"
echo "  config=${CONFIG} episodes=${EPISODES} seeds=[${SEEDS}] save_dir=${SAVE_DIR}"
echo "=================================================="

# 1/3: All 3 baselines (DQN, DDPG, MP-DQN), all 3 seeds in one call.
echo -e "\n>>> [1/3] Baseline benchmarks (DQN, DDPG, MP-DQN)..."
python -m oran_training.train_oran_baselines \
  --config "$CONFIG" \
  --episodes "$EPISODES" \
  --save-dir "$SAVE_DIR"

# 2/3: Proposed BMPP-DQN agent -- one invocation per seed.
echo -e "\n>>> [2/3] Proposed BMPP-DQN agent..."
for seed in $SEEDS; do
  echo "--- Training seed ${seed} (${EPISODES} episodes) ---"
  python -m oran_training.train_bmpp_dqn \
    --config "$CONFIG" \
    --seed "$seed" \
    --episodes "$EPISODES" \
    --save-dir "$SAVE_DIR"
done

# 3/3: Statistical aggregation (95% CIs, paired t-test, Cohen's d) and the
# inference-latency benchmark (Concept Note Section 6.1/7.1's focused
# single-gNB scope -- a single scenario, not a scalability sweep).
echo -e "\n>>> [3/3] Aggregating results + inference-latency benchmark..."
python -c "
from oran_evaluation import analyze_convergence, run_latency_benchmark
analyze_convergence(results_dir='${SAVE_DIR}', save_dir='${FIGURES_DIR}', table_save_dir='${TABLES_DIR}')
run_latency_benchmark(config_path='${CONFIG}', save_dir='${FIGURES_DIR}')
"

echo -e "\n=================================================="
echo "O-RAN experiment suite complete."
echo "Figures: ${FIGURES_DIR}  Tables: ${TABLES_DIR}  Raw results: ${SAVE_DIR}"
echo "=================================================="
