#!/usr/bin/env bash
# Run the full required experiment suite for the C-RAN / Branching MP-DQN + TD3
# track (manuscript/MPhil_Thesis_Concept_Note_v4.md), i.e. docs/workflow.md's
# Phase 4 deliverables. See docs/cran_experiment_guide.md for the full
# step-by-step writeup this script automates.
#
# Setup (once):
#   python -m venv venv && source venv/bin/activate
#   pip install -r requirements.txt
#   pre-commit install
#
# Usage:
#   bash scripts/run_cran_experiments.sh
#
# Override any default via environment variables, e.g. a fast smoke test:
#   EPISODES=5 bash scripts/run_cran_experiments.sh
#
# This does NOT include the Section 12.11 hyperparameter proxy sweep --
# that's a pre-training gate, already run and logged (docs/daily_log.md,
# 2026-08-05 entry). To re-run it: `python -m training.hyperparam_search`.

set -eo pipefail

CONFIG="${CONFIG:-config/default.yaml}"
EPISODES="${EPISODES:-3000}"   # <=3000 per the convergence target (README.md)
SAVE_DIR="${SAVE_DIR:-data/results}"
FIGURES_DIR="${FIGURES_DIR:-thesis/figures}"
TABLES_DIR="${TABLES_DIR:-thesis/tables}"
RUN_CSI_AND_GENERALIZATION="${RUN_CSI_AND_GENERALIZATION:-true}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=================================================="
echo "C-RAN full experiment suite"
echo "  config=${CONFIG} episodes=${EPISODES} save_dir=${SAVE_DIR}"
echo "=================================================="

# 1/7: Core matrix -- all 10 baselines + the proposed method, all 10 seeds
# (Concept Note v4.0 Section 12; docs/workflow.md Phase 4's Experiment
# Matrix), plus CSI-robustness/generalization per seed (reusing each seed's
# just-trained checkpoint) and the final statistical aggregation
# (evaluation/convergence.py). training/run_extended_sweeps.py already
# orchestrates this entire pipeline in one call.
echo -e "\n>>> [1/7] Core matrix: baselines + proposed method, 10 seeds..."
extra_args=()
if [ "$RUN_CSI_AND_GENERALIZATION" = "true" ]; then
  extra_args+=(--run-csi-and-generalization)
fi
python training/run_extended_sweeps.py \
  --config "$CONFIG" \
  --episodes "$EPISODES" \
  --save-dir "$SAVE_DIR" \
  "${extra_args[@]}"

# 2/7: Scalability sweep (5 network sizes; R=50 is a stretch goal, not
# committed -- kept at its own smaller default episode count, since this
# trains one fresh agent per network size, not a single full-scale run).
echo -e "\n>>> [2/7] Scalability sweep (R=5,12,20,35,50)..."
python -c "
from evaluation import analyze_scalability
analyze_scalability(config_path='${CONFIG}', save_dir='${FIGURES_DIR}')
"

# 3/7: Inference-latency benchmark (R=5,12,20,35,50; P-DQN/MP-DQN capped at
# R<=12, skipped gracefully above that).
echo -e "\n>>> [3/7] Inference-latency benchmark..."
python -c "
from evaluation import run_latency_benchmark
run_latency_benchmark(config_path='${CONFIG}', save_dir='${FIGURES_DIR}')
"

# 4/7: Ablation study (RQ3: hybrid vs pure-DDPG continuous relaxation;
# RQ4: hybrid vs P-DQN/MP-DQN).
echo -e "\n>>> [4/7] Ablation study..."
python -c "
from evaluation import run_ablation_study
run_ablation_study(config_path='${CONFIG}', save_dir='${FIGURES_DIR}')
"

# 5/7: Demand-response curve (EE/power vs. user-demand multiplier).
echo -e "\n>>> [5/7] Demand-response evaluation..."
python -c "
from evaluation import run_demand_response_evaluation
run_demand_response_evaluation(config_path='${CONFIG}', save_dir='${FIGURES_DIR}')
"

# 6/7: Power-vs-time-of-day profile.
echo -e "\n>>> [6/7] Power/time-of-day profile evaluation..."
python -c "
from evaluation import run_power_time_profile_evaluation
run_power_time_profile_evaluation(config_path='${CONFIG}', save_dir='${FIGURES_DIR}')
"

# 7/7: Reward-weight (gamma_switch) sensitivity sweep.
echo -e "\n>>> [7/7] Reward-weight sensitivity sweep..."
python -c "
from evaluation import run_reward_sensitivity_sweep
run_reward_sensitivity_sweep(config_path='${CONFIG}', save_dir='${FIGURES_DIR}')
"

echo -e "\n=================================================="
echo "C-RAN experiment suite complete."
echo "Figures: ${FIGURES_DIR}  Tables: ${TABLES_DIR}  Raw results: ${SAVE_DIR}"
echo "=================================================="
