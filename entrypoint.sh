#!/usr/bin/env bash
set -e

case "$1" in
  hybrid|train_hybrid)
    shift
    exec python training/train_hybrid.py "$@"
    ;;
  baselines|train_baselines)
    shift
    exec python training/train_baselines.py "$@"
    ;;
  hpsearch|hyperparam_search)
    shift
    exec python -m training.hyperparam_search "$@"
    ;;
  sweeps|run_extended_sweeps)
    shift
    exec python training/run_extended_sweeps.py "$@"
    ;;
  oran_hybrid|oran_bmpp_dqn|train_bmpp_dqn)
    shift
    exec python -m oran_training.train_bmpp_dqn "$@"
    ;;
  oran_baselines|train_oran_baselines)
    shift
    exec python -m oran_training.train_oran_baselines "$@"
    ;;
  full_cran|run_cran_experiments)
    # Runs the entire required C-RAN experiment matrix (docs/workflow.md
    # Phase 4 / manuscript/MPhil_Thesis_Concept_Note_v4.md Section 12),
    # not a single training run -- see scripts/run_cran_experiments.sh's
    # own header for every environment-variable override it honors
    # (CONFIG, EPISODES, SAVE_DIR, FIGURES_DIR, TABLES_DIR, ...).
    shift
    exec bash scripts/run_cran_experiments.sh "$@"
    ;;
  full_oran|run_oran_experiments)
    # Runs the entire required O-RAN experiment matrix (Concept Note
    # ORAN_BMPP_DQN_Concept_Note_v1.md Section 5.3), not a single training
    # run -- see scripts/run_oran_experiments.sh's own header for every
    # environment-variable override it honors (CONFIG, SEEDS, EPISODES,
    # SAVE_DIR, FIGURES_DIR, TABLES_DIR).
    shift
    exec bash scripts/run_oran_experiments.sh "$@"
    ;;
  oran_power_sensitivity|power_sensitivity)
    # oran_evaluation/power_sensitivity.py: perturbs oran_env/power_model.py's
    # needs-validation constants and checks whether the 4-method reward
    # ranking is robust -- see that module's own --help for all flags.
    shift
    exec python -m oran_evaluation.power_sensitivity "$@"
    ;;
  "")
    exec python training/train_hybrid.py --config config/default.yaml
    ;;
  *)
    # Arbitrary command passthrough, e.g. `bash`, `pytest tests/test_env.py -v`
    exec "$@"
    ;;
esac
