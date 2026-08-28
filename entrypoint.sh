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
  "")
    exec python training/train_hybrid.py --config config/default.yaml
    ;;
  *)
    # Arbitrary command passthrough, e.g. `bash`, `pytest tests/test_env.py -v`
    exec "$@"
    ;;
esac
