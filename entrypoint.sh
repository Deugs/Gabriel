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
  "")
    exec python training/train_hybrid.py --config config/default.yaml
    ;;
  *)
    # Arbitrary command passthrough, e.g. `bash`, `pytest tests/test_env.py -v`
    exec "$@"
    ;;
esac
