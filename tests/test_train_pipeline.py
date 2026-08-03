"""Unit tests for end-to-end training pipeline execution."""

import os
from pathlib import Path
import tempfile
import pytest

from training.train_hybrid import train_hybrid_agent
from training.train_baselines import run_baseline_benchmarks


def test_train_hybrid_agent_short_run():
    """Verify train_hybrid_agent executes end-to-end without error."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary = train_hybrid_agent(
            config_path="config/small_network.yaml",
            seed=42,
            episodes=2,
            eval_freq=2,
            save_dir=tmp_dir,
        )

        assert summary["algorithm"] == "Branching_MP_DQN"
        assert summary["episodes"] == 2
        assert len(summary["history"]["episode_rewards"]) == 2
        assert os.path.exists(Path(tmp_dir) / "branching_mp_dqn_seed42" / "summary.json")
        assert os.path.exists(Path(tmp_dir) / "branching_mp_dqn_seed42" / "final_model.pt")


def test_run_baseline_benchmarks_short_run():
    """Verify run_baseline_benchmarks executes all 7 baselines without error."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_baseline_benchmarks(
            config_path="config/small_network.yaml",
            seeds=[42],
            episodes=2,
            algorithms=["all_on", "greedy", "nmbs", "convex", "ddqn", "ann_gsbf", "ddqn_socp"],
            save_dir=tmp_dir,
        )

        assert len(results) == 7
        for algo_name in ["all_on", "greedy", "nmbs", "convex", "ddqn", "ann_gsbf", "ddqn_socp"]:
            assert algo_name in results
            assert os.path.exists(Path(tmp_dir) / f"benchmark_{algo_name}" / "summary.json")
