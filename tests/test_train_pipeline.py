"""Unit tests for end-to-end training pipeline execution."""

import os
from pathlib import Path
import tempfile

from training.train_hybrid import train_hybrid_agent
from training.train_baselines import run_baseline_benchmarks
from training.run_extended_sweeps import run_extended_sweeps


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
        assert os.path.exists(
            Path(tmp_dir) / "branching_mp_dqn_seed42" / "summary.json"
        )
        assert os.path.exists(
            Path(tmp_dir) / "branching_mp_dqn_seed42" / "final_model.pt"
        )


def test_run_baseline_benchmarks_short_run():
    """Verify run_baseline_benchmarks executes all 7 baselines without error."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_baseline_benchmarks(
            config_path="config/small_network.yaml",
            seeds=[42],
            episodes=2,
            algorithms=[
                "all_on",
                "greedy",
                "nmbs",
                "convex",
                "ddqn",
                "ann_gsbf",
                "ddqn_socp",
            ],
            save_dir=tmp_dir,
        )

        assert len(results) == 7
        for algo_name in [
            "all_on",
            "greedy",
            "nmbs",
            "convex",
            "ddqn",
            "ann_gsbf",
            "ddqn_socp",
        ]:
            assert algo_name in results
            assert os.path.exists(
                Path(tmp_dir) / f"benchmark_{algo_name}" / "summary.json"
            )


def test_run_extended_sweeps_short_run(tmp_path, monkeypatch):
    """training/run_extended_sweeps.py had no test coverage at all — verify
    the full orchestration (baselines -> hybrid agent -> convergence
    analysis) runs end-to-end without error. analyze_convergence() defaults
    to writing under thesis/figures and thesis/tables (real repo paths), so
    chdir into tmp_path first to avoid polluting the actual repository."""
    monkeypatch.chdir(tmp_path)
    results_dir = str(tmp_path / "results")

    run_extended_sweeps(
        config_path=str(Path(__file__).parent.parent / "config" / "small_network.yaml"),
        episodes=2,
        seeds=[42],
        results_dir=results_dir,
    )

    assert (Path(results_dir) / "branching_mp_dqn_seed42" / "summary.json").exists()
    assert (tmp_path / "thesis" / "figures").exists()
    assert (tmp_path / "thesis" / "tables").exists()


def test_run_extended_sweeps_reuses_checkpoint_for_csi_and_generalization(
    tmp_path, monkeypatch
):
    """run_csi_and_generalization=True must reuse each seed's just-trained
    checkpoint (Concept Note v4.0 Section 14), not just be a no-op flag."""
    monkeypatch.chdir(tmp_path)
    results_dir = str(tmp_path / "results")

    run_extended_sweeps(
        config_path=str(Path(__file__).parent.parent / "config" / "small_network.yaml"),
        episodes=2,
        seeds=[42],
        results_dir=results_dir,
        run_csi_and_generalization=True,
    )

    assert (Path(results_dir) / "branching_mp_dqn_seed42" / "final_model.pt").exists()
    assert (
        Path(results_dir) / "csi_robustness_seed42" / "csi_robustness_ee.pdf"
    ).exists()
    assert (
        Path(results_dir) / "generalization_seed42" / "generalization_ee.pdf"
    ).exists()
