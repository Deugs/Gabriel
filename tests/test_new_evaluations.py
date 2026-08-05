"""Short-run tests for the CSI-robustness, generalization, and inference-latency
evaluation modules added per Concept Note v3.0/v4.0 Section 12.3/12.5 (S3, A3, A5).
"""

from pathlib import Path

from evaluation import (
    run_csi_robustness_evaluation,
    run_generalization_evaluation,
    run_latency_benchmark,
)


def test_csi_robustness_short_run(tmp_path):
    fig_dir = str(tmp_path / "figures")

    results = run_csi_robustness_evaluation(
        config_path="config/small_network.yaml",
        methods=["branching_mp_dqn", "ddqn"],
        sigmas=[0.0, 0.05],
        train_episodes=2,
        eval_episodes=1,
        batch_size=16,
        save_dir=fig_dir,
    )

    assert set(results.keys()) == {"branching_mp_dqn", "ddqn"}
    for method_results in results.values():
        for sigma_metrics in method_results.values():
            assert "ee_mbit_per_joule" in sigma_metrics
            assert "qos_violation_rate" in sigma_metrics
    assert (Path(fig_dir) / "csi_robustness_ee.pdf").exists()
    assert (Path(fig_dir) / "csi_robustness_qos.pdf").exists()


def test_generalization_short_run(tmp_path):
    fig_dir = str(tmp_path / "figures")

    results = run_generalization_evaluation(
        config_path="config/small_network.yaml",
        methods=["branching_mp_dqn"],
        train_episodes=2,
        eval_episodes=1,
        batch_size=16,
        save_dir=fig_dir,
    )

    assert "branching_mp_dqn" in results
    assert "weekday_urban_matched" in results["branching_mp_dqn"]
    assert "weekend_suburban_generalization" in results["branching_mp_dqn"]
    assert (Path(fig_dir) / "generalization_ee.pdf").exists()


def test_latency_benchmark_short_run(tmp_path):
    fig_dir = str(tmp_path / "figures")

    results = run_latency_benchmark(
        config_path="config/small_network.yaml",
        methods=["branching_mp_dqn", "pdqn", "mpdqn"],
        n_rrh_values=[4, 25],
        n_repeats=3,
        save_dir=fig_dir,
    )

    assert set(results.keys()) == {"branching_mp_dqn", "pdqn", "mpdqn"}
    # Tractable at R=4 for every method.
    for method in ["branching_mp_dqn", "pdqn", "mpdqn"]:
        assert results[method][4] is not None
    # P-DQN/MP-DQN must be explicitly skipped (None), not crash, at R=25 (> cap).
    assert results["pdqn"][25] is None
    assert results["mpdqn"][25] is None
    # Branching scales to R=25 with no cap.
    assert results["branching_mp_dqn"][25] is not None
    assert (Path(fig_dir) / "latency_benchmark.pdf").exists()
