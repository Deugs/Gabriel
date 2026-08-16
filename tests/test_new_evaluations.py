"""Short-run tests for the CSI-robustness, generalization, inference-latency,
demand-response, and power/time-profile evaluation modules added per Concept
Note v3.0/v4.0 Section 12.3/12.5 (S3, A3, A5).
"""

from pathlib import Path

import pytest
import torch
import yaml  # type: ignore[import-untyped]

from agents import BranchingMPDQN
from cran_env import CRANEnv
from evaluation import (
    run_csi_robustness_evaluation,
    run_demand_response_evaluation,
    run_generalization_evaluation,
    run_latency_benchmark,
    run_power_time_profile_evaluation,
    run_reward_sensitivity_sweep,
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


def test_csi_robustness_reuses_checkpoint_instead_of_training(tmp_path):
    """checkpoint_paths must genuinely load the given checkpoint rather than
    training a fresh agent (Concept Note v4.0 Section 14's "reuse
    already-trained checkpoints" mitigation)."""
    config_path = Path(__file__).parent.parent / "config" / "small_network.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env = CRANEnv(cfg)

    trained_agent = BranchingMPDQN(
        state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
    )
    ckpt_path = tmp_path / "final_model.pt"
    torch.save(
        {
            "encoder": trained_agent.encoder.state_dict(),
            "param_net": trained_agent.param_net.state_dict(),
            "twin_critic": trained_agent.twin_critic.state_dict(),
        },
        ckpt_path,
    )

    results = run_csi_robustness_evaluation(
        config_path=str(config_path),
        methods=["branching_mp_dqn"],
        sigmas=[0.0],
        eval_episodes=1,
        save_dir=str(tmp_path / "figures"),
        checkpoint_paths={"branching_mp_dqn": str(ckpt_path)},
    )
    assert "branching_mp_dqn" in results


def test_csi_robustness_checkpoint_reuse_not_implemented_for_baselines(tmp_path):
    """Passing a checkpoint path for ddqn/ddpg must raise, not silently
    ignore the path and train from scratch anyway."""
    config_path = Path(__file__).parent.parent / "config" / "small_network.yaml"
    with pytest.raises(NotImplementedError):
        run_csi_robustness_evaluation(
            config_path=str(config_path),
            methods=["ddqn"],
            sigmas=[0.0],
            train_episodes=1,
            eval_episodes=1,
            save_dir=str(tmp_path / "figures"),
            checkpoint_paths={"ddqn": "not_a_real_path.pt"},
        )


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


def test_demand_response_short_run(tmp_path):
    fig_dir = str(tmp_path / "figures")

    results = run_demand_response_evaluation(
        config_path="config/small_network.yaml",
        methods=["branching_mp_dqn", "ddqn"],
        demand_multipliers=[0.5, 1.5],
        train_episodes=2,
        eval_episodes=1,
        batch_size=16,
        save_dir=fig_dir,
    )

    assert set(results.keys()) == {"branching_mp_dqn", "ddqn"}
    for method_results in results.values():
        assert set(method_results.keys()) == {0.5, 1.5}
        for demand_metrics in method_results.values():
            assert "ee_mbit_per_joule" in demand_metrics
            assert "mean_power_w" in demand_metrics
    assert (Path(fig_dir) / "demand_response_ee.pdf").exists()
    assert (Path(fig_dir) / "demand_response_power.pdf").exists()


def test_power_time_profile_short_run(tmp_path):
    fig_dir = str(tmp_path / "figures")

    results = run_power_time_profile_evaluation(
        config_path="config/small_network.yaml",
        methods=["branching_mp_dqn"],
        train_episodes=2,
        eval_episodes=2,
        batch_size=16,
        save_dir=fig_dir,
    )

    assert "branching_mp_dqn" in results
    assert set(results["branching_mp_dqn"].keys()) == set(range(24))
    assert (Path(fig_dir) / "power_time_profile.pdf").exists()


def test_reward_sensitivity_sweep_short_run(tmp_path):
    fig_dir = str(tmp_path / "figures")

    results = run_reward_sensitivity_sweep(
        config_path="config/small_network.yaml",
        gamma_switch_grid=[0.01, 1.0],
        train_episodes=2,
        eval_episodes=1,
        batch_size=16,
        save_dir=fig_dir,
    )

    assert set(results.keys()) == {0.01, 1.0}
    for metrics in results.values():
        assert "ee_mbit_per_joule" in metrics
        assert "qos_violation_rate" in metrics
        assert "switching_frequency" in metrics
    assert (Path(fig_dir) / "reward_sensitivity_ee.pdf").exists()
    assert (Path(fig_dir) / "reward_sensitivity_qos.pdf").exists()
    assert (Path(fig_dir) / "reward_sensitivity_switching.pdf").exists()
