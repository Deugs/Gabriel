"""Unit tests for the CSI-robustness evaluation (evaluation/csi_robustness.py
and CRANEnv's observation_noise_std)."""

from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from agents import BranchingMPDQN
from cran_env import CRANEnv
from evaluation.csi_robustness import evaluate_csi_robustness
from training.eval_utils import run_eval_episodes


@pytest.fixture
def small_config():
    config_path = Path(__file__).parent.parent / "config" / "small_network.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def agent(small_config):
    env = CRANEnv(small_config)
    return BranchingMPDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=small_config,
    )


def test_sigma_zero_reproduces_baseline(agent, small_config):
    result = evaluate_csi_robustness(
        agent, small_config, sigmas=[0.0], episodes=2, seed=7
    )
    baseline_env = CRANEnv(small_config)
    baseline = run_eval_episodes(baseline_env, agent, episodes=2, seed=7)

    assert result[0.0] == baseline


def test_observation_noise_does_not_alter_environment_trajectory(small_config):
    env_clean = CRANEnv(small_config, observation_noise_std=0.0)
    env_noisy = CRANEnv(small_config, observation_noise_std=0.1)

    obs_clean, _ = env_clean.reset(seed=5)
    obs_noisy, _ = env_noisy.reset(seed=5)

    # sigma>0 must not change the true channel state.
    assert np.allclose(env_clean.channel_gains, env_noisy.channel_gains)

    n_rrh = env_clean.n_rrh
    action = {
        "rrh_on": np.ones(n_rrh, dtype=np.int64),
        "power": np.full(n_rrh, env_clean.p_max_w, dtype=np.float32),
        "bandwidth": np.full(n_rrh, 1.0 / n_rrh, dtype=np.float32),
    }

    next_obs_clean, reward_clean, _, _, info_clean = env_clean.step(action)
    next_obs_noisy, reward_noisy, _, _, info_noisy = env_noisy.step(action)

    assert reward_clean == reward_noisy
    assert info_clean["total_power_w"] == info_noisy["total_power_w"]
    assert info_clean["ee_mbit_per_joule"] == info_noisy["ee_mbit_per_joule"]
    assert np.allclose(env_clean.channel_gains, env_noisy.channel_gains)

    # But the *observed* channel-magnitude slice must actually differ once
    # sigma > 0 (channel-gain block is the first n_rrh*n_ue entries of obs).
    n_gain_entries = env_clean.n_rrh * env_clean.n_ue
    assert not np.allclose(
        next_obs_clean[:n_gain_entries], next_obs_noisy[:n_gain_entries]
    )


def test_csi_robustness_output_shape(agent, small_config):
    sigmas = [0.0, 0.01, 0.05, 0.1]
    result = evaluate_csi_robustness(
        agent, small_config, sigmas=sigmas, episodes=2, seed=42
    )
    assert set(result.keys()) == set(sigmas)
    for metrics in result.values():
        for value in metrics.values():
            assert np.isfinite(value)
