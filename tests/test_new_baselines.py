"""Unit tests for the P-DQN, MP-DQN and pure-DDPG baselines added per Concept
Note v3.0/v4.0 Section 12.1 (S2, RQ3): agents/pdqn_agent.py, agents/mpdqn_agent.py,
agents/ddpg_agent.py.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from agents import DDPGAgent, MPDQNAgent, PDQNAgent
from cran_env import CRANEnv
from training.train_baselines import run_baseline_benchmarks


@pytest.fixture
def small_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Keep n_rrh tiny so the 2^R flat joint-action head (P-DQN/MP-DQN) stays cheap.
    cfg = deepcopy(cfg)
    cfg["network"]["n_rrh"] = 4
    cfg["network"]["n_ue"] = 3
    return cfg


def test_pdqn_action_selection_and_update(small_config):
    env = CRANEnv(small_config)
    obs, _ = env.reset(seed=42)

    agent = PDQNAgent(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=small_config,
    )

    action = agent.select_action(obs, evaluate=False)
    assert action["rrh_on"].shape == (env.n_rrh,)
    assert action["power"].shape == (env.n_rrh,)
    assert np.all(action["power"] >= 0.0)
    assert np.all(action["power"] <= env.p_max_w + 1e-5)
    assert 0 <= action["action_idx"] < 2**env.n_rrh

    for _ in range(30):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.memory.push(
            obs,
            action["action_idx"],
            action["continuous"],
            reward,
            next_obs,
            terminated,
        )
        obs = next_obs

    metrics = agent.update(batch_size=16)
    assert "critic_loss" in metrics
    assert not np.isnan(metrics["critic_loss"])


def test_pdqn_rejects_intractable_n_rrh():
    with pytest.raises(ValueError):
        PDQNAgent(state_dim=100, n_rrh=25)


def test_mpdqn_action_selection_and_update(small_config):
    env = CRANEnv(small_config)
    obs, _ = env.reset(seed=42)

    agent = MPDQNAgent(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=small_config,
    )

    action = agent.select_action(obs, evaluate=False)
    assert action["rrh_on"].shape == (env.n_rrh,)
    assert 0 <= action["action_idx"] < 2**env.n_rrh

    for _ in range(30):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.memory.push(
            obs,
            action["action_idx"],
            action["continuous"],
            reward,
            next_obs,
            terminated,
        )
        obs = next_obs

    metrics = agent.update(batch_size=16)
    assert "critic_loss" in metrics
    assert not np.isnan(metrics["critic_loss"])


def test_mpdqn_masks_inactive_rrh_params(small_config):
    """The masked Q-value for a candidate action must not depend on params of RRHs OFF under it."""
    env = CRANEnv(small_config)
    obs, _ = env.reset(seed=42)
    agent = MPDQNAgent(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=small_config,
    )

    import torch

    state_t = torch.FloatTensor(obs).unsqueeze(0)
    feat = agent.encoder(state_t)
    cont_params_a = torch.rand(1, env.n_rrh, 2)
    cont_params_b = cont_params_a.clone()

    # Action index 0 means every RRH is OFF; perturbing the (irrelevant, masked)
    # continuous params must not change action 0's masked Q-value.
    cont_params_b[0, 0, 0] += 5.0

    q_a = agent._compute_q_all_actions(agent.q_net, feat, cont_params_a)
    q_b = agent._compute_q_all_actions(agent.q_net, feat, cont_params_b)

    assert torch.allclose(q_a[0, 0], q_b[0, 0], atol=1e-5)


def test_ddpg_action_selection_and_update(small_config):
    env = CRANEnv(small_config)
    obs, _ = env.reset(seed=42)

    agent = DDPGAgent(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=small_config,
    )

    action = agent.select_action(obs, evaluate=False)
    assert action["rrh_on"].shape == (env.n_rrh,)
    assert action["power"].shape == (env.n_rrh,)
    assert set(np.unique(action["rrh_on"]).tolist()) <= {0, 1}

    for _ in range(30):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.memory.push(
            obs, action["continuous_action"], reward, next_obs, terminated
        )
        obs = next_obs

    metrics = agent.update(batch_size=16)
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert not np.isnan(metrics["critic_loss"])


def test_pdqn_reads_algorithm_config_section(small_config):
    """Regression test: `getattr(cfg, "algorithm", cfg)` does not perform
    dict key lookup, so for a plain dict config this previously always
    resolved to the whole cfg object (not cfg["algorithm"]), silently
    discarding every algorithm: hyperparameter regardless of the YAML."""
    cfg = deepcopy(small_config)
    cfg["algorithm"]["buffer_size"] = 777

    agent = PDQNAgent(
        state_dim=20, n_rrh=cfg["network"]["n_rrh"], p_max_w=1.0, config=cfg
    )
    assert agent.memory.buffer.maxlen == 777


def test_ddpg_reads_algorithm_config_section(small_config):
    cfg = deepcopy(small_config)
    cfg["algorithm"]["buffer_size"] = 777

    agent = DDPGAgent(
        state_dim=20, n_rrh=cfg["network"]["n_rrh"], p_max_w=1.0, config=cfg
    )
    assert agent.memory.buffer.maxlen == 777


def test_run_baseline_benchmarks_includes_new_methods(tmp_path):
    """agents/ddpg_agent.py, pdqn_agent.py, mpdqn_agent.py wired into the
    unified baseline runner (Concept Note v3.0/v4.0 Section 12.1, S2)."""
    results = run_baseline_benchmarks(
        config_path="config/small_network.yaml",
        seeds=[42],
        episodes=2,
        algorithms=["ddpg", "pdqn", "mpdqn"],
        save_dir=str(tmp_path / "benchmarks"),
    )

    assert set(results.keys()) == {"ddpg", "pdqn", "mpdqn"}
    for algo, algo_results in results.items():
        assert len(algo_results) == 1
        assert not np.isnan(algo_results[0]["mean_reward"])
