"""Unit tests for Proposed Hybrid SAC-DDQN Agent (agents/hybrid_sac_dqn.py)."""

from pathlib import Path
import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]

from agents import (
    ContinuousActor,
    DiscreteActor,
    HybridCritic,
    HybridSACDDQN,
)
from cran_env import CRANEnv


@pytest.fixture
def default_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_discrete_actor():
    state_dim, n_rrh = 134, 12
    actor = DiscreteActor(state_dim, n_rrh)

    state_t = torch.randn(4, state_dim)
    q_vals = actor(state_t)
    assert q_vals.shape == (4, n_rrh, 2)

    action = actor.select_action(torch.randn(state_dim), epsilon=0.0)
    assert action.shape == (n_rrh,)
    assert torch.all((action == 0) | (action == 1))


def test_continuous_actor():
    state_dim, n_rrh = 134, 12
    actor = ContinuousActor(state_dim, n_rrh)

    state_t = torch.randn(4, state_dim)
    mean, log_std = actor(state_t)
    assert mean.shape == (4, n_rrh)
    assert log_std.shape == (4, n_rrh)
    assert torch.all(mean >= 0.0) and torch.all(mean <= 1.0)

    action, log_prob = actor.sample(state_t)
    assert action.shape == (4, n_rrh)
    assert log_prob.shape == (4, 1)
    assert torch.all(action >= 0.0) and torch.all(action <= 1.0)
    assert not torch.isnan(log_prob).any()


def test_hybrid_critic():
    state_dim, n_rrh = 134, 12
    critic = HybridCritic(state_dim, n_rrh)

    state_t = torch.randn(4, state_dim)
    disc_act = torch.randint(0, 2, (4, n_rrh))
    cont_act = torch.rand(4, n_rrh)

    q1, q2 = critic(state_t, disc_act, cont_act)
    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)
    assert not torch.isnan(q1).any()
    assert not torch.isnan(q2).any()


def test_hybrid_sac_ddqn_action_selection(default_config):
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    agent = HybridSACDDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=default_config,
    )

    action = agent.select_action(obs, evaluate=False)
    assert "rrh_on" in action and "power" in action
    assert action["rrh_on"].shape == (env.n_rrh,)
    assert action["power"].shape == (env.n_rrh,)
    assert np.all(action["power"] >= 0.0)
    assert np.all(action["power"] <= env.p_max_w + 1e-5)

    eval_action = agent.select_action(obs, evaluate=True)
    assert np.all(eval_action["power"] >= 0.0)


def test_hybrid_sac_ddqn_update(default_config):
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    agent = HybridSACDDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=default_config,
    )

    # Populate replay memory
    for _ in range(30):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.memory.push(
            obs,
            action["rrh_on"],
            action["power"] / env.p_max_w,
            reward,
            next_obs,
            terminated,
        )
        obs = next_obs

    metrics = agent.update(batch_size=16)

    assert "critic_loss" in metrics
    assert "disc_loss" in metrics
    assert "actor_loss" in metrics
    assert "alpha" in metrics
    assert "epsilon" in metrics

    assert not np.isnan(metrics["critic_loss"])
    assert not np.isnan(metrics["actor_loss"])


def test_hybrid_agent_in_cran_env(default_config):
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    agent = HybridSACDDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=default_config,
    )

    rewards = []
    for _ in range(20):
        action = agent.select_action(obs, evaluate=False)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            obs, _ = env.reset()

    assert len(rewards) == 20
    assert not np.isnan(rewards).any()
