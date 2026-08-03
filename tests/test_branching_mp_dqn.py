"""Unit tests for Branching MP-DQN + TD3 Agent (agents/branching_mp_dqn.py)."""

from pathlib import Path
import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]

from agents import BranchingMPDQN
from cran_env import CRANEnv


@pytest.fixture
def default_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_branching_mp_dqn_action_selection(default_config):
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    agent = BranchingMPDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=default_config,
    )

    action = agent.select_action(obs, evaluate=False)
    assert "rrh_on" in action and "power" in action and "bandwidth" in action
    assert action["rrh_on"].shape == (env.n_rrh,)
    assert action["power"].shape == (env.n_rrh,)
    assert action["bandwidth"].shape == (env.n_rrh,)
    assert np.all(action["power"] >= 0.0)
    assert np.all(action["power"] <= env.p_max_w + 1e-5)

    eval_action = agent.select_action(obs, evaluate=True)
    assert np.all(eval_action["power"] >= 0.0)


def test_branching_mp_dqn_update(default_config):
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    agent = BranchingMPDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=default_config,
    )

    # Populate replay buffer with 30 steps
    for _ in range(30):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        cont_params = np.stack(
            [action["power"] / env.p_max_w, action["bandwidth"]], axis=-1
        )
        agent.memory.push(
            obs,
            action["rrh_on"],
            cont_params,
            reward,
            next_obs,
            terminated,
        )
        obs = next_obs

    metrics = agent.update(batch_size=16)
    assert "critic_loss" in metrics
    assert "param_loss" in metrics
    assert "epsilon" in metrics
    assert not np.isnan(metrics["critic_loss"])
