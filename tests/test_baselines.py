"""Unit tests for Non-DRL & Simple-DRL Baseline Algorithms (baselines/ and agents/ddqn_agent.py)."""

import numpy as np
from pathlib import Path
import pytest
import yaml

from agents import DDQNAgent
from baselines import (
    AllOnUniformBaseline,
    ConvexPowerBaseline,
    GreedyHeuristicBaseline,
    NMBSBinPackingBaseline,
)
from cran_env import CRANEnv


@pytest.fixture
def default_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_all_on_uniform_baseline(default_config):
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    baseline = AllOnUniformBaseline(n_rrh=env.n_rrh, p_max_w=env.p_max_w)
    action = baseline.select_action(obs)

    assert "rrh_on" in action and "power" in action
    assert np.all(action["rrh_on"] == 1)
    assert np.allclose(action["power"], env.p_max_w)

    next_obs, reward, _, _, info = env.step(action)
    assert not np.isnan(reward)
    assert info["active_rrhs"] == env.n_rrh


def test_greedy_heuristic_baseline(default_config):
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    baseline = GreedyHeuristicBaseline(
        n_rrh=env.n_rrh, n_ue=env.n_ue, p_max_w=env.p_max_w
    )
    action = baseline.select_action(obs)

    assert "rrh_on" in action and "power" in action
    assert action["rrh_on"].shape == (env.n_rrh,)
    assert action["power"].shape == (env.n_rrh,)
    assert np.sum(action["rrh_on"]) > 0

    _, reward, _, _, info = env.step(action)
    assert not np.isnan(reward)


def test_nmbs_binpack_baseline(default_config):
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    baseline = NMBSBinPackingBaseline(
        n_rrh=env.n_rrh, n_ue=env.n_ue, p_max_w=env.p_max_w
    )
    action = baseline.select_action(obs)

    assert "rrh_on" in action and "power" in action
    assert np.sum(action["rrh_on"]) >= 1

    _, reward, _, _, info = env.step(action)
    assert not np.isnan(reward)


def test_nmbs_binpack_docstring_does_not_overclaim_reproduction():
    """Guards against re-introducing the mischaracterization that this
    per-slot FFD heuristic is a reproduction of Al-Zubaedi (2019)'s actual
    NMBS algorithm, which Concept Note v4.0's own literature table (Section 4)
    describes as a deployment/planning-timescale metaheuristic, not a
    slot-by-slot resource-allocation heuristic."""
    import baselines.nmbs_binpack as nmbs_binpack_module

    docstring = nmbs_binpack_module.__doc__ or ""
    assert "NOT a reproduction" in docstring


def test_convex_power_baseline(default_config):
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    baseline = ConvexPowerBaseline(n_rrh=env.n_rrh, n_ue=env.n_ue, p_max_w=env.p_max_w)
    action = baseline.select_action(obs)

    assert "rrh_on" in action and "power" in action
    assert np.all(action["power"] >= 0.0)
    assert np.all(action["power"] <= env.p_max_w + 1e-5)

    _, reward, _, _, info = env.step(action)
    assert not np.isnan(reward)


def test_ddqn_agent(default_config):
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    agent = DDQNAgent(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        batch_size=16,
    )

    action = agent.select_action(obs, evaluate=False)
    assert "rrh_on" in action and "power" in action

    next_obs, reward, terminated, truncated, _ = env.step(action)

    # Push to replay buffer
    agent.memory.push(obs, action["rrh_on"], reward, next_obs, terminated)

    # Fill buffer to batch size
    for _ in range(20):
        agent.memory.push(obs, action["rrh_on"], reward, next_obs, terminated)

    # Perform training update
    metrics = agent.update()
    assert "loss" in metrics and "epsilon" in metrics
    assert not np.isnan(metrics["loss"])


def test_baselines_run_against_cran_env(default_config):
    env = CRANEnv(default_config)

    baselines = {
        "AllOn": AllOnUniformBaseline(env.n_rrh, env.p_max_w),
        "Greedy": GreedyHeuristicBaseline(env.n_rrh, env.n_ue, env.p_max_w),
        "NMBS": NMBSBinPackingBaseline(env.n_rrh, env.n_ue, env.p_max_w),
        "Convex": ConvexPowerBaseline(env.n_rrh, env.n_ue, env.p_max_w),
    }

    results = {}
    for name, policy in baselines.items():
        obs, _ = env.reset(seed=42)
        total_power_sum = 0.0

        for _ in range(10):
            action = policy.select_action(obs)
            obs, _, _, _, info = env.step(action)
            total_power_sum += info["total_power_w"]

        results[name] = total_power_sum / 10.0

    # Verify All-ON consumes the most power
    assert results["AllOn"] > results["NMBS"]
    assert results["AllOn"] > results["Greedy"]
