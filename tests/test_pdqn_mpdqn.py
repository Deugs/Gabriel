"""Unit tests for the P-DQN / MP-DQN baselines (agents/pdqn_mpdqn.py)."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]

from agents import MAX_SAFE_N_RRH, MPDQNAgent, PDQNAgent
from agents.pdqn_mpdqn import JointActionSpace, JointQNetwork
from cran_env import CRANEnv
from training.train_baselines import run_baseline_benchmarks

AGENT_CLASSES = [PDQNAgent, MPDQNAgent]


def _load_config(name):
    config_path = Path(__file__).parent.parent / "config" / name
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def small_config():
    return _load_config("small_network.yaml")  # R=5


@pytest.fixture
def default_config():
    return _load_config("default.yaml")  # R=12


@pytest.mark.parametrize("agent_cls", AGENT_CLASSES)
@pytest.mark.parametrize("config_name", ["small_config", "default_config"])
def test_action_selection_shapes(agent_cls, config_name, request):
    cfg = request.getfixturevalue(config_name)
    env = CRANEnv(cfg)
    obs, _ = env.reset(seed=42)

    agent = agent_cls(
        state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
    )

    for evaluate in (False, True):
        action = agent.select_action(obs, evaluate=evaluate)
        assert "rrh_on" in action and "power" in action and "bandwidth" in action
        assert "config_idx" in action

        assert action["rrh_on"].shape == (env.n_rrh,)
        assert set(np.unique(action["rrh_on"]).tolist()) <= {0, 1}
        assert action["power"].shape == (env.n_rrh,)
        assert np.all(action["power"] >= 0.0)
        assert np.all(action["power"] <= env.p_max_w + 1e-5)
        assert action["bandwidth"].shape == (env.n_rrh,)
        assert np.all(action["bandwidth"] >= 0.0)

        # select_action and JointActionSpace.encode must agree on the same
        # joint configuration index.
        encoded = agent.action_space.encode(
            torch.as_tensor(action["rrh_on"], dtype=torch.float32)
        )
        assert int(encoded.item()) == action["config_idx"]


@pytest.mark.parametrize("agent_cls", AGENT_CLASSES)
@pytest.mark.parametrize("config_name", ["small_config", "default_config"])
def test_update_finite_nonnegative_loss(agent_cls, config_name, request):
    cfg = request.getfixturevalue(config_name)
    env = CRANEnv(cfg)
    obs, _ = env.reset(seed=42)

    agent = agent_cls(
        state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
    )

    for _ in range(40):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        cont_params = np.stack(
            [action["power"] / env.p_max_w, action["bandwidth"]], axis=-1
        )
        agent.memory.push(
            obs, action["config_idx"], cont_params, reward, next_obs, terminated
        )
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    metrics = agent.update(batch_size=16)
    assert "critic_loss" in metrics and "param_loss" in metrics and "epsilon" in metrics
    assert not np.isnan(metrics["critic_loss"])
    assert metrics["critic_loss"] >= 0.0  # MSE loss is non-negative by construction
    assert not np.isnan(metrics["param_loss"])
    assert np.isfinite(metrics["param_loss"])


def test_joint_action_space_roundtrip():
    space = JointActionSpace(n_rrh=12, device=torch.device("cpu"))
    idx = torch.randint(0, space.n_configs, (200,))
    assert torch.equal(space.encode(space.decode(idx)), idx)


@pytest.mark.parametrize("mode", ["single_pass", "multi_pass"])
def test_forward_single_matches_full_forward(mode):
    torch.manual_seed(0)
    n_rrh = 5
    state_dim = 10
    space = JointActionSpace(n_rrh=n_rrh, device=torch.device("cpu"))
    net = JointQNetwork(state_dim, n_rrh, space, mode=mode)

    batch = 8
    state = torch.randn(batch, state_dim)
    x = torch.rand(batch, n_rrh, 2)
    config_idx = torch.randint(0, space.n_configs, (batch,))

    full = net.forward(state, x)  # (batch, K)
    expected = full.gather(-1, config_idx.unsqueeze(-1)).squeeze(-1)
    actual = net.forward_single(state, x, config_idx)

    assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("agent_cls", AGENT_CLASSES)
def test_max_safe_n_rrh_guard_raises(agent_cls):
    with pytest.raises(ValueError):
        agent_cls(state_dim=50, n_rrh=MAX_SAFE_N_RRH + 1)


def test_train_baselines_skips_pdqn_mpdqn_at_large_r():
    results = run_baseline_benchmarks(
        config_path="config/large_network.yaml",
        algorithms=["pdqn", "mpdqn"],
        episodes=1,
        seeds=[42],
    )
    assert results == {"pdqn": [], "mpdqn": []}
