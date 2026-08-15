"""Unit tests for Branching MP-DQN + TD3 Agent (agents/branching_mp_dqn.py)."""

from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml  # type: ignore[import-untyped]

from agents import BranchingMPDQN
from agents.branching_mp_dqn import SharedEncoder
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


def test_shared_encoder_hidden_dims_is_config_driven():
    """algorithm.hidden_dims must actually change SharedEncoder's widths, not
    just be accepted and ignored (config/default.yaml's [256, 256] used to be
    silently overridden by a hardcoded [256, 128])."""
    encoder = SharedEncoder(state_dim=20, hidden_dims=[64, 32])
    assert encoder.output_dim == 32
    linear_layers = [m for m in encoder.network if isinstance(m, nn.Linear)]
    assert [layer.out_features for layer in linear_layers] == [64, 32]


def test_shared_encoder_activation_is_config_driven():
    encoder = SharedEncoder(state_dim=10, hidden_dims=[16], activation="tanh")
    assert any(isinstance(m, nn.Tanh) for m in encoder.network)
    assert not any(isinstance(m, nn.ReLU) for m in encoder.network)


def test_shared_encoder_rejects_unknown_activation():
    with pytest.raises(ValueError):
        SharedEncoder(
            state_dim=10, hidden_dims=[16], activation="not_a_real_activation"
        )


def test_shared_encoder_use_layer_norm_toggle():
    with_norm = SharedEncoder(state_dim=10, hidden_dims=[16], use_layer_norm=True)
    without_norm = SharedEncoder(state_dim=10, hidden_dims=[16], use_layer_norm=False)
    assert any(isinstance(m, nn.LayerNorm) for m in with_norm.network)
    assert not any(isinstance(m, nn.LayerNorm) for m in without_norm.network)


def test_branching_mp_dqn_reads_architecture_and_stability_config(default_config):
    """hidden_dims/activation/use_layer_norm/gradient_clip_norm/reward_scale
    (config/default.yaml's algorithm: block) must all reach BranchingMPDQN,
    not just the previously-wired hyperparameters."""
    env = CRANEnv(default_config)

    cfg = deepcopy(default_config)
    cfg["algorithm"]["hidden_dims"] = [64, 32]
    cfg["algorithm"]["activation"] = "tanh"
    cfg["algorithm"]["use_layer_norm"] = False
    cfg["algorithm"]["gradient_clip_norm"] = 0.5
    cfg["algorithm"]["reward_scale"] = 2.0

    agent = BranchingMPDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=cfg,
    )

    assert agent.encoder.output_dim == 32
    assert agent.gradient_clip_norm == pytest.approx(0.5)
    assert agent.reward_scale == pytest.approx(2.0)
    assert any(isinstance(m, nn.Tanh) for m in agent.encoder.network)
    assert not any(isinstance(m, nn.LayerNorm) for m in agent.encoder.network)


def test_branching_mp_dqn_defaults_to_spec_architecture_when_config_omits_keys():
    """config/small_network.yaml has no hidden_dims/activation/use_layer_norm/
    gradient_clip_norm/reward_scale keys — defaults must match the pre-wiring
    behavior (Concept Note Section 10.3's [256, 128] spec, ReLU, LayerNorm on,
    gradient_clip_norm=1.0, reward_scale=1.0) so those runs are unaffected."""
    config_path = Path(__file__).parent.parent / "config" / "small_network.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env = CRANEnv(cfg)

    agent = BranchingMPDQN(
        state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
    )

    assert agent.encoder.output_dim == 128
    assert agent.gradient_clip_norm == pytest.approx(1.0)
    assert agent.reward_scale == pytest.approx(1.0)
    assert any(isinstance(m, nn.ReLU) for m in agent.encoder.network)
    assert any(isinstance(m, nn.LayerNorm) for m in agent.encoder.network)


def test_branching_mp_dqn_reward_scale_affects_update(default_config):
    """reward_scale must actually change the Bellman target used in the
    critic loss, not just be stored and ignored."""
    torch.manual_seed(0)
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    transitions = []
    for _ in range(20):
        action = {
            "rrh_on": np.random.randint(0, 2, size=env.n_rrh),
            "continuous": np.random.uniform(0.0, 1.0, size=(env.n_rrh, 2)),
        }
        next_obs, reward, terminated, truncated, _ = env.step(
            {
                "rrh_on": action["rrh_on"],
                "power": action["continuous"][:, 0] * env.p_max_w,
                "bandwidth": action["continuous"][:, 1],
            }
        )
        transitions.append(
            (obs, action["rrh_on"], action["continuous"], reward, next_obs, terminated)
        )
        obs = next_obs if not (terminated or truncated) else env.reset(seed=42)[0]

    def make_agent(reward_scale):
        torch.manual_seed(1)
        cfg = deepcopy(default_config)
        cfg["algorithm"]["reward_scale"] = reward_scale
        agent = BranchingMPDQN(
            state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
        )
        for s, k, x, r, s2, d in transitions:
            agent.memory.push(s, k, x, r, s2, d)
        return agent

    agent_scale_1 = make_agent(1.0)
    agent_scale_10 = make_agent(10.0)

    torch.manual_seed(2)
    metrics_1 = agent_scale_1.update(batch_size=16)
    torch.manual_seed(2)
    metrics_10 = agent_scale_10.update(batch_size=16)

    assert metrics_1["critic_loss"] != pytest.approx(metrics_10["critic_loss"])


def test_branching_mp_dqn_device_defaults_from_hardware_config(default_config):
    """hardware.device (config/default.yaml) supplies BranchingMPDQN's device
    default; falls back to cpu regardless when no GPU is present, matching
    pre-wiring behavior in this (GPU-less) test environment."""
    env = CRANEnv(default_config)
    agent = BranchingMPDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=default_config,
    )
    assert agent.device.type == "cpu"

    # An explicit device= argument always wins over hardware.device.
    agent_explicit = BranchingMPDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=default_config,
        device="cpu",
    )
    assert agent_explicit.device.type == "cpu"
