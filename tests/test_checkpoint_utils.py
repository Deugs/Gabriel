"""Unit tests for training/checkpoint_utils.py (save_checkpoint/load_checkpoint)."""

from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from agents import BranchingMPDQN, HybridSACDDQN
from cran_env import CRANEnv
from training.checkpoint_utils import load_checkpoint, save_checkpoint


@pytest.fixture
def small_config():
    config_path = Path(__file__).parent.parent / "config" / "small_network.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _train_a_little(agent, env, obs, steps=30):
    """Push a handful of transitions and take a few update steps to move weights
    off their initial values, so a save/load round-trip test is non-trivial."""
    for _ in range(steps):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        cont_action = action.get(
            "continuous",
            np.stack([action["power"] / env.p_max_w, action["bandwidth"]], axis=-1),
        )
        agent.memory.push(obs, action["rrh_on"], cont_action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    for _ in range(5):
        agent.update(batch_size=8)
    return obs


def test_branching_mp_dqn_checkpoint_roundtrip(tmp_path, small_config):
    env = CRANEnv(small_config)
    obs, _ = env.reset(seed=42)

    agent = BranchingMPDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=small_config,
    )
    _train_a_little(agent, env, obs)

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        agent,
        ckpt_path,
        meta={"config": small_config, "ctor_kwargs": {"config": small_config}},
    )

    loaded = load_checkpoint(BranchingMPDQN, ckpt_path)

    obs, _ = env.reset(seed=99)
    original_action = agent.select_action(obs, evaluate=True)
    loaded_action = loaded.select_action(obs, evaluate=True)

    assert np.array_equal(original_action["rrh_on"], loaded_action["rrh_on"])
    assert np.allclose(original_action["power"], loaded_action["power"], atol=1e-6)
    assert np.allclose(
        original_action["bandwidth"], loaded_action["bandwidth"], atol=1e-6
    )


def test_hybrid_sac_ddqn_checkpoint_roundtrip(tmp_path, small_config):
    env = CRANEnv(small_config)
    obs, _ = env.reset(seed=42)

    agent = HybridSACDDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=small_config,
    )
    for _ in range(30):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.memory.push(obs, action["rrh_on"], action["power"] / env.p_max_w, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    for _ in range(5):
        agent.update(batch_size=8)

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        agent,
        ckpt_path,
        meta={"config": small_config, "ctor_kwargs": {"config": small_config}},
    )
    loaded = load_checkpoint(HybridSACDDQN, ckpt_path)

    obs, _ = env.reset(seed=99)
    original_action = agent.select_action(obs, evaluate=True)
    loaded_action = loaded.select_action(obs, evaluate=True)

    assert np.array_equal(original_action["rrh_on"], loaded_action["rrh_on"])
    assert np.allclose(original_action["power"], loaded_action["power"], atol=1e-6)


def test_load_checkpoint_needs_no_extra_hyperparams(tmp_path, small_config):
    """The caller should not need to remember state_dim/n_rrh/p_max_w -- only
    the agent class and the checkpoint path."""
    env = CRANEnv(small_config)
    agent = BranchingMPDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=small_config,
    )
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        agent,
        ckpt_path,
        meta={"config": small_config, "ctor_kwargs": {"config": small_config}},
    )

    loaded = load_checkpoint(BranchingMPDQN, ckpt_path)
    assert loaded.state_dim == env.state_dim
    assert loaded.n_rrh == env.n_rrh
    assert loaded.p_max_w == env.p_max_w
