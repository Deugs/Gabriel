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


def test_mpdqn_known_action_and_greedy_optimizations_are_exactly_equivalent(
    small_config,
):
    """agents/mpdqn_agent.py's `_compute_q_for_known_action`/
    `_compute_greedy_q_with_grad` overrides avoid an O(n_joint_actions)
    forward+backward cost (evaluating and backpropagating through every
    joint action just to use one) that was the root cause of MP-DQN taking
    dramatically longer per update() than every other baseline. This test
    proves both overrides are exact algebraic identities, not
    approximations: it reimplements the naive "evaluate all actions, then
    gather one" formula directly and asserts the values (and, for the
    gradient-carrying path, every gradient w.r.t. the continuous
    parameters) match the optimized override to floating-point precision.
    """
    import torch

    agent = MPDQNAgent(
        state_dim=12, n_rrh=4, p_max_w=1.0, config=small_config, device="cpu"
    )
    batch = 5
    feat = torch.randn(batch, agent.encoder.output_dim)

    # -- _compute_q_for_known_action: known action, no gradient needed --
    known_action_idx = torch.randint(0, agent.n_joint_actions, (batch,))
    cont_params = torch.rand(batch, agent.n_rrh, 2)
    q_naive = agent._compute_q_all_actions(agent.q_net, feat, cont_params).gather(
        -1, known_action_idx.unsqueeze(-1)
    )
    q_optimized = agent._compute_q_for_known_action(
        agent.q_net, feat, cont_params, known_action_idx
    )
    assert torch.allclose(q_naive, q_optimized, atol=1e-6)

    # -- _compute_greedy_q_with_grad: greedy action, gradient must match --
    torch.manual_seed(0)
    p_ratio_old, bw_share_old = agent.param_net(feat)
    pred_params_old = torch.stack([p_ratio_old, bw_share_old], dim=-1)
    q_pred_all = agent._compute_q_all_actions(agent.q_net, feat, pred_params_old)
    greedy_idx_old = q_pred_all.argmax(dim=-1, keepdim=True).detach()
    param_loss_old = -q_pred_all.gather(-1, greedy_idx_old).mean()
    grads_old = torch.autograd.grad(param_loss_old, list(agent.param_net.parameters()))

    agent.param_net.zero_grad()
    p_ratio_new, bw_share_new = agent.param_net(feat)
    pred_params_new = torch.stack([p_ratio_new, bw_share_new], dim=-1)
    q_pred_sel, _greedy_idx_new = agent._compute_greedy_q_with_grad(
        agent.q_net, feat, pred_params_new
    )
    param_loss_new = -q_pred_sel.mean()
    grads_new = torch.autograd.grad(param_loss_new, list(agent.param_net.parameters()))

    assert torch.allclose(param_loss_old, param_loss_new, atol=1e-6)
    for g_old, g_new in zip(grads_old, grads_new):
        assert torch.allclose(g_old, g_new, atol=1e-6)


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
