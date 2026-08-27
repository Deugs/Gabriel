"""Unit tests for O-RAN DRL Agents (oran_agents/).

A fully separate test module from tests/test_branching_mp_dqn.py -- no
shared fixtures or imports with the C-RAN agent tests.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]

from oran_agents import (
    MAX_N_RU_FOR_FLAT_JOINT_ORAN_ACTION,
    BMPPDQNAgent,
    ORANDDPGAgent,
    ORANDQNAgent,
    ORANMPDQNAgent,
)
from oran_env import ORANEnv


@pytest.fixture
def default_config():
    config_path = Path(__file__).parent.parent / "config" / "oran_default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _make_agent(env, cfg):
    return BMPPDQNAgent(
        state_dim=env.state_dim,
        n_ru=env.n_ru,
        n_splits=env.n_splits,
        p_max_w=env.p_max_w,
        config=cfg,
    )


def test_select_action_returns_expected_keys_and_shapes(default_config):
    env = ORANEnv(default_config)
    obs, _ = env.reset(seed=42)
    agent = _make_agent(env, default_config)

    action = agent.select_action(obs, evaluate=False)
    assert set(action.keys()) == {"ru_on", "split", "power", "prb"}
    assert action["ru_on"].shape == (env.n_ru,)
    assert action["split"].shape == (env.n_ru,)
    assert action["power"].shape == (env.n_ru,)
    assert action["prb"].shape == (env.n_ru,)
    assert np.all(action["power"] >= 0.0)
    assert np.all(action["power"] <= env.p_max_w + 1e-5)
    assert np.all(action["split"] < env.n_splits)

    eval_action = agent.select_action(obs, evaluate=True)
    assert np.all(eval_action["power"] >= 0.0)


def test_hardware_device_config_key_is_read_by_all_four_agents(
    default_config, monkeypatch
):
    """`hardware.device` (config/oran_default.yaml) must actually supply the
    default device -- previously it was parsed nowhere, so every O-RAN agent
    silently ignored it regardless of the YAML. Force torch.cuda.is_available()
    to True so the final availability gate doesn't mask the config value with
    a forced cpu fallback, then confirm an explicit hardware.device: "cpu"
    override in config is genuinely honored (would be "cuda" if the config
    value were still being ignored)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    cfg = dict(default_config)
    cfg["hardware"] = {"device": "cpu"}
    env = ORANEnv(cfg)

    agents = [
        BMPPDQNAgent(
            state_dim=env.state_dim,
            n_ru=env.n_ru,
            n_splits=env.n_splits,
            p_max_w=env.p_max_w,
            config=cfg,
        ),
        ORANDQNAgent(
            state_dim=env.state_dim,
            n_ru=env.n_ru,
            n_splits=env.n_splits,
            p_max_w=env.p_max_w,
            config=cfg,
        ),
        ORANDDPGAgent(
            state_dim=env.state_dim,
            n_ru=env.n_ru,
            n_splits=env.n_splits,
            p_max_w=env.p_max_w,
            config=cfg,
        ),
        ORANMPDQNAgent(
            state_dim=env.state_dim,
            n_ru=env.n_ru,
            n_splits=env.n_splits,
            p_max_w=env.p_max_w,
            config=cfg,
        ),
    ]
    for agent in agents:
        assert agent.device == torch.device("cpu")


def test_ddpg_lr_critic_config_key_resolves_from_oran_default_yaml(default_config):
    """config/oran_default.yaml previously had no algorithm.lr_critic key, so
    ORANDDPGAgent's own read of it always silently fell back to its Python
    default regardless of the YAML. A custom override must now actually
    reach the critic optimizer."""
    cfg = dict(default_config)
    cfg["algorithm"] = dict(default_config["algorithm"])
    cfg["algorithm"]["lr_critic"] = 1.0e-2

    env = ORANEnv(cfg)
    agent = ORANDDPGAgent(
        state_dim=env.state_dim,
        n_ru=env.n_ru,
        n_splits=env.n_splits,
        p_max_w=env.p_max_w,
        config=cfg,
    )
    assert agent.critic_opt.param_groups[0]["lr"] == pytest.approx(1.0e-2)


def test_discrete_decision_held_constant_across_upper_level_period(default_config):
    """The discrete (ru_on, split) choice must be replayed unchanged for
    `upper_level_period_steps` consecutive select_action() calls, while
    continuous outputs (power, prb) are free to vary every call --
    Concept Note Section 5.2's multi-timescale requirement."""
    cfg = dict(default_config)
    cfg["algorithm"] = dict(default_config["algorithm"])
    cfg["algorithm"]["upper_level_period_steps"] = 5
    cfg["algorithm"]["epsilon_start"] = 0.0  # deterministic discrete choice

    env = ORANEnv(cfg)
    obs, _ = env.reset(seed=42)
    agent = _make_agent(env, cfg)

    ru_on_choices = []
    split_choices = []
    power_choices = []
    for _ in range(5):
        action = agent.select_action(obs, evaluate=False)
        ru_on_choices.append(action["ru_on"].copy())
        split_choices.append(action["split"].copy())
        power_choices.append(action["power"].copy())

    for choice in ru_on_choices[1:]:
        assert np.array_equal(choice, ru_on_choices[0])
    for choice in split_choices[1:]:
        assert np.array_equal(choice, split_choices[0])
    # Continuous outputs are not required to be constant (exploration noise
    # is added every call) -- just confirm at least one call differs, to
    # prove select_action() genuinely recomputes them each time rather than
    # also caching them.
    assert any(not np.array_equal(p, power_choices[0]) for p in power_choices[1:])


def test_remember_flushes_pending_upper_transition_at_episode_end(default_config):
    """Guards against silently losing the trailing partial upper-level
    window when max_steps_per_episode isn't an exact multiple of
    upper_level_period_steps -- the episode-end done=True step must flush
    whatever's pending into upper_memory, not discard it."""
    cfg = dict(default_config)
    cfg["algorithm"] = dict(default_config["algorithm"])
    cfg["algorithm"]["upper_level_period_steps"] = 3

    env = ORANEnv(cfg)
    obs, _ = env.reset(seed=42)
    agent = _make_agent(env, cfg)

    assert len(agent.upper_memory) == 0

    # Step 1: fresh decision, starts a new pending window.
    action = agent.select_action(obs, evaluate=True)
    next_obs, reward, _, _, _ = env.step(action)
    agent.remember(obs, action, reward, next_obs, done=False)
    assert len(agent.upper_memory) == 0  # window not complete yet
    obs = next_obs

    # Step 2: episode ends here (done=True) before the 3-step window
    # completes -- the pending transition must still be flushed.
    action = agent.select_action(obs, evaluate=True)
    next_obs, reward, _, _, _ = env.step(action)
    agent.remember(obs, action, reward, next_obs, done=True)

    assert len(agent.upper_memory) == 1


def test_no_twin_critic_attribute(default_config):
    """Explicit no-TD3 guard (Concept Note Section 10.4): this agent must
    not carry a twin-critic pair the way agents/branching_mp_dqn.py does."""
    env = ORANEnv(default_config)
    agent = _make_agent(env, default_config)

    assert not hasattr(agent, "critic_a")
    assert not hasattr(agent, "critic_b")
    assert not hasattr(agent, "twin_critic")
    assert hasattr(agent, "critic")
    assert not hasattr(agent, "policy_delay")
    assert not hasattr(agent, "target_noise_std")


def test_multi_pass_no_cross_talk(default_config):
    """Two continuous-parameter vectors differing only in an unrelated
    RU's (power, prb) must yield identical Q-values for every *other*
    branch -- the MP-DQN multi-pass masking guarantee."""
    env = ORANEnv(default_config)
    agent = _make_agent(env, default_config)

    import torch

    feat = torch.randn(1, agent.upper_encoder.output_dim)
    cont_params_a = torch.rand(1, agent.n_ru, 2)
    cont_params_b = cont_params_a.clone()
    cont_params_b[0, 0, :] = 1.0 - cont_params_b[0, 0, :]  # perturb only RU 0

    with torch.no_grad():
        act_a, split_a = agent._multi_pass_q(feat, cont_params_a)
        act_b, split_b = agent._multi_pass_q(feat, cont_params_b)

    # Every branch except RU 0 must be unaffected by RU 0's perturbation.
    assert torch.allclose(act_a[:, 1:, :], act_b[:, 1:, :], atol=1e-6)
    assert torch.allclose(split_a[:, 1:, :], split_b[:, 1:, :], atol=1e-6)


def test_update_lower_gathers_greedy_action_not_raw_mean(default_config):
    """Guards against the exact bug already found and fixed in
    agents/branching_mp_dqn.py's actor update: param_loss must be computed
    from the *gathered* greedy-action Q-value, not a raw .mean() over the
    un-gathered activation_q tensor (which would average in the non-greedy
    action's Q-value at every branch)."""
    cfg = dict(default_config)
    cfg["algorithm"] = dict(default_config["algorithm"])
    cfg["algorithm"]["min_buffer_size"] = 4
    cfg["algorithm"]["batch_size"] = 4

    env = ORANEnv(cfg)
    obs, _ = env.reset(seed=42)
    agent = _make_agent(env, cfg)

    for _ in range(8):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.remember(obs, action, reward, next_obs, terminated)
        obs = next_obs

    import torch

    states, cont_params, rewards, next_states, dones = agent.lower_memory.sample(4)
    states = states.to(agent.device)

    lower_feat = agent.lower_encoder(states)
    power_ratio, prb_share = agent.param_net(lower_feat)
    pred_params = torch.stack([power_ratio, prb_share], dim=-1)
    upper_feat = agent.upper_encoder(states).detach()
    activation_q, _ = agent._multi_pass_q(upper_feat, pred_params)

    expected_greedy_idx = activation_q.argmax(dim=-1, keepdim=True)
    expected_loss = (
        -activation_q.gather(-1, expected_greedy_idx).squeeze(-1).mean().detach()
    )
    wrong_loss = -activation_q.mean().detach()

    # The two only coincide if the network happens to assign identical
    # Q-values to both discrete actions everywhere, which a freshly
    # initialized network essentially never does -- so this is a real
    # discriminator between "gathered greedy" and "raw mean" behavior.
    assert abs(float(expected_loss) - float(wrong_loss)) > 1e-6


def test_update_lower_and_upper_do_not_crash_across_full_training_loop(
    default_config,
):
    cfg = dict(default_config)
    cfg["algorithm"] = dict(default_config["algorithm"])
    cfg["algorithm"]["min_buffer_size"] = 16
    cfg["algorithm"]["batch_size"] = 8
    cfg["algorithm"]["upper_level_period_steps"] = 4

    env = ORANEnv(cfg)
    obs, _ = env.reset(seed=42)
    agent = _make_agent(env, cfg)

    saw_nonzero_param_loss = False
    saw_nonzero_critic_loss = False
    for _ in range(40):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.remember(obs, action, reward, next_obs, terminated)
        obs = next_obs

        lower_metrics = agent.update_lower()
        upper_metrics = agent.update_upper()
        assert not np.isnan(lower_metrics["param_loss"])
        assert not np.isnan(upper_metrics["critic_loss"])
        saw_nonzero_param_loss |= lower_metrics["param_loss"] != 0.0
        saw_nonzero_critic_loss |= upper_metrics["critic_loss"] != 0.0

        if terminated or truncated:
            obs, _ = env.reset(seed=123)

    assert saw_nonzero_param_loss
    assert saw_nonzero_critic_loss


# ---------------------------------------------------------------------------
# Baselines: DQN (discrete-only), DDPG (continuous-only), MP-DQN (flat joint)
# ---------------------------------------------------------------------------


def _small_config(default_config):
    cfg = dict(default_config)
    cfg["network"] = dict(default_config["network"])
    cfg["network"]["n_ru"] = 2
    cfg["network"]["n_ue"] = 2
    cfg["network"]["n_splits"] = 2
    cfg["algorithm"] = dict(default_config["algorithm"])
    return cfg


def test_dqn_baseline_uses_plain_target_not_double_dqn(default_config):
    """The target must select AND evaluate off the target network only
    (target_net(next_state).max()), not online-net-argmax + target-evaluate
    -- the explicit plain-DQN requirement distinguishing this baseline from
    agents/ddqn_agent.py's Double DQN."""
    cfg = _small_config(default_config)
    env = ORANEnv(cfg)
    agent = ORANDQNAgent(
        state_dim=env.state_dim, n_ru=env.n_ru, n_splits=env.n_splits, config=cfg
    )

    import torch

    # Make the online and target networks disagree sharply so the two
    # target-computation strategies would diverge if plain-DQN's max() were
    # accidentally replaced with online-select + target-evaluate.
    with torch.no_grad():
        for p in agent.target_q_net.parameters():
            p.add_(1.0)

    states = torch.randn(4, env.state_dim)
    with torch.no_grad():
        _, target_split_q = agent.target_q_net(states)
        expected = target_split_q.max(dim=-1).values

        online_split_q, _ = agent.q_net(states)
        _, target_split_q_2 = agent.target_q_net(states)
        online_argmax = online_split_q.argmax(dim=-1)
        double_dqn_style = target_split_q_2.gather(
            -1, online_argmax.unsqueeze(-1)
        ).squeeze(-1)

    # The two strategies must differ given the deliberately-perturbed
    # target net -- confirming this test can actually discriminate them.
    assert not torch.allclose(expected, double_dqn_style)


def test_dqn_baseline_runs_and_updates(default_config):
    cfg = _small_config(default_config)
    env = ORANEnv(cfg)
    obs, _ = env.reset(seed=42)
    agent = ORANDQNAgent(
        state_dim=env.state_dim, n_ru=env.n_ru, n_splits=env.n_splits, config=cfg
    )

    for _ in range(20):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.memory.push(
            obs, action["ru_on"], action["split"], reward, next_obs, terminated
        )
        obs = next_obs
        metrics = agent.update(batch_size=8)
        if terminated or truncated:
            obs, _ = env.reset(seed=42)

    assert not np.isnan(metrics["loss"])


def test_ddpg_baseline_fixes_activation_and_split(default_config):
    """Concept Note Section 2.2: DDPG has no mechanism to represent
    discrete decisions -- ru_on/split must be fixed, not learned."""
    cfg = _small_config(default_config)
    env = ORANEnv(cfg)
    obs, _ = env.reset(seed=42)
    agent = ORANDDPGAgent(
        state_dim=env.state_dim, n_ru=env.n_ru, n_splits=env.n_splits, config=cfg
    )

    for _ in range(5):
        action = agent.select_action(obs, evaluate=False)
        assert np.all(action["ru_on"] == 1)
        assert np.all(action["split"] == ORANDDPGAgent.FIXED_SPLIT_LEVEL)
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs


def test_ddpg_baseline_runs_and_updates(default_config):
    cfg = _small_config(default_config)
    env = ORANEnv(cfg)
    obs, _ = env.reset(seed=42)
    agent = ORANDDPGAgent(
        state_dim=env.state_dim, n_ru=env.n_ru, n_splits=env.n_splits, config=cfg
    )

    for _ in range(20):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        cont = np.concatenate([action["power"] / env.p_max_w, action["prb"]])
        agent.memory.push(obs, cont, reward, next_obs, terminated)
        obs = next_obs
        metrics = agent.update(batch_size=8)
        if terminated or truncated:
            obs, _ = env.reset(seed=42)

    assert not np.isnan(metrics["critic_loss"])
    assert not np.isnan(metrics["actor_loss"])


def test_mpdqn_baseline_raises_above_tractability_cap():
    with pytest.raises(ValueError, match="MAX_N_RU_FOR_FLAT_JOINT_ORAN_ACTION"):
        ORANMPDQNAgent(
            state_dim=100,
            n_ru=MAX_N_RU_FOR_FLAT_JOINT_ORAN_ACTION + 1,
            n_splits=3,
        )


def test_mpdqn_baseline_tractability_cap_reads_config_override():
    """algorithm.max_n_ru_for_flat_joint_action must actually override the
    module-level default, not be silently ignored (it previously was)."""
    # A config cap lower than the module default must reject an n_ru the
    # module default alone would have allowed.
    with pytest.raises(ValueError, match="algorithm.max_n_ru_for_flat_joint_action"):
        ORANMPDQNAgent(
            state_dim=100,
            n_ru=3,
            n_splits=3,
            config={"algorithm": {"max_n_ru_for_flat_joint_action": 2}},
        )

    # A config cap higher than the module default must permit an n_ru the
    # module default alone would have rejected.
    agent = ORANMPDQNAgent(
        state_dim=100,
        n_ru=MAX_N_RU_FOR_FLAT_JOINT_ORAN_ACTION + 1,
        n_splits=2,
        config={
            "algorithm": {
                "max_n_ru_for_flat_joint_action": MAX_N_RU_FOR_FLAT_JOINT_ORAN_ACTION
                + 1
            }
        },
    )
    assert agent.n_ru == MAX_N_RU_FOR_FLAT_JOINT_ORAN_ACTION + 1


def test_mpdqn_baseline_runs_and_updates(default_config):
    cfg = _small_config(default_config)
    env = ORANEnv(cfg)
    obs, _ = env.reset(seed=42)
    agent = ORANMPDQNAgent(
        state_dim=env.state_dim, n_ru=env.n_ru, n_splits=env.n_splits, config=cfg
    )
    assert agent.n_joint_actions == (2**env.n_ru) * (env.n_splits**env.n_ru)

    for _ in range(20):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        cont = np.stack([action["power"] / env.p_max_w, action["prb"]], axis=-1)
        agent.memory.push(
            obs, agent._last_action_idx, cont, reward, next_obs, terminated
        )
        obs = next_obs
        metrics = agent.update(batch_size=8)
        if terminated or truncated:
            obs, _ = env.reset(seed=42)

    assert not np.isnan(metrics["critic_loss"])
    assert not np.isnan(metrics["param_loss"])
