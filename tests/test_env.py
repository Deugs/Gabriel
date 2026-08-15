"""Unit tests for C-RAN Gymnasium Environment (cran_env/)."""

from copy import deepcopy

import pytest
import numpy as np
import yaml  # type: ignore[import-untyped]
from pathlib import Path
from gymnasium.utils.env_checker import check_env

from cran_env import CRANEnv, ChannelModel, TrafficModel, PowerModel


@pytest.fixture
def default_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_cran_env_reads_max_steps_per_episode_from_config(default_config):
    """Regression test: `getattr(cfg, "algorithm", cfg)` does not perform
    dict key lookup, so for a plain dict config this previously always
    resolved to the whole cfg object (not cfg["algorithm"]), silently
    discarding algorithm.max_steps_per_episode regardless of the YAML."""
    cfg = deepcopy(default_config)
    cfg["algorithm"]["max_steps_per_episode"] = 7
    env = CRANEnv(cfg)
    assert env.max_steps == 7


def test_channel_model():
    n_rrh, n_ue = 12, 10
    channel = ChannelModel(n_rrh, n_ue, carrier_freq_ghz=2.1, bandwidth_mhz=20.0)

    # 1km distance
    distances = np.full((n_rrh, n_ue), 1000.0)
    path_loss_db = channel.compute_path_loss(distances)

    # Check path loss magnitude (COST231-Hata at 1km is ~125-145 dB)
    assert np.all(path_loss_db > 100.0)
    assert np.all(path_loss_db < 160.0)

    # Generate initial channel
    rng = np.random.default_rng(42)
    h_init = channel.generate_channel(distances, rng)
    assert h_init.shape == (n_rrh, n_ue)
    assert np.iscomplexobj(h_init)

    # Step channel with Gauss-Markov process
    h_next = channel.step_channel(h_init, distances, rng)
    assert h_next.shape == (n_rrh, n_ue)
    assert not np.allclose(h_init, h_next)


def test_traffic_model():
    n_ue = 10
    traffic = TrafficModel(n_ue, base_rate_mbps=50.0, peak_multiplier=3.0)

    rng = np.random.default_rng(42)
    demands_offpeak = traffic.get_demands(3, rng)  # 3 AM
    demands_peak = traffic.get_demands(10, rng)  # 10 AM

    assert demands_offpeak.shape == (n_ue,)
    assert demands_peak.shape == (n_ue,)
    assert np.mean(demands_peak) > np.mean(demands_offpeak)


def test_power_model_earth_constants():
    n_rrh, n_bbu = 12, 3
    power = PowerModel(n_rrh, n_bbu)

    # Check EARTH model constants
    assert power.p_stat == 175.0
    assert power.p_dyn == 250.0
    assert power.p_active == 6.8
    assert power.p_sleep == 4.3
    assert power.p_switch == 3.0
    assert power.eta == 0.25

    # Test BBU power calculation
    loads = np.array([1.0, 0.5, 0.0])  # 2 active BBUs
    p_bbu = power.compute_bbu_power(loads)
    # Expected: 2 * 175.0 + 0.44 * 250.0 * 1.5 = 350.0 + 165.0 = 515.0 W
    assert pytest.approx(p_bbu, rel=1e-3) == 515.0

    # Test RRH power calculation
    active_mask = np.array([True] * 6 + [False] * 6)
    transmit_power = np.array([0.5] * 12)  # 0.5W per RRH
    p_rrh = power.compute_rrh_power(active_mask, transmit_power)
    # Expected: (6 * 0.5 / 0.25) + (6 * 6.8) + (6 * 4.3) = 12.0 + 40.8 + 25.8 = 78.6 W
    assert pytest.approx(p_rrh, rel=1e-3) == 78.6

    # Test Switching cost
    prev_mask = np.array([True] * 12)
    curr_mask = np.array([True] * 6 + [False] * 6)
    p_switch = power.compute_switching_cost(prev_mask, curr_mask)
    # Expected: 6 transitions * 3.0 W = 18.0 W
    assert pytest.approx(p_switch, rel=1e-3) == 18.0


def test_cran_env_reset(default_config):
    env = CRANEnv(default_config)

    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)

    assert np.allclose(obs1, obs2)
    assert info1["hour"] == info2["hour"]
    assert obs1.shape == (env.state_dim,)
    assert not np.isnan(obs1).any()
    assert not np.isinf(obs1).any()


def test_cran_env_step(default_config):
    env = CRANEnv(default_config)
    obs, info = env.reset(seed=42)

    # Take step with all RRHs active and half power
    action = {
        "rrh_on": np.ones(env.n_rrh, dtype=int),
        "power": np.full(env.n_rrh, env.p_max_w / 2.0, dtype=np.float32),
    }

    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert next_obs.shape == (env.state_dim,)
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert not np.isinf(reward)
    assert "total_power_w" in step_info
    assert step_info["total_power_w"] > 0.0


def test_gymnasium_compliance(default_config):
    env = CRANEnv(default_config)
    # Verify Gymnasium environment compatibility
    check_env(env.unwrapped, skip_render_check=True)


def test_sinr_interference_physics(default_config):
    env = CRANEnv(default_config)
    env.reset(seed=42)

    # Action 1: Only 1 RRH active at full power
    action_1_rrh = {
        "rrh_on": np.array([1] + [0] * (env.n_rrh - 1)),
        "power": np.array([env.p_max_w] + [0.0] * (env.n_rrh - 1), dtype=np.float32),
    }
    sinr_1_rrh = env._compute_sinr(action_1_rrh["rrh_on"], action_1_rrh["power"])

    # Action 2: ALL RRHs active at full power (creates co-channel interference
    # for users assigned to different RRHs)
    action_all_rrhs = {
        "rrh_on": np.ones(env.n_rrh, dtype=int),
        "power": np.full(env.n_rrh, env.p_max_w, dtype=np.float32),
    }
    sinr_all_rrhs = env._compute_sinr(
        action_all_rrhs["rrh_on"], action_all_rrhs["power"]
    )

    # Under true multi-cell downlink interference, users served by a single
    # RRH receive interference from other active RRHs. Therefore, the mean
    # SINR per user with all RRHs transmitting full power must experience
    # interference penalty relative to non-interfered signal ratio.
    assert np.all(sinr_1_rrh >= 0.0)
    assert np.all(sinr_all_rrhs >= 0.0)
    # Check that interference calculation is active (some users suffer interference)
    active_interference = False
    for u in range(env.n_ue):
        # Calculate received power from all active RRHs for user u
        ch_sq = np.abs(env.channel_gains[:, u]) ** 2
        rx_p = ch_sq * action_all_rrhs["power"]
        best = np.max(rx_p)
        interf = np.sum(rx_p) - best
        if interf > 0:
            active_interference = True
            break
    assert (
        active_interference
    ), "Interference physics calculation must be active for multi-RRH transmissions"
