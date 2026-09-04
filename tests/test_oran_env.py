"""Unit tests for the O-RAN Gymnasium Environment (oran_env/).

A fully separate test module from tests/test_env.py -- no shared fixtures
or imports with the C-RAN environment tests.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]
from gymnasium.utils.env_checker import check_env

from oran_env import ORANChannelModel, ORANEnv, ORANPowerModel, ORANTrafficModel


@pytest.fixture
def default_config():
    config_path = Path(__file__).parent.parent / "config" / "oran_default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_gymnasium_compliance(default_config):
    env = ORANEnv(default_config)
    check_env(env.unwrapped, skip_render_check=True)


def test_observation_and_action_shapes(default_config):
    env = ORANEnv(default_config)
    obs, info = env.reset(seed=42)

    expected_dim = (
        env.n_ru * env.n_ue + env.n_ru + env.n_ru * env.n_splits + env.n_ue + 4
    )
    assert env.state_dim == expected_dim
    assert obs.shape == (expected_dim,)

    action = env.action_space.sample()
    assert set(action.keys()) == {"ru_on", "split", "power", "prb"}
    assert action["ru_on"].shape == (env.n_ru,)
    assert action["split"].shape == (env.n_ru,)
    assert action["power"].shape == (env.n_ru,)
    assert action["prb"].shape == (env.n_ru,)

    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert next_obs.shape == (expected_dim,)
    assert isinstance(reward, float)


def test_env_accepts_full_action_dict_regardless_of_upper_level_period(
    default_config,
):
    """The environment must stay timescale-agnostic: step() always accepts
    the full 4-key action dict every call, with no cadence-aware gating of
    its own -- the two-timescale behavior belongs entirely to the agent
    (docs/skills/skill_oran_env.md Rule 4). This guards against that
    responsibility being accidentally pulled into the env later."""
    cfg = deepcopy(default_config)
    cfg["algorithm"]["upper_level_period_steps"] = 5
    env = ORANEnv(cfg)
    env.reset(seed=42)

    for _ in range(12):  # spans more than one upper-level period
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (env.state_dim,)


def test_rolling_window_propagates_lower_level_metrics_into_state(default_config):
    """Concept Note Section 5.2: lower-level throughput/power metrics must
    feed into the state the upper-level decision sees -- realized here as
    the observation's trailing two scalars (rolling-window means)."""
    cfg = deepcopy(default_config)
    cfg["algorithm"]["upper_level_period_steps"] = 4
    env = ORANEnv(cfg)
    obs, _ = env.reset(seed=42)

    # Before any steps, the rolling window is empty -> both propagated
    # fields are exactly zero.
    assert obs[-1] == 0.0
    assert obs[-2] == 0.0

    action = {
        "ru_on": np.ones(env.n_ru, dtype=int),
        "split": np.zeros(env.n_ru, dtype=int),
        "power": np.full(env.n_ru, env.p_max_w / 2.0, dtype=np.float32),
        "prb": np.ones(env.n_ru, dtype=np.float32),
    }
    obs, _, _, _, info = env.step(action)
    # After one step with active RUs, both rolling means should reflect
    # that step's throughput/power (window has exactly one sample so far).
    assert obs[-2] == pytest.approx(info["throughput_mbps"], rel=1e-4)
    assert obs[-1] == pytest.approx(info["total_power_w"] / 1000.0, rel=1e-4)


def test_demand_sampled_once_per_hour_not_twice(default_config, monkeypatch):
    """Guards against the demand-double-sampling bug already found and
    fixed in cran_env/cran_env.py: get_demands() must be called exactly
    once per step (for the *next* observation), not once for the
    observation and again independently for the reward's QoS calc."""
    env = ORANEnv(default_config)
    env.reset(seed=42)

    call_count = {"n": 0}
    original = env.traffic.get_demands

    def counting_get_demands(hour, rng):
        call_count["n"] += 1
        return original(hour, rng)

    monkeypatch.setattr(env.traffic, "get_demands", counting_get_demands)

    action = env.action_space.sample()
    env.step(action)
    assert call_count["n"] == 1


def test_power_model_monotonic_in_split_centralization_level():
    """docs/skills/skill_oran_env.md's monotonicity guarantee: RU power
    must strictly decrease, and DU + fronthaul power must strictly
    increase, as the split centralization level c increases from Option 2
    (c=0) to Option 8 (c=2) -- Concept Note Section 10.2/10.5."""
    pm = ORANPowerModel(n_ru=2)
    active = np.array([True, True])
    tx = np.array([0.0, 0.0])

    ru_powers, du_powers, fh_powers = [], [], []
    for c in range(pm.n_splits):
        split = np.full(2, c, dtype=np.int64)
        ru_powers.append(pm.compute_ru_power(active, split, tx))
        du_powers.append(pm.compute_du_power(active, split))
        fh_powers.append(pm.compute_fronthaul_power(active, split))

    assert ru_powers == sorted(ru_powers, reverse=True)
    assert du_powers == sorted(du_powers)
    assert fh_powers == sorted(fh_powers)


def test_power_model_per_split_arrays_read_from_config(default_config):
    """oran_env.py never passed power.ru.p_proc_by_split_w /
    power.du.p_per_ru_by_split_w / power.fronthaul.p_per_ru_by_split_w to
    ORANPowerModel -- the per-split-level arrays that actually encode the
    Section 10.2 monotonic centralization trade-off always silently used
    ORANPowerModel's own Python-side defaults regardless of config,
    the exact "dead config key" bug class this codebase has repeatedly
    found and fixed elsewhere. A config override must now actually reach
    ORANPowerModel."""
    cfg = deepcopy(default_config)
    cfg["power"]["ru"]["p_proc_by_split_w"] = [111.0, 22.0, 3.0]
    cfg["power"]["du"]["p_per_ru_by_split_w"] = [4.0, 5.0, 6.0]
    cfg["power"]["fronthaul"]["p_per_ru_by_split_w"] = [7.0, 8.0, 9.0]

    env = ORANEnv(cfg)

    assert list(env.power.p_ru_proc_by_split) == [111.0, 22.0, 3.0]
    assert list(env.power.p_du_per_ru_by_split) == [4.0, 5.0, 6.0]
    assert list(env.power.p_fh_per_ru_by_split) == [7.0, 8.0, 9.0]


def test_power_model_switching_cost_counts_flips_and_split_changes():
    pm = ORANPowerModel(n_ru=3)
    prev_active = np.array([True, True, False])
    curr_active = np.array([True, False, True])  # RU1 off, RU2 on: 2 flips
    prev_split = np.array([0, 1, 0])
    curr_split = np.array([1, 1, 0])  # RU0's split changed while active in both

    cost = pm.compute_switching_cost(prev_active, curr_active, prev_split, curr_split)
    expected = 2 * pm.p_switch_ru_w + 1 * pm.p_switch_split_w
    assert cost == pytest.approx(expected)


def test_traffic_model_defaults_match_3gpp_ftp_model_3(default_config):
    """lambda_peak/packet_size_bits are derived from 3GPP TR 38.864 Annex A's
    FTP Model 3 (0.5 MB packet size, 200 ms mean inter-arrival time), not a
    guess -- see oran_env/traffic_model.py's docstring."""
    tm = ORANTrafficModel(n_ue=1)
    assert tm.lambda_peak == pytest.approx(0.5)
    assert tm.packet_size_bits == pytest.approx(4.0e6)

    traffic_cfg = default_config["traffic"]
    assert traffic_cfg["lambda_peak"] == pytest.approx(0.5)
    assert traffic_cfg["packet_size_bits"] == pytest.approx(4.0e6)


def test_traffic_model_trapezoidal_shape():
    tm = ORANTrafficModel(
        n_ue=1, lambda_peak=5.0, floor_ratio=0.2, t1=7, t2=10, t3=20, t4=23
    )

    assert tm._envelope(0.0) == pytest.approx(1.0)
    assert tm._envelope(7.0) == pytest.approx(1.0)
    assert tm._envelope(10.0) == pytest.approx(5.0)
    assert tm._envelope(15.0) == pytest.approx(5.0)
    assert tm._envelope(23.0) == pytest.approx(1.0)
    # Midway through the rise: halfway between floor and peak
    assert tm._envelope(8.5) == pytest.approx(3.0)


def test_channel_model_generates_complex_gains():
    cm = ORANChannelModel(n_ru=3, n_ue=2)
    rng = np.random.default_rng(0)
    distances = np.full((3, 2), 100.0)
    gains = cm.generate_channel(distances, rng)
    assert gains.shape == (3, 2)
    assert np.iscomplexobj(gains)
    assert np.all(np.abs(gains) > 0.0)
