"""Unit tests for the second traffic profile (cran_env/traffic_model.py) and
its wiring through CRANEnv."""

from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from cran_env import CRANEnv, TrafficModel


def _mean_demand_per_hour(model: TrafficModel, seed: int = 0):
    rng = np.random.default_rng(seed)
    return np.array([np.mean(model.get_demands(h, rng)) for h in range(24)])


def test_default_profile_matches_weekday_urban():
    default_model = TrafficModel(n_ue=10)
    explicit_model = TrafficModel(n_ue=10, profile="weekday_urban")

    default_demands = _mean_demand_per_hour(default_model)
    explicit_demands = _mean_demand_per_hour(explicit_model)

    assert np.allclose(default_demands, explicit_demands)
    assert default_model.profile == "weekday_urban"


def test_weekend_suburban_differs_from_weekday_urban():
    weekday = _mean_demand_per_hour(TrafficModel(n_ue=20, profile="weekday_urban"))
    weekend = _mean_demand_per_hour(TrafficModel(n_ue=20, profile="weekend_suburban"))

    weekday_peak_hour = int(np.argmax(weekday))
    weekend_peak_hour = int(np.argmax(weekend))
    assert weekday_peak_hour != weekend_peak_hour
    assert weekday_peak_hour in (10, 11, 12)
    assert weekend_peak_hour in (20, 21, 22, 23)

    weekday_ratio = np.max(weekday) / np.min(weekday)
    weekend_ratio = np.max(weekend) / np.min(weekend)
    assert weekday_ratio > weekend_ratio  # weekday is more sharply peaked


def test_invalid_profile_raises():
    with pytest.raises(ValueError):
        TrafficModel(n_ue=10, profile="bogus")


def test_cran_env_wires_traffic_profile():
    config_path = Path(__file__).parent.parent / "config" / "small_network.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("traffic", {})
    cfg["traffic"]["profile"] = "weekend_suburban"

    env = CRANEnv(cfg)
    assert env.traffic.profile == "weekend_suburban"

    # Default (no traffic.profile key at all) still falls back to weekday_urban.
    default_env = CRANEnv({k: v for k, v in cfg.items() if k != "traffic"})
    assert default_env.traffic.profile == "weekday_urban"
