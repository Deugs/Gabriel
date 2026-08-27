"""Unit tests for Concept Note v2.0 Baselines (ANNGSBFBaseline and DDQNSOCPBaseline)."""

from pathlib import Path
import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from baselines import ANNGSBFBaseline, DDQNSOCPBaseline
from cran_env import CRANEnv


@pytest.fixture
def default_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_ann_gsbf_baseline(default_config):
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    model = ANNGSBFBaseline(env.n_rrh, env.n_ue, env.p_max_w)
    action = model.select_action(obs)

    assert "rrh_on" in action and "power" in action and "bandwidth" in action
    assert action["rrh_on"].shape == (env.n_rrh,)
    assert action["power"].shape == (env.n_rrh,)
    assert np.all(action["power"] >= 0.0)


def test_ddqn_socp_baseline(default_config):
    env = CRANEnv(default_config)
    obs, _ = env.reset(seed=42)

    model = DDQNSOCPBaseline(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        n_ue=env.n_ue,
        p_max_w=env.p_max_w,
        config=default_config,
    )

    action = model.select_action(obs, evaluate=True)
    assert "rrh_on" in action and "power" in action
    assert action["rrh_on"].shape == (env.n_rrh,)
    assert action["power"].shape == (env.n_rrh,)
    assert np.all(action["power"] >= 0.0)


def test_ddqn_socp_defaults_to_nominal_csi_matching_convex_baseline(default_config):
    """The main benchmark's "identical simulation conditions" framing
    (Concept Note v4.0 Section 12.2) requires nominal (perfect) CSI by
    default, matching sibling ConvexPowerBaseline's csi_uncertainty=0.0
    default -- not a baked-in robust-SOCP assumption no other baseline
    shares."""
    model = DDQNSOCPBaseline(
        state_dim=10,
        n_rrh=5,
        n_ue=2,
        p_max_w=1.0,
    )
    assert model.convex_solver.csi_uncertainty == 0.0
