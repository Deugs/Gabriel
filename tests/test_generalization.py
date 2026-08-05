"""Unit tests for the cross-traffic-profile generalization evaluation
(evaluation/generalization.py)."""

from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from agents import BranchingMPDQN
from cran_env import CRANEnv
from evaluation.generalization import evaluate_generalization


@pytest.fixture
def small_config():
    config_path = Path(__file__).parent.parent / "config" / "small_network.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def test_evaluate_generalization_shape(small_config):
    env = CRANEnv(small_config)
    agent = BranchingMPDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=small_config,
    )

    result = evaluate_generalization(agent, small_config, episodes=2, seed=42)

    assert set(result.keys()) == {"matched", "generalized", "degradation"}
    for key in ("matched", "generalized"):
        for value in result[key].values():
            assert np.isfinite(value)
    for value in result["degradation"].values():
        assert np.isfinite(value)

    # The base config itself must be untouched (in-memory override only).
    assert "traffic" not in small_config or "profile" not in small_config.get(
        "traffic", {}
    )
