"""Equation-code consistency tests, per docs/equation_code_mapping.md's Rule 1."""

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]
from pathlib import Path

from cran_env import CRANEnv


@pytest.fixture
def default_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


class TestEquationConsistency:
    def test_power_model_matches_equation_3_5(self, default_config):
        """Verify total power matches Eq. (3.5): P_total = P_RRH + P_BBU + P_FH (+ switching)."""
        env = CRANEnv(default_config)
        env.reset(seed=42)

        action = {
            "rrh_on": np.ones(env.n_rrh, dtype=int),
            "power": np.ones(env.n_rrh, dtype=np.float32) * (env.p_max_w / 2.0),
        }
        _, _, _, _, info = env.step(action)

        expected_total = (
            info["rrh_power_w"]
            + info["bbu_power_w"]
            + info["fronthaul_power_w"]
            + info["switching_power_w"]
        )
        assert info["total_power_w"] == pytest.approx(expected_total, rel=1e-5)
