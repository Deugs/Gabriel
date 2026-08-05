"""Unit tests for evaluation/inference_latency.py."""

from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from agents import BranchingMPDQN
from cran_env import CRANEnv
from evaluation.inference_latency import benchmark_inference_latency


@pytest.fixture
def small_config():
    config_path = Path(__file__).parent.parent / "config" / "small_network.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def agent(small_config):
    env = CRANEnv(small_config)
    return BranchingMPDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=small_config,
    )


def test_benchmark_inference_latency_positive_finite(agent, small_config):
    result = benchmark_inference_latency(
        agent, small_config, n_warmup=2, n_trials=5
    )
    assert result["mean_latency_ms"] > 0.0
    assert np.isfinite(result["mean_latency_ms"])
    assert result["n_trials"] == 5


def test_benchmark_inference_latency_excludes_training_cost(agent, small_config):
    benchmark_inference_latency(agent, small_config, n_warmup=2, n_trials=5)
    # select_action alone never touches the replay buffer or calls update() --
    # proves no backprop path was exercised by the benchmark.
    assert len(agent.memory) == 0
