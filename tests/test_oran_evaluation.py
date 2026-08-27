"""Unit tests for O-RAN Evaluation Modules (oran_evaluation/).

A fully separate test module from tests/test_evaluation.py -- no shared
fixtures or imports with the C-RAN evaluation tests.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from oran_evaluation import PROPOSED_ALGO, analyze_convergence, run_latency_benchmark
from oran_evaluation.plot_utils import compute_confidence_interval


@pytest.fixture
def default_config():
    config_path = Path(__file__).parent.parent / "config" / "oran_default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["network"]["n_ru"] = 2
    cfg["network"]["n_ue"] = 2
    cfg["algorithm"]["min_buffer_size"] = 8
    cfg["algorithm"]["batch_size"] = 4
    cfg["algorithm"]["max_steps_per_episode"] = 5
    return cfg


def test_compute_confidence_interval_not_degenerate_for_real_seed_spread():
    """Guards against the exact bug already found and fixed in
    evaluation/plot_utils.py: a real 1-D per-seed array must not collapse
    to a zero-width interval."""
    data = np.array([10.0, 12.0, 8.0, 15.0, 9.0, 11.0, 13.0, 7.0, 14.0, 10.0])
    mean, lower, upper = compute_confidence_interval(data)
    assert lower < mean < upper

    single = np.array([5.0])
    mean_s, lower_s, upper_s = compute_confidence_interval(single)
    assert mean_s == lower_s == upper_s == 5.0


def test_analyze_convergence_pairs_by_seed(tmp_path, default_config):
    """analyze_convergence() must pair proposed-vs-baseline by seed, not
    list position -- written correctly from the start for this track
    (mirrors the already-fixed evaluation/convergence.py)."""
    res_dir = tmp_path / "results"

    for algo, seed, reward in [
        (PROPOSED_ALGO, 42, 150.5),
        (PROPOSED_ALGO, 123, 148.0),
        ("dqn", 42, 120.0),
        ("dqn", 123, 118.5),
    ]:
        algo_dir = res_dir / f"{algo.lower()}_seed{seed}"
        algo_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "algorithm": algo,
            "seed": seed,
            "final_eval_reward": reward,
            "final_eval_power_w": 90.0,
            "final_qos_rate": 0.95,
        }
        with open(algo_dir / "summary.json", "w") as f:
            json.dump(summary, f)

    report = analyze_convergence(
        results_dir=str(res_dir),
        save_dir=str(tmp_path / "figures"),
        table_save_dir=str(tmp_path / "tables"),
    )

    assert PROPOSED_ALGO in report["algorithms"]
    assert "dqn" in report["paired_ttests"]
    assert report["paired_ttests"]["dqn"]["n_paired_seeds"] == 2
    assert (tmp_path / "tables" / "convergence_summary_oran.tex").exists()


def test_analyze_convergence_handles_baseline_list_shape(tmp_path):
    res_dir = tmp_path / "results"
    proposed_dir = res_dir / "bmpp_dqn_seed42"
    proposed_dir.mkdir(parents=True, exist_ok=True)
    with open(proposed_dir / "summary.json", "w") as f:
        json.dump(
            {
                "algorithm": PROPOSED_ALGO,
                "seed": 42,
                "final_eval_reward": 100.0,
                "final_eval_power_w": 80.0,
                "final_qos_rate": 0.9,
            },
            f,
        )

    ddpg_dir = res_dir / "oran_benchmark_ddpg"
    ddpg_dir.mkdir(parents=True, exist_ok=True)
    with open(ddpg_dir / "summary.json", "w") as f:
        json.dump(
            [
                {
                    "algorithm": "ddpg",
                    "seed": 42,
                    "mean_reward": 50.0,
                    "mean_power_w": 95.0,
                    "qos_satisfaction_rate": 0.8,
                }
            ],
            f,
        )

    report = analyze_convergence(
        results_dir=str(res_dir),
        save_dir=str(tmp_path / "figures"),
        table_save_dir=str(tmp_path / "tables"),
    )
    assert "ddpg" in report["algorithms"]
    assert report["algorithms"]["ddpg"]["mean_reward"] == pytest.approx(50.0)


def test_latency_benchmark_runs_all_four_methods(tmp_path, default_config):
    config_path = tmp_path / "oran_test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(default_config, f)

    results = run_latency_benchmark(
        config_path=str(config_path),
        n_repeats=3,
        save_dir=str(tmp_path / "figures"),
    )

    assert set(results.keys()) == {"bmpp_dqn", "dqn", "ddpg", "mpdqn"}
    for method, latency in results.items():
        assert latency is not None
        assert latency > 0.0

    assert (tmp_path / "figures" / "latency_benchmark_oran.pdf").exists()


def test_latency_benchmark_skips_mpdqn_above_tractability_cap(tmp_path, default_config):
    cfg = deepcopy(default_config)
    cfg["network"]["n_ru"] = 8
    config_path = tmp_path / "oran_test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    results = run_latency_benchmark(
        config_path=str(config_path),
        methods=["mpdqn"],
        n_repeats=1,
        save_dir=str(tmp_path / "figures"),
    )
    assert results["mpdqn"] is None
