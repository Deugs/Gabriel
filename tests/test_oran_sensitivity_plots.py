"""Unit tests for oran_evaluation/sensitivity_plots.py.

A fully separate test module from tests/test_oran_multicriteria.py and
tests/test_oran_evaluation.py, matching this repo's one-test-file-per-
module convention for the O-RAN track.
"""

import json
from pathlib import Path

from oran_evaluation.sensitivity_plots import (
    plot_sensitivity_reward_power,
    plot_sensitivity_topsis,
)


def _write_fake_results(base_dir: Path, reward_bmpp: float, power_bmpp: float) -> str:
    """Build a minimal 4-method results directory (1 seed each) that
    load_algo_seed_metrics() can parse, with BMPP-DQN's reward/power
    controllable per call so two fake "configs" can differ."""
    algos = {
        "BMPP_DQN": {"reward": reward_bmpp, "power": power_bmpp},
        "ddpg": {"reward": -50000.0, "power": 200.0},
        "dqn": {"reward": -20000.0, "power": 210.0},
        "mpdqn": {"reward": -15000.0, "power": 160.0},
    }
    for algo, vals in algos.items():
        summary = {
            "algorithm": algo,
            "seed": 42,
            "final_eval_reward": vals["reward"],
            "final_eval_power_w": vals["power"],
            "final_qos_rate": 0.5,
            "final_qos_per_ue_rate": 0.9,
            "final_switching_events": 0.2,
            "final_eval_throughput_mbps": 1000.0,
        }
        if algo == "BMPP_DQN":
            algo_dir = base_dir / "bmpp_dqn_seed42"
            algo_dir.mkdir(parents=True, exist_ok=True)
            with open(algo_dir / "summary.json", "w") as f:
                json.dump(summary, f)
        else:
            algo_dir = base_dir / f"oran_benchmark_{algo}"
            algo_dir.mkdir(parents=True, exist_ok=True)
            with open(algo_dir / "summary.json", "w") as f:
                json.dump([{**summary, "mean_reward": vals["reward"],
                            "mean_power_w": vals["power"],
                            "qos_satisfaction_rate": 0.5,
                            "mean_throughput_mbps": 1000.0}], f)
    return str(base_dir)


def test_plot_sensitivity_topsis_writes_pdf_and_png(tmp_path):
    default_dir = _write_fake_results(tmp_path / "default", -21000.0, 160.0)
    perturbed_dir = _write_fake_results(tmp_path / "perturbed", -30000.0, 300.0)
    configs = [("Default", default_dir), ("Perturbed x10", perturbed_dir)]

    save_path = tmp_path / "sensitivity_topsis.pdf"
    plot_sensitivity_topsis(configs, save_path=str(save_path))

    assert save_path.exists()
    assert save_path.with_suffix(".png").exists()


def test_plot_sensitivity_reward_power_writes_pdf_and_png(tmp_path):
    default_dir = _write_fake_results(tmp_path / "default", -21000.0, 160.0)
    perturbed_dir = _write_fake_results(tmp_path / "perturbed", -30000.0, 300.0)
    configs = [("Default", default_dir), ("Perturbed x10", perturbed_dir)]

    save_path = tmp_path / "sensitivity_reward_power.pdf"
    plot_sensitivity_reward_power(configs, save_path=str(save_path))

    assert save_path.exists()
    assert save_path.with_suffix(".png").exists()
