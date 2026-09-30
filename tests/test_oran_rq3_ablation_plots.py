"""Unit tests for oran_evaluation/rq3_ablation_plots.py.

A fully separate test module, matching this repo's one-test-file-per-
module convention for the O-RAN track.
"""

import json
from pathlib import Path

from oran_evaluation.rq3_ablation_plots import plot_rq3_ablation


def _write_fake_bmpp_results(base_dir: Path, reward: float, switching: float) -> str:
    algo_dir = base_dir / "bmpp_dqn_seed42"
    algo_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "algorithm": "BMPP_DQN",
        "seed": 42,
        "final_eval_reward": reward,
        "final_eval_power_w": 160.0,
        "final_qos_rate": 0.5,
        "final_qos_per_ue_rate": 0.9,
        "final_switching_events": switching,
        "final_eval_throughput_mbps": 1000.0,
    }
    with open(algo_dir / "summary.json", "w") as f:
        json.dump(summary, f)
    return str(base_dir)


def test_plot_rq3_ablation_writes_pdf_and_png(tmp_path):
    two_ts_dir = _write_fake_bmpp_results(tmp_path / "two_ts", -20000.0, 0.2)
    one_ts_dir = _write_fake_bmpp_results(tmp_path / "one_ts", -19000.0, 1.2)

    save_path = tmp_path / "rq3_ablation.pdf"
    plot_rq3_ablation(two_ts_dir, one_ts_dir, save_path=str(save_path))

    assert save_path.exists()
    assert save_path.with_suffix(".png").exists()
