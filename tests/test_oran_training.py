"""Unit tests for O-RAN Training Scripts (oran_training/).

A fully separate test module from tests/test_training.py -- no shared
fixtures or imports with the C-RAN training tests.
"""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from oran_training.train_bmpp_dqn import train_bmpp_dqn_agent
from oran_training.train_oran_baselines import run_oran_baseline_benchmarks


@pytest.fixture
def make_config_path(tmp_path):
    orig_path = Path(__file__).parent.parent / "config" / "oran_default.yaml"
    with open(orig_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    def _make(overrides=None, name="oran_test_config.yaml"):
        cfg = deepcopy(base_cfg)
        cfg["network"]["n_ru"] = 2
        cfg["network"]["n_ue"] = 2
        cfg["algorithm"]["min_buffer_size"] = 8
        cfg["algorithm"]["batch_size"] = 4
        cfg["algorithm"]["upper_level_period_steps"] = 3
        cfg["algorithm"]["max_steps_per_episode"] = 10
        cfg["evaluation"]["eval_freq"] = 2
        cfg["evaluation"]["n_eval_episodes"] = 2
        cfg["evaluation"]["checkpoint_freq"] = 2
        if overrides:
            for section, values in overrides.items():
                cfg[section].update(values)
        cfg_file = tmp_path / name
        with open(cfg_file, "w") as f:
            yaml.dump(cfg, f)
        return str(cfg_file)

    return _make


def test_train_bmpp_dqn_agent_short_run(make_config_path, tmp_path):
    config_path = make_config_path()
    save_dir = str(tmp_path / "results")

    summary = train_bmpp_dqn_agent(
        config_path=config_path, seed=42, episodes=4, save_dir=save_dir
    )

    expected_keys = {
        "algorithm",
        "seed",
        "episodes",
        "total_training_time_sec",
        "final_train_reward",
        "final_eval_reward",
        "final_eval_power_w",
        "final_qos_rate",
        "final_switching_events",
        "final_eval_throughput_mbps",
        "final_upper_level_decisions",
        "history",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["algorithm"] == "BMPP_DQN"

    out_folder = Path(save_dir) / "bmpp_dqn_seed42"
    assert (out_folder / "summary.json").exists()
    assert (out_folder / "final_model.pt").exists()
    assert (out_folder / "config.yaml").exists()


def test_train_bmpp_dqn_agent_enforces_max_episodes_cap(make_config_path):
    config_path = make_config_path({"algorithm": {"max_episodes": 3}})

    with pytest.raises(ValueError, match="max_episodes"):
        train_bmpp_dqn_agent(
            config_path=config_path, seed=42, episodes=4, save_dir=None
        )


def test_train_bmpp_dqn_agent_writes_intermediate_checkpoints(
    make_config_path, tmp_path
):
    save_dir = str(tmp_path / "results")
    config_path = make_config_path()

    train_bmpp_dqn_agent(
        config_path=config_path, seed=42, episodes=4, save_dir=save_dir
    )

    out_folder = Path(save_dir) / "bmpp_dqn_seed42"
    assert (out_folder / "checkpoint_ep2.pt").exists()
    assert (out_folder / "checkpoint_ep4.pt").exists()


def test_run_oran_baseline_benchmarks_short_run(make_config_path, tmp_path):
    config_path = make_config_path()
    save_dir = str(tmp_path / "results")

    res = run_oran_baseline_benchmarks(
        config_path=config_path,
        seeds=[42],
        episodes=2,
        algorithms=["dqn", "ddpg"],
        save_dir=save_dir,
    )

    assert "dqn" in res and "ddpg" in res
    assert len(res["dqn"]) == 1
    assert res["dqn"][0]["seed"] == 42
    assert "mean_reward" in res["dqn"][0]
    assert "mean_throughput_mbps" in res["dqn"][0]

    assert (Path(save_dir) / "oran_benchmark_dqn" / "summary.json").exists()
    assert (Path(save_dir) / "oran_benchmark_ddpg" / "summary.json").exists()


def test_run_oran_baseline_benchmarks_skips_mpdqn_above_tractability_cap(
    make_config_path, tmp_path
):
    config_path = make_config_path({"network": {"n_ru": 8}})
    save_dir = str(tmp_path / "results")

    res = run_oran_baseline_benchmarks(
        config_path=config_path,
        seeds=[42],
        episodes=1,
        algorithms=["mpdqn"],
        save_dir=save_dir,
    )

    assert res["mpdqn"] == []
    assert not (Path(save_dir) / "oran_benchmark_mpdqn").exists()


def test_run_oran_baseline_benchmarks_default_seeds_match_n_random_seeds(
    make_config_path, tmp_path
):
    """evaluation.n_random_seeds must match the hardcoded default seed
    list's length, mirroring training/train_baselines.py's own guard."""
    config_path = make_config_path({"evaluation": {"n_random_seeds": 5}})

    with pytest.raises(ValueError, match="n_random_seeds"):
        run_oran_baseline_benchmarks(
            config_path=config_path,
            episodes=1,
            algorithms=["dqn"],
            save_dir=str(tmp_path / "results"),
        )
