"""Unit tests for Training Infrastructure (training/)."""

from pathlib import Path
import pytest
import yaml  # type: ignore[import-untyped]

from training import (
    HyperparameterSearch,
    apply_config_overrides,
    run_baseline_benchmarks,
    run_proxy_sensitivity_sweep,
    train_hybrid_agent,
)


@pytest.fixture
def config_path(tmp_path):
    orig_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(orig_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_file = tmp_path / "test_config.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(cfg, f)

    return str(cfg_file)


def test_train_hybrid_agent_short_run(config_path, tmp_path):
    save_dir = str(tmp_path / "results")
    res = train_hybrid_agent(
        config_path=config_path,
        seed=42,
        episodes=5,
        eval_freq=5,
        save_dir=save_dir,
    )

    assert "final_train_reward" in res
    assert "final_eval_reward" in res
    assert "history" in res
    assert len(res["history"]["episode_rewards"]) == 5

    out_folder = Path(save_dir) / "branching_mp_dqn_seed42"
    assert (out_folder / "summary.json").exists()
    assert (out_folder / "final_model.pt").exists()


def test_apply_config_overrides_updates_nested_section():
    cfg = {"algorithm": {"lr_actor": 3.0e-4}, "reward": {"beta_qos": 10.0}}
    updated = apply_config_overrides(cfg, {"lr_actor": 0.0, "beta_qos": 5.0})

    assert updated["algorithm"]["lr_actor"] == 0.0
    assert updated["reward"]["beta_qos"] == 5.0
    # The input config must not be mutated.
    assert cfg["algorithm"]["lr_actor"] == 3.0e-4
    assert cfg["reward"]["beta_qos"] == 10.0


def test_apply_config_overrides_no_overrides_is_a_no_op():
    cfg = {"algorithm": {"lr_actor": 3.0e-4}}
    assert apply_config_overrides(cfg, None) is cfg
    assert apply_config_overrides(cfg, {}) is cfg


def test_apply_config_overrides_rejects_unknown_key():
    cfg = {"algorithm": {"lr_actor": 3.0e-4}}
    with pytest.raises(ValueError):
        apply_config_overrides(cfg, {"not_a_real_key": 1.0})


def test_train_hybrid_agent_rejects_unknown_override_key(config_path):
    """Proves config_overrides is actually wired into train_hybrid_agent,
    not just accepted and ignored (the bug found in evaluation/ablation.py)."""
    with pytest.raises(ValueError):
        train_hybrid_agent(
            config_path=config_path,
            seed=42,
            episodes=1,
            eval_freq=1,
            save_dir=None,
            config_overrides={"not_a_real_key": 1.0},
        )


def test_run_baseline_benchmarks_short_run(config_path, tmp_path):
    save_dir = str(tmp_path / "results")
    res = run_baseline_benchmarks(
        config_path=config_path,
        seeds=[42],
        episodes=2,
        algorithms=["all_on", "greedy"],
        save_dir=save_dir,
    )

    assert "all_on" in res and "greedy" in res
    assert len(res["all_on"]) == 1
    assert res["all_on"][0]["seed"] == 42
    assert "mean_reward" in res["all_on"][0]

    assert (Path(save_dir) / "benchmark_all_on" / "summary.json").exists()
    assert (Path(save_dir) / "benchmark_greedy" / "summary.json").exists()


def test_hyperparameter_search(config_path, tmp_path):
    save_dir = str(tmp_path / "grid_search")
    searcher = HyperparameterSearch(
        base_config_path=config_path,
        save_dir=save_dir,
    )

    grid = {
        "lr_actor": [1e-4, 3e-4],
    }

    results = searcher.run_grid_search(
        param_grid=grid,
        episodes_per_trial=2,
        seeds=[42],
    )

    assert len(results) == 2
    assert (Path(save_dir) / "grid_search_results.json").exists()


def test_proxy_sensitivity_sweep_short_run(config_path, tmp_path):
    """Concept Note v4.0 Section 12.11's lightweight lr-pair/tau sensitivity sweep."""
    save_dir = str(tmp_path / "proxy_sweep")

    summary = run_proxy_sensitivity_sweep(
        base_config_path=config_path,
        episodes=2,
        seeds=[42],
        save_dir=save_dir,
    )

    assert summary["scenario"]["n_rrh"] == 5
    assert summary["scenario"]["n_ue"] == 2

    expected_variants = {
        "lr_pair_down",
        "lr_pair_default",
        "lr_pair_up",
        "tau_down",
        "tau_default",
        "tau_up",
    }
    assert set(summary["results"].keys()) == expected_variants
    assert set(summary["decisions"].keys()) == {"lr_pair", "tau"}
    for decision in summary["decisions"].values():
        assert "default_kept" in decision and "reason" in decision

    assert (Path(save_dir) / "proxy_sweep_summary.json").exists()
