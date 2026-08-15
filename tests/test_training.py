"""Unit tests for Training Infrastructure (training/)."""

from copy import deepcopy
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


@pytest.fixture
def make_config_path(tmp_path):
    """Factory fixture: write config/default.yaml with an `evaluation:`
    override applied, for tests of eval_freq/n_eval_episodes/save_checkpoints/
    checkpoint_freq wiring in train_hybrid_agent()."""
    orig_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(orig_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    def _make(evaluation_overrides, name="test_config.yaml"):
        cfg = deepcopy(base_cfg)
        cfg["evaluation"].update(evaluation_overrides)
        cfg_file = tmp_path / name
        with open(cfg_file, "w") as f:
            yaml.dump(cfg, f)
        return str(cfg_file)

    return _make


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


def test_train_hybrid_agent_enforces_max_episodes_cap(config_path):
    """algorithm.max_episodes (Concept Note v4.0 Section 12.2) is documented
    as an upper cap — previously unread anywhere, so nothing actually capped
    episodes at this value."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["algorithm"]["max_episodes"] = 3
    capped_path = Path(config_path).parent / "capped_config.yaml"
    capped_path.write_text(yaml.dump(cfg))

    with pytest.raises(ValueError, match="max_episodes"):
        train_hybrid_agent(
            config_path=str(capped_path), seed=42, episodes=4, save_dir=None
        )

    # Within the cap: runs normally.
    res = train_hybrid_agent(
        config_path=str(capped_path), seed=42, episodes=3, save_dir=None
    )
    assert res["episodes"] == 3


def test_train_hybrid_agent_eval_freq_defaults_from_config(make_config_path, tmp_path):
    """eval_freq=None (the default) must read evaluation.eval_freq from the
    config, not silently fall back to a hardcoded value that ignores it."""
    config_path = make_config_path({"eval_freq": 2, "n_eval_episodes": 1})
    res = train_hybrid_agent(
        config_path=config_path, seed=42, episodes=4, save_dir=None
    )

    # eval_freq=2 over 4 episodes -> evaluated at episodes 2 and 4.
    assert [h["episode"] for h in res["history"]["eval_history"]] == [2, 4]


def test_train_hybrid_agent_n_eval_episodes_from_config(make_config_path, tmp_path):
    config_path = make_config_path({"eval_freq": 1, "n_eval_episodes": 3})
    res = train_hybrid_agent(
        config_path=config_path, seed=42, episodes=1, save_dir=None
    )

    eval_metrics = res["history"]["eval_history"][0]
    # eval_mean_reward is a mean over n_eval_episodes episodes; just confirm
    # evaluate_agent ran (no crash) and the config path was actually used.
    assert "eval_mean_reward" in eval_metrics


def test_train_hybrid_agent_writes_intermediate_checkpoints(make_config_path, tmp_path):
    """evaluation.save_checkpoints/checkpoint_freq must produce intermediate
    checkpoint files during training, not just the always-saved final_model.pt."""
    save_dir = str(tmp_path / "results")
    config_path = make_config_path(
        {
            "eval_freq": 4,
            "n_eval_episodes": 1,
            "save_checkpoints": True,
            "checkpoint_freq": 2,
        }
    )
    train_hybrid_agent(config_path=config_path, seed=42, episodes=4, save_dir=save_dir)

    out_folder = Path(save_dir) / "branching_mp_dqn_seed42"
    assert (out_folder / "checkpoint_ep2.pt").exists()
    assert (out_folder / "checkpoint_ep4.pt").exists()
    assert (out_folder / "final_model.pt").exists()


def test_train_hybrid_agent_save_checkpoints_false_skips_intermediate_checkpoints(
    make_config_path, tmp_path
):
    save_dir = str(tmp_path / "results")
    config_path = make_config_path(
        {
            "eval_freq": 4,
            "n_eval_episodes": 1,
            "save_checkpoints": False,
            "checkpoint_freq": 1,
        }
    )
    train_hybrid_agent(config_path=config_path, seed=42, episodes=3, save_dir=save_dir)

    out_folder = Path(save_dir) / "branching_mp_dqn_seed42"
    assert not list(out_folder.glob("checkpoint_ep*.pt"))
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


def test_run_baseline_benchmarks_default_seeds_match_n_random_seeds(
    config_path, tmp_path
):
    """evaluation.n_random_seeds must match the hardcoded default seed
    list's length — a config drift here was previously unread and would
    have gone unnoticed."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["evaluation"]["n_random_seeds"] = 3  # actual default list has 10
    mismatched_path = Path(config_path).parent / "mismatched_config.yaml"
    mismatched_path.write_text(yaml.dump(cfg))

    with pytest.raises(ValueError, match="n_random_seeds"):
        run_baseline_benchmarks(
            config_path=str(mismatched_path),
            episodes=1,
            algorithms=["all_on"],
            save_dir=str(tmp_path / "results"),
        )


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
