"""Unit tests for Training Infrastructure (training/)."""

from pathlib import Path
import pytest
import yaml  # type: ignore[import-untyped]

from training import HyperparameterSearch, run_baseline_benchmarks, train_hybrid_agent
import training.hyperparam_search as hyperparam_search_module


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
    assert all(r["n_unstable"] == 0 for r in results)


def test_run_sensitivity_check(tmp_path, monkeypatch):
    """G9 lightweight hyperparameter-sensitivity protocol (Concept Note v4.0
    Section 12.11): base_config_path="config/small_network.yaml" (R=5) is
    the concept note's prescribed scenario. train_hybrid_agent is stubbed out
    here -- this test verifies run_sensitivity_check builds and executes the
    right 3x3x3 grid (values swept around the base config's own defaults),
    not that 27 real training runs individually converge (that's covered,
    for one real trial, by test_hyperparameter_search and
    test_run_grid_search_records_instability_without_crashing)."""
    small_config_path = Path(__file__).parent.parent / "config" / "small_network.yaml"
    save_dir = str(tmp_path / "sensitivity")

    def fake_train_hybrid_agent(*, config_path, seed, episodes, eval_freq, save_dir):
        return {"final_eval_reward": 1.0, "final_eval_power_w": 2.0, "final_qos_rate": 0.9}

    monkeypatch.setattr(
        hyperparam_search_module, "train_hybrid_agent", fake_train_hybrid_agent
    )

    searcher = HyperparameterSearch(
        base_config_path=str(small_config_path),
        save_dir=save_dir,
    )
    results = searcher.run_sensitivity_check(episodes_per_trial=2, seeds=[42])

    # 3 values each for lr_discrete, lr_actor, tau -> 27 combinations.
    assert len(results) == 27
    assert all("lr_discrete" in r["params"] for r in results)
    assert all("lr_actor" in r["params"] for r in results)
    assert all("tau" in r["params"] for r in results)
    assert all(r["n_unstable"] == 0 for r in results)

    swept_lr_discrete = sorted({r["params"]["lr_discrete"] for r in results})
    assert len(swept_lr_discrete) == 3
    assert swept_lr_discrete[1] == pytest.approx(1.0e-3)  # small_network.yaml's default

    assert (Path(save_dir) / "grid_search_results.json").exists()


def test_run_grid_search_records_instability_without_crashing(config_path, tmp_path, monkeypatch):
    """A RuntimeError from one seed's training run (e.g. a NaN/Inf reward, per
    train_hybrid_agent's own anomaly check) must be caught and recorded as
    "unstable" for that trial, not abort the whole sweep -- required for the
    sensitivity check's own purpose of finding *which* settings are unstable."""
    save_dir = str(tmp_path / "grid_search_unstable")
    searcher = HyperparameterSearch(base_config_path=config_path, save_dir=save_dir)

    call_count = {"n": 0}
    real_train_hybrid_agent = hyperparam_search_module.train_hybrid_agent

    def flaky_train_hybrid_agent(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Numerical instability detected at episode 1: reward=nan")
        return real_train_hybrid_agent(*args, **kwargs)

    monkeypatch.setattr(
        hyperparam_search_module, "train_hybrid_agent", flaky_train_hybrid_agent
    )

    results = searcher.run_grid_search(
        param_grid={"lr_actor": [1e-4]},
        episodes_per_trial=2,
        seeds=[42, 123],
    )

    assert len(results) == 1
    assert results[0]["n_unstable"] == 1
    statuses = {s["seed"]: s["status"] for s in results[0]["seed_statuses"]}
    assert statuses == {42: "unstable", 123: "ok"}
