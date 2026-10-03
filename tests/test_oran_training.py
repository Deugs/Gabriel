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


def test_run_oran_baseline_benchmarks_trains_with_exploration_enabled(
    make_config_path, tmp_path, monkeypatch
):
    """Guards against training rollout calling select_action(evaluate=True)
    for baselines, which would silently disable exploration for the entire
    training run."""
    import oran_agents.dqn_agent as dqn_module

    seen_evaluate_flags = []
    original_select_action = dqn_module.ORANDQNAgent.select_action

    def spy_select_action(self, obs, evaluate=False):
        seen_evaluate_flags.append(evaluate)
        return original_select_action(self, obs, evaluate=evaluate)

    monkeypatch.setattr(dqn_module.ORANDQNAgent, "select_action", spy_select_action)

    config_path = make_config_path()
    run_oran_baseline_benchmarks(
        config_path=config_path,
        seeds=[42],
        episodes=2,
        algorithms=["dqn"],
        save_dir=str(tmp_path / "results"),
    )

    assert False in seen_evaluate_flags
    assert True in seen_evaluate_flags


def test_run_oran_baseline_benchmarks_reports_held_out_eval_separately(
    make_config_path, tmp_path
):
    config_path = make_config_path()
    res = run_oran_baseline_benchmarks(
        config_path=config_path,
        seeds=[42],
        episodes=2,
        algorithms=["dqn"],
        save_dir=str(tmp_path / "results"),
    )
    seed_summary = res["dqn"][0]
    assert "mean_reward" in seed_summary
    assert "train_mean_reward" in seed_summary


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


@pytest.mark.parametrize("algo", ["dqn", "mpdqn"])
def test_discrete_hold_steps_holds_ru_on_and_split_fixed_within_window(
    make_config_path, tmp_path, monkeypatch, algo
):
    """Fair-cadence control regression guard: with discrete_hold_steps=N,
    DQN/MP-DQN's (ru_on, split) actually sent to env.step() must change
    only every N steps, matching BMPP-DQN's own two-timescale cadence --
    not every step, which is what made the original switching-frequency
    comparison conflate decision cadence with architecture (the critique
    this fair-cadence control responds to)."""
    config_path = make_config_path()
    hold = 3
    seen_actions = []

    import oran_env.oran_env as oran_env_module

    original_step = oran_env_module.ORANEnv.step

    def spy_step(self, action):
        seen_actions.append(
            (action["ru_on"].copy(), action["split"].copy())
        )
        return original_step(self, action)

    monkeypatch.setattr(oran_env_module.ORANEnv, "step", spy_step)

    run_oran_baseline_benchmarks(
        config_path=config_path,
        seeds=[42],
        episodes=2,
        algorithms=[algo],
        save_dir=str(tmp_path / "results"),
        discrete_hold_steps=hold,
    )

    # Training + held-out eval both ran under the hold; every window of
    # `hold` consecutive env.step() calls within a single episode must
    # share the same (ru_on, split) -- windows are reset at each episode
    # boundary (reset() restarts steps_since_decision at 0), so this
    # checks within-window equality rather than across the whole log.
    assert len(seen_actions) > hold, "not enough steps recorded to test a hold window"
    for i in range(1, hold):
        ru_on_i, split_i = seen_actions[i]
        ru_on_0, split_0 = seen_actions[0]
        assert (ru_on_i == ru_on_0).all() and (split_i == split_0).all(), (
            f"step {i} within the first hold window changed (ru_on, split) "
            f"without a decision boundary"
        )


def test_reward_scale_scales_the_replay_buffer_not_the_reported_metrics(
    make_config_path, tmp_path, monkeypatch
):
    """Reward-scaling spot check (Section~oran-training-budget-check):
    reward_scale must multiply what DQN's replay buffer (and therefore its
    Bellman target) sees, while total_reward/ep_rewards and
    _evaluate_oran_baseline's held-out metrics stay computed from the raw
    env reward -- otherwise a reward_scale != 1.0 run's reported numbers
    would silently stop being comparable to every other result in this
    chapter."""
    config_path = make_config_path()
    pushed_rewards = []

    import oran_agents.dqn_agent as dqn_agent_module

    original_push = dqn_agent_module.ReplayBuffer.push

    def spy_push(self, state, ru_on, split, reward, next_state, done):
        pushed_rewards.append(reward)
        return original_push(self, state, ru_on, split, reward, next_state, done)

    monkeypatch.setattr(dqn_agent_module.ReplayBuffer, "push", spy_push)

    scale = 0.01
    results = run_oran_baseline_benchmarks(
        config_path=config_path,
        seeds=[42],
        episodes=2,
        algorithms=["dqn"],
        save_dir=str(tmp_path / "results"),
        reward_scale=scale,
    )

    assert len(pushed_rewards) > 0, "no rewards were pushed to the replay buffer"
    assert all(abs(r) < 50 for r in pushed_rewards), (
        "pushed rewards look unscaled -- reward_scale did not reach "
        "model.memory.push"
    )

    summary = results["dqn"][0]
    assert abs(summary["train_mean_reward"]) > 50, (
        "train_mean_reward looks scaled -- it must stay computed from the "
        "raw env reward, not the scaled replay-buffer reward"
    )
    assert abs(summary["mean_reward"]) > 50, (
        "mean_reward (held-out eval) looks scaled -- "
        "_evaluate_oran_baseline must use the raw env reward regardless "
        "of reward_scale"
    )
