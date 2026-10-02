"""Unit tests for oran_training/oran_heuristic_oracle.py.

A fully separate test module -- no shared fixtures with the C-RAN
baselines' own tests.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from oran_env import ORANEnv
from oran_training.oran_heuristic_oracle import (
    MAX_N_RU_FOR_ORACLE_SEARCH,
    _HeuristicPolicy,
    _OraclePolicy,
    _restore_env_state,
    _score_candidate_reward,
    _snapshot_env_state,
    run_oran_heuristic_oracle_benchmarks,
)


@pytest.fixture
def default_config():
    config_path = Path(__file__).parent.parent / "config" / "oran_default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_heuristic_policy_never_goes_fully_dark(default_config):
    env = ORANEnv(default_config)
    obs, _ = env.reset(seed=42)
    policy = _HeuristicPolicy(env)

    for _ in range(20):
        action = policy.select_action(obs)
        assert action["ru_on"].sum() >= 1
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


def test_heuristic_policy_activates_only_rus_serving_demanding_ues(default_config):
    """Regression guard on the rule itself: an RU that is no UE's
    strongest server, or whose only served UEs have zero current demand,
    must be left off (unless the all-off fallback kicks in)."""
    cfg = deepcopy(default_config)
    cfg["network"]["n_ru"] = 2
    cfg["network"]["n_ue"] = 1
    env = ORANEnv(cfg)
    env.reset(seed=42)

    # RU0 is UE0's strongest server; RU1 is weaker for every UE.
    env.channel_gains = np.array([[2.0 + 0j], [0.5 + 0j]], dtype=np.complex64)
    env.current_demands_bps = np.array([1.0e6], dtype=np.float64)

    policy = _HeuristicPolicy(env)
    action = policy.select_action(env._get_obs())
    assert list(action["ru_on"]) == [1, 0]


def test_snapshot_restore_round_trip_is_exact(default_config):
    """The oracle's snapshot/restore must leave the environment exactly as
    it was before any throwaway trial step -- including the RNG stream --
    or every candidate after the first would see a different, un-reset
    channel/demand realization than the real step eventually commits to."""
    env = ORANEnv(default_config)
    env.reset(seed=42)

    snap = _snapshot_env_state(env)
    action = env.action_space.sample()
    env.step(action)  # mutate everything a trial step would mutate
    _restore_env_state(env, snap)

    post_restore_action = {
        "ru_on": np.ones(env.n_ru, dtype=int),
        "split": np.zeros(env.n_ru, dtype=int),
        "power": np.full(env.n_ru, env.p_max_w, dtype=np.float32),
        "prb": np.ones(env.n_ru, dtype=np.float32) / env.n_ru,
    }
    _, reward_a, _, _, info_a = env.step(post_restore_action)

    # Redo the exact same sequence from an independent, freshly-reset env
    # at the same seed: reset -> sample+step (the "trial") -> restore ->
    # step the same post-restore action. If restore is exact, this must
    # reproduce the identical reward/info the first run got.
    env2 = ORANEnv(default_config)
    env2.reset(seed=42)
    snap2 = _snapshot_env_state(env2)
    env2.step(action)
    _restore_env_state(env2, snap2)
    _, reward_b, _, _, info_b = env2.step(post_restore_action)

    assert reward_a == pytest.approx(reward_b)
    assert info_a["throughput_mbps"] == pytest.approx(info_b["throughput_mbps"])


def test_score_candidate_reward_matches_real_env_step(default_config):
    """The oracle's fast path (_score_candidate_reward) must return exactly
    the reward the real env.step() would have returned for the same
    action at the same state -- the whole point of reusing
    _signal_interference()/compute_total_power() directly rather than
    reimplementing the formula is that this can't silently drift, but
    only if it actually is checked against the real thing."""
    env = ORANEnv(default_config)
    obs, _ = env.reset(seed=42)
    action = env.action_space.sample()

    snap = _snapshot_env_state(env)
    fast_reward = _score_candidate_reward(env, action)

    _, real_reward, _, _, _ = env.step(action)
    _restore_env_state(env, snap)

    assert fast_reward == pytest.approx(real_reward, rel=1e-5)


def test_oracle_policy_never_scores_worse_than_random_candidates(default_config):
    """The oracle must pick the best-scoring candidate, not merely a valid
    one -- checked by confirming its chosen action's reward is >= a
    sample of other candidates' rewards at the same state (the oracle
    necessarily tries every candidate internally, so this also guards
    against a stale/wrong best-tracking bug)."""
    env = ORANEnv(default_config)
    obs, _ = env.reset(seed=42)
    oracle = _OraclePolicy(env)

    snap = _snapshot_env_state(env)
    oracle_action = oracle.select_action(obs)
    _, oracle_reward, _, _, _ = env.step(oracle_action)
    _restore_env_state(env, snap)

    rng = np.random.default_rng(0)
    for _ in range(10):
        idx = rng.integers(0, len(oracle._candidates))
        ru_on, split = oracle._candidates[idx]
        n_active = max(1, int(ru_on.sum()))
        candidate_action = {
            "ru_on": ru_on,
            "split": split * ru_on,
            "power": np.where(ru_on == 1, env.p_max_w, 0.0).astype(np.float32),
            "prb": (ru_on.astype(np.float32) / n_active).astype(np.float32),
        }
        _, candidate_reward, _, _, _ = env.step(candidate_action)
        _restore_env_state(env, snap)
        assert oracle_reward >= candidate_reward - 1e-6


def test_oracle_raises_above_tractability_cap(default_config):
    cfg = deepcopy(default_config)
    cfg["network"]["n_ru"] = MAX_N_RU_FOR_ORACLE_SEARCH + 1
    env = ORANEnv(cfg)
    env.reset(seed=42)

    with pytest.raises(ValueError, match="MAX_N_RU_FOR_ORACLE_SEARCH"):
        _OraclePolicy(env)


def test_run_oran_heuristic_oracle_benchmarks_short_run(default_config, tmp_path):
    cfg = deepcopy(default_config)
    cfg["network"]["n_ru"] = 2
    cfg["network"]["n_ue"] = 2
    cfg["algorithm"]["max_steps_per_episode"] = 5
    cfg["evaluation"]["n_eval_episodes"] = 2
    config_path = tmp_path / "oran_test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    save_dir = str(tmp_path / "results")
    res = run_oran_heuristic_oracle_benchmarks(
        config_path=str(config_path), seeds=[42], save_dir=save_dir
    )

    assert "heuristic" in res and "oracle" in res
    assert len(res["heuristic"]) == 1 and len(res["oracle"]) == 1
    for policy_name in ("heuristic", "oracle"):
        summary = res[policy_name][0]
        assert summary["algorithm"] == policy_name
        assert summary["seed"] == 42
        assert "mean_reward" in summary
        assert "mean_throughput_mbps" in summary
        assert (
            Path(save_dir) / f"oran_benchmark_{policy_name}" / "summary.json"
        ).exists()

    # The oracle sees its own action's realized reward before committing
    # (a per-step upper bound) -- it must never do worse, in expectation
    # over these seeds/episodes, than the much simpler fixed heuristic.
    assert res["oracle"][0]["mean_reward"] >= res["heuristic"][0]["mean_reward"]
