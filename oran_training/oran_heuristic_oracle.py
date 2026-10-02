"""Heuristic and Oracle Baselines for the O-RAN track.

Neither needs training: both are fixed/per-step-search policies evaluated
with the exact same held-out protocol every trained baseline already uses
(_evaluate_oran_baseline, oran_training/train_oran_baselines.py), so their
numbers are directly comparable to DQN/DDPG/MP-DQN/BMPP-DQN's own.
Responds to the critique that this thesis's >=15% energy-saving objective
(Concept Note Section 4.2) was benchmarked only against three trained
baselines, with no simple reference policy or upper-bound-style oracle to
calibrate what that target even means in this environment.

Zero imports from cran_env/agents/baselines -- this track's own
zero-shared-code guarantee, same as every other oran_* module.
"""

import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml  # type: ignore[import-untyped]

from oran_env import ORANEnv
from oran_training.train_oran_baselines import _evaluate_oran_baseline

# Mirrors oran_agents/mpdqn_agent.py's own flat-joint-action tractability
# cap -- the oracle's per-step exhaustive search is exactly that same
# 2^n_ru * n_splits^n_ru space, so it is intractable at the same scale.
MAX_N_RU_FOR_ORACLE_SEARCH = 6


class _HeuristicPolicy:
    """Fixed rule, no learning, no per-step search: activate every RU that
    is the strongest server for at least one UE with nonzero current
    demand (idle RUs serving no one go to sleep), full power and an equal
    PRB split among the active RUs, split=0 (most RU-side processing,
    cheapest fronthaul) throughout. The simplest policy this thesis's
    >=15% objective was always implicitly benchmarked against but never
    itself implemented."""

    def __init__(self, env: ORANEnv):
        self.env = env

    def select_action(
        self, obs: np.ndarray, evaluate: bool = True
    ) -> Dict[str, np.ndarray]:
        env = self.env
        gains_sq = np.abs(env.channel_gains) ** 2  # (n_ru, n_ue)
        strongest_ru_per_ue = np.argmax(gains_sq, axis=0)  # (n_ue,)
        has_demand = env.current_demands_bps > 0.0

        ru_on = np.zeros(env.n_ru, dtype=int)
        for ru in range(env.n_ru):
            if np.any((strongest_ru_per_ue == ru) & has_demand):
                ru_on[ru] = 1
        if not ru_on.any():
            # Never go fully dark -- an all-off network can never recover
            # QoS no matter what the next step's demand realization is.
            ru_on[:] = 1

        split = np.zeros(env.n_ru, dtype=np.int64)
        n_active = max(1, int(ru_on.sum()))
        power = np.where(ru_on == 1, env.p_max_w, 0.0).astype(np.float32)
        prb = (ru_on.astype(np.float32) / n_active).astype(np.float32)
        return {"ru_on": ru_on, "split": split, "power": power, "prb": prb}


def _snapshot_env_state(env: ORANEnv) -> Dict[str, Any]:
    """Generic env-state snapshot/restore pair. Not used by _OraclePolicy's
    own hot path (see _score_candidate_reward below for why), but kept as
    a general-purpose utility and exercised directly by this module's own
    tests as a correctness check in its own right."""
    return {
        "active_mask": env.active_mask.copy(),
        "split_idx": env.split_idx.copy(),
        "prev_power_w": env.prev_power_w,
        "hour": env.hour,
        "step_count": env.step_count,
        "channel_gains": env.channel_gains.copy(),
        "current_demands_bps": env.current_demands_bps.copy(),
        "throughput_window": list(env._throughput_window),
        "power_window": list(env._power_window),
        "rng_state": env.rng.bit_generator.state,
    }


def _restore_env_state(env: ORANEnv, snap: Dict[str, Any]) -> None:
    env.active_mask = snap["active_mask"]
    env.split_idx = snap["split_idx"]
    env.prev_power_w = snap["prev_power_w"]
    env.hour = snap["hour"]
    env.step_count = snap["step_count"]
    env.channel_gains = snap["channel_gains"]
    env.current_demands_bps = snap["current_demands_bps"]
    env._throughput_window = snap["throughput_window"]
    env._power_window = snap["power_window"]
    env.rng.bit_generator.state = snap["rng_state"]


def _score_candidate_reward(env: ORANEnv, action: Dict[str, np.ndarray]) -> float:
    """Pure replica of oran_env.py's step() reward computation for a
    hypothetical `action`, given the environment's *current* channel
    gains/demand/active_mask/split_idx -- deliberately stops short of
    everything step() does *after* computing the reward (advancing the
    clock, regenerating the channel, resampling demand, appending to the
    rolling windows), none of which affects this step's own reward and
    all of which would otherwise burn RNG draws for every one of the
    oracle's 1296 throwaway trial evaluations. Reuses
    env._signal_interference()/env.power.compute_total_power() directly
    (both already pure functions of their arguments) so this cannot drift
    from the real reward formula without also changing the function it
    calls."""
    ru_on = np.asarray(action["ru_on"], dtype=bool)
    split = np.asarray(action["split"], dtype=np.int64)
    power_w = np.asarray(action["power"], dtype=np.float32)
    prb_raw = np.asarray(action["prb"], dtype=np.float32)

    active_prb = prb_raw * ru_on.astype(np.float32)
    prb_sum = np.sum(active_prb)
    if prb_sum > 1e-12:
        prb_share = active_prb / prb_sum
    else:
        n_active = max(1, int(np.sum(ru_on)))
        prb_share = ru_on.astype(np.float32) / n_active

    signal, interference, serving_ru = env._signal_interference(ru_on, power_w)
    sinr = np.where(
        signal > 0.0, signal / (interference + env.noise_power_w), 0.0
    ).astype(np.float32)

    served_ue_count = np.zeros(env.n_ru, dtype=np.float32)
    served_mask = serving_ru >= 0
    np.add.at(served_ue_count, serving_ru[served_mask], 1.0)
    serving_ru_clipped = np.clip(serving_ru, 0, env.n_ru - 1)
    n_co_served = np.where(served_mask, served_ue_count[serving_ru_clipped], 1.0)

    user_bandwidth_hz = (
        np.where(serving_ru >= 0, prb_share[serving_ru_clipped] / n_co_served, 0.0)
        * env.channel.bandwidth
    )
    achievable_capacity_bps = user_bandwidth_hz * np.log2(1.0 + sinr)
    total_throughput_mbps = float(np.sum(achievable_capacity_bps) / 1e6)

    qos_violations_bps = np.maximum(
        0.0, env.current_demands_bps - achievable_capacity_bps
    )

    power_dict = env.power.compute_total_power(
        active_mask=ru_on,
        split_idx=split,
        transmit_power_w=power_w,
        prev_active_mask=env.active_mask,
        prev_split_idx=env.split_idx,
    )
    p_total = power_dict["total"]

    exact_switching_count = int(
        np.sum(ru_on != env.active_mask)
        + np.sum((split != env.split_idx) & (ru_on & env.active_mask))
    )

    ee_mbit_per_joule = total_throughput_mbps / (p_total + 1e-6)
    reward = (
        env.alpha_energy * ee_mbit_per_joule
        - env.beta_qos * (float(np.sum(qos_violations_bps)) / 1e6)
        - env.gamma_switch * exact_switching_count
    )
    return float(reward)


class _OraclePolicy:
    """Privileged, non-causal per-step upper bound, not a deployable
    policy: exhaustively searches every (ru_on, split) combination
    (tractable at this scope -- see MAX_N_RU_FOR_ORACLE_SEARCH, the same
    cap oran_agents/mpdqn_agent.py applies to its own flat joint action
    space), paired with a fixed max-power/equal-PRB continuous policy (the
    same reference policy this model's own bandwidth-headroom checks
    use), and greedily picks whichever discrete combination maximizes
    *that single step's own* reward. It is allowed to "see" the step's
    result before committing to an action -- real policies cannot -- so
    read this as a ceiling on single-step reward, not an achievable
    trained policy.

    Implementation: each candidate is scored by `_score_candidate_reward`,
    a pure function replicating oran_env.py's own reward computation
    exactly (same `_signal_interference`/`compute_total_power` calls) but
    without calling the real env.step() -- which would also regenerate
    the channel and resample demand (RNG draws needed only for the
    *next* state, irrelevant to scoring the *current* one) 1296 times
    per real decision for nothing. Only the finally-chosen best candidate
    is actually stepped, once, so the random channel/demand trajectory
    every other baseline experiences at this seed is unaffected by how
    many candidates were scored to reach it.
    """

    def __init__(self, env: ORANEnv):
        if env.n_ru > MAX_N_RU_FOR_ORACLE_SEARCH:
            raise ValueError(
                f"n_ru={env.n_ru} exceeds MAX_N_RU_FOR_ORACLE_SEARCH="
                f"{MAX_N_RU_FOR_ORACLE_SEARCH}; the oracle's exhaustive "
                "per-step search (2^n_ru * n_splits^n_ru) is intractable "
                "beyond this, mirroring MP-DQN's own flat-action cap."
            )
        self.env = env
        ru_on_combos = list(itertools.product([0, 1], repeat=env.n_ru))
        split_combos = list(itertools.product(range(env.n_splits), repeat=env.n_ru))
        self._candidates = [
            (np.array(r, dtype=int), np.array(s, dtype=np.int64))
            for r in ru_on_combos
            for s in split_combos
        ]

    def select_action(
        self, obs: np.ndarray, evaluate: bool = True
    ) -> Dict[str, np.ndarray]:
        env = self.env

        best_reward = -np.inf
        best_action: Optional[Dict[str, np.ndarray]] = None
        for ru_on, split in self._candidates:
            split_masked = split * ru_on  # off RUs: split forced to 0
            n_active = max(1, int(ru_on.sum()))
            power = np.where(ru_on == 1, env.p_max_w, 0.0).astype(np.float32)
            prb = (ru_on.astype(np.float32) / n_active).astype(np.float32)
            candidate_action = {
                "ru_on": ru_on,
                "split": split_masked,
                "power": power,
                "prb": prb,
            }

            reward = _score_candidate_reward(env, candidate_action)

            if reward > best_reward:
                best_reward = reward
                best_action = candidate_action

        assert best_action is not None
        return best_action


def run_oran_heuristic_oracle_benchmarks(
    config_path: str = "config/oran_default.yaml",
    seeds: Optional[List[int]] = None,
    save_dir: str = "data/results_oran",
    policies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate the heuristic and/or oracle policies over `seeds`, writing
    oran_benchmark_<policy>/summary.json in the same format every trained
    baseline's run_oran_baseline_benchmarks() output already uses, so the
    existing table/figure-generation code can read these in unchanged."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    n_eval_episodes = int(cfg.get("evaluation", {}).get("n_eval_episodes", 5))

    if seeds is None:
        seeds = [42, 123, 456]
    if policies is None:
        policies = ["heuristic", "oracle"]

    results: Dict[str, Any] = {}

    for policy_name in policies:
        print(
            f"\n================ Running Benchmark: "
            f"{policy_name.upper()} ================"
        )
        policy_results = []
        skipped = False

        for seed in seeds:
            env = ORANEnv(cfg)
            env.reset(seed=seed)

            model: Any
            try:
                if policy_name == "heuristic":
                    model = _HeuristicPolicy(env)
                elif policy_name == "oracle":
                    model = _OraclePolicy(env)
                else:
                    raise ValueError(f"Unknown policy: {policy_name}")
            except ValueError as exc:
                if policy_name == "oracle":
                    print(
                        f"  n_ru={env.n_ru:3d} | oracle | "
                        f"SKIPPED (intractable): {exc}"
                    )
                    skipped = True
                    break
                raise

            eval_metrics = _evaluate_oran_baseline(
                env, model, eval_episodes=n_eval_episodes
            )
            seed_summary = {"algorithm": policy_name, "seed": seed, **eval_metrics}
            policy_results.append(seed_summary)

            qos_pct = eval_metrics["qos_satisfaction_rate"] * 100
            qos_per_ue_pct = eval_metrics["qos_per_ue_rate"] * 100
            print(
                f"Policy: {policy_name:9s} | Seed: {seed:4d} | "
                f"Reward: {eval_metrics['mean_reward']:8.2f} | "
                f"Power: {eval_metrics['mean_power_w']:6.1f}W | QoS: {qos_pct:5.1f}% "
                f"(per-UE: {qos_per_ue_pct:5.1f}%)"
            )

        if skipped:
            results[policy_name] = []
            continue

        results[policy_name] = policy_results

        out_path = Path(save_dir) / f"oran_benchmark_{policy_name}"
        out_path.mkdir(parents=True, exist_ok=True)
        import json

        with open(out_path / "summary.json", "w") as f:
            json.dump(policy_results, f, indent=2)

        run_record = dict(cfg)
        run_record["_run"] = {
            "config_path": config_path,
            "policy": policy_name,
            "seeds": seeds,
        }
        with open(out_path / "config.yaml", "w") as f:
            yaml.dump(run_record, f)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Heuristic/Oracle Benchmarks for O-RAN"
    )
    parser.add_argument(
        "--config", type=str, default="config/oran_default.yaml", help="Config path"
    )
    parser.add_argument(
        "--save-dir", type=str, default="data/results_oran", help="Save directory"
    )
    args = parser.parse_args()

    run_oran_heuristic_oracle_benchmarks(
        config_path=args.config, save_dir=args.save_dir
    )
