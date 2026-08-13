"""Unified Baseline Evaluation Benchmark Runner for C-RAN Simulation."""

import argparse
import gc
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

import numpy as np
import yaml  # type: ignore[import-untyped]

from agents import DDPGAgent, DDQNAgent, MPDQNAgent, PDQNAgent
from baselines import (
    AllOnUniformBaseline,
    ANNGSBFBaseline,
    ConvexPowerBaseline,
    DDQNSOCPBaseline,
    GreedyHeuristicBaseline,
    NMBSBinPackingBaseline,
)
from cran_env import CRANEnv


def set_seed(seed: int):
    """Set random seeds across Python and NumPy for fair baseline comparison."""
    random.seed(seed)
    np.random.seed(seed)


def run_baseline_benchmarks(
    config_path: str = "config/default.yaml",
    seeds: Optional[List[int]] = None,
    episodes: int = 50,
    algorithms: Optional[List[str]] = None,
    save_dir: str = "data/results",
) -> Dict[str, Any]:
    """Run baseline benchmark algorithms over specified random seeds."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011, 1337, 2024, 2718, 3141, 4242]

    if algorithms is None:
        algorithms = [
            "all_on",
            "greedy",
            "nmbs",
            "convex",
            "ddqn",
            "ann_gsbf",
            "ddqn_socp",
            "ddpg",
            "pdqn",
            "mpdqn",
        ]

    drl_trained_algorithms = {"ddqn", "ddpg", "pdqn", "mpdqn"}

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    results: Dict[str, Any] = {}

    for algo in algorithms:
        print(f"\n================ Running Benchmark: {algo.upper()} ================")
        algo_results = []

        for seed in seeds:
            set_seed(seed)
            env = CRANEnv(cfg)
            obs, _ = env.reset(seed=seed)

            # Instantiate baseline model
            model: Any
            if algo == "all_on":
                model = AllOnUniformBaseline(env.n_rrh, env.p_max_w)
            elif algo == "greedy":
                model = GreedyHeuristicBaseline(env.n_rrh, env.n_ue, env.p_max_w)
            elif algo == "nmbs":
                model = NMBSBinPackingBaseline(env.n_rrh, env.n_ue, env.p_max_w)
            elif algo == "convex":
                model = ConvexPowerBaseline(env.n_rrh, env.n_ue, env.p_max_w)
            elif algo == "ddqn":
                model = DDQNAgent(env.state_dim, env.n_rrh)
            elif algo == "ann_gsbf":
                model = ANNGSBFBaseline(env.n_rrh, env.n_ue, env.p_max_w)
            elif algo == "ddqn_socp":
                model = DDQNSOCPBaseline(
                    state_dim=env.state_dim,
                    n_rrh=env.n_rrh,
                    n_ue=env.n_ue,
                    p_max_w=env.p_max_w,
                    config=cfg,
                )
            elif algo == "ddpg":
                model = DDPGAgent(
                    state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
                )
            elif algo == "pdqn":
                model = PDQNAgent(
                    state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
                )
            elif algo == "mpdqn":
                model = MPDQNAgent(
                    state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
                )
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            ep_rewards = []
            ep_powers = []
            ep_qos_rates = []
            ep_active_rrhs = []

            for ep in range(episodes):
                obs, _ = env.reset()
                total_reward = 0.0
                powers = []
                qos_flags = []
                actives = []

                done = False
                while not done:
                    if algo in drl_trained_algorithms:
                        action = model.select_action(obs, evaluate=True)
                    else:
                        action = model.select_action(obs)

                    next_obs, reward, terminated, truncated, info = env.step(action)

                    if algo == "ddqn":
                        model.memory.push(
                            obs, action["rrh_on"], reward, next_obs, terminated
                        )
                        model.update()
                    elif algo == "ddpg":
                        model.memory.push(
                            obs, action["continuous_action"], reward, next_obs, terminated
                        )
                        model.update()
                    elif algo in ("pdqn", "mpdqn"):
                        model.memory.push(
                            obs,
                            action["action_idx"],
                            action["continuous"],
                            reward,
                            next_obs,
                            terminated,
                        )
                        model.update()

                    total_reward += reward
                    powers.append(info.get("total_power_w", 0.0))
                    qos_flags.append(
                        1.0 if info.get("qos_violations_count", 0) == 0 else 0.0
                    )
                    actives.append(info.get("active_rrhs", 0))

                    obs = next_obs
                    done = terminated or truncated

                ep_rewards.append(float(total_reward))
                ep_powers.append(float(np.mean(powers)))
                ep_qos_rates.append(float(np.mean(qos_flags)))
                ep_active_rrhs.append(float(np.mean(actives)))

            seed_summary: Dict[str, Any] = {
                "algorithm": algo,
                "seed": seed,
                "mean_reward": float(np.mean(ep_rewards)),
                "std_reward": float(np.std(ep_rewards)),
                "mean_power_w": float(np.mean(ep_powers)),
                "qos_satisfaction_rate": float(np.mean(ep_qos_rates)),
                "mean_active_rrhs": float(np.mean(ep_active_rrhs)),
            }
            algo_results.append(seed_summary)

            qos_pct = seed_summary["qos_satisfaction_rate"] * 100
            print(
                f"Algo: {algo:8s} | Seed: {seed:4d} | "
                f"Reward: {seed_summary['mean_reward']:8.2f} | "
                f"Power: {seed_summary['mean_power_w']:6.1f}W | QoS: {qos_pct:5.1f}%"
            )

            del model, env
            gc.collect()

        results[algo] = algo_results

        # Save algorithm benchmark summary
        out_path = Path(save_dir) / f"benchmark_{algo}"
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "summary.json", "w") as f:
            json.dump(algo_results, f, indent=2)

        # Save the exact config and run-level args used, so this summary is
        # reproducible on its own (Reproducibility Commitment, Concept Note
        # v4.0 Section 12.10).
        run_record = dict(cfg)
        run_record["_run"] = {
            "config_path": config_path,
            "algorithm": algo,
            "seeds": seeds,
            "episodes": episodes,
        }
        with open(out_path / "config.yaml", "w") as f:
            yaml.dump(run_record, f)

    gc.collect()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline Benchmarks for C-RAN")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Config file path"
    )
    parser.add_argument(
        "--episodes", type=int, default=50, help="Number of episodes per seed"
    )
    parser.add_argument(
        "--save-dir", type=str, default="data/results", help="Save directory"
    )
    args = parser.parse_args()

    run_baseline_benchmarks(
        config_path=args.config,
        episodes=args.episodes,
        save_dir=args.save_dir,
    )
