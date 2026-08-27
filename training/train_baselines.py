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


def _evaluate_baseline(
    env: CRANEnv,
    algo: str,
    model: Any,
    drl_trained_algorithms: set,
    eval_episodes: int = 5,
) -> Dict[str, float]:
    """Deterministic held-out evaluation, mirroring training/train_hybrid.py's
    evaluate_agent(): dedicated eval seeds, no training/no memory writes.

    Comparing a baseline's training-time running-average reward against the
    proposed method's held-out final_eval_reward (as this module previously
    did) is not a fair comparison -- this gives every baseline the same
    held-out-eval treatment the proposed method already gets."""
    eval_rewards, eval_powers, eval_qos, eval_active, eval_switch = [], [], [], [], []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        total_reward = 0.0
        powers, qos_flags, actives, switches = [], [], [], []

        done = False
        while not done:
            if algo in drl_trained_algorithms:
                action = model.select_action(obs, evaluate=True)
            else:
                action = model.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            powers.append(info.get("total_power_w", 0.0))
            qos_flags.append(1.0 if info.get("qos_violations_count", 0) == 0 else 0.0)
            actives.append(info.get("active_rrhs", 0))
            switches.append(info.get("switching_events", 0))
            done = terminated or truncated

        eval_rewards.append(float(total_reward))
        eval_powers.append(float(np.mean(powers)))
        eval_qos.append(float(np.mean(qos_flags)))
        eval_active.append(float(np.mean(actives)))
        eval_switch.append(float(np.mean(switches)))

    return {
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "mean_power_w": float(np.mean(eval_powers)),
        "qos_satisfaction_rate": float(np.mean(eval_qos)),
        "mean_active_rrhs": float(np.mean(eval_active)),
        "mean_switching_events": float(np.mean(eval_switch)),
    }


def run_baseline_benchmarks(
    config_path: str = "config/default.yaml",
    seeds: Optional[List[int]] = None,
    episodes: int = 50,
    algorithms: Optional[List[str]] = None,
    save_dir: str = "data/results",
) -> Dict[str, Any]:
    """Run baseline benchmark algorithms over specified random seeds."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    n_eval_episodes = int(cfg.get("evaluation", {}).get("n_eval_episodes", 5))

    if seeds is None:
        seeds = [42, 123, 456, 789, 1011, 1337, 2024, 2718, 3141, 4242]
        # evaluation.n_random_seeds (Concept Note v4.0 Section 12.4) is a
        # consistency guard on this default list, not a generator for it —
        # the specific seed values are a deliberate, documented choice
        # (README.md's Key Decisions Log), not derivable from a count alone.
        n_random_seeds = cfg.get("evaluation", {}).get("n_random_seeds")
        if n_random_seeds is not None and len(seeds) != int(n_random_seeds):
            raise ValueError(
                f"Default seed list has {len(seeds)} seeds but "
                f"evaluation.n_random_seeds={n_random_seeds} in {config_path} "
                "— update one to match the other."
            )

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

    drl_trained_algorithms = {"ddqn", "ddqn_socp", "ddpg", "pdqn", "mpdqn"}

    results: Dict[str, Any] = {}

    for algo in algorithms:
        print(f"\n================ Running Benchmark: {algo.upper()} ================")
        algo_results = []
        skipped = False

        for seed in seeds:
            set_seed(seed)
            env = CRANEnv(cfg)
            obs, _ = env.reset(seed=seed)

            # Instantiate baseline model
            model: Any
            try:
                if algo == "all_on":
                    model = AllOnUniformBaseline(env.n_rrh, env.p_max_w)
                elif algo == "greedy":
                    model = GreedyHeuristicBaseline(env.n_rrh, env.n_ue, env.p_max_w)
                elif algo == "nmbs":
                    model = NMBSBinPackingBaseline(env.n_rrh, env.n_ue, env.p_max_w)
                elif algo == "convex":
                    model = ConvexPowerBaseline(
                        env.n_rrh,
                        env.n_ue,
                        env.p_max_w,
                        target_sinr_db=float(
                            cfg.get("reward", {}).get("qos_target_sinr_db", 0.0)
                        ),
                        noise_power_w=env.noise_power_w,
                    )
                elif algo == "ddqn":
                    algo_cfg = cfg.get("algorithm", {})
                    model = DDQNAgent(
                        env.state_dim,
                        env.n_rrh,
                        hidden_dims=algo_cfg.get("hidden_dims"),
                        activation=algo_cfg.get("activation", "relu"),
                        use_layer_norm=algo_cfg.get("use_layer_norm", True),
                    )
                elif algo == "ann_gsbf":
                    model = ANNGSBFBaseline(
                        env.n_rrh,
                        env.n_ue,
                        env.p_max_w,
                        noise_power_w=env.noise_power_w,
                        bandwidth_hz=env.channel.bandwidth,
                    )
                elif algo == "ddqn_socp":
                    model = DDQNSOCPBaseline(
                        state_dim=env.state_dim,
                        n_rrh=env.n_rrh,
                        n_ue=env.n_ue,
                        p_max_w=env.p_max_w,
                        config=cfg,
                        noise_power_w=env.noise_power_w,
                    )
                elif algo == "ddpg":
                    model = DDPGAgent(
                        state_dim=env.state_dim,
                        n_rrh=env.n_rrh,
                        p_max_w=env.p_max_w,
                        config=cfg,
                    )
                elif algo == "pdqn":
                    model = PDQNAgent(
                        state_dim=env.state_dim,
                        n_rrh=env.n_rrh,
                        p_max_w=env.p_max_w,
                        config=cfg,
                    )
                elif algo == "mpdqn":
                    model = MPDQNAgent(
                        state_dim=env.state_dim,
                        n_rrh=env.n_rrh,
                        p_max_w=env.p_max_w,
                        config=cfg,
                    )
                else:
                    raise ValueError(f"Unknown algorithm: {algo}")
            except ValueError as exc:
                # P-DQN/MP-DQN's flat joint discrete action space is
                # intractable above MAX_N_RRH_FOR_FLAT_JOINT_ACTION (Section
                # 10.3.1/12.1) — reported as a finding (matching
                # evaluation/latency_benchmark.py's same graceful skip),
                # not left to crash the whole benchmark run.
                if algo in ("pdqn", "mpdqn"):
                    print(
                        f"  R={env.n_rrh:3d} | {algo:10s} | SKIPPED (intractable): {exc}"
                    )
                    skipped = True
                    break
                raise

            ep_rewards = []
            ep_powers = []
            ep_qos_rates = []
            ep_active_rrhs = []
            ep_switching_events = []

            for ep in range(episodes):
                obs, _ = env.reset()
                total_reward = 0.0
                powers = []
                qos_flags = []
                actives = []
                switches = []

                done = False
                while not done:
                    if algo in drl_trained_algorithms:
                        # evaluate=False: exploration (epsilon-greedy /
                        # continuous noise) must stay on during training
                        # rollout -- evaluate=True here previously trained
                        # every DRL baseline with exploration permanently
                        # disabled, so each agent only ever exploited
                        # whatever its randomly-initialized network produced.
                        action = model.select_action(obs, evaluate=False)
                    else:
                        action = model.select_action(obs)

                    next_obs, reward, terminated, truncated, info = env.step(action)

                    if algo == "ddqn":
                        model.memory.push(
                            obs, action["rrh_on"], reward, next_obs, terminated
                        )
                        model.update()
                    elif algo == "ddqn_socp":
                        # DDQNSOCPBaseline wraps a plain DDQNAgent (Stage 1,
                        # discrete activation) at self.ddqn; Stage 2 (SOCP
                        # power) is a solver, not a learned component, so
                        # only the Stage 1 DDQN needs training here.
                        model.ddqn.memory.push(
                            obs, action["rrh_on"], reward, next_obs, terminated
                        )
                        model.ddqn.update()
                    elif algo == "ddpg":
                        model.memory.push(
                            obs,
                            action["continuous_action"],
                            reward,
                            next_obs,
                            terminated,
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
                    switches.append(info.get("switching_events", 0))

                    obs = next_obs
                    done = terminated or truncated

                # epsilon_decay (config/default.yaml) is a per-episode rate,
                # not per environment step.
                if algo in ("ddqn", "pdqn", "mpdqn"):
                    model.decay_exploration()
                elif algo == "ddqn_socp":
                    model.ddqn.decay_exploration()

                ep_rewards.append(float(total_reward))
                ep_powers.append(float(np.mean(powers)))
                ep_qos_rates.append(float(np.mean(qos_flags)))
                ep_active_rrhs.append(float(np.mean(actives)))
                ep_switching_events.append(float(np.mean(switches)))

            # Held-out deterministic evaluation, same treatment
            # training/train_hybrid.py already gives the proposed method —
            # the training-time ep_rewards/etc. above are not a fair
            # like-for-like comparison against final_eval_reward (dragged
            # down by early exploration, not the converged policy).
            eval_metrics = _evaluate_baseline(
                env, algo, model, drl_trained_algorithms, eval_episodes=n_eval_episodes
            )

            seed_summary: Dict[str, Any] = {
                "algorithm": algo,
                "seed": seed,
                "mean_reward": eval_metrics["mean_reward"],
                "std_reward": eval_metrics["std_reward"],
                "mean_power_w": eval_metrics["mean_power_w"],
                "qos_satisfaction_rate": eval_metrics["qos_satisfaction_rate"],
                "mean_active_rrhs": eval_metrics["mean_active_rrhs"],
                "mean_switching_events": eval_metrics["mean_switching_events"],
                "train_mean_reward": float(np.mean(ep_rewards)),
                "train_mean_power_w": float(np.mean(ep_powers)),
                "train_qos_satisfaction_rate": float(np.mean(ep_qos_rates)),
                "train_mean_active_rrhs": float(np.mean(ep_active_rrhs)),
                "train_mean_switching_events": float(np.mean(ep_switching_events)),
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

        if skipped:
            results[algo] = []
            continue

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
