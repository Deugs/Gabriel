"""Unified Baseline Benchmark Runner for the O-RAN track.

Mirrors training/train_baselines.py's structure, adapted to the 3
baselines this track's scope requires (Concept Note Section 6.3/7.1):
DQN, DDPG, MP-DQN. Zero imports from training/ or agents/.
"""

import gc
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

import numpy as np
import yaml  # type: ignore[import-untyped]

from oran_agents import ORANDDPGAgent, ORANDQNAgent, ORANMPDQNAgent
from oran_env import ORANEnv


def set_seed(seed: int):
    """Set random seeds across Python and NumPy for fair baseline comparison."""
    random.seed(seed)
    np.random.seed(seed)


def _evaluate_oran_baseline(
    env: ORANEnv, model: Any, eval_episodes: int = 5
) -> Dict[str, float]:
    """Deterministic held-out evaluation, mirroring
    oran_training/train_bmpp_dqn.py's evaluate_agent(): dedicated eval
    seeds, no training/no memory writes. Comparing a baseline's
    training-time running-average reward against the proposed method's
    held-out final_eval_reward is not a fair comparison -- this gives
    every baseline the same held-out-eval treatment BMPP-DQN already gets."""
    eval_rewards, eval_powers, eval_qos = [], [], []
    eval_active, eval_switch, eval_throughput = [], [], []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        total_reward = 0.0
        powers, qos_flags, actives, switches, throughputs = [], [], [], [], []

        done = False
        while not done:
            action = model.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            powers.append(info.get("total_power_w", 0.0))
            qos_flags.append(1.0 if info.get("qos_violations_count", 0) == 0 else 0.0)
            actives.append(info.get("active_rus", 0))
            switches.append(info.get("switching_events", 0))
            throughputs.append(info.get("throughput_mbps", 0.0))
            done = terminated or truncated

        eval_rewards.append(float(total_reward))
        eval_powers.append(float(np.mean(powers)))
        eval_qos.append(float(np.mean(qos_flags)))
        eval_active.append(float(np.mean(actives)))
        eval_switch.append(float(np.mean(switches)))
        eval_throughput.append(float(np.mean(throughputs)))

    return {
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "mean_power_w": float(np.mean(eval_powers)),
        "qos_satisfaction_rate": float(np.mean(eval_qos)),
        "mean_active_rus": float(np.mean(eval_active)),
        "mean_switching_events": float(np.mean(eval_switch)),
        "mean_throughput_mbps": float(np.mean(eval_throughput)),
    }


def run_oran_baseline_benchmarks(
    config_path: str = "config/oran_default.yaml",
    seeds: Optional[List[int]] = None,
    episodes: int = 50,
    algorithms: Optional[List[str]] = None,
    save_dir: str = "data/results_oran",
) -> Dict[str, Any]:
    """Run O-RAN baseline benchmark algorithms over specified random seeds."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    n_eval_episodes = int(cfg.get("evaluation", {}).get("n_eval_episodes", 5))

    if seeds is None:
        seeds = [42, 123, 456]
        # Concept Note Section 5.3: "3 random seeds for statistical
        # confidence" -- the first 3 of the C-RAN track's own 10-seed
        # convention, for consistency across tracks rather than an
        # unrelated arbitrary choice.
        n_random_seeds = cfg.get("evaluation", {}).get("n_random_seeds")
        if n_random_seeds is not None and len(seeds) != int(n_random_seeds):
            raise ValueError(
                f"Default seed list has {len(seeds)} seeds but "
                f"evaluation.n_random_seeds={n_random_seeds} in {config_path} "
                "— update one to match the other."
            )

    if algorithms is None:
        algorithms = ["dqn", "ddpg", "mpdqn"]

    results: Dict[str, Any] = {}

    for algo in algorithms:
        print(f"\n================ Running Benchmark: {algo.upper()} ================")
        algo_results = []
        skipped = False

        for seed in seeds:
            set_seed(seed)
            env = ORANEnv(cfg)
            obs, _ = env.reset(seed=seed)

            model: Any
            try:
                if algo == "dqn":
                    model = ORANDQNAgent(
                        state_dim=env.state_dim,
                        n_ru=env.n_ru,
                        n_splits=env.n_splits,
                        p_max_w=env.p_max_w,
                        config=cfg,
                    )
                elif algo == "ddpg":
                    model = ORANDDPGAgent(
                        state_dim=env.state_dim,
                        n_ru=env.n_ru,
                        n_splits=env.n_splits,
                        p_max_w=env.p_max_w,
                        config=cfg,
                    )
                elif algo == "mpdqn":
                    model = ORANMPDQNAgent(
                        state_dim=env.state_dim,
                        n_ru=env.n_ru,
                        n_splits=env.n_splits,
                        p_max_w=env.p_max_w,
                        config=cfg,
                    )
                else:
                    raise ValueError(f"Unknown algorithm: {algo}")
            except ValueError as exc:
                # MP-DQN's flat joint discrete action space is intractable
                # above MAX_N_RU_FOR_FLAT_JOINT_ORAN_ACTION -- reported as a
                # graceful skip, not left to crash the whole benchmark run
                # (mirrors training/train_baselines.py's identical pattern).
                if algo == "mpdqn":
                    print(
                        f"  n_ru={env.n_ru:3d} | {algo:6s} | SKIPPED (intractable): {exc}"
                    )
                    skipped = True
                    break
                raise

            batch_size = int(cfg.get("algorithm", {}).get("batch_size", 128))
            ep_rewards, ep_powers, ep_qos_rates = [], [], []
            ep_active_rus, ep_switching_events, ep_throughputs = [], [], []

            for ep in range(episodes):
                obs, _ = env.reset()
                total_reward = 0.0
                powers, qos_flags, actives, switches, throughputs = [], [], [], [], []

                done = False
                while not done:
                    # evaluate=False: exploration must stay on during
                    # training rollout -- evaluate=True here previously
                    # trained every baseline with exploration permanently
                    # disabled (only ever exploiting the randomly
                    # initialized network's greedy output).
                    action = model.select_action(obs, evaluate=False)
                    next_obs, reward, terminated, truncated, info = env.step(action)

                    if algo == "dqn":
                        model.memory.push(
                            obs,
                            action["ru_on"],
                            action["split"],
                            reward,
                            next_obs,
                            terminated,
                        )
                    elif algo == "ddpg":
                        cont = np.concatenate(
                            [action["power"] / env.p_max_w, action["prb"]]
                        )
                        model.memory.push(obs, cont, reward, next_obs, terminated)
                    elif algo == "mpdqn":
                        cont = np.stack(
                            [action["power"] / env.p_max_w, action["prb"]], axis=-1
                        )
                        model.memory.push(
                            obs,
                            model._last_action_idx,
                            cont,
                            reward,
                            next_obs,
                            terminated,
                        )
                    model.update(batch_size=batch_size)

                    total_reward += reward
                    powers.append(info.get("total_power_w", 0.0))
                    qos_flags.append(
                        1.0 if info.get("qos_violations_count", 0) == 0 else 0.0
                    )
                    actives.append(info.get("active_rus", 0))
                    switches.append(info.get("switching_events", 0))
                    throughputs.append(info.get("throughput_mbps", 0.0))

                    obs = next_obs
                    done = terminated or truncated

                if algo in ("dqn", "mpdqn"):
                    model.decay_exploration()

                ep_rewards.append(float(total_reward))
                ep_powers.append(float(np.mean(powers)))
                ep_qos_rates.append(float(np.mean(qos_flags)))
                ep_active_rus.append(float(np.mean(actives)))
                ep_switching_events.append(float(np.mean(switches)))
                ep_throughputs.append(float(np.mean(throughputs)))

            # Held-out deterministic evaluation, same treatment
            # oran_training/train_bmpp_dqn.py already gives the proposed
            # method -- the training-time ep_rewards/etc. above are not a
            # fair like-for-like comparison against final_eval_reward.
            eval_metrics = _evaluate_oran_baseline(
                env, model, eval_episodes=n_eval_episodes
            )

            seed_summary: Dict[str, Any] = {
                "algorithm": algo,
                "seed": seed,
                "mean_reward": eval_metrics["mean_reward"],
                "std_reward": eval_metrics["std_reward"],
                "mean_power_w": eval_metrics["mean_power_w"],
                "qos_satisfaction_rate": eval_metrics["qos_satisfaction_rate"],
                "mean_active_rus": eval_metrics["mean_active_rus"],
                "mean_switching_events": eval_metrics["mean_switching_events"],
                "mean_throughput_mbps": eval_metrics["mean_throughput_mbps"],
                "train_mean_reward": float(np.mean(ep_rewards)),
                "train_mean_power_w": float(np.mean(ep_powers)),
                "train_qos_satisfaction_rate": float(np.mean(ep_qos_rates)),
                "train_mean_active_rus": float(np.mean(ep_active_rus)),
                "train_mean_switching_events": float(np.mean(ep_switching_events)),
                "train_mean_throughput_mbps": float(np.mean(ep_throughputs)),
            }
            algo_results.append(seed_summary)

            qos_pct = seed_summary["qos_satisfaction_rate"] * 100
            print(
                f"Algo: {algo:6s} | Seed: {seed:4d} | "
                f"Reward: {seed_summary['mean_reward']:8.2f} | "
                f"Power: {seed_summary['mean_power_w']:6.1f}W | QoS: {qos_pct:5.1f}%"
            )

            del model, env
            gc.collect()

        if skipped:
            results[algo] = []
            continue

        results[algo] = algo_results

        out_path = Path(save_dir) / f"oran_benchmark_{algo}"
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "summary.json", "w") as f:
            json.dump(algo_results, f, indent=2)

        run_record = dict(cfg)
        run_record["_run"] = {
            "config_path": config_path,
            "algorithm": algo,
            "seeds": seeds,
            "episodes": episodes,
        }
        with open(out_path / "config.yaml", "w") as f:
            yaml.dump(run_record, f)

    return results


if __name__ == "__main__":
    run_oran_baseline_benchmarks()
