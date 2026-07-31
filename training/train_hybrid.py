"""Training pipeline for Proposed Hybrid SAC-DDQN Agent in 5G C-RAN Simulation."""

import argparse
import gc
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

from agents import HybridSACDDQN
from cran_env import CRANEnv


def set_seed(seed: int):
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_hardware_constants(env: CRANEnv):
    """Verify EARTH hardware power model constants match validated thesis bounds."""
    earth_stat = env.power.p_stat
    earth_dyn = env.power.p_dyn
    assert (
        abs(earth_stat - 175.0) < 1e-3
    ), f"EARTH BBU static power mutated: {earth_stat} != 175.0 W"
    assert (
        abs(earth_dyn - 250.0) < 1e-3
    ), f"EARTH BBU dynamic power mutated: {earth_dyn} != 250.0 W"


def evaluate_agent(
    env: CRANEnv, agent: HybridSACDDQN, eval_episodes: int = 5
) -> Dict[str, float]:
    """Evaluate current policy in deterministic mode over N episodes."""
    eval_rewards = []
    eval_powers = []
    eval_qos_rates = []
    eval_active_rrhs = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        ep_reward = 0.0
        ep_power = []
        ep_qos = []
        ep_active = []

        done = False
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_power.append(info.get("power_metrics", {}).get("p_total_w", 0.0))
            ep_qos.append(1.0 if info.get("qos_satisfied", False) else 0.0)
            ep_active.append(info.get("active_rrhs", 0))
            done = terminated or truncated

        eval_rewards.append(ep_reward)
        eval_powers.append(float(np.mean(ep_power)))
        eval_qos_rates.append(float(np.mean(ep_qos)))
        eval_active_rrhs.append(float(np.mean(ep_active)))

    return {
        "eval_mean_reward": float(np.mean(eval_rewards)),
        "eval_std_reward": float(np.std(eval_rewards)),
        "eval_mean_power_w": float(np.mean(eval_powers)),
        "eval_qos_satisfaction_rate": float(np.mean(eval_qos_rates)),
        "eval_mean_active_rrhs": float(np.mean(eval_active_rrhs)),
    }


def train_hybrid_agent(
    config_path: str,
    seed: int = 42,
    episodes: int = 100,
    eval_freq: int = 10,
    save_dir: Optional[str] = None,
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """Train Hybrid SAC-DDQN agent and log training metrics."""
    set_seed(seed)

    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    env = CRANEnv(cfg)
    validate_hardware_constants(env)

    agent = HybridSACDDQN(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        config=cfg,
    )

    batch_size = cfg.get("algorithm", {}).get("batch_size", 256)
    start_time = time.time()

    history: Dict[str, Any] = {
        "episode_rewards": [],
        "episode_powers": [],
        "qos_rates": [],
        "active_rrhs": [],
        "critic_losses": [],
        "eval_history": [],
    }

    print(
        f"--- Starting Training Hybrid SAC-DDQN | Seed: {seed} | Episodes: {episodes} ---"
    )

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_powers = []
        ep_qos = []
        ep_active = []
        critic_loss_list = []

        done = False
        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store transition in replay buffer
            agent.memory.push(
                obs,
                action["rrh_on"],
                action["power"] / env.p_max_w,
                reward,
                next_obs,
                terminated,
            )

            # Optimization step
            metrics = agent.update(batch_size=batch_size)
            if metrics["critic_loss"] > 0.0:
                critic_loss_list.append(metrics["critic_loss"])

            ep_reward += reward
            ep_powers.append(info.get("power_metrics", {}).get("p_total_w", 0.0))
            ep_qos.append(1.0 if info.get("qos_satisfied", False) else 0.0)
            ep_active.append(info.get("active_rrhs", 0))

            obs = next_obs
            done = terminated or truncated

        # Log episode metrics
        mean_power = float(np.mean(ep_powers)) if ep_powers else 0.0
        qos_rate = float(np.mean(ep_qos)) if ep_qos else 0.0
        mean_active = float(np.mean(ep_active)) if ep_active else 0.0
        mean_loss = float(np.mean(critic_loss_list)) if critic_loss_list else 0.0

        history["episode_rewards"].append(float(ep_reward))
        history["episode_powers"].append(mean_power)
        history["qos_rates"].append(qos_rate)
        history["active_rrhs"].append(mean_active)
        history["critic_losses"].append(mean_loss)

        # Anomaly checking
        if np.isnan(ep_reward) or np.isinf(ep_reward):
            raise RuntimeError(
                f"Numerical instability detected at episode {ep}: reward={ep_reward}"
            )

        # Evaluation phase
        if ep % eval_freq == 0 or ep == episodes:
            eval_metrics = evaluate_agent(env, agent, eval_episodes=5)
            eval_metrics["episode"] = ep
            history["eval_history"].append(eval_metrics)

            print(
                f"Ep {ep:4d}/{episodes} | Train Reward: {ep_reward:8.2f} | "
                f"Eval Reward: {eval_metrics['eval_mean_reward']:8.2f} | "
                f"Power: {eval_metrics['eval_mean_power_w']:6.1f}W | "
                f"QoS: {eval_metrics['eval_qos_satisfaction_rate']*100:5.1f}% | "
                f"Active RRHs: {eval_metrics['eval_mean_active_rrhs']:4.1f}/{env.n_rrh}"
            )

    elapsed_time = time.time() - start_time

    # Save summary and model checkpoint if save_dir specified
    summary = {
        "algorithm": "Hybrid_SAC_DDQN",
        "seed": seed,
        "episodes": episodes,
        "total_training_time_sec": elapsed_time,
        "final_train_reward": history["episode_rewards"][-1],
        "final_eval_reward": (
            history["eval_history"][-1]["eval_mean_reward"]
            if history["eval_history"]
            else 0.0
        ),
        "final_eval_power_w": (
            history["eval_history"][-1]["eval_mean_power_w"]
            if history["eval_history"]
            else 0.0
        ),
        "final_qos_rate": (
            history["eval_history"][-1]["eval_qos_satisfaction_rate"]
            if history["eval_history"]
            else 0.0
        ),
        "history": history,
    }

    if save_dir is not None:
        out_path = Path(save_dir) / f"hybrid_sac_dqn_seed{seed}"
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        torch.save(
            {
                "discrete_actor": agent.discrete_actor.state_dict(),
                "continuous_actor": agent.continuous_actor.state_dict(),
                "critic": agent.critic.state_dict(),
            },
            out_path / "final_model.pt",
        )
        print(f"Saved results and model checkpoint to {out_path}")

    gc.collect()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Hybrid SAC-DDQN Agent for C-RAN"
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Config file path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument(
        "--eval-freq", type=int, default=10, help="Evaluation frequency"
    )
    parser.add_argument(
        "--save-dir", type=str, default="data/results", help="Directory to save results"
    )
    args = parser.parse_args()

    train_hybrid_agent(
        config_path=args.config,
        seed=args.seed,
        episodes=args.episodes,
        eval_freq=args.eval_freq,
        save_dir=args.save_dir,
    )
