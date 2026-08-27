"""Training loop for the proposed BMPP-DQN agent (O-RAN track).

Mirrors training/train_hybrid.py's single-seed entrypoint signature and
summary.json schema conventions, adapted for BMPP-DQN's two-timescale
update structure. No W&B integration (out of this track's scope,
ORAN_BMPP_DQN_Concept_Note_v1.md Section 5.3: "single-GPU setup", no
logging-service requirement). Zero imports from training/ or agents/.
"""

import gc
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

from oran_agents.bmpp_dqn import BMPPDQNAgent
from oran_env import ORANEnv


def set_seed(seed: int):
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_agent(
    env: ORANEnv, agent: BMPPDQNAgent, eval_episodes: int = 5
) -> Dict[str, float]:
    """Evaluate current policy in deterministic mode over N episodes."""
    eval_rewards = []
    eval_powers = []
    eval_qos_rates = []
    eval_active_rus = []
    eval_switching_events = []
    eval_throughputs = []

    for ep in range(eval_episodes):
        agent.reset_decision_cadence()
        obs, _ = env.reset(seed=1000 + ep)
        ep_reward = 0.0
        ep_power, ep_qos, ep_active, ep_switch, ep_throughput = [], [], [], [], []

        done = False
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_power.append(info.get("total_power_w", 0.0))
            ep_qos.append(1.0 if info.get("qos_violations_count", 0) == 0 else 0.0)
            ep_active.append(info.get("active_rus", 0))
            ep_switch.append(info.get("switching_events", 0))
            ep_throughput.append(info.get("throughput_mbps", 0.0))
            done = terminated or truncated

        eval_rewards.append(ep_reward)
        eval_powers.append(float(np.mean(ep_power)))
        eval_qos_rates.append(float(np.mean(ep_qos)))
        eval_active_rus.append(float(np.mean(ep_active)))
        eval_switching_events.append(float(np.mean(ep_switch)))
        eval_throughputs.append(float(np.mean(ep_throughput)))

    return {
        "eval_mean_reward": float(np.mean(eval_rewards)),
        "eval_std_reward": float(np.std(eval_rewards)),
        "eval_mean_power_w": float(np.mean(eval_powers)),
        "eval_qos_satisfaction_rate": float(np.mean(eval_qos_rates)),
        "eval_mean_active_rus": float(np.mean(eval_active_rus)),
        "eval_mean_switching_events": float(np.mean(eval_switching_events)),
        "eval_mean_throughput_mbps": float(np.mean(eval_throughputs)),
    }


def train_bmpp_dqn_agent(
    config_path: str,
    seed: int = 42,
    episodes: int = 100,
    eval_freq: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Train the BMPP-DQN agent and return/save a summary dict.

    eval_freq=None (the default) reads config's evaluation.eval_freq, same
    convention as training/train_hybrid.py.
    """
    set_seed(seed)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    evaluation_cfg = cfg.get("evaluation", {})
    if eval_freq is None:
        eval_freq = int(evaluation_cfg.get("eval_freq", 50))
    n_eval_episodes = int(evaluation_cfg.get("n_eval_episodes", 5))
    save_checkpoints = bool(evaluation_cfg.get("save_checkpoints", True))
    checkpoint_freq = int(evaluation_cfg.get("checkpoint_freq", 250))

    max_episodes = cfg.get("algorithm", {}).get("max_episodes")
    if max_episodes is not None and episodes > int(max_episodes):
        raise ValueError(
            f"episodes={episodes} exceeds algorithm.max_episodes={max_episodes} "
            f"in {config_path}"
        )

    env = ORANEnv(cfg)
    agent = BMPPDQNAgent(
        state_dim=env.state_dim,
        n_ru=env.n_ru,
        n_splits=env.n_splits,
        p_max_w=env.p_max_w,
        config=cfg,
    )

    batch_size = cfg.get("algorithm", {}).get("batch_size", 128)
    start_time = time.time()

    history: Dict[str, Any] = {
        "episode_rewards": [],
        "episode_powers": [],
        "qos_rates": [],
        "active_rus": [],
        "switching_events": [],
        "param_losses": [],
        "critic_losses": [],
        "eval_history": [],
    }

    print(f"--- Starting Training BMPP-DQN | Seed: {seed} | Episodes: {episodes} ---")

    for ep in range(1, episodes + 1):
        agent.reset_decision_cadence()
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_powers, ep_qos, ep_active, ep_switch = [], [], [], []
        param_loss_list, critic_loss_list = [], []

        done = False
        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.remember(obs, action, reward, next_obs, terminated)

            lower_metrics = agent.update_lower(batch_size=batch_size)
            upper_metrics = agent.update_upper(batch_size=batch_size)
            if lower_metrics.get("param_loss", 0.0) != 0.0:
                param_loss_list.append(lower_metrics["param_loss"])
            if upper_metrics.get("critic_loss", 0.0) != 0.0:
                critic_loss_list.append(upper_metrics["critic_loss"])

            ep_reward += reward
            ep_powers.append(info.get("total_power_w", 0.0))
            ep_qos.append(1.0 if info.get("qos_violations_count", 0) == 0 else 0.0)
            ep_active.append(info.get("active_rus", 0))
            ep_switch.append(info.get("switching_events", 0))

            obs = next_obs
            done = terminated or truncated

        agent.decay_exploration()

        mean_power = float(np.mean(ep_powers)) if ep_powers else 0.0
        qos_rate = float(np.mean(ep_qos)) if ep_qos else 0.0
        mean_active = float(np.mean(ep_active)) if ep_active else 0.0
        mean_switching = float(np.mean(ep_switch)) if ep_switch else 0.0
        mean_param_loss = float(np.mean(param_loss_list)) if param_loss_list else 0.0
        mean_critic_loss = float(np.mean(critic_loss_list)) if critic_loss_list else 0.0

        history["episode_rewards"].append(float(ep_reward))
        history["episode_powers"].append(mean_power)
        history["qos_rates"].append(qos_rate)
        history["active_rus"].append(mean_active)
        history["switching_events"].append(mean_switching)
        history["param_losses"].append(mean_param_loss)
        history["critic_losses"].append(mean_critic_loss)

        if np.isnan(ep_reward) or np.isinf(ep_reward):
            raise RuntimeError(
                f"Numerical instability detected at episode {ep}: reward={ep_reward}"
            )

        is_eval_ep = ep % eval_freq == 0 or ep == episodes
        if is_eval_ep:
            eval_metrics = evaluate_agent(env, agent, eval_episodes=n_eval_episodes)
            eval_metrics["episode"] = ep
            history["eval_history"].append(eval_metrics)
            agent.reset_decision_cadence()  # resume training cleanly after eval

            print(
                f"Ep {ep:4d}/{episodes} | Train Reward: {ep_reward:8.2f} | "
                f"Eval Reward: {eval_metrics['eval_mean_reward']:8.2f} | "
                f"Power: {eval_metrics['eval_mean_power_w']:6.1f}W | "
                f"QoS: {eval_metrics['eval_qos_satisfaction_rate']*100:5.1f}% | "
                f"Active RUs: {eval_metrics['eval_mean_active_rus']:4.1f}/{env.n_ru}"
            )

        if save_dir is not None and save_checkpoints and ep % checkpoint_freq == 0:
            ckpt_dir = Path(save_dir) / f"bmpp_dqn_seed{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "episode": ep,
                    "upper_encoder": agent.upper_encoder.state_dict(),
                    "lower_encoder": agent.lower_encoder.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "param_net": agent.param_net.state_dict(),
                },
                ckpt_dir / f"checkpoint_ep{ep}.pt",
            )

        gc.collect()

    elapsed_time = time.time() - start_time

    last_eval = history["eval_history"][-1] if history["eval_history"] else {}
    summary = {
        "algorithm": "BMPP_DQN",
        "seed": seed,
        "episodes": episodes,
        "total_training_time_sec": elapsed_time,
        "final_train_reward": history["episode_rewards"][-1],
        "final_eval_reward": last_eval.get("eval_mean_reward", 0.0),
        "final_eval_power_w": last_eval.get("eval_mean_power_w", 0.0),
        "final_qos_rate": last_eval.get("eval_qos_satisfaction_rate", 0.0),
        "final_switching_events": last_eval.get("eval_mean_switching_events", 0.0),
        "final_eval_throughput_mbps": last_eval.get("eval_mean_throughput_mbps", 0.0),
        "final_upper_level_decisions": episodes
        * env.max_steps
        // agent.upper_level_period_steps,
        "history": history,
    }

    if save_dir is not None:
        out_path = Path(save_dir) / f"bmpp_dqn_seed{seed}"
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        run_record = dict(cfg)
        run_record["_run"] = {
            "config_path": config_path,
            "seed": seed,
            "episodes": episodes,
            "eval_freq": eval_freq,
        }
        with open(out_path / "config.yaml", "w") as f:
            yaml.dump(run_record, f)

        torch.save(
            {
                "upper_encoder": agent.upper_encoder.state_dict(),
                "lower_encoder": agent.lower_encoder.state_dict(),
                "critic": agent.critic.state_dict(),
                "param_net": agent.param_net.state_dict(),
            },
            out_path / "final_model.pt",
        )
        print(f"Saved results and model checkpoint to {out_path}")

    return summary


if __name__ == "__main__":
    train_bmpp_dqn_agent(
        config_path="config/oran_default.yaml",
        seed=42,
        episodes=10,
        save_dir="data/results_oran",
    )
