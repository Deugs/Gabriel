"""Training pipeline for Proposed Branching MP-DQN + TD3 Agent in 5G C-RAN Simulation."""

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

try:
    import wandb
except ImportError:
    wandb = None

from agents import BranchingMPDQN
from cran_env import CRANEnv


def apply_config_overrides(
    cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply flat {key: value} overrides to whichever top-level config section
    already defines that key, e.g. {"lr_actor": 0.0} -> cfg["algorithm"]["lr_actor"],
    {"beta_qos": 5.0} -> cfg["reward"]["beta_qos"].

    Returns a new config dict; does not mutate the input. Raises ValueError if
    an override key isn't found in any top-level section, since a silently
    ignored override (e.g. a typo) is exactly the failure mode this exists to
    prevent (see evaluation/ablation.py's variants, which previously defined
    such overrides but never applied them).
    """
    if not overrides:
        return cfg
    cfg = deepcopy(cfg)
    for key, value in overrides.items():
        for section in cfg.values():
            if isinstance(section, dict) and key in section:
                section[key] = value
                break
        else:
            raise ValueError(
                f"Config override key '{key}' not found in any top-level "
                "section of the config; check for a typo or a missing default."
            )
    return cfg


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
    env: CRANEnv, agent: BranchingMPDQN, eval_episodes: int = 5
) -> Dict[str, float]:
    """Evaluate current policy in deterministic mode over N episodes."""
    eval_rewards = []
    eval_powers = []
    eval_qos_rates = []
    eval_active_rrhs = []
    eval_switching_events = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        ep_reward = 0.0
        ep_power = []
        ep_qos = []
        ep_active = []
        ep_switch = []

        done = False
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_power.append(info.get("total_power_w", 0.0))
            ep_qos.append(1.0 if info.get("qos_violations_count", 0) == 0 else 0.0)
            ep_active.append(info.get("active_rrhs", 0))
            ep_switch.append(info.get("switching_events", 0))
            done = terminated or truncated

        eval_rewards.append(ep_reward)
        eval_powers.append(float(np.mean(ep_power)))
        eval_qos_rates.append(float(np.mean(ep_qos)))
        eval_active_rrhs.append(float(np.mean(ep_active)))
        eval_switching_events.append(float(np.mean(ep_switch)))

    return {
        "eval_mean_reward": float(np.mean(eval_rewards)),
        "eval_std_reward": float(np.std(eval_rewards)),
        "eval_mean_power_w": float(np.mean(eval_powers)),
        "eval_qos_satisfaction_rate": float(np.mean(eval_qos_rates)),
        "eval_mean_active_rrhs": float(np.mean(eval_active_rrhs)),
        "eval_mean_switching_events": float(np.mean(eval_switching_events)),
    }


def train_hybrid_agent(
    config_path: str,
    seed: int = 42,
    episodes: int = 100,
    eval_freq: Optional[int] = None,
    save_dir: Optional[str] = None,
    use_wandb: Optional[bool] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train Branching MP-DQN + TD3 agent and log training metrics.

    eval_freq=None (the default) reads config's evaluation.eval_freq; passing
    an explicit int always overrides the config, same pattern as use_wandb.
    """
    set_seed(seed)

    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_config_overrides(cfg, config_overrides)

    evaluation_cfg = cfg.get("evaluation", {})
    if eval_freq is None:
        eval_freq = int(evaluation_cfg.get("eval_freq", 10))
    n_eval_episodes = int(evaluation_cfg.get("n_eval_episodes", 5))
    save_checkpoints = bool(evaluation_cfg.get("save_checkpoints", True))
    checkpoint_freq = int(evaluation_cfg.get("checkpoint_freq", 500))

    logging_cfg = cfg.get("logging", {})
    log_freq = int(logging_cfg.get("log_freq", 10))
    use_wandb_explicit = use_wandb is not None
    if use_wandb is None:
        use_wandb = logging_cfg.get("use_wandb", False)
    if use_wandb and wandb is None:
        if use_wandb_explicit:
            raise RuntimeError(
                "use_wandb=True but the 'wandb' package is not installed. "
                "Install it with `pip install wandb` or pass --no-wandb."
            )
        # W&B was only requested via the config default, not explicitly by
        # the caller/CLI — don't let a missing optional package (e.g. in a
        # lightweight test/CI environment) crash training outright.
        print(
            "W&B logging requested by config/default.yaml but the 'wandb' "
            "package is not installed — continuing without it. Install it "
            "with `pip install wandb`, or pass --no-wandb to silence this."
        )
        use_wandb = False
    if use_wandb:
        if not os.environ.get("WANDB_API_KEY"):
            # Avoid hanging on an interactive login prompt in headless/remote
            # containers (e.g. docker compose run) when no API key is set.
            os.environ.setdefault("WANDB_MODE", "offline")
            print(
                "WANDB_API_KEY not set — running W&B in offline mode "
                "(run `wandb sync` on the run directory later to upload)."
            )
        wandb.init(
            project=logging_cfg.get("wandb_project", "cran-drl-thesis"),
            name=f"branching_mp_dqn_seed{seed}",
            config=cfg,
        )

    env = CRANEnv(cfg)
    validate_hardware_constants(env)

    agent = BranchingMPDQN(
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
        "switching_events": [],
        "critic_losses": [],
        "eval_history": [],
    }

    print(
        f"--- Starting Training Branching MP-DQN + TD3 | Seed: {seed} | Episodes: {episodes} ---"
    )

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_powers = []
        ep_qos = []
        ep_active = []
        ep_switch = []
        critic_loss_list = []

        done = False
        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, info = env.step(action)

            cont_action = action.get(
                "continuous",
                np.stack([action["power"] / env.p_max_w, action["bandwidth"]], axis=-1),
            )
            # Store transition in replay buffer
            agent.memory.push(
                obs,
                action["rrh_on"],
                cont_action,
                reward,
                next_obs,
                terminated,
            )

            # Optimization step
            metrics = agent.update(batch_size=batch_size)
            if metrics.get("critic_loss", 0.0) > 0.0:
                critic_loss_list.append(metrics["critic_loss"])

            ep_reward += reward
            ep_powers.append(info.get("total_power_w", 0.0))
            ep_qos.append(1.0 if info.get("qos_violations_count", 0) == 0 else 0.0)
            ep_active.append(info.get("active_rrhs", 0))
            ep_switch.append(info.get("switching_events", 0))

            obs = next_obs
            done = terminated or truncated

        # Log episode metrics
        mean_power = float(np.mean(ep_powers)) if ep_powers else 0.0
        qos_rate = float(np.mean(ep_qos)) if ep_qos else 0.0
        mean_active = float(np.mean(ep_active)) if ep_active else 0.0
        mean_switching = float(np.mean(ep_switch)) if ep_switch else 0.0
        mean_loss = float(np.mean(critic_loss_list)) if critic_loss_list else 0.0

        history["episode_rewards"].append(float(ep_reward))
        history["episode_powers"].append(mean_power)
        history["qos_rates"].append(qos_rate)
        history["active_rrhs"].append(mean_active)
        history["switching_events"].append(mean_switching)
        history["critic_losses"].append(mean_loss)

        if use_wandb:
            wandb.log(
                {
                    "episode": ep,
                    "train/episode_reward": float(ep_reward),
                    "train/mean_power_w": mean_power,
                    "train/qos_satisfaction_rate": qos_rate,
                    "train/mean_active_rrhs": mean_active,
                    "train/mean_switching_events": mean_switching,
                    "train/critic_loss": mean_loss,
                },
                step=ep,
            )

        # Anomaly checking
        if np.isnan(ep_reward) or np.isinf(ep_reward):
            raise RuntimeError(
                f"Numerical instability detected at episode {ep}: reward={ep_reward}"
            )

        # Evaluation phase
        is_eval_ep = ep % eval_freq == 0 or ep == episodes
        if is_eval_ep:
            eval_metrics = evaluate_agent(env, agent, eval_episodes=n_eval_episodes)
            eval_metrics["episode"] = ep
            history["eval_history"].append(eval_metrics)

            print(
                f"Ep {ep:4d}/{episodes} | Train Reward: {ep_reward:8.2f} | "
                f"Eval Reward: {eval_metrics['eval_mean_reward']:8.2f} | "
                f"Power: {eval_metrics['eval_mean_power_w']:6.1f}W | "
                f"QoS: {eval_metrics['eval_qos_satisfaction_rate']*100:5.1f}% | "
                f"Active RRHs: {eval_metrics['eval_mean_active_rrhs']:4.1f}/{env.n_rrh} | "
                f"Switches/step: {eval_metrics['eval_mean_switching_events']:4.2f}"
            )

            if use_wandb:
                wandb.log(
                    {f"eval/{k}": v for k, v in eval_metrics.items() if k != "episode"},
                    step=ep,
                )
        elif ep % log_freq == 0:
            # Lighter console heartbeat between the fuller eval-cadence prints
            # above (logging.log_freq).
            print(
                f"Ep {ep:4d}/{episodes} | Train Reward: {ep_reward:8.2f} | "
                f"Critic Loss: {mean_loss:.4f} | Epsilon: {agent.epsilon:.3f}"
            )

        # Intermediate checkpoint (evaluation.save_checkpoints/checkpoint_freq)
        # — distinct from the always-saved final_model.pt below, which is the
        # actual deliverable rather than a crash-recovery snapshot.
        if save_dir is not None and save_checkpoints and ep % checkpoint_freq == 0:
            ckpt_dir = Path(save_dir) / f"branching_mp_dqn_seed{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "episode": ep,
                    "encoder": agent.encoder.state_dict(),
                    "param_net": agent.param_net.state_dict(),
                    "twin_critic": agent.twin_critic.state_dict(),
                },
                ckpt_dir / f"checkpoint_ep{ep}.pt",
            )

        gc.collect()

    elapsed_time = time.time() - start_time

    # Save summary and model checkpoint if save_dir specified
    summary = {
        "algorithm": "Branching_MP_DQN",
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
        "final_switching_events": (
            history["eval_history"][-1]["eval_mean_switching_events"]
            if history["eval_history"]
            else 0.0
        ),
        "history": history,
    }

    if save_dir is not None:
        out_path = Path(save_dir) / f"branching_mp_dqn_seed{seed}"
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save the exact config used, plus the run-level args not captured in
        # cfg itself, so this checkpoint/summary is reproducible on its own
        # (Reproducibility Commitment, Concept Note v4.0 Section 12.10).
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
                "encoder": agent.encoder.state_dict(),
                "param_net": agent.param_net.state_dict(),
                "twin_critic": agent.twin_critic.state_dict(),
            },
            out_path / "final_model.pt",
        )
        print(f"Saved results and model checkpoint to {out_path}")

    if use_wandb:
        wandb.log(
            {
                "summary/final_train_reward": summary["final_train_reward"],
                "summary/final_eval_reward": summary["final_eval_reward"],
                "summary/final_eval_power_w": summary["final_eval_power_w"],
                "summary/final_qos_rate": summary["final_qos_rate"],
                "summary/final_switching_events": summary["final_switching_events"],
                "summary/total_training_time_sec": elapsed_time,
            }
        )
        wandb.finish()

    gc.collect()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Branching MP-DQN + TD3 Agent for C-RAN"
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Config file path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="Evaluation frequency (overrides config's evaluation.eval_freq if set)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="data/results", help="Directory to save results"
    )
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--use-wandb",
        dest="use_wandb",
        action="store_true",
        default=None,
        help="Force-enable W&B logging (overrides config/default.yaml's logging.use_wandb)",
    )
    wandb_group.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        default=None,
        help="Force-disable W&B logging (overrides config/default.yaml's logging.use_wandb)",
    )
    args = parser.parse_args()

    train_hybrid_agent(
        config_path=args.config,
        seed=args.seed,
        episodes=args.episodes,
        eval_freq=args.eval_freq,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
    )
