"""Shared deterministic evaluation loop, reused by training/train_hybrid.py's
evaluate_agent and by the CSI-robustness/generalization/scalability evaluation
harnesses (evaluation/csi_robustness.py, evaluation/generalization.py,
evaluation/scalability.py).

Lives under training/, not evaluation/, to avoid a circular import:
evaluation/__init__.py already imports from training/ (e.g. train_hybrid_agent),
so a module under evaluation/ imported by training/train_hybrid.py would deadlock
package initialization; training/ has no existing dependency on evaluation/, and
this preserves that direction.
"""

from typing import Any, Dict

import numpy as np

from cran_env import CRANEnv


def run_eval_episodes(
    env: CRANEnv, agent: Any, episodes: int = 5, seed: int = 1000
) -> Dict[str, float]:
    """Run `episodes` deterministic (evaluate=True) episodes on `env`, seeded
    `seed, seed+1, ..., seed+episodes-1`, and return aggregate metrics.

    `agent` may be any object exposing `select_action(obs, evaluate=True)`.
    """
    eval_rewards = []
    eval_powers = []
    eval_qos_rates = []
    eval_qos_violation_rates = []
    eval_active_rrhs = []
    eval_ee = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_power = []
        ep_qos = []
        ep_active = []
        ep_ee = []

        done = False
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_power.append(info.get("total_power_w", 0.0))
            ep_qos.append(1.0 if info.get("qos_violations_count", 0) == 0 else 0.0)
            ep_active.append(info.get("active_rrhs", 0))
            ep_ee.append(info.get("ee_mbit_per_joule", 0.0))
            done = terminated or truncated

        eval_rewards.append(ep_reward)
        eval_powers.append(float(np.mean(ep_power)))
        eval_qos_rates.append(float(np.mean(ep_qos)))
        eval_qos_violation_rates.append(1.0 - float(np.mean(ep_qos)))
        eval_active_rrhs.append(float(np.mean(ep_active)))
        eval_ee.append(float(np.mean(ep_ee)))

    return {
        "eval_mean_reward": float(np.mean(eval_rewards)),
        "eval_std_reward": float(np.std(eval_rewards)),
        "eval_mean_power_w": float(np.mean(eval_powers)),
        "eval_qos_satisfaction_rate": float(np.mean(eval_qos_rates)),
        "eval_qos_violation_rate": float(np.mean(eval_qos_violation_rates)),
        "eval_mean_active_rrhs": float(np.mean(eval_active_rrhs)),
        "eval_mean_ee_mbit_per_joule": float(np.mean(eval_ee)),
    }
