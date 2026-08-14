"""Reward-Weight Sensitivity Sweep (Concept Note v3.0/v4.0 Section 12.6, S5).

Section 12.6 sets lambda1 (beta_qos) via a violation-dominance criterion and
lambda2 (gamma_switch) so one switching event is penalized on the same order
of magnitude as the RRH switch power cost, then refines both "via a coarse
grid sweep (e.g., lambda2 in {0.01, 0.05, 0.1, 0.5, 1.0} at fixed lambda1)
reporting how EE, QoS-violation rate and switching frequency shift... the
final operating point is the one that meets the QoS target (Section 12.7)
at the lowest switching frequency."

This module implements that grid sweep. Unlike the CSI-robustness sweep
(evaluation/csi_robustness.py), which perturbs a *frozen* trained policy's
observations at evaluation time only, gamma_switch is a training-time reward
weight: a policy optimized under one gamma_switch value behaves differently
from one optimized under another, so each grid point requires training a
fresh agent from scratch under that reward weight, not just re-evaluating a
single trained policy.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml  # type: ignore[import-untyped]

from cran_env import CRANEnv
from evaluation.csi_robustness import _TRAINERS
from evaluation.plot_utils import plot_degradation_curve

DEFAULT_GAMMA_SWITCH_GRID = (0.01, 0.05, 0.1, 0.5, 1.0)


def _evaluate_operating_point(
    env: CRANEnv, agent: Any, eval_episodes: int, seed_offset: int = 5000
) -> Dict[str, float]:
    """Evaluate a trained (deterministic) policy: EE, QoS-violation rate, switching frequency."""
    ee_values: List[float] = []
    qos_violation_flags: List[float] = []
    switching_events: List[float] = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, _, terminated, truncated, info = env.step(action)
            ee_values.append(float(info.get("ee_mbit_per_joule", 0.0)))
            qos_violation_flags.append(
                1.0 if info.get("qos_violations_count", 0) > 0 else 0.0
            )
            switching_events.append(float(info.get("switching_events", 0)))
            done = terminated or truncated

    return {
        "ee_mbit_per_joule": float(np.mean(ee_values)) if ee_values else 0.0,
        "qos_violation_rate": (
            float(np.mean(qos_violation_flags)) if qos_violation_flags else 0.0
        ),
        "switching_frequency": (
            float(np.mean(switching_events)) if switching_events else 0.0
        ),
    }


def run_reward_sensitivity_sweep(
    config_path: str = "config/default.yaml",
    gamma_switch_grid: Optional[List[float]] = None,
    train_episodes: int = 30,
    eval_episodes: int = 5,
    batch_size: int = 64,
    save_dir: str = "thesis/figures",
) -> Dict[float, Dict[str, float]]:
    """Sweep lambda2 (reward.gamma_switch) at fixed lambda1 (reward.beta_qos).

    Returns {gamma_switch_value: {"ee_mbit_per_joule", "qos_violation_rate",
    "switching_frequency"}}.
    """
    if gamma_switch_grid is None:
        gamma_switch_grid = list(DEFAULT_GAMMA_SWITCH_GRID)

    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    fixed_beta_qos = base_cfg.get("reward", {}).get("beta_qos", 10.0)
    train_fn = _TRAINERS["branching_mp_dqn"]

    results: Dict[float, Dict[str, float]] = {}
    print(
        "--- Reward-Weight Sensitivity Sweep (Section 12.6): lambda2 "
        f"(gamma_switch) grid at fixed lambda1 (beta_qos)={fixed_beta_qos} ---"
    )
    for gamma_switch in gamma_switch_grid:
        cfg = deepcopy(base_cfg)
        cfg.setdefault("reward", {})["gamma_switch"] = gamma_switch

        env = CRANEnv(cfg)
        agent = train_fn(env, cfg, train_episodes, batch_size)
        metrics = _evaluate_operating_point(env, agent, eval_episodes)
        results[gamma_switch] = metrics

        print(
            f"  gamma_switch={gamma_switch:.3f} | "
            f"EE={metrics['ee_mbit_per_joule']:.3f} Mbit/J | "
            f"QoS violation rate={metrics['qos_violation_rate']*100:.1f}% | "
            f"Switching freq={metrics['switching_frequency']:.3f} events/step"
        )

    series_name = "Hybrid MP-DQN + TD3"
    ee_curve = {
        series_name: {g: results[g]["ee_mbit_per_joule"] for g in gamma_switch_grid}
    }
    qos_curve = {
        series_name: {g: results[g]["qos_violation_rate"] for g in gamma_switch_grid}
    }
    switch_curve = {
        series_name: {g: results[g]["switching_frequency"] for g in gamma_switch_grid}
    }

    save_path = Path(save_dir)
    plot_degradation_curve(
        ee_curve,
        xlabel="lambda2 (gamma_switch)",
        ylabel="Energy Efficiency (Mbit/Joule)",
        title="Reward-Weight Sensitivity: Energy Efficiency vs. Switching-Cost Weight",
        save_path=str(save_path / "reward_sensitivity_ee.pdf"),
    )
    plot_degradation_curve(
        qos_curve,
        xlabel="lambda2 (gamma_switch)",
        ylabel="QoS Violation Rate",
        title="Reward-Weight Sensitivity: QoS Violation Rate vs. Switching-Cost Weight",
        save_path=str(save_path / "reward_sensitivity_qos.pdf"),
    )
    plot_degradation_curve(
        switch_curve,
        xlabel="lambda2 (gamma_switch)",
        ylabel="RRH Switching Frequency (events/step)",
        title="Reward-Weight Sensitivity: Switching Frequency vs. Switching-Cost Weight",
        save_path=str(save_path / "reward_sensitivity_switching.pdf"),
    )

    return results


if __name__ == "__main__":
    run_reward_sensitivity_sweep(train_episodes=10, eval_episodes=2)
