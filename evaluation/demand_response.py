"""Energy-Efficiency / Power vs. User-Demand Evaluation (Concept Note v4.0
Section 12.3 — "comparable to Iqbal et al.'s Figs. 3 and 5").

Trains each method once under the default demand level, then evaluates the
frozen policy across a sweep of demand multipliers (holding the trained
policy fixed — no retraining per demand level, matching the evaluation-only
philosophy already used for CSI robustness), reporting energy efficiency
(Mbit/Joule) and mean power (W) as a function of user demand.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml  # type: ignore[import-untyped]

from cran_env import CRANEnv
from evaluation.csi_robustness import _TRAINERS
from evaluation.plot_utils import plot_degradation_curve

DEFAULT_DEMAND_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 2.5)


def _evaluate_under_demand_multiplier(
    base_cfg: Dict[str, Any],
    agent: Any,
    demand_multiplier: float,
    eval_episodes: int,
    seed_offset: int = 3000,
) -> Dict[str, float]:
    """Evaluate a frozen policy under a scaled traffic demand level."""
    cfg = deepcopy(base_cfg)
    traffic_cfg = cfg.setdefault("traffic", {})
    base_rate = float(traffic_cfg.get("base_rate_mbps", 50.0))
    traffic_cfg["base_rate_mbps"] = base_rate * demand_multiplier

    env = CRANEnv(cfg)
    ee_values: List[float] = []
    power_values: List[float] = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, _, terminated, truncated, info = env.step(action)
            ee_values.append(float(info.get("ee_mbit_per_joule", 0.0)))
            power_values.append(float(info.get("total_power_w", 0.0)))
            done = terminated or truncated

    return {
        "ee_mbit_per_joule": float(np.mean(ee_values)) if ee_values else 0.0,
        "mean_power_w": float(np.mean(power_values)) if power_values else 0.0,
    }


def run_demand_response_evaluation(
    config_path: str = "config/default.yaml",
    methods: Optional[List[str]] = None,
    demand_multipliers: Optional[List[float]] = None,
    train_episodes: int = 30,
    eval_episodes: int = 5,
    batch_size: int = 64,
    save_dir: str = "thesis/figures",
) -> Dict[str, Dict[float, Dict[str, float]]]:
    """Train each method once, then sweep demand multipliers on the frozen policy."""
    if methods is None:
        methods = ["branching_mp_dqn", "ddqn", "ddpg"]
    if demand_multipliers is None:
        demand_multipliers = list(DEFAULT_DEMAND_MULTIPLIERS)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    results: Dict[str, Dict[float, Dict[str, float]]] = {}

    for method in methods:
        if method not in _TRAINERS:
            raise ValueError(
                f"Unknown method '{method}'; expected one of {list(_TRAINERS)}"
            )

        print(f"\n--- Demand Response: training {method} under default demand ---")
        env = CRANEnv(deepcopy(cfg))
        agent = _TRAINERS[method](env, cfg, train_episodes, batch_size)

        results[method] = {}
        for mult in demand_multipliers:
            metrics = _evaluate_under_demand_multiplier(
                cfg, agent, mult, eval_episodes
            )
            results[method][mult] = metrics
            print(
                f"  demand x{mult:.2f} | EE={metrics['ee_mbit_per_joule']:.3f} Mbit/J | "
                f"Power={metrics['mean_power_w']:.1f}W"
            )

    ee_curve = {
        m: {d: results[m][d]["ee_mbit_per_joule"] for d in demand_multipliers}
        for m in methods
    }
    power_curve = {
        m: {d: results[m][d]["mean_power_w"] for d in demand_multipliers}
        for m in methods
    }

    save_path = Path(save_dir)
    plot_degradation_curve(
        ee_curve,
        xlabel="Demand multiplier (x base rate)",
        ylabel="Energy Efficiency (Mbit/Joule)",
        title="Energy Efficiency vs. User Demand",
        save_path=str(save_path / "demand_response_ee.pdf"),
    )
    plot_degradation_curve(
        power_curve,
        xlabel="Demand multiplier (x base rate)",
        ylabel="Mean Total Power (W)",
        title="Power Consumption vs. User Demand",
        save_path=str(save_path / "demand_response_power.pdf"),
    )

    return results


if __name__ == "__main__":
    run_demand_response_evaluation(train_episodes=10, eval_episodes=2)
