"""Power-Consumption-vs-Time-of-Day Evaluation (Concept Note v4.0 Section
12.3 — "comparable to Iqbal et al.'s Fig. 4").

Trains each method once, then rolls out many episodes under the frozen
policy, bucketing per-step power consumption by hour-of-day (0-23) to
produce a diurnal power profile per method — the time-domain counterpart to
the scalar power/EE summaries reported elsewhere.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml  # type: ignore[import-untyped]

from cran_env import CRANEnv
from evaluation.csi_robustness import _TRAINERS
from evaluation.plot_utils import plot_degradation_curve


def _evaluate_power_by_hour(
    env: CRANEnv, agent: Any, eval_episodes: int, seed_offset: int = 4000
) -> Dict[int, float]:
    """Roll out under the frozen policy, bucketing power by env.hour."""
    power_by_hour: Dict[int, List[float]] = {h: [] for h in range(24)}

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        while not done:
            hour = env.hour
            action = agent.select_action(obs, evaluate=True)
            obs, _, terminated, truncated, info = env.step(action)
            power_by_hour[hour].append(float(info.get("total_power_w", 0.0)))
            done = terminated or truncated

    return {
        hour: (float(np.mean(values)) if values else 0.0)
        for hour, values in power_by_hour.items()
    }


def run_power_time_profile_evaluation(
    config_path: str = "config/default.yaml",
    methods: Optional[List[str]] = None,
    train_episodes: int = 30,
    eval_episodes: int = 20,
    batch_size: int = 64,
    save_dir: str = "thesis/figures",
) -> Dict[str, Dict[int, float]]:
    """Train each method once, then report its power-vs-hour-of-day profile."""
    if methods is None:
        methods = ["branching_mp_dqn", "ddqn", "ddpg"]

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    results: Dict[str, Dict[int, float]] = {}

    for method in methods:
        if method not in _TRAINERS:
            raise ValueError(
                f"Unknown method '{method}'; expected one of {list(_TRAINERS)}"
            )

        print(f"\n--- Power/Time Profile: training {method} ---")
        env = CRANEnv(deepcopy(cfg))
        agent = _TRAINERS[method](env, cfg, train_episodes, batch_size)

        profile = _evaluate_power_by_hour(env, agent, eval_episodes)
        results[method] = profile
        values = list(profile.values())
        print(
            f"  {method}: mean power range {min(values):.1f}-{max(values):.1f}W across 24h"
        )

    power_curve = {m: results[m] for m in methods}

    save_path = Path(save_dir)
    plot_degradation_curve(
        power_curve,
        xlabel="Hour of day",
        ylabel="Mean Total Power (W)",
        title="Power Consumption vs. Time of Day",
        save_path=str(save_path / "power_time_profile.pdf"),
    )

    return results


if __name__ == "__main__":
    run_power_time_profile_evaluation(train_episodes=10, eval_episodes=5)
