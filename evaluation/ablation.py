from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from evaluation.plot_utils import plot_ablation_comparison
from training.train_hybrid import train_hybrid_agent


def run_ablation_study(
    config_path: str = "config/default.yaml",
    seeds: Optional[List[int]] = None,
    episodes: int = 30,
    save_dir: str = "thesis/figures",
) -> Dict[str, float]:
    """Run ablation study evaluating component contributions."""
    if seeds is None:
        seeds = [42]

    variants = {
        "1. Full Proposed (Branching MP-DQN + TD3)": {},
        "2. No Continuous Parameter Tuning (Fixed Power)": {"lr_actor": 0.0},
        "3. No Discrete Branch Selection (All-ON)": {
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
        },
        "4. No Fronthaul Reward Term": {"beta_qos": 5.0},
    }

    results: Dict[str, float] = {}

    print("--- Starting Ablation Study ---")
    for name, override in variants.items():
        variant_rewards = []
        for seed in seeds:
            res = train_hybrid_agent(
                config_path=config_path,
                seed=seed,
                episodes=episodes,
                eval_freq=episodes,
                save_dir=None,
            )
            variant_rewards.append(res["final_eval_reward"])

        mean_reward = float(np.mean(variant_rewards))
        results[name] = mean_reward
        print(f"Variant: {name:45s} | Mean Eval Reward: {mean_reward:8.2f}")

    # Plot ablation chart
    fig_path = Path(save_dir) / "ablation_study.pdf"
    plot_ablation_comparison(results, save_path=str(fig_path))
    print(f"Saved ablation chart to {fig_path}")

    return results


if __name__ == "__main__":
    run_ablation_study(episodes=20)
