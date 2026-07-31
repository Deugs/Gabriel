from copy import deepcopy
from pathlib import Path
import time
from typing import Dict

import yaml  # type: ignore[import-untyped]

from evaluation.plot_utils import plot_scalability_analysis
from training.train_hybrid import train_hybrid_agent


def analyze_scalability(
    config_path: str = "config/default.yaml",
    episodes: int = 20,
    save_dir: str = "thesis/figures",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate network performance and execution time across network topology scales."""
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    scales = {
        "Small (6x5)": {"n_rrh": 6, "n_ue": 5},
        "Medium (12x10)": {"n_rrh": 12, "n_ue": 10},
        "Large (24x20)": {"n_rrh": 24, "n_ue": 20},
    }

    scalability_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("--- Starting Scalability Analysis ---")

    for scale_name, params in scales.items():
        print(f"\nEvaluating Scale: {scale_name}")
        trial_cfg = deepcopy(base_cfg)

        if "network" not in trial_cfg:
            trial_cfg["network"] = {}

        trial_cfg["network"]["n_rrh"] = params["n_rrh"]
        trial_cfg["network"]["n_ue"] = params["n_ue"]

        temp_cfg = save_path / f"temp_{params['n_rrh']}x{params['n_ue']}.yaml"
        with open(temp_cfg, "w") as f:
            yaml.dump(trial_cfg, f)

        t_start = time.time()
        res = train_hybrid_agent(
            config_path=str(temp_cfg),
            seed=42,
            episodes=episodes,
            eval_freq=episodes,
            save_dir=None,
        )
        t_elapsed = time.time() - t_start
        step_time_ms = (t_elapsed / (episodes * 24)) * 1000.0  # 24 steps per episode

        scalability_results[scale_name] = {
            "Hybrid_SAC_DDQN": {
                "power": float(res["final_eval_power_w"]),
                "time": float(step_time_ms),
            }
        }

        if temp_cfg.exists():
            temp_cfg.unlink()

    # Plot scalability chart
    plot_scalability_analysis(
        scalability_results, save_path=str(save_path / "scalability_analysis.pdf")
    )
    print(
        f"Saved scalability analysis figure to {save_path / 'scalability_analysis.pdf'}"
    )

    return scalability_results


if __name__ == "__main__":
    analyze_scalability(episodes=10)
