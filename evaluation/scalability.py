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
    """Evaluate power, execution time, QoS satisfaction and switching frequency
    across network topology scales -- the three-way energy/QoS/switching
    trade-off RQ5 (Section 6) asks about, not just the power/time half of it."""
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    # Matches docs/workflow.md's committed Experiment Matrix (R=5/12/20/35/50);
    # R=35's UE count isn't listed there (only R=5/12/20/50 are), so 25 is used
    # as an interpolation between the R=20/U=20 and R=50/U=30 rows.
    scales = {
        "R=5, U=2": {"n_rrh": 5, "n_ue": 2},
        "R=12, U=10": {"n_rrh": 12, "n_ue": 10},
        "R=20, U=20": {"n_rrh": 20, "n_ue": 20},
        "R=35, U=25": {"n_rrh": 35, "n_ue": 25},
        "R=50, U=30 (stretch)": {"n_rrh": 50, "n_ue": 30},
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
            "Branching_MP_DQN": {
                "power": float(res["final_eval_power_w"]),
                "time": float(step_time_ms),
                "qos_rate": float(res["final_qos_rate"]),
                "switching_events": float(res["final_switching_events"]),
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
