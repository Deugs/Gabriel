"""Cross-Profile Generalization Evaluation (Concept Note v3.0/v4.0 Section 12.3, A5).

The policy is trained on the weekday/urban tidal traffic profile (the
project's default) and evaluated, without retraining, on a weekend/suburban
profile variant (`cran_env.traffic_model.TrafficModel`'s "weekend_suburban"
profile: flatter daytime demand, later and lower residential peak). EE and
QoS-violation-rate degradation relative to the matched (weekday-trained,
weekday-evaluated) case is reported as a robustness indicator, mirroring the
CSI-robustness protocol in `evaluation/csi_robustness.py`.
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import yaml  # type: ignore[import-untyped]

from cran_env import CRANEnv
from evaluation.csi_robustness import _TRAINERS, _evaluate_under_csi_noise
from evaluation.plot_utils import plot_energy_efficiency_bar

import numpy as np


def run_generalization_evaluation(
    config_path: str = "config/default.yaml",
    methods: Optional[List[str]] = None,
    train_episodes: int = 30,
    eval_episodes: int = 5,
    batch_size: int = 64,
    seed: int = 42,
    save_dir: str = "thesis/figures",
    checkpoint_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Train on the default (weekday/urban) profile; evaluate on both profiles.

    checkpoint_paths, when given, maps a method name to an already-trained
    checkpoint file to load instead of training that method from scratch
    (Concept Note v4.0 Section 14's "reuse already-trained checkpoints"
    mitigation), the same as evaluation/csi_robustness.py's parameter of the
    same name. Only "branching_mp_dqn" currently supports this.
    """
    if methods is None:
        methods = ["branching_mp_dqn"]
    if checkpoint_paths is None:
        checkpoint_paths = {}

    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    rng = np.random.default_rng(seed)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for method in methods:
        if method not in _TRAINERS:
            raise ValueError(
                f"Unknown method '{method}'; expected one of {list(_TRAINERS)}"
            )

        weekday_cfg = deepcopy(base_cfg)
        weekday_cfg.setdefault("traffic", {})["profile"] = "weekday_urban"

        checkpoint_path = checkpoint_paths.get(method)
        if checkpoint_path is not None:
            print(f"\n--- Generalization: loading {method} from {checkpoint_path} ---")
        else:
            print(
                f"\n--- Generalization: training {method} on weekday_urban profile ---"
            )
        train_env = CRANEnv(deepcopy(weekday_cfg))
        agent = _TRAINERS[method](
            train_env,
            weekday_cfg,
            train_episodes,
            batch_size,
            checkpoint_path=checkpoint_path,
        )

        matched_env = CRANEnv(deepcopy(weekday_cfg))
        matched_metrics = _evaluate_under_csi_noise(
            matched_env, agent, sigma=0.0, eval_episodes=eval_episodes, rng=rng
        )

        weekend_cfg = deepcopy(base_cfg)
        weekend_cfg.setdefault("traffic", {})["profile"] = "weekend_suburban"
        weekend_env = CRANEnv(weekend_cfg)
        weekend_metrics = _evaluate_under_csi_noise(
            weekend_env, agent, sigma=0.0, eval_episodes=eval_episodes, rng=rng
        )

        results[method] = {
            "weekday_urban_matched": matched_metrics,
            "weekend_suburban_generalization": weekend_metrics,
        }

        ee_drop_pct = (
            100.0
            * (
                matched_metrics["ee_mbit_per_joule"]
                - weekend_metrics["ee_mbit_per_joule"]
            )
            / (matched_metrics["ee_mbit_per_joule"] + 1e-9)
        )
        print(
            f"  {method}: matched EE={matched_metrics['ee_mbit_per_joule']:.3f} Mbit/J, "
            f"generalization EE={weekend_metrics['ee_mbit_per_joule']:.3f} Mbit/J "
            f"({ee_drop_pct:+.1f}% change)"
        )

    save_path = Path(save_dir)
    bar_metrics = {}
    for method in methods:
        bar_metrics[f"{method} (weekday, matched)"] = {
            "mean": results[method]["weekday_urban_matched"]["ee_mbit_per_joule"],
            "std": 0.0,
        }
        bar_metrics[f"{method} (weekend, generalization)"] = {
            "mean": results[method]["weekend_suburban_generalization"][
                "ee_mbit_per_joule"
            ],
            "std": 0.0,
        }
    plot_energy_efficiency_bar(
        bar_metrics,
        title="Cross-Profile Generalization: Energy Efficiency",
        save_path=str(save_path / "generalization_ee.pdf"),
    )

    return results


if __name__ == "__main__":
    run_generalization_evaluation(train_episodes=10, eval_episodes=2)
