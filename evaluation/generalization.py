"""Cross-traffic-profile generalization evaluation (Concept Note v4.0 Section
12.3 / A5): evaluate an already-trained, frozen policy on a traffic profile it
was never trained on, with no retraining, to test robustness to a demand
pattern the policy has never seen.
"""

import copy
from typing import Any, Dict, Union

from cran_env import CRANEnv
from training.eval_utils import run_eval_episodes


def _with_traffic_profile(env_config: Union[dict, Any], profile: str) -> dict:
    """Deep-copy `env_config` and override traffic.profile in-memory only --
    never touching the on-disk YAML, mirroring how observation_noise_std is a
    runtime kwarg rather than a persisted config field (evaluation/csi_robustness.py)."""
    cfg = copy.deepcopy(env_config)
    if not isinstance(cfg, dict):
        raise TypeError(
            "evaluate_generalization expects env_config as a plain dict "
            "(load the YAML with yaml.safe_load before calling)."
        )
    cfg.setdefault("traffic", {})
    cfg["traffic"]["profile"] = profile
    return cfg


def evaluate_generalization(
    agent: Any,
    env_config: Union[dict, Any],
    train_profile: str = "weekday_urban",
    eval_profile: str = "weekend_suburban",
    episodes: int = 10,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a frozen `agent` on both its matched and a generalized traffic
    profile, reporting the degradation between them.

    Returns:
        {"matched": {...}, "generalized": {...}, "degradation": {
            "ee_mbit_per_joule_drop": float,
            "ee_mbit_per_joule_pct_drop": float,
            "qos_violation_rate_increase": float,
        }}
    """
    matched_env = CRANEnv(_with_traffic_profile(env_config, train_profile))
    generalized_env = CRANEnv(_with_traffic_profile(env_config, eval_profile))

    matched = run_eval_episodes(matched_env, agent, episodes=episodes, seed=seed)
    generalized = run_eval_episodes(
        generalized_env, agent, episodes=episodes, seed=seed
    )

    ee_drop = matched["eval_mean_ee_mbit_per_joule"] - generalized[
        "eval_mean_ee_mbit_per_joule"
    ]
    ee_pct_drop = (
        100.0 * ee_drop / matched["eval_mean_ee_mbit_per_joule"]
        if matched["eval_mean_ee_mbit_per_joule"] != 0.0
        else 0.0
    )
    qos_increase = (
        generalized["eval_qos_violation_rate"] - matched["eval_qos_violation_rate"]
    )

    return {
        "matched": matched,
        "generalized": generalized,
        "degradation": {
            "ee_mbit_per_joule_drop": ee_drop,
            "ee_mbit_per_joule_pct_drop": ee_pct_drop,
            "qos_violation_rate_increase": qos_increase,
        },
    }
