"""Power-Model Constant Sensitivity Analysis (O-RAN / BMPP-DQN track).

Operationalizes ORAN_BMPP_DQN_Concept_Note_v1.md Section 6.3's own
"Mitigation" paragraph: oran_env/power_model.py's RU/DU/CU/fronthaul power
constants are literature-style placeholders, explicitly flagged "needs
validation" after nine independent literature-verification passes that
found no source giving a matching split-level, component-decomposed
wattage table at this model's small-cell scale (see power_model.py's own
docstring, and docs/oran_thesis_guide.md's "Needs-Validation Flags"
checklist). Since no such source exists to validate the constants
directly -- Abubakar et al. 2023's own survey conclusion states this is a
field-wide gap in the literature at large, not a failure of this
repository's search -- this module instead tests something weaker and
more defensible: whether Chapter 4's headline comparison (BMPP-DQN placing
last among all four methods on every metric, Concept Note Section 6.4) is
an artifact of any single unvalidated power-model constant, by perturbing
each constant group within the ranges the literature review itself
brackets and re-running the full 4-method comparison at each perturbation
point.

This does NOT validate the power model's absolute values. It validates
whether the *comparative* ranking among BMPP-DQN/DQN/DDPG/MP-DQN is robust
to plausible error in these constants -- the same logic
evaluation/reward_sensitivity.py already applies to a training-time reward
weight, applied here to power-model constants instead. Each perturbation
scenario requires training all four methods from scratch (a policy
optimized under one power model behaves differently from one optimized
under another), not just re-evaluating already-trained policies.

Zero imports from cran_env/agents/training/evaluation (this track's own
decoupling guarantee, mirrored from oran_env/*.py's own docstrings).
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

from oran_agents import BMPPDQNAgent, ORANDDPGAgent, ORANDQNAgent, ORANMPDQNAgent
from oran_env import ORANEnv
from oran_evaluation.plot_utils import plot_bar_comparison

ALGORITHMS = ("bmpp_dqn", "dqn", "ddpg", "mpdqn")
ALGO_LABELS = {"bmpp_dqn": "BMPP-DQN", "dqn": "DQN", "ddpg": "DDPG", "mpdqn": "MP-DQN"}


def _mk(
    multiply: Optional[Dict[str, float]] = None, set_: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {"multiply": multiply or {}, "set": set_ or {}}


# ---------------------------------------------------------------------------
# Perturbation scenarios. Each entry perturbs one or more dotted
# "section.key" leaves of config/oran_default.yaml's `power:` section,
# either by a multiplicative `factor` (applied element-wise to list-valued
# constants) or a direct `set` value. Ranges are drawn directly from the
# specific literature findings disclosed in oran_env/power_model.py's
# docstring -- not arbitrary round numbers -- as annotated below.
# ---------------------------------------------------------------------------
PERTURBATION_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "baseline": _mk(),
    # RU-side scale mismatch: Al-Tahmeesschi et al. 2025's real, component-
    # decomposed Split-8 RU measurement (~43-45 W) sits ~4x above this
    # model's own composite c=2 RU estimate (~11 W) -- the closest real
    # anchor found across all nine literature passes. 5x moves past it.
    "ru_power_high_5x": _mk(
        multiply={"ru.p_proc_by_split_w": 5.0, "ru.p_sleep_w": 5.0}
    ),
    "ru_power_low_0.5x": _mk(
        multiply={"ru.p_proc_by_split_w": 0.5, "ru.p_sleep_w": 0.5}
    ),
    # DU/CU scale mismatch: real enterprise-class O-DU/O-CU compute-host
    # power (Rutgers/ONF/ORCID white paper: ~280-310 W DU, ~230 W CU) runs
    # roughly 10-50x above this model's own placeholder scale (~50-130 W
    # DU, ~30-34 W CU). 5x is a moderate move within that disclosed range.
    "du_cu_power_high_5x": _mk(
        multiply={
            "du.p_static_w": 5.0,
            "du.p_per_ru_by_split_w": 5.0,
            "cu.p_static_w": 5.0,
            "cu.p_dyn_per_ru_w": 5.0,
        }
    ),
    # Fronthaul under-weighting: Lopez-Perez et al. (cited via Abubakar et
    # al. 2023) report fronthaul at ~60% of total C-RAN power for the
    # most-centralized split, vs. this model's own implied ~18% at c=2 --
    # a >3x gap in fronthaul-power *share*. 4x directly targets that
    # specific, quantified disclosed gap.
    "fronthaul_power_high_4x": _mk(
        multiply={"fronthaul.p_common_w": 4.0, "fronthaul.p_per_ru_by_split_w": 4.0}
    ),
    # PA efficiency IS already validated (three independent real sources:
    # 0.29-0.39, 0.14-0.32, 0.35, all bracketing this model's own
    # eta=0.25) -- swept anyway to the bracket's own low/high ends, as a
    # check that even a validated constant's remaining uncertainty doesn't
    # flip the ranking.
    "pa_efficiency_low_0.14": _mk(set_={"ru.pa_efficiency": 0.14}),
    "pa_efficiency_high_0.39": _mk(set_={"ru.pa_efficiency": 0.39}),
    # Omnibus stress test: every RU/DU/CU/fronthaul constant simultaneously
    # at the upper end of the disclosed 10-50x real-hardware scale
    # mismatch -- the most aggressive single perturbation tested.
    "order_of_magnitude_high_10x": _mk(
        multiply={
            "ru.p_proc_by_split_w": 10.0,
            "ru.p_sleep_w": 10.0,
            "du.p_static_w": 10.0,
            "du.p_per_ru_by_split_w": 10.0,
            "cu.p_static_w": 10.0,
            "cu.p_dyn_per_ru_w": 10.0,
            "fronthaul.p_common_w": 10.0,
            "fronthaul.p_per_ru_by_split_w": 10.0,
        }
    ),
}


def apply_scenario(
    base_cfg: Dict[str, Any], scenario: Dict[str, Any]
) -> Dict[str, Any]:
    """Return a deep-copied config with one scenario applied to `power:`."""
    cfg = deepcopy(base_cfg)
    power_cfg = cfg.setdefault("power", {})

    for dotted_key, factor in scenario.get("multiply", {}).items():
        section, key = dotted_key.split(".")
        sec = power_cfg.setdefault(section, {})
        val = sec.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            sec[key] = [v * factor for v in val]
        else:
            sec[key] = val * factor

    for dotted_key, value in scenario.get("set", {}).items():
        section, key = dotted_key.split(".")
        power_cfg.setdefault(section, {})[key] = value

    return cfg


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate(
    env: ORANEnv, model: Any, eval_episodes: int, seed_offset: int = 9000
) -> Dict[str, float]:
    """Deterministic held-out evaluation, shared across all 4 agent types
    (each exposes select_action(obs, evaluate=True)); mirrors
    oran_training/train_bmpp_dqn.py's evaluate_agent()."""
    rewards: List[float] = []
    powers: List[float] = []
    qos: List[float] = []
    for ep in range(eval_episodes):
        if hasattr(model, "reset_decision_cadence"):
            model.reset_decision_cadence()
        obs, _ = env.reset(seed=seed_offset + ep)
        total_reward = 0.0
        ep_powers: List[float] = []
        ep_qos: List[float] = []
        done = False
        while not done:
            action = model.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            ep_powers.append(info.get("total_power_w", 0.0))
            ep_qos.append(1.0 if info.get("qos_violations_count", 0) == 0 else 0.0)
            done = terminated or truncated
        rewards.append(total_reward)
        powers.append(float(np.mean(ep_powers)) if ep_powers else 0.0)
        qos.append(float(np.mean(ep_qos)) if ep_qos else 0.0)
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_power_w": float(np.mean(powers)),
        "qos_satisfaction_rate": float(np.mean(qos)),
    }


def _train_bmpp_dqn(
    cfg: Dict[str, Any], seed: int, train_episodes: int, eval_episodes: int
) -> Dict[str, float]:
    """Lightweight from-scratch BMPP-DQN trainer, adapted from
    oran_training/train_bmpp_dqn.py's loop but stripped of checkpointing/
    history logging (this sweep only needs the final held-out eval)."""
    _set_seed(seed)
    env = ORANEnv(cfg)
    agent = BMPPDQNAgent(
        state_dim=env.state_dim,
        n_ru=env.n_ru,
        n_splits=env.n_splits,
        p_max_w=env.p_max_w,
        config=cfg,
    )
    batch_size = int(cfg.get("algorithm", {}).get("batch_size", 128))

    for _ in range(train_episodes):
        agent.reset_decision_cadence()
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            agent.remember(obs, action, reward, next_obs, terminated)
            agent.update_lower(batch_size=batch_size)
            agent.update_upper(batch_size=batch_size)
            obs = next_obs
            done = terminated or truncated
        agent.decay_exploration()

    agent.reset_decision_cadence()
    return _evaluate(env, agent, eval_episodes)


def _train_baseline(
    algo: str, cfg: Dict[str, Any], seed: int, train_episodes: int, eval_episodes: int
) -> Dict[str, float]:
    """Lightweight from-scratch baseline trainer, adapted from
    oran_training/train_oran_baselines.py's loop."""
    _set_seed(seed)
    env = ORANEnv(cfg)

    if algo == "dqn":
        model: Any = ORANDQNAgent(
            state_dim=env.state_dim, n_ru=env.n_ru, n_splits=env.n_splits,
            p_max_w=env.p_max_w, config=cfg,
        )
    elif algo == "ddpg":
        model = ORANDDPGAgent(
            state_dim=env.state_dim, n_ru=env.n_ru, n_splits=env.n_splits,
            p_max_w=env.p_max_w, config=cfg,
        )
    elif algo == "mpdqn":
        # MP-DQN's flat joint discrete action space is intractable above
        # config's algorithm.max_n_ru_for_flat_joint_action -- reported as
        # a graceful skip (raised as ValueError by the constructor),
        # mirroring oran_training/train_oran_baselines.py's own handling.
        model = ORANMPDQNAgent(
            state_dim=env.state_dim, n_ru=env.n_ru, n_splits=env.n_splits,
            p_max_w=env.p_max_w, config=cfg,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    batch_size = int(cfg.get("algorithm", {}).get("batch_size", 128))

    for _ in range(train_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = model.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _info = env.step(action)

            if algo == "dqn":
                model.memory.push(
                    obs, action["ru_on"], action["split"], reward, next_obs, terminated
                )
            elif algo == "ddpg":
                cont = np.concatenate([action["power"] / env.p_max_w, action["prb"]])
                model.memory.push(obs, cont, reward, next_obs, terminated)
            elif algo == "mpdqn":
                cont = np.stack([action["power"] / env.p_max_w, action["prb"]], axis=-1)
                model.memory.push(
                    obs, model._last_action_idx, cont, reward, next_obs, terminated
                )
            model.update(batch_size=batch_size)

            obs = next_obs
            done = terminated or truncated

        if algo in ("dqn", "mpdqn"):
            model.decay_exploration()

    return _evaluate(env, model, eval_episodes)


def run_power_sensitivity_analysis(
    config_path: str = "config/oran_default.yaml",
    scenarios: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    train_episodes: int = 100,
    eval_episodes: int = 5,
    save_dir: str = "thesis/figures_oran",
    results_dir: str = "data/results_oran/power_sensitivity",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run the full {scenario x algorithm x seed} sweep and report, per
    scenario, whether the reward-based ranking among the 4 methods matches
    the unperturbed baseline's ranking.

    Returns {scenario_name: {algo: {"mean_reward", "mean_power_w",
    "qos_satisfaction_rate"}}}, each averaged over `seeds`.
    """
    if scenarios is None:
        scenarios = list(PERTURBATION_SCENARIOS.keys())
    if seeds is None:
        seeds = [42]

    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    fig_path = Path(save_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    robustness: Dict[str, Optional[bool]] = {}
    baseline_ranking: Optional[List[str]] = None
    start_time = time.time()

    for scenario_name in scenarios:
        scenario = PERTURBATION_SCENARIOS[scenario_name]
        cfg = apply_scenario(base_cfg, scenario)
        print(f"\n================ Scenario: {scenario_name} ================")

        algo_metrics: Dict[str, Dict[str, float]] = {}
        for algo in ALGORITHMS:
            per_seed: List[Dict[str, float]] = []
            for seed in seeds:
                try:
                    if algo == "bmpp_dqn":
                        metrics = _train_bmpp_dqn(
                            cfg, seed, train_episodes, eval_episodes
                        )
                    else:
                        metrics = _train_baseline(
                            algo, cfg, seed, train_episodes, eval_episodes
                        )
                except ValueError as exc:
                    if algo == "mpdqn":
                        print(
                            f"  {algo:8s} | seed {seed} | "
                            f"SKIPPED (intractable): {exc}"
                        )
                        continue
                    raise
                per_seed.append(metrics)

            if not per_seed:
                continue
            algo_metrics[algo] = {
                key: float(np.mean([m[key] for m in per_seed]))
                for key in ("mean_reward", "mean_power_w", "qos_satisfaction_rate")
            }
            m = algo_metrics[algo]
            print(
                f"  {ALGO_LABELS[algo]:9s} | reward={m['mean_reward']:10.2f} | "
                f"power={m['mean_power_w']:7.1f} W | "
                f"QoS={m['qos_satisfaction_rate']*100:5.1f}%"
            )

        results[scenario_name] = algo_metrics

        ranking = sorted(
            algo_metrics, key=lambda a: algo_metrics[a]["mean_reward"], reverse=True
        )
        rank_str = " > ".join(ALGO_LABELS[a] for a in ranking)
        print(f"  Ranking (best to worst, by mean eval reward): {rank_str}")

        if scenario_name == "baseline":
            baseline_ranking = ranking
            robustness[scenario_name] = None
        else:
            is_robust = ranking == baseline_ranking
            robustness[scenario_name] = is_robust
            print(f"  Ranking matches baseline: {is_robust}")

        reward_by_label = {
            ALGO_LABELS[a]: {"mean": algo_metrics[a]["mean_reward"]}
            for a in algo_metrics
        }
        plot_bar_comparison(
            reward_by_label,
            ylabel="Mean Evaluation Reward",
            title=f"Power-Model Sensitivity: {scenario_name}",
            save_path=str(fig_path / f"power_sensitivity_{scenario_name}_reward.pdf"),
        )

    elapsed = time.time() - start_time
    n_robust = sum(1 for v in robustness.values() if v is True)
    n_tested = sum(1 for v in robustness.values() if v is not None)
    print(
        f"\n=== Summary: ranking robust in {n_robust}/{n_tested} perturbation "
        f"scenarios ({elapsed:.0f}s total) ==="
    )

    with open(res_path / "summary.json", "w") as f:
        json.dump(
            {
                "config_path": config_path,
                "seeds": seeds,
                "train_episodes": train_episodes,
                "eval_episodes": eval_episodes,
                "results": results,
                "robustness": robustness,
                "baseline_ranking": (
                    [ALGO_LABELS[a] for a in baseline_ranking]
                    if baseline_ranking
                    else None
                ),
            },
            f,
            indent=2,
        )
    print(f"Saved results to {res_path / 'summary.json'}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Power-model constant sensitivity analysis (O-RAN track): "
            "perturbs oran_env/power_model.py's needs-validation constants "
            "and checks whether the 4-method reward ranking is robust."
        )
    )
    parser.add_argument("--config", type=str, default="config/oran_default.yaml")
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help=f"Subset of {list(PERTURBATION_SCENARIOS)} to run (default: all).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="thesis/figures_oran")
    parser.add_argument(
        "--results-dir", type=str, default="data/results_oran/power_sensitivity"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Override to a fast smoke-test run "
            "(train_episodes=5, eval_episodes=2, seeds=[42])."
        ),
    )
    args = parser.parse_args()

    if args.quick:
        run_power_sensitivity_analysis(
            config_path=args.config,
            scenarios=args.scenarios,
            seeds=[42],
            train_episodes=5,
            eval_episodes=2,
            save_dir=args.save_dir,
            results_dir=args.results_dir,
        )
    else:
        run_power_sensitivity_analysis(
            config_path=args.config,
            scenarios=args.scenarios,
            seeds=args.seeds,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            save_dir=args.save_dir,
            results_dir=args.results_dir,
        )
