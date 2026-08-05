"""Hyperparameter Tuning Search Utility for the Branching MP-DQN + TD3 Agent."""

from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml  # type: ignore[import-untyped]

from training.train_hybrid import train_hybrid_agent


class HyperparameterSearch:
    """Grid/Random hyperparameter search runner for the Branching MP-DQN + TD3 agent."""

    def __init__(
        self,
        base_config_path: str = "config/default.yaml",
        save_dir: str = "data/results/grid_search",
    ):
        self.base_config_path = base_config_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        with open(base_config_path, "r") as f:
            self.base_cfg = yaml.safe_load(f)

    def run_grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        episodes_per_trial: int = 30,
        seeds: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Run grid search over specified parameter combinations.

        A trial/seed combination that raises (e.g. train_hybrid_agent's
        RuntimeError on a NaN/Inf reward) is caught and recorded as
        "unstable" rather than aborting the whole sweep -- required for
        run_sensitivity_check's own purpose (Concept Note v4.0 Section
        12.11 / G9): finding *which* settings are unstable, not just the
        first one that crashes.
        """
        if seeds is None:
            seeds = [42]

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        search_results: List[Dict[str, Any]] = []
        print(f"--- Starting Hyperparameter Search ({len(combinations)} Trials) ---")

        for idx, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            print(f"\n[Trial {idx+1}/{len(combinations)}] Parameters: {params}")

            trial_cfg = deepcopy(self.base_cfg)
            if "algorithm" not in trial_cfg:
                trial_cfg["algorithm"] = {}

            for k, v in params.items():
                trial_cfg["algorithm"][k] = v

            # Save temporary trial config
            temp_cfg_path = self.save_dir / f"temp_trial_{idx}.yaml"
            with open(temp_cfg_path, "w") as f:
                yaml.dump(trial_cfg, f)

            trial_rewards: List[float] = []
            trial_powers: List[float] = []
            trial_qos: List[float] = []
            seed_statuses: List[Dict[str, Any]] = []

            for seed in seeds:
                try:
                    res = train_hybrid_agent(
                        config_path=str(temp_cfg_path),
                        seed=seed,
                        episodes=episodes_per_trial,
                        eval_freq=episodes_per_trial,
                        save_dir=None,
                    )
                except RuntimeError as e:
                    print(f"  [seed {seed}] UNSTABLE: {e}")
                    seed_statuses.append(
                        {"seed": seed, "status": "unstable", "error": str(e)}
                    )
                    continue

                trial_rewards.append(float(res["final_eval_reward"]))
                trial_powers.append(float(res["final_eval_power_w"]))
                trial_qos.append(float(res["final_qos_rate"]))
                seed_statuses.append({"seed": seed, "status": "ok"})

            n_unstable = sum(1 for s in seed_statuses if s["status"] == "unstable")
            trial_summary: Dict[str, Any] = {
                "trial_id": idx + 1,
                "params": params,
                "seed_statuses": seed_statuses,
                "n_unstable": n_unstable,
                "mean_eval_reward": (
                    float(sum(trial_rewards) / len(trial_rewards))
                    if trial_rewards
                    else float("nan")
                ),
                "mean_eval_power_w": (
                    float(sum(trial_powers) / len(trial_powers))
                    if trial_powers
                    else float("nan")
                ),
                "mean_qos_rate": (
                    float(sum(trial_qos) / len(trial_qos)) if trial_qos else float("nan")
                ),
            }
            search_results.append(trial_summary)

            if temp_cfg_path.exists():
                temp_cfg_path.unlink()

        # Rank trials by mean evaluation reward; fully-unstable trials (no
        # surviving seed, mean_eval_reward = NaN) sort last rather than
        # corrupting the ordering of the rest.
        search_results.sort(
            key=lambda x: (
                float("-inf") if math.isnan(x["mean_eval_reward"]) else x["mean_eval_reward"]
            ),
            reverse=True,
        )

        with open(self.save_dir / "grid_search_results.json", "w") as f:
            json.dump(search_results, f, indent=2)

        print("\n=== Hyperparameter Search Top 3 Configurations ===")
        for rank, res in enumerate(search_results[:3], 1):
            qos_pct = float(res["mean_qos_rate"]) * 100 if res["mean_qos_rate"] == res["mean_qos_rate"] else float("nan")
            print(
                f"Rank {rank}: Reward={res['mean_eval_reward']:.2f} | "
                f"QoS={qos_pct:.1f}% | Unstable={res['n_unstable']} | Params={res['params']}"
            )

        return search_results

    def run_sensitivity_check(
        self,
        episodes_per_trial: int = 100,
        seeds: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Lightweight hyperparameter-sensitivity protocol (Concept Note v4.0
        Section 12.11 / G9): a targeted sensitivity check, not a search for a
        new optimum. Sweeps the branch/continuous-net learning rates and tau
        roughly half an order of magnitude up/down from their defaults, at
        R=5 (this instance should be constructed with
        `base_config_path="config/small_network.yaml"`), 100 episodes, 2
        seeds. If the default operating point is not visibly unstable
        relative to the swept alternatives, the default is kept -- see the
        printed ranking and each trial's `n_unstable` count.
        """
        if seeds is None:
            seeds = [42, 123]

        algo_cfg = self.base_cfg.get("algorithm", {})
        default_lr_discrete = float(algo_cfg.get("lr_discrete", 1e-3))
        default_lr_actor = float(algo_cfg.get("lr_actor", 1e-4))
        default_tau = float(algo_cfg.get("tau", 0.005))

        grid: Dict[str, List[Any]] = {
            "lr_discrete": [
                default_lr_discrete / 3.0,
                default_lr_discrete,
                default_lr_discrete * 3.0,
            ],
            "lr_actor": [
                default_lr_actor / 3.0,
                default_lr_actor,
                default_lr_actor * 3.0,
            ],
            "tau": [default_tau / 3.0, default_tau, default_tau * 3.0],
        }
        return self.run_grid_search(
            grid, episodes_per_trial=episodes_per_trial, seeds=seeds
        )


if __name__ == "__main__":
    searcher = HyperparameterSearch()
    grid: Dict[str, List[Any]] = {
        "lr_actor": [1e-4, 3e-4],
        "lr_discrete": [1e-4, 3e-4],
        "batch_size": [128, 256],
    }
    searcher.run_grid_search(grid, episodes_per_trial=20, seeds=[42])
