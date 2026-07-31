"""Hyperparameter Tuning Search Utility for Hybrid SAC-DDQN Agent."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List
import yaml  # type: ignore[import-untyped]

from training.train_hybrid import train_hybrid_agent


class HyperparameterSearch:
    """Grid/Random hyperparameter search runner for Hybrid SAC-DDQN agent."""

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
        seeds: List[int] = [42],
    ) -> List[Dict[str, Any]]:
        """Run grid search over specified parameter combinations."""
        import itertools

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

            for seed in seeds:
                res = train_hybrid_agent(
                    config_path=str(temp_cfg_path),
                    seed=seed,
                    episodes=episodes_per_trial,
                    eval_freq=episodes_per_trial,
                    save_dir=None,
                )
                trial_rewards.append(float(res["final_eval_reward"]))
                trial_powers.append(float(res["final_eval_power_w"]))
                trial_qos.append(float(res["final_qos_rate"]))

            trial_summary: Dict[str, Any] = {
                "trial_id": idx + 1,
                "params": params,
                "mean_eval_reward": float(sum(trial_rewards) / len(trial_rewards)),
                "mean_eval_power_w": float(sum(trial_powers) / len(trial_powers)),
                "mean_qos_rate": float(sum(trial_qos) / len(trial_qos)),
            }
            search_results.append(trial_summary)

            if temp_cfg_path.exists():
                temp_cfg_path.unlink()

        # Rank trials by mean evaluation reward
        search_results.sort(key=lambda x: float(x["mean_eval_reward"]), reverse=True)

        with open(self.save_dir / "grid_search_results.json", "w") as f:
            json.dump(search_results, f, indent=2)

        print("\n=== Hyperparameter Search Top 3 Configurations ===")
        for rank, res in enumerate(search_results[:3], 1):
            qos_pct = float(res["mean_qos_rate"]) * 100
            print(
                f"Rank {rank}: Reward={res['mean_eval_reward']:.2f} | "
                f"QoS={qos_pct:.1f}% | Params={res['params']}"
            )

        return search_results


if __name__ == "__main__":
    searcher = HyperparameterSearch()
    grid: Dict[str, List[Any]] = {
        "lr_actor": [1e-4, 3e-4],
        "lr_critic": [1e-4, 3e-4],
        "batch_size": [128, 256],
    }
    searcher.run_grid_search(grid, episodes_per_trial=20, seeds=[42])
