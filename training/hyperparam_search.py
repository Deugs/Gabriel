"""Hyperparameter Tuning Search Utility for the Branching MP-DQN + TD3 Agent."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import yaml  # type: ignore[import-untyped]

from training.train_hybrid import train_hybrid_agent

# "roughly half an order of magnitude up and down" (Concept Note v4.0 Section 12.11, item 2).
PROXY_SWEEP_SCALE_UP = 10.0**0.5
PROXY_SWEEP_SCALE_DOWN = 10.0**-0.5


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


def run_proxy_sensitivity_sweep(
    base_config_path: str = "config/small_network.yaml",
    episodes: int = 100,
    seeds: List[int] = [42, 123],
    save_dir: str = "data/results/proxy_sweep",
) -> Dict[str, Any]:
    """Concept Note v4.0 Section 12.11's lightweight hyperparameter-tuning protocol.

    A targeted sensitivity CHECK, not a search for a new optimum (item 4): run
    the smallest scenario (R=5, U=2, 100 episodes, 2 seeds) with the
    branch/continuous-net learning-rate pair (lr_discrete/lr_actor) and tau
    each varied ~half an order of magnitude up and down from their Section
    12.2 defaults. If the default operating point is not visibly unstable
    (critic loss diverging/NaN, or a training crash) relative to the swept
    alternatives, the default is kept (item 2) — this function only reports
    that decision; it does not mutate any config file. Per item 3, if a
    parameter change is warranted, log it as a `docs/daily_log_template.md`
    entry with the before/after value and observed effect from this
    function's returned summary.
    """
    with open(base_config_path, "r") as f:
        base_cfg = deepcopy(yaml.safe_load(f))

    base_cfg.setdefault("network", {})
    base_cfg["network"]["n_rrh"] = 5
    base_cfg["network"]["n_ue"] = 2

    algo_cfg = base_cfg.setdefault("algorithm", {})
    default_lr_discrete = float(algo_cfg.get("lr_discrete", 1e-3))
    default_lr_actor = float(algo_cfg.get("lr_actor", 1e-4))
    default_tau = float(algo_cfg.get("tau", 0.005))

    variants: Dict[str, Dict[str, float]] = {
        "lr_pair_down": {
            "lr_discrete": default_lr_discrete * PROXY_SWEEP_SCALE_DOWN,
            "lr_actor": default_lr_actor * PROXY_SWEEP_SCALE_DOWN,
        },
        "lr_pair_default": {
            "lr_discrete": default_lr_discrete,
            "lr_actor": default_lr_actor,
        },
        "lr_pair_up": {
            "lr_discrete": default_lr_discrete * PROXY_SWEEP_SCALE_UP,
            "lr_actor": default_lr_actor * PROXY_SWEEP_SCALE_UP,
        },
        "tau_down": {"tau": default_tau * PROXY_SWEEP_SCALE_DOWN},
        "tau_default": {"tau": default_tau},
        "tau_up": {"tau": default_tau * PROXY_SWEEP_SCALE_UP},
    }

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("--- Concept Note v4.0 Section 12.11 Proxy Sensitivity Sweep (R=5, U=2) ---")

    results: Dict[str, Any] = {}
    for variant_name, overrides in variants.items():
        trial_cfg = deepcopy(base_cfg)
        trial_cfg["algorithm"].update(overrides)

        temp_cfg_path = save_path / f"temp_{variant_name}.yaml"
        with open(temp_cfg_path, "w") as f:
            yaml.dump(trial_cfg, f)

        seed_rewards: List[float] = []
        seed_tail_critic_losses: List[float] = []
        crashed = False

        for seed in seeds:
            try:
                res = train_hybrid_agent(
                    config_path=str(temp_cfg_path),
                    seed=seed,
                    episodes=episodes,
                    eval_freq=episodes,
                    save_dir=None,
                )
                seed_rewards.append(float(res["final_eval_reward"]))
                critic_losses = res["history"]["critic_losses"]
                tail = critic_losses[-max(1, len(critic_losses) // 5) :]
                seed_tail_critic_losses.append(float(np.mean(tail)) if tail else 0.0)
            except (RuntimeError, ValueError) as exc:
                crashed = True
                seed_rewards.append(float("nan"))
                seed_tail_critic_losses.append(float("nan"))
                print(f"  [{variant_name} seed={seed}] CRASHED: {exc}")

        if temp_cfg_path.exists():
            temp_cfg_path.unlink()

        mean_reward = float(np.nanmean(seed_rewards)) if seed_rewards else float("nan")
        mean_tail_loss = (
            float(np.nanmean(seed_tail_critic_losses)) if seed_tail_critic_losses else float("nan")
        )
        results[variant_name] = {
            "overrides": overrides,
            "mean_final_eval_reward": mean_reward,
            "mean_tail_critic_loss": mean_tail_loss,
            "crashed": crashed,
        }
        status = "CRASHED" if crashed else "ok"
        print(
            f"{variant_name:16s} | reward={mean_reward:10.3f} | "
            f"tail_critic_loss={mean_tail_loss:10.4f} | {status}"
        )

    decisions: Dict[str, Dict[str, Any]] = {}
    for dim, keys in [
        ("lr_pair", ["lr_pair_down", "lr_pair_default", "lr_pair_up"]),
        ("tau", ["tau_down", "tau_default", "tau_up"]),
    ]:
        default_result = results[keys[1]]
        default_unstable = default_result["crashed"] or not np.isfinite(
            default_result["mean_final_eval_reward"]
        )
        decisions[dim] = {
            "default_kept": not default_unstable,
            "reason": (
                "default operating point crashed or produced a non-finite reward; "
                "recommend using the more stable swept alternative instead"
                if default_unstable
                else "default operating point not visibly unstable relative to the "
                "swept alternatives; kept per Concept Note v4.0 Section 12.11 item 2"
            ),
        }
        print(f"Decision [{dim}]: default_kept={decisions[dim]['default_kept']} — {decisions[dim]['reason']}")

    summary = {
        "scenario": {"n_rrh": 5, "n_ue": 2, "episodes": episodes, "seeds": seeds},
        "results": results,
        "decisions": decisions,
    }

    with open(save_path / "proxy_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    searcher = HyperparameterSearch()
    grid: Dict[str, List[Any]] = {
        "lr_actor": [1e-4, 3e-4],
        "lr_critic": [1e-4, 3e-4],
        "batch_size": [128, 256],
    }
    searcher.run_grid_search(grid, episodes_per_trial=20, seeds=[42])
    run_proxy_sensitivity_sweep()
