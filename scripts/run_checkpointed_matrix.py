"""Checkpoint-resumable full experiment matrix runner, for both tracks.

Unlike `scripts/run_cran_experiments.sh`/`scripts/run_oran_experiments.sh`
(which run their whole matrix as one call and lose all progress if
interrupted), this script runs one (method, seed) pair at a time and
persists a checkpoint manifest after each one completes. Re-running the
exact same command after an interruption (container reclaim, crash, manual
stop) skips every (method, seed) pair already marked done and resumes with
whichever pair was next -- at most one in-flight pair's work is repeated.

Usage:
    python3 scripts/run_checkpointed_matrix.py --track cran \
        --episodes 3000 --seeds 42 123 456

    python3 scripts/run_checkpointed_matrix.py --track oran \
        --episodes 500 --seeds 42 123 456

Defaults (episodes, seeds, config, save-dir) match this repo's own
established conventions: C-RAN's 3000-episode convergence target and full
10-seed list (README.md's Key Decisions Log; `scripts/run_cran_experiments.sh`),
O-RAN's 500-episode cap and 3-seed list (`config/oran_default.yaml`;
`scripts/run_oran_experiments.sh`) -- override `--episodes`/`--seeds`
explicitly for a reduced-scope first pass.
"""

import argparse
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.checkpointed_runner import run_checkpointed  # noqa: E402

CRAN_METHODS = [
    "all_on",
    "greedy",
    "nmbs",
    "convex",
    "ddqn",
    "ann_gsbf",
    "ddqn_socp",
    "ddpg",
    "pdqn",
    "mpdqn",
    "hybrid",
]
CRAN_DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 1337, 2024, 2718, 3141, 4242]

ORAN_METHODS = ["dqn", "ddpg", "mpdqn", "bmpp_dqn"]
ORAN_DEFAULT_SEEDS = [42, 123, 456]


def _make_cran_job_fn(config_path: str):
    def job_fn(job: Dict[str, Any]) -> Dict[str, Any]:
        method, seed, episodes, save_dir = (
            job["method"],
            job["seed"],
            job["episodes"],
            job["save_dir"],
        )
        os.makedirs(save_dir, exist_ok=True)
        if method == "hybrid":
            from training.train_hybrid import train_hybrid_agent

            res = train_hybrid_agent(
                config_path=config_path,
                seed=seed,
                episodes=episodes,
                eval_freq=max(episodes // 20, 1),
                save_dir=save_dir,
                use_wandb=False,
            )
            return {"final_eval_reward": res.get("final_eval_reward")}
        from training.train_baselines import run_baseline_benchmarks

        res = run_baseline_benchmarks(
            config_path=config_path,
            seeds=[seed],
            episodes=episodes,
            algorithms=[method],
            save_dir=save_dir,
        )
        algo_results = res.get(method) or []
        return (
            {"mean_reward": algo_results[0]["mean_reward"]}
            if algo_results
            else {"skipped": True}
        )

    return job_fn


def _make_oran_job_fn(config_path: str):
    def job_fn(job: Dict[str, Any]) -> Dict[str, Any]:
        method, seed, episodes, save_dir = (
            job["method"],
            job["seed"],
            job["episodes"],
            job["save_dir"],
        )
        os.makedirs(save_dir, exist_ok=True)
        if method == "bmpp_dqn":
            from oran_training.train_bmpp_dqn import train_bmpp_dqn_agent

            res = train_bmpp_dqn_agent(
                config_path=config_path, seed=seed, episodes=episodes, save_dir=save_dir
            )
            return {"final_eval_reward": res.get("final_eval_reward")}
        from oran_training.train_oran_baselines import run_oran_baseline_benchmarks

        res = run_oran_baseline_benchmarks(
            config_path=config_path,
            seeds=[seed],
            episodes=episodes,
            algorithms=[method],
            save_dir=save_dir,
        )
        algo_results = res.get(method) or []
        return (
            {"mean_reward": algo_results[0]["mean_reward"]}
            if algo_results
            else {"skipped": True}
        )

    return job_fn


def build_jobs(methods, seeds, episodes, save_root):
    jobs = []
    for method in methods:
        for seed in seeds:
            jobs.append(
                {
                    "method": method,
                    "seed": seed,
                    "episodes": episodes,
                    "save_dir": os.path.join(save_root, method, f"seed{seed}"),
                }
            )
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint-resumable full experiment matrix runner"
    )
    parser.add_argument("--track", choices=["cran", "oran"], required=True)
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--save-dir", default=None, help="Root save directory")
    parser.add_argument("--manifest", default=None, help="Checkpoint manifest path")
    args = parser.parse_args()

    if args.track == "cran":
        methods = CRAN_METHODS
        config_path = args.config or "config/default.yaml"
        seeds = args.seeds or CRAN_DEFAULT_SEEDS
        save_root = args.save_dir or "data/results/checkpointed_matrix"
        job_fn = _make_cran_job_fn(config_path)
    else:
        methods = ORAN_METHODS
        config_path = args.config or "config/oran_default.yaml"
        seeds = args.seeds or ORAN_DEFAULT_SEEDS
        save_root = args.save_dir or "data/results_oran/checkpointed_matrix"
        job_fn = _make_oran_job_fn(config_path)

    manifest_path = args.manifest or os.path.join(save_root, "manifest.json")
    jobs = build_jobs(methods, seeds, args.episodes, save_root)

    print(
        f"Checkpointed {args.track.upper()} matrix: {len(jobs)} (method, seed) "
        f"jobs, {args.episodes} episodes each, manifest={manifest_path}"
    )
    run_checkpointed(
        jobs,
        manifest_path,
        job_fn,
        job_key_fn=lambda j: f"{j['method']}/seed{j['seed']}",
    )
    print("Checkpointed matrix run complete (or resumed to completion).")


if __name__ == "__main__":
    main()
