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
explicitly for a reduced-scope first pass. Use `--methods` to exclude a
specific baseline whose per-seed cost is impractical in an ephemeral,
reclaim-prone sandbox -- e.g. C-RAN's MP-DQN baseline projects to ~87h/seed
at real n_rrh=12 scale (deliberately so, per its own module docstring), and
this script only checkpoints at (method, seed) job boundaries, so an
interrupted single ~87h job restarts from episode 0 rather than resuming
mid-job:

    python3 scripts/run_checkpointed_matrix.py --track cran \
        --episodes 3000 --seeds 42 123 456 \
        --methods all_on greedy nmbs convex ddqn ann_gsbf ddqn_socp ddpg pdqn hybrid

Runs one track per invocation by design: PyTorch defaults to using every
available core, so two unpinned invocations racing for the same cores thrash
each other rather than truly parallelizing (measured ~1.6x slower in
aggregate than running one track at a time on this project's own 4-core
sandbox). On a genuinely multi-core machine, run both tracks as separate,
non-competing processes by pinning each to half the cores with
`--num-threads`, e.g. on a 16-core box:

    python3 scripts/run_checkpointed_matrix.py --track cran \
        --episodes 3000 --num-threads 8 &
    python3 scripts/run_checkpointed_matrix.py --track oran \
        --episodes 500 --num-threads 8 &
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


def resolve_methods(track, methods_override):
    """`--methods` override, or the track's full default list unchanged."""
    default = CRAN_METHODS if track == "cran" else ORAN_METHODS
    return methods_override or default


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
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Override the default method list for this track (e.g. to "
            "exclude a specific baseline for a reduced-scope pass). Omit "
            "for the full default list (CRAN_METHODS/ORAN_METHODS)."
        ),
    )
    parser.add_argument("--save-dir", default=None, help="Root save directory")
    parser.add_argument("--manifest", default=None, help="Checkpoint manifest path")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help=(
            "Pin this process to N CPU threads instead of PyTorch's default "
            "(all available cores). Needed to run both tracks as genuinely "
            "parallel, non-competing processes on a multi-core machine -- "
            "e.g. `--track cran --num-threads 8` and `--track oran "
            "--num-threads 8` together on a 16-core box. Without this, two "
            "unpinned invocations both try to use every core and thrash "
            "each other (measured ~1.6x slower in aggregate on this "
            "project's own 4-core sandbox -- see docs/daily_log.md's "
            "2026-09-07 entry). Omit for the default, single-track-at-a-time "
            "usage this script was originally built for."
        ),
    )
    args = parser.parse_args()

    if args.num_threads is not None:
        # Must happen before torch is imported anywhere (including inside
        # job_fn, which imports it lazily per job) -- setting these env vars
        # after import is too late for the underlying BLAS thread pools.
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = str(args.num_threads)
        import torch

        torch.set_num_threads(args.num_threads)

    methods = resolve_methods(args.track, args.methods)
    if args.track == "cran":
        config_path = args.config or "config/default.yaml"
        seeds = args.seeds or CRAN_DEFAULT_SEEDS
        save_root = args.save_dir or "data/results/checkpointed_matrix"
        job_fn = _make_cran_job_fn(config_path)
    else:
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
