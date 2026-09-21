"""Parallel, CPU-efficient runner for BOTH tracks' core training matrix.

Unlike `scripts/run_cran_experiments.sh` / `scripts/run_oran_experiments.sh`
(each track run as one long sequential process) and
`scripts/run_checkpointed_matrix.py` (one track at a time, one (method,
seed) unit at a time, but resumable across interruptions), this script
runs every (track, method) unit from BOTH tracks -- baseline algorithms
(one job per algorithm, covering all of that track's seeds internally,
exactly like a normal full-suite run) plus the proposed method (one job
per seed) -- as an independent OS subprocess in a single shared worker
pool, sized to the host's available CPU count. C-RAN and O-RAN therefore
train concurrently with each other, and each track's own methods/seeds
also run concurrently with one another, not just two-way (one process per
track).

Why subprocesses, not threads: `set_seed()` mutates *global* Python/NumPy/
PyTorch random state. Running units as Python threads inside one
interpreter would race on that shared global state and silently break
every unit's reproducibility. Each unit here is its own OS process with
its own interpreter and RNG state -- the same reason
`scripts/run_checkpointed_matrix.py` runs one unit at a time in-process
rather than threading them.

CPU budget: PyTorch (and NumPy's BLAS backend) default to using every
available core per process. Running many such processes concurrently
without capping each one causes exactly the thrashing
`scripts/run_checkpointed_matrix.py`'s own docstring already documents
(~1.6x slower in aggregate than sequential, measured on a 4-core sandbox
there). Every subprocess this script launches therefore has
OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS/NUMEXPR_NUM_THREADS
capped to max(1, cpu_count // concurrency) via that subprocess's own
environment, keeping total thread demand within the host's actual core
count regardless of how many jobs run at once.

Output layout: each job calls the exact same, already-tested entry point
(`run_baseline_benchmarks` / `train_hybrid_agent` for C-RAN,
`run_oran_baseline_benchmarks` / `train_bmpp_dqn_agent` for O-RAN) with
the same `save_dir` a normal full-suite run would use -- `data/results/`
for C-RAN, `data/results_oran/` for O-RAN, already kept separate by every
existing script and evaluation module (see README.md's "Running the
Experiments" section). This script only parallelizes how those calls are
scheduled; it does not change what they write or where.

Scope: this covers the CORE training matrix for both tracks -- all
baseline algorithms plus the proposed method, all seeds. It does not run
the C-RAN track's additional sweeps (scalability, ablation, demand-
response, power-time-profile, reward-sensitivity --
`scripts/run_cran_experiments.sh`'s steps 2-7) or either track's final
statistical-aggregation step; run those separately once this matrix
finishes -- this script prints the exact follow-up commands at the end.

Resumability: before launching a job, this script checks whether that
job's expected output file (`.../benchmark_<algo>/summary.json`,
`.../branching_mp_dqn_seed<N>/summary.json`, etc. -- exactly what that
job's own entry point writes on successful completion) already exists,
and skips it if so, printing a SKIP line instead of spawning a
subprocess. Re-running the same command after an interruption (crash,
Ctrl-C, container reclaim) therefore only repeats whichever jobs hadn't
finished yet -- no shared manifest file or locking is needed for this,
unlike `scripts/run_checkpointed_matrix.py`'s resumability, because each
job's own "done" state is just "does its one output file exist", checked
independently per job. Pass --force to disable this and re-run everything
regardless.

Usage:
    python scripts/run_all_parallel.py
    python scripts/run_all_parallel.py --jobs 8
    python scripts/run_all_parallel.py --dry-run
    python scripts/run_all_parallel.py --force   # ignore existing output, re-run all
    # Smoke test (fast, tiny episode counts, one seed per track):
    python scripts/run_all_parallel.py \\
        --cran-episodes 5 --oran-episodes 5 \\
        --cran-seeds 42 --oran-seeds 42
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_checkpointed_matrix import (  # noqa: E402
    CRAN_DEFAULT_SEEDS,
    CRAN_METHODS,
    ORAN_DEFAULT_SEEDS,
    ORAN_METHODS,
)

CRAN_PROPOSED_METHOD = "hybrid"
ORAN_PROPOSED_METHOD = "bmpp_dqn"


@dataclass
class Job:
    track: str  # "cran" or "oran"
    method: str
    label: str  # e.g. "cran/ddqn" or "oran/bmpp_dqn/seed42"
    code: str  # Python source passed to `python -c`
    env: Dict[str, str]
    output_marker: Path  # summary.json this job writes on success -> resume check


def _job_env(n_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[var] = str(max(1, n_threads))
    # `python -c` resolves imports against cwd (REPO_ROOT, set on each
    # subprocess call below) via sys.path[0]="" -- PYTHONPATH is set
    # anyway for parity with this repo's documented setup instructions
    # (README.md's "Running the Experiments") and as a safety net.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


def _cran_baseline_job(
    method: str, seeds: List[int], episodes: int, config_path: str,
    save_dir: str, n_threads: int,
) -> Job:
    code = (
        "from training.train_baselines import run_baseline_benchmarks; "
        f"run_baseline_benchmarks(config_path={config_path!r}, seeds={seeds!r}, "
        f"episodes={episodes}, algorithms=[{method!r}], save_dir={save_dir!r})"
    )
    marker = Path(save_dir) / f"benchmark_{method}" / "summary.json"
    return Job("cran", method, f"cran/{method}", code, _job_env(n_threads), marker)


def _cran_proposed_job(
    seed: int, episodes: int, config_path: str, save_dir: str, n_threads: int,
) -> Job:
    eval_freq = max(episodes // 20, 1)
    code = (
        "from training.train_hybrid import train_hybrid_agent; "
        f"train_hybrid_agent(config_path={config_path!r}, seed={seed}, "
        f"episodes={episodes}, eval_freq={eval_freq}, save_dir={save_dir!r}, "
        "use_wandb=False)"
    )
    marker = Path(save_dir) / f"branching_mp_dqn_seed{seed}" / "summary.json"
    return Job(
        "cran", CRAN_PROPOSED_METHOD, f"cran/{CRAN_PROPOSED_METHOD}/seed{seed}",
        code, _job_env(n_threads), marker,
    )


def _oran_baseline_job(
    method: str, seeds: List[int], episodes: int, config_path: str,
    save_dir: str, n_threads: int,
) -> Job:
    code = (
        "from oran_training.train_oran_baselines import run_oran_baseline_benchmarks; "
        f"run_oran_baseline_benchmarks(config_path={config_path!r}, seeds={seeds!r}, "
        f"episodes={episodes}, algorithms=[{method!r}], save_dir={save_dir!r})"
    )
    marker = Path(save_dir) / f"oran_benchmark_{method}" / "summary.json"
    return Job("oran", method, f"oran/{method}", code, _job_env(n_threads), marker)


def _oran_proposed_job(
    seed: int, episodes: int, config_path: str, save_dir: str, n_threads: int,
) -> Job:
    code = (
        "from oran_training.train_bmpp_dqn import train_bmpp_dqn_agent; "
        f"train_bmpp_dqn_agent(config_path={config_path!r}, seed={seed}, "
        f"episodes={episodes}, save_dir={save_dir!r})"
    )
    marker = Path(save_dir) / f"bmpp_dqn_seed{seed}" / "summary.json"
    return Job(
        "oran", ORAN_PROPOSED_METHOD, f"oran/{ORAN_PROPOSED_METHOD}/seed{seed}",
        code, _job_env(n_threads), marker,
    )


def build_jobs(args, n_threads: int) -> List[Job]:
    # The proposed method always gets its own per-seed job (below), never a
    # baseline-style job -- filter it out of the baseline-methods list even
    # when the caller explicitly passed it in --cran-methods/--oran-methods
    # (run_baseline_benchmarks/run_oran_baseline_benchmarks both raise
    # ValueError on an algorithm name they don't recognize).
    cran_methods = [
        m for m in (args.cran_methods or CRAN_METHODS) if m != CRAN_PROPOSED_METHOD
    ]
    cran_seeds = args.cran_seeds or CRAN_DEFAULT_SEEDS
    oran_methods = [
        m for m in (args.oran_methods or ORAN_METHODS) if m != ORAN_PROPOSED_METHOD
    ]
    oran_seeds = args.oran_seeds or ORAN_DEFAULT_SEEDS

    jobs: List[Job] = []

    if not args.skip_cran:
        for method in cran_methods:
            jobs.append(
                _cran_baseline_job(
                    method, cran_seeds, args.cran_episodes, args.cran_config,
                    args.cran_save_dir, n_threads,
                )
            )
        if args.cran_methods is None or CRAN_PROPOSED_METHOD in args.cran_methods:
            for seed in cran_seeds:
                jobs.append(
                    _cran_proposed_job(
                        seed, args.cran_episodes, args.cran_config,
                        args.cran_save_dir, n_threads,
                    )
                )

    if not args.skip_oran:
        for method in oran_methods:
            jobs.append(
                _oran_baseline_job(
                    method, oran_seeds, args.oran_episodes, args.oran_config,
                    args.oran_save_dir, n_threads,
                )
            )
        if args.oran_methods is None or ORAN_PROPOSED_METHOD in args.oran_methods:
            for seed in oran_seeds:
                jobs.append(
                    _oran_proposed_job(
                        seed, args.oran_episodes, args.oran_config,
                        args.oran_save_dir, n_threads,
                    )
                )

    return jobs


def run_job(job: Job, log_dir: Path) -> dict:
    log_path = log_dir / f"{job.label.replace('/', '_')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            [sys.executable, "-c", job.code],
            cwd=str(REPO_ROOT),
            env=job.env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - start
    return {
        "label": job.label,
        "track": job.track,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "log": str(log_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--jobs", type=int, default=None,
        help="Max concurrent subprocesses (default: os.cpu_count()).",
    )
    parser.add_argument("--cran-config", default="config/default.yaml")
    parser.add_argument("--oran-config", default="config/oran_default.yaml")
    parser.add_argument(
        "--cran-episodes", type=int, default=3000,
        help="Episodes per seed for the C-RAN track (default: 3000, the convergence target).",
    )
    parser.add_argument(
        "--oran-episodes", type=int, default=500,
        help=(
            "Episodes per seed for the O-RAN track "
            "(default: 500, config/oran_default.yaml's cap)."
        ),
    )
    parser.add_argument("--cran-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--oran-seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--cran-methods", type=str, nargs="+", default=None,
        choices=CRAN_METHODS,
        help="Subset of C-RAN methods to run (default: all 11).",
    )
    parser.add_argument(
        "--oran-methods", type=str, nargs="+", default=None,
        choices=ORAN_METHODS,
        help="Subset of O-RAN methods to run (default: all 4).",
    )
    parser.add_argument("--cran-save-dir", default="data/results")
    parser.add_argument("--oran-save-dir", default="data/results_oran")
    parser.add_argument("--log-dir", default="data/results/parallel_run_logs")
    parser.add_argument("--skip-cran", action="store_true")
    parser.add_argument("--skip-oran", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned job list and exit without running anything.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "Re-run every job even if its output file already exists "
            "(default: skip jobs whose output is already present, so "
            "re-running after an interruption only repeats unfinished work)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_cran and args.skip_oran:
        raise SystemExit("--skip-cran and --skip-oran together leave nothing to run.")

    cpu_count = os.cpu_count() or 1
    n_jobs = args.jobs or cpu_count
    n_threads_per_job = max(1, cpu_count // n_jobs)

    all_jobs = build_jobs(args, n_threads_per_job)
    n_cran = sum(1 for j in all_jobs if j.track == "cran")
    n_oran = sum(1 for j in all_jobs if j.track == "oran")

    print(
        f"Planned {len(all_jobs)} jobs ({n_cran} C-RAN, {n_oran} O-RAN) across up to "
        f"{n_jobs} concurrent workers on {cpu_count} logical CPUs "
        f"({n_threads_per_job} thread(s)/job)."
    )
    if args.dry_run:
        for job in all_jobs:
            done = job.output_marker.exists() and not args.force
            print(f"  [{job.label}]{'  (already done, would SKIP)' if done else ''}")
        return

    jobs = []
    for job in all_jobs:
        if not args.force and job.output_marker.exists():
            print(f"[{job.label}] SKIP (already done): {job.output_marker}")
        else:
            jobs.append(job)
    if len(jobs) < len(all_jobs):
        print(f"Skipped {len(all_jobs) - len(jobs)} already-completed job(s).")
    if not jobs:
        print("Nothing to do -- every planned job's output already exists "
              "(pass --force to re-run anyway).")
        return

    log_dir = Path(args.log_dir)
    results: List[dict] = []
    start_all = time.time()

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(run_job, job, log_dir): job for job in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            status = "OK" if res["returncode"] == 0 else f"FAILED (exit {res['returncode']})"
            print(
                f"[{res['label']}] {status} in {res['elapsed_sec'] / 60:.1f} min "
                f"-- log: {res['log']}"
            )
            results.append(res)

    total_elapsed = time.time() - start_all
    failed = [r for r in results if r["returncode"] != 0]
    print(
        f"\nDone: {len(results)} jobs in {total_elapsed / 60:.1f} min wall-clock, "
        f"{len(failed)} failed."
    )
    if failed:
        print("Failed jobs (see their log files for details):")
        for r in failed:
            print(f"  {r['label']}: {r['log']}")

    print(
        "\nNext steps (not run automatically by this script):\n"
        "  C-RAN aggregation:\n"
        "    python -c \"from evaluation import analyze_convergence; "
        f"analyze_convergence(results_dir={args.cran_save_dir!r})\"\n"
        "  O-RAN aggregation + latency benchmark:\n"
        "    python -c \"from oran_evaluation import analyze_convergence, "
        "run_latency_benchmark; "
        f"analyze_convergence(results_dir={args.oran_save_dir!r}); "
        f"run_latency_benchmark(config_path={args.oran_config!r})\"\n"
        "  C-RAN's additional sweeps (scalability/ablation/demand-response/"
        "power-time-profile/reward-sensitivity):\n"
        "    bash scripts/run_cran_experiments.sh   # steps 2-7; re-uses the "
        "matrix this script just trained\n"
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
