"""Fair-cadence-control runner for the O-RAN track.

Runs DQN/MP-DQN with the same `discrete_hold_steps`-step discrete hold
BMPP-DQN's own two-timescale design uses (oran_training.train_oran_baselines
.run_oran_baseline_benchmarks(discrete_hold_steps=...)), so the switching-
frequency comparison in Chapter 4 isolates the architecture rather than
conflating it with an unmatched decision cadence -- responds directly to
the critique that the original comparison gave BMPP-DQN a cadence
advantage no baseline shared.

One (algorithm, seed) pair per subprocess, same thread-capping/concurrency
pattern as scripts/run_all_parallel.py (not imported from there, to avoid
touching that shared, already-tested script for this one-off need).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent


class Job(NamedTuple):
    label: str
    code: str
    env: Dict[str, str]
    output_marker: Path


def _job_env(n_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[var] = str(max(1, n_threads))
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


def build_jobs(
    methods: List[str],
    seeds: List[int],
    episodes: int,
    config_path: str,
    save_dir: str,
    discrete_hold_steps: int,
    n_threads: int,
) -> List[Job]:
    jobs = []
    for method in methods:
        for seed in seeds:
            per_seed_dir = str(Path(save_dir) / f"{method}_seed{seed}")
            code = (
                "from oran_training.train_oran_baselines import "
                "run_oran_baseline_benchmarks; "
                f"run_oran_baseline_benchmarks(config_path={config_path!r}, "
                f"seeds=[{seed}], episodes={episodes}, algorithms=[{method!r}], "
                f"save_dir={per_seed_dir!r}, "
                f"discrete_hold_steps={discrete_hold_steps})"
            )
            marker = Path(per_seed_dir) / f"oran_benchmark_{method}" / "summary.json"
            jobs.append(
                Job(f"{method}/seed{seed}", code, _job_env(n_threads), marker)
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
    return {
        "label": job.label,
        "returncode": proc.returncode,
        "elapsed_sec": time.time() - start,
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/oran_default.yaml")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--methods", nargs="+", default=["dqn", "mpdqn"])
    parser.add_argument("--discrete-hold-steps", type=int, default=10)
    parser.add_argument("--save-dir", default="data/results_oran_cadence_control")
    parser.add_argument("--log-dir", default="logs/cadence_control")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cpu_count = os.cpu_count() or 1
    n_jobs = args.jobs or cpu_count
    n_threads_per_job = max(1, cpu_count // n_jobs)

    all_jobs = build_jobs(
        args.methods, args.seeds, args.episodes, args.config, args.save_dir,
        args.discrete_hold_steps, n_threads_per_job,
    )
    print(
        f"Planned {len(all_jobs)} cadence-control jobs across up to {n_jobs} "
        f"concurrent workers on {cpu_count} logical CPUs "
        f"({n_threads_per_job} thread(s)/job)."
    )

    jobs = []
    for job in all_jobs:
        if not args.force and job.output_marker.exists():
            print(f"[{job.label}] SKIP (already done): {job.output_marker}")
        else:
            jobs.append(job)
    if not jobs:
        print("Nothing to do.")
        return

    log_dir = Path(args.log_dir)
    results = []
    start_all = time.time()
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(run_job, job, log_dir): job for job in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            status = (
                "OK" if res["returncode"] == 0 else f"FAILED (exit {res['returncode']})"
            )
            print(
                f"[{res['label']}] {status} in {res['elapsed_sec'] / 60:.1f} min "
                f"-- log: {res['log']}"
            )
            results.append(res)

    total_elapsed = time.time() - start_all
    failed = [r for r in results if r["returncode"] != 0]
    print(
        f"\nDone: {len(results)} jobs in {total_elapsed / 60:.1f} min, "
        f"{len(failed)} failed."
    )
    if failed:
        for r in failed:
            print(f"  FAILED: {r['label']} -- see {r['log']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
