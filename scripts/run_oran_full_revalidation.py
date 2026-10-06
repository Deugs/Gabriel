"""Full O-RAN revalidation: n_eval_episodes 5->20, n=10 canonical seeds,
fresh fair-cadence-control + RQ3 single-timescale reruns at the new eval
count, and a fresh power-model sensitivity sweep.

Responds to the two remaining critique items left open after the
critique-response rerun (Chapter 4's bug-fix disclosure/calibration/
cadence-control sections): the thin n_eval_episodes=5 evaluation, and the
sensitivity/n=10 checks never having been rerun against the corrected
environment (both explicitly flagged as deferred future work until now).

Writes to NEW, "_v2"-suffixed directories rather than overwriting the
current canonical data/results_oran/ etc. -- inspect and validate before
promoting (rename) over the existing canonical directories, the same
promote-after-validate discipline used for the original bug-fix rerun.

One (job-type, algo, seed) unit per subprocess, same
Job/ThreadPoolExecutor/_job_env pattern as
scripts/run_oran_cadence_control.py (not imported from there, to avoid
touching that shared, already-tested script for this one-off need).
Each job's own expected output file is checked before (re)running it, so
an interrupted run only repeats what did not finish -- pass --force to
disable this.

Usage:
    python scripts/run_oran_full_revalidation.py --phase all
    python scripts/run_oran_full_revalidation.py --phase canonical
    python scripts/run_oran_full_revalidation.py --phase cadence
    python scripts/run_oran_full_revalidation.py --phase rq3
    python scripts/run_oran_full_revalidation.py --phase calibration
    python scripts/run_oran_full_revalidation.py --phase sensitivity
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

SEEDS_10 = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
SEEDS_3 = [42, 123, 456]
BASELINE_ALGOS = ["dqn", "ddpg", "mpdqn"]

CANONICAL_SAVE_DIR = "data/results_oran_v2"
CADENCE_SAVE_DIR = "data/results_oran_v2_cadence_control"
RQ3_SAVE_DIR = "data/results_oran_v2_rq3_single_timescale"
RQ3_CONFIG = "config/rq3_single_timescale.yaml"
SENSITIVITY_RESULTS_DIR = "data/results_oran_v2_sensitivity"
SENSITIVITY_FIGS_DIR = "thesis/figures_oran_v2_sensitivity"
SENSITIVITY_SCENARIOS = [
    "baseline",
    "ru_power_high_5x",
    "ru_power_low_0.5x",
    "du_cu_power_high_5x",
    "fronthaul_power_high_4x",
    "pa_efficiency_low_0.14",
    "pa_efficiency_high_0.39",
    "order_of_magnitude_high_10x",
]

# The ORIGINAL (pre-env-bugfix) sensitivity-sweep convention -- full
# canonical-scale (3 seeds, 500 episodes) per perturbation config, matching
# data/results_oran_archive/pre_env_bugfix_20261001/results_oran_sensitivity's
# own ru_10x/ru_0p1x/etc. layout and the scratchpad's
# run_sensitivity_sweep.sh (same 8 configs, same scripts/run_all_parallel.py
# entry points, just run here through this script's own Job/ThreadPoolExecutor
# pool instead of 8 sequential scripts/run_all_parallel.py invocations, so
# all 96 jobs bin-pack across every available worker slot at once rather
# than one config's 12 jobs at a time).
SENSITIVITY_FULL_CONFIGS = [
    "ru_10x",
    "ru_0p1x",
    "du_10x",
    "du_0p1x",
    "cu_10x",
    "cu_0p1x",
    "fronthaul_10x",
    "fronthaul_0p1x",
]
SENSITIVITY_FULL_SAVE_DIR = "data/results_oran_sensitivity_v2"


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


def build_canonical_jobs(n_threads: int) -> List[Job]:
    jobs = []
    for seed in SEEDS_10:
        # train_bmpp_dqn_agent appends f"bmpp_dqn_seed{seed}" to save_dir
        # itself (oran_training/train_bmpp_dqn.py's out_path construction)
        # -- pass the PARENT dir here, not the already-seed-suffixed path.
        code = (
            "from oran_training.train_bmpp_dqn import train_bmpp_dqn_agent; "
            f"train_bmpp_dqn_agent(config_path='config/oran_default.yaml', "
            f"seed={seed}, episodes=500, save_dir={CANONICAL_SAVE_DIR!r})"
        )
        marker = Path(CANONICAL_SAVE_DIR) / f"bmpp_dqn_seed{seed}" / "final_model.pt"
        jobs.append(Job(f"bmpp_dqn/seed{seed}", code, _job_env(n_threads), marker))

    for algo in BASELINE_ALGOS:
        for seed in SEEDS_10:
            per_seed_dir = f"{CANONICAL_SAVE_DIR}/per_seed_baselines/{algo}_seed{seed}"
            code = (
                "from oran_training.train_oran_baselines import "
                "run_oran_baseline_benchmarks; "
                f"run_oran_baseline_benchmarks(config_path='config/oran_default.yaml', "
                f"seeds=[{seed}], episodes=500, algorithms=[{algo!r}], "
                f"save_dir={per_seed_dir!r})"
            )
            marker = Path(per_seed_dir) / f"oran_benchmark_{algo}" / "summary.json"
            jobs.append(Job(f"{algo}/seed{seed}", code, _job_env(n_threads), marker))
    return jobs


def build_cadence_jobs(n_threads: int) -> List[Job]:
    jobs = []
    for algo in ("dqn", "mpdqn"):
        for seed in SEEDS_3:
            per_seed_dir = f"{CADENCE_SAVE_DIR}/{algo}_seed{seed}"
            code = (
                "from oran_training.train_oran_baselines import "
                "run_oran_baseline_benchmarks; "
                f"run_oran_baseline_benchmarks(config_path='config/oran_default.yaml', "
                f"seeds=[{seed}], episodes=500, algorithms=[{algo!r}], "
                f"save_dir={per_seed_dir!r}, discrete_hold_steps=10)"
            )
            marker = Path(per_seed_dir) / f"oran_benchmark_{algo}" / "summary.json"
            jobs.append(
                Job(f"cadence/{algo}/seed{seed}", code, _job_env(n_threads), marker)
            )
    return jobs


def build_rq3_jobs(n_threads: int) -> List[Job]:
    jobs = []
    for seed in SEEDS_3:
        # Same save_dir convention as build_canonical_jobs above: pass the
        # parent dir, train_bmpp_dqn_agent appends bmpp_dqn_seed{seed} itself.
        code = (
            "from oran_training.train_bmpp_dqn import train_bmpp_dqn_agent; "
            f"train_bmpp_dqn_agent(config_path={RQ3_CONFIG!r}, "
            f"seed={seed}, episodes=500, save_dir={RQ3_SAVE_DIR!r})"
        )
        marker = Path(RQ3_SAVE_DIR) / f"bmpp_dqn_seed{seed}" / "final_model.pt"
        jobs.append(Job(f"rq3/bmpp_dqn/seed{seed}", code, _job_env(n_threads), marker))
    return jobs


def build_calibration_job(n_threads: int) -> List[Job]:
    code = (
        "from oran_training.oran_heuristic_oracle import "
        "run_oran_heuristic_oracle_benchmarks; "
        "run_oran_heuristic_oracle_benchmarks(config_path='config/oran_default.yaml', "
        f"seeds={SEEDS_10!r}, save_dir={CANONICAL_SAVE_DIR!r})"
    )
    marker = Path(CANONICAL_SAVE_DIR) / "oran_benchmark_oracle" / "summary.json"
    return [Job("heuristic_oracle", code, _job_env(n_threads), marker)]


def build_sensitivity_full_jobs(n_threads: int) -> List[Job]:
    jobs = []
    for config_name in SENSITIVITY_FULL_CONFIGS:
        config_path = f"config/sensitivity/oran_{config_name}.yaml"
        save_dir = f"{SENSITIVITY_FULL_SAVE_DIR}/{config_name}"

        # Same save_dir convention as build_canonical_jobs: pass the
        # parent dir, train_bmpp_dqn_agent appends bmpp_dqn_seed{seed}.
        for seed in SEEDS_3:
            code = (
                "from oran_training.train_bmpp_dqn import train_bmpp_dqn_agent; "
                f"train_bmpp_dqn_agent(config_path={config_path!r}, "
                f"seed={seed}, episodes=500, save_dir={save_dir!r})"
            )
            marker = Path(save_dir) / f"bmpp_dqn_seed{seed}" / "final_model.pt"
            jobs.append(
                Job(
                    f"sensitivity_full/{config_name}/bmpp_dqn/seed{seed}",
                    code,
                    _job_env(n_threads),
                    marker,
                )
            )

        for algo in BASELINE_ALGOS:
            for seed in SEEDS_3:
                per_seed_dir = f"{save_dir}/per_seed_baselines/{algo}_seed{seed}"
                code = (
                    "from oran_training.train_oran_baselines import "
                    "run_oran_baseline_benchmarks; "
                    f"run_oran_baseline_benchmarks(config_path={config_path!r}, "
                    f"seeds=[{seed}], episodes=500, algorithms=[{algo!r}], "
                    f"save_dir={per_seed_dir!r})"
                )
                marker = (
                    Path(per_seed_dir) / f"oran_benchmark_{algo}" / "summary.json"
                )
                jobs.append(
                    Job(
                        f"sensitivity_full/{config_name}/{algo}/seed{seed}",
                        code,
                        _job_env(n_threads),
                        marker,
                    )
                )
    return jobs


def build_sensitivity_job(n_threads: int) -> List[Job]:
    # run_power_sensitivity_analysis loops every scenario SEQUENTIALLY in
    # one process and writes one combined summary.json at the end -- as a
    # single job that would be ~8x longer than any other job in this
    # script and dominate total wall-clock (it does not parallelize
    # internally). Split into one job per scenario instead, each to its
    # OWN results_dir (no shared-file race, since each call's summary.json
    # lives in a different directory) -- aggregate_sensitivity_results()
    # below recombines them (incl. the baseline-ranking-robustness
    # comparison run_power_sensitivity_analysis itself would have done)
    # once every per-scenario job has finished.
    jobs = []
    for scenario in SENSITIVITY_SCENARIOS:
        scenario_dir = f"{SENSITIVITY_RESULTS_DIR}/{scenario}"
        code = (
            "from oran_evaluation.power_sensitivity import "
            "run_power_sensitivity_analysis; "
            "run_power_sensitivity_analysis(config_path='config/oran_default.yaml', "
            f"scenarios=[{scenario!r}], "
            f"save_dir={SENSITIVITY_FIGS_DIR!r}, results_dir={scenario_dir!r})"
        )
        marker = Path(scenario_dir) / "summary.json"
        jobs.append(
            Job(f"sensitivity/{scenario}", code, _job_env(n_threads), marker)
        )
    return jobs


def aggregate_sensitivity_results() -> None:
    """Recombine the per-scenario summary.json files
    build_sensitivity_job() produces into the single combined summary.json
    oran_evaluation/sensitivity_plots.py (and Chapter 4's sensitivity
    section) expects -- the same ranking-robustness comparison
    run_power_sensitivity_analysis itself does when run as one call over
    every scenario, just computed here instead since each scenario ran in
    its own process/call."""
    import json

    algo_labels = {
        "bmpp_dqn": "BMPP-DQN",
        "dqn": "DQN",
        "ddpg": "DDPG",
        "mpdqn": "MP-DQN",
    }

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scenario in SENSITIVITY_SCENARIOS:
        with open(Path(SENSITIVITY_RESULTS_DIR) / scenario / "summary.json") as f:
            per_scenario = json.load(f)
        results[scenario] = per_scenario["results"][scenario]

    baseline_ranking = sorted(
        results["baseline"],
        key=lambda a: results["baseline"][a]["mean_reward"],
        reverse=True,
    )
    robustness: Dict[str, object] = {"baseline": None}
    for scenario in SENSITIVITY_SCENARIOS:
        if scenario == "baseline":
            continue
        ranking = sorted(
            results[scenario],
            key=lambda a: results[scenario][a]["mean_reward"],
            reverse=True,
        )
        robustness[scenario] = ranking == baseline_ranking

    out_path = Path(SENSITIVITY_RESULTS_DIR) / "summary.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "config_path": "config/oran_default.yaml",
                "results": results,
                "robustness": robustness,
                "baseline_ranking": [algo_labels[a] for a in baseline_ranking],
            },
            f,
            indent=2,
        )
    print(f"Aggregated sensitivity summary written to {out_path}")


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


PHASE_BUILDERS = {
    "canonical": build_canonical_jobs,
    "cadence": build_cadence_jobs,
    "rq3": build_rq3_jobs,
    "calibration": build_calibration_job,
    "sensitivity": build_sensitivity_job,
    "sensitivity_full": build_sensitivity_full_jobs,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=list(PHASE_BUILDERS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--log-dir", default="logs/full_revalidation")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cpu_count = os.cpu_count() or 1
    n_jobs = args.jobs or cpu_count
    n_threads_per_job = max(1, cpu_count // n_jobs)

    phases = (
        list(PHASE_BUILDERS.keys()) if args.phase == "all" else [args.phase]
    )

    all_jobs: List[Job] = []
    for phase in phases:
        all_jobs += PHASE_BUILDERS[phase](n_threads_per_job)

    print(
        f"Planned {len(all_jobs)} jobs across phases {phases}, up to {n_jobs} "
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

    if "sensitivity" in phases:
        aggregate_sensitivity_results()


if __name__ == "__main__":
    main()
