"""Checkpoint-resumable job runner, shared by both tracks' full experiment
matrices.

Motivation: a full experiment matrix (many (method, seed) training runs) can
take hours, and the environment it runs in may be interrupted at any point
(container reclaim, crash, manual stop) with no guarantee the whole matrix
finishes in one sitting. `run_baseline_benchmarks()`/`train_hybrid_agent()`/
etc. only persist their own output once their *entire* call returns, so an
interruption mid-call loses that call's work even if most of it was already
computed. This module adds a per-(method, seed)-unit checkpoint layer on top
of those existing entry points, without changing their own behavior: each
unit's result is written to disk and recorded in a manifest immediately
after that unit completes, so re-running the same job list after an
interruption skips every unit already marked done and only repeats the one
that was in flight (at most) when the process was cut off.

This is a thin wrapper, not a new training system: `job_fn` is expected to
call the existing, already-tested `run_baseline_benchmarks`/
`train_hybrid_agent`/`run_oran_baseline_benchmarks`/`train_bmpp_dqn_agent`
entry points scoped to a single (method, seed) pair (see
`scripts/run_checkpointed_matrix.py`), which already persist their own
per-call results to disk exactly as they do for a non-checkpointed run.
"""

import json
import os
from typing import Any, Callable, Dict, List


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """Load a checkpoint manifest, or an empty one if it doesn't exist yet."""
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest_path: str, manifest: Dict[str, Any]) -> None:
    """Atomically write the manifest (write-then-rename) so a crash mid-write
    can never leave a truncated/corrupt manifest that would misreport which
    jobs are actually done."""
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    tmp_path = f"{manifest_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, manifest_path)


def run_checkpointed(
    jobs: List[Dict[str, Any]],
    manifest_path: str,
    job_fn: Callable[[Dict[str, Any]], Any],
    job_key_fn: Callable[[Dict[str, Any]], str],
) -> Dict[str, Any]:
    """Run each job in `jobs` at most once across possibly-many process
    restarts, skipping any job whose key is already marked "done" in the
    manifest at `manifest_path`.

    Args:
        jobs: Job specs, each passed as-is to `job_fn`.
        manifest_path: Where to persist the checkpoint manifest (JSON).
        job_fn: Does the actual work for one job and returns a small,
            JSON-serializable summary (e.g. `{"mean_reward": ...}`) --
            `job_fn` itself is responsible for persisting the job's full
            results to disk (e.g. via the existing `save_dir`-based
            functions); this manifest only records a lightweight summary
            plus done/failed status, not the full results.
        job_key_fn: Maps a job spec to a stable string key used to check
            manifest membership (e.g. `f"{method}/seed{seed}"`).

    Returns:
        The final manifest (also persisted to `manifest_path`).

    Raises:
        Whatever `job_fn` raises, after first recording that job as
        "failed" in the manifest (so a subsequent re-run doesn't silently
        treat a crashed job as done, but also doesn't lose track of which
        job failed).
    """
    manifest = load_manifest(manifest_path)
    for job in jobs:
        key = job_key_fn(job)
        if manifest.get(key, {}).get("status") == "done":
            print(f"[checkpointed_runner] SKIP (already done): {key}")
            continue
        print(f"[checkpointed_runner] RUN: {key}")
        try:
            result = job_fn(job)
        except Exception as exc:
            manifest[key] = {"status": "failed", "error": str(exc)}
            save_manifest(manifest_path, manifest)
            raise
        manifest[key] = {"status": "done", "result_summary": result}
        save_manifest(manifest_path, manifest)
    return manifest
