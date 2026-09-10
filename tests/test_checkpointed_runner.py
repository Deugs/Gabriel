"""Unit tests for training/checkpointed_runner.py's resumability logic, plus
a lightweight end-to-end smoke test of scripts/run_checkpointed_matrix.py's
job wiring (real training calls, but at a trivial episode count)."""

import json
from pathlib import Path

import pytest

from training.checkpointed_runner import load_manifest, run_checkpointed, save_manifest


def test_run_checkpointed_executes_every_job_once(tmp_path):
    manifest_path = str(tmp_path / "manifest.json")
    calls = []

    def job_fn(job):
        calls.append(job["id"])
        return {"ok": True}

    jobs = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    run_checkpointed(jobs, manifest_path, job_fn, job_key_fn=lambda j: j["id"])

    assert calls == ["a", "b", "c"]
    manifest = load_manifest(manifest_path)
    assert all(manifest[k]["status"] == "done" for k in ("a", "b", "c"))


def test_run_checkpointed_skips_jobs_already_marked_done(tmp_path):
    manifest_path = str(tmp_path / "manifest.json")
    # Simulate a prior run that already completed job "a".
    save_manifest(manifest_path, {"a": {"status": "done", "result_summary": {}}})

    calls = []

    def job_fn(job):
        calls.append(job["id"])
        return {"ok": True}

    jobs = [{"id": "a"}, {"id": "b"}]
    run_checkpointed(jobs, manifest_path, job_fn, job_key_fn=lambda j: j["id"])

    # "a" must NOT be re-run -- this is the whole point of resumability.
    assert calls == ["b"]


def test_run_checkpointed_records_failure_without_marking_done(tmp_path):
    manifest_path = str(tmp_path / "manifest.json")

    def job_fn(job):
        if job["id"] == "bad":
            raise RuntimeError("boom")
        return {"ok": True}

    jobs = [{"id": "good"}, {"id": "bad"}]
    with pytest.raises(RuntimeError):
        run_checkpointed(jobs, manifest_path, job_fn, job_key_fn=lambda j: j["id"])

    manifest = load_manifest(manifest_path)
    assert manifest["good"]["status"] == "done"
    assert manifest["bad"]["status"] == "failed"
    # A subsequent re-run must retry "bad" (not silently skip a failed job).
    calls = []

    def job_fn_retry(job):
        calls.append(job["id"])
        return {"ok": True}

    run_checkpointed(jobs, manifest_path, job_fn_retry, job_key_fn=lambda j: j["id"])
    assert calls == ["bad"]  # "good" is skipped (already done), "bad" retried


def test_manifest_write_is_atomic_via_rename(tmp_path):
    manifest_path = str(tmp_path / "manifest.json")
    save_manifest(manifest_path, {"x": {"status": "done"}})
    assert not (tmp_path / "manifest.json.tmp").exists()
    assert json.loads(Path(manifest_path).read_text()) == {"x": {"status": "done"}}


def test_checkpointed_matrix_script_resumes_across_two_invocations(tmp_path):
    """End-to-end smoke test: build the real C-RAN job wiring
    (scripts/run_checkpointed_matrix.py's helpers) for two cheap, non-DRL
    baseline methods at a trivial episode count, run the checkpointed
    runner twice against the same manifest, and confirm the second
    invocation skips both jobs (already done) rather than re-running them."""
    save_dir = str(tmp_path / "cran_matrix")

    from scripts.run_checkpointed_matrix import build_jobs, _make_cran_job_fn
    from training.checkpointed_runner import run_checkpointed, load_manifest

    manifest_path = str(Path(save_dir) / "manifest.json")
    jobs = build_jobs(
        methods=["all_on", "greedy"], seeds=[42], episodes=2, save_root=save_dir
    )
    job_fn = _make_cran_job_fn("config/default.yaml")

    calls = []

    def counting_job_fn(job):
        calls.append(job["method"])
        return job_fn(job)

    run_checkpointed(
        jobs,
        manifest_path,
        counting_job_fn,
        job_key_fn=lambda j: f"{j['method']}/seed{j['seed']}",
    )
    assert calls == ["all_on", "greedy"]

    # Second "run" (same manifest) must skip both -- neither method re-runs.
    calls_second_run = []
    run_checkpointed(
        jobs,
        manifest_path,
        lambda job: calls_second_run.append(job["method"]),
        job_key_fn=lambda j: f"{j['method']}/seed{j['seed']}",
    )
    assert calls_second_run == []

    manifest = load_manifest(manifest_path)
    assert manifest["all_on/seed42"]["status"] == "done"
    assert manifest["greedy/seed42"]["status"] == "done"
