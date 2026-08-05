"""Single-command experiment runner (docs/rules.md Rule 4: Reproducibility).

Every experiment is defined by a YAML file under experiments/, naming a
dispatch `type` and the underlying config/*.yaml (network/power/reward
parameters) it runs against. This script is the one command Rule 4
requires to reproduce any of them:

    python run_experiment.py --config experiments/hybrid_medium.yaml --seed 42

`--seed` overrides the experiment file's own seed(s) for a single run,
matching how a cloud job array would fan a multi-seed experiment out across
one container per seed. Every run's output directory gets a
`run_manifest.json` recording a SHA-256 hash of both the experiment YAML
and the config/*.yaml it references, so a result can always be tied back
to the exact inputs that produced it — not just a filename.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # type: ignore[import-untyped]

from evaluation.csi_robustness import run_csi_robustness_evaluation
from evaluation.generalization import run_generalization_evaluation
from evaluation.latency_benchmark import run_latency_benchmark
from training.hyperparam_search import run_proxy_sensitivity_sweep
from training.train_baselines import run_baseline_benchmarks
from training.train_hybrid import train_hybrid_agent


def _file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_manifest(save_dir: str, experiment_path: str, spec: Dict[str, Any], seed_override: Optional[int]):
    manifest = {
        "experiment_file": experiment_path,
        "experiment_sha256": _file_sha256(experiment_path),
        "referenced_config_file": spec.get("config"),
        "referenced_config_sha256": _file_sha256(spec["config"]) if spec.get("config") else None,
        "seed_override": seed_override,
        "spec": spec,
    }
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote run manifest (config hash included) to {out_dir / 'run_manifest.json'}")


def run_train_hybrid(spec: Dict[str, Any], seed: Optional[int]):
    seed = seed if seed is not None else int(spec.get("seeds", [42])[0])
    save_dir = spec.get("save_dir", "data/results")
    _write_manifest(save_dir, spec["_experiment_path"], spec, seed)
    return train_hybrid_agent(
        config_path=spec["config"],
        seed=seed,
        episodes=spec["episodes"],
        eval_freq=spec.get("eval_freq", spec["episodes"]),
        save_dir=save_dir,
    )


def run_baseline_matrix(spec: Dict[str, Any], seed: Optional[int]):
    seeds = [seed] if seed is not None else spec.get("seeds")
    save_dir = spec.get("save_dir", "data/results")
    _write_manifest(save_dir, spec["_experiment_path"], spec, seed)
    return run_baseline_benchmarks(
        config_path=spec["config"],
        seeds=seeds,
        episodes=spec["episodes"],
        algorithms=spec.get("algorithms"),
        save_dir=save_dir,
    )


def run_csi_robustness(spec: Dict[str, Any], seed: Optional[int]):
    save_dir = spec.get("save_dir", "thesis/figures")
    _write_manifest(save_dir, spec["_experiment_path"], spec, seed)
    return run_csi_robustness_evaluation(
        config_path=spec["config"],
        methods=spec.get("methods"),
        sigmas=spec.get("sigmas"),
        train_episodes=spec["episodes"],
        eval_episodes=spec.get("eval_episodes", 5),
        seed=seed if seed is not None else spec.get("seed", 42),
        save_dir=save_dir,
    )


def run_generalization(spec: Dict[str, Any], seed: Optional[int]):
    save_dir = spec.get("save_dir", "thesis/figures")
    _write_manifest(save_dir, spec["_experiment_path"], spec, seed)
    return run_generalization_evaluation(
        config_path=spec["config"],
        methods=spec.get("methods"),
        train_episodes=spec["episodes"],
        eval_episodes=spec.get("eval_episodes", 5),
        seed=seed if seed is not None else spec.get("seed", 42),
        save_dir=save_dir,
    )


def run_latency(spec: Dict[str, Any], seed: Optional[int]):
    save_dir = spec.get("save_dir", "thesis/figures")
    _write_manifest(save_dir, spec["_experiment_path"], spec, seed)
    return run_latency_benchmark(
        config_path=spec["config"],
        methods=spec.get("methods"),
        n_rrh_values=spec.get("n_rrh_values"),
        n_repeats=spec.get("n_repeats", 50),
        save_dir=save_dir,
    )


def run_proxy_sweep(spec: Dict[str, Any], seed: Optional[int]):
    save_dir = spec.get("save_dir", "data/results/proxy_sweep")
    _write_manifest(save_dir, spec["_experiment_path"], spec, seed)
    seeds = [seed] if seed is not None else spec.get("seeds", [42, 123])
    return run_proxy_sensitivity_sweep(
        base_config_path=spec["config"],
        episodes=spec["episodes"],
        seeds=seeds,
        save_dir=save_dir,
    )


DISPATCH = {
    "train_hybrid": run_train_hybrid,
    "baseline_matrix": run_baseline_matrix,
    "csi_robustness": run_csi_robustness,
    "generalization": run_generalization,
    "latency_benchmark": run_latency,
    "proxy_sweep": run_proxy_sweep,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run a single experiment defined in experiments/*.yaml (docs/rules.md Rule 4)."
    )
    parser.add_argument(
        "--config", required=True, help="Path to an experiments/*.yaml experiment file"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override the experiment file's seed(s) for this run"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        spec = yaml.safe_load(f)
    spec["_experiment_path"] = args.config

    exp_type = spec.get("type")
    if exp_type not in DISPATCH:
        raise ValueError(
            f"Unknown experiment type '{exp_type}' in {args.config}; expected one of {list(DISPATCH)}"
        )

    print(f"=== Running experiment '{spec.get('name', Path(args.config).stem)}' ({exp_type}) ===")
    DISPATCH[exp_type](spec, args.seed)


if __name__ == "__main__":
    main()
