"""Extended Multi-Seed Training Sweeps for 5G C-RAN Energy Optimization."""

import argparse
from pathlib import Path
import time

from evaluation.convergence import analyze_convergence
from evaluation.csi_robustness import run_csi_robustness_evaluation
from evaluation.generalization import run_generalization_evaluation
from training.train_baselines import run_baseline_benchmarks
from training.train_hybrid import train_hybrid_agent


def run_extended_sweeps(
    config_path: str = "config/default.yaml",
    episodes: int = 150,
    seeds: list = [42, 123, 456, 789, 1011, 1337, 2024, 2718, 3141, 4242],
    results_dir: str = "data/results",
    run_csi_and_generalization: bool = False,
):
    """Execute extended training sweeps across all evaluation seeds.

    run_csi_and_generalization=True additionally runs CSI-robustness
    (evaluation/csi_robustness.py) and cross-profile generalization
    (evaluation/generalization.py) for the proposed method after each seed's
    training completes, reusing that seed's just-trained checkpoint rather
    than training a fresh agent for those evaluations (Concept Note v4.0
    Section 14's "reuse already-trained checkpoints" mitigation). Defaults
    to False so existing callers see no change in runtime/behavior — this
    roughly doubles or triples wall-clock time per seed when enabled.
    """
    print("==================================================")
    print(
        f"Starting Extended Multi-Seed Sweeps ({len(seeds)} seeds, {episodes} episodes)"
    )
    print("==================================================")

    start_time = time.time()

    # 1. Run Baseline Benchmarks
    print("\n>>> Step 1/3: Running Baseline Algorithms Across Seeds...")
    run_baseline_benchmarks(config_path, seeds=seeds, save_dir=results_dir)

    # 2. Run Proposed Branching MP-DQN + TD3 Agent Across Seeds
    print(
        "\n>>> Step 2/3: Training Proposed Branching MP-DQN + TD3 Agent Across Seeds..."
    )
    for s in seeds:
        print(f"\n--- Training Seed {s} ({episodes} Episodes) ---")
        train_hybrid_agent(
            config_path=config_path,
            seed=s,
            episodes=episodes,
            save_dir=results_dir,
        )

        if run_csi_and_generalization:
            checkpoint_path = str(
                Path(results_dir) / f"branching_mp_dqn_seed{s}" / "final_model.pt"
            )
            print(
                f"\n--- Seed {s}: CSI-robustness/generalization (reusing checkpoint) ---"
            )
            run_csi_robustness_evaluation(
                config_path=config_path,
                methods=["branching_mp_dqn"],
                seed=s,
                save_dir=str(Path(results_dir) / f"csi_robustness_seed{s}"),
                checkpoint_paths={"branching_mp_dqn": checkpoint_path},
            )
            run_generalization_evaluation(
                config_path=config_path,
                methods=["branching_mp_dqn"],
                seed=s,
                save_dir=str(Path(results_dir) / f"generalization_seed{s}"),
                checkpoint_paths={"branching_mp_dqn": checkpoint_path},
            )

    # 3. Aggregate Statistical Convergence & Re-render Manuscript Figures
    print("\n>>> Step 3/3: Aggregating Empirical Results & Re-rendering Figures...")
    analyze_convergence(results_dir=results_dir)

    total_time = time.time() - start_time
    print("==================================================")
    print(f"Extended Multi-Seed Sweeps Complete in {total_time / 60.0:.2f} minutes.")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Extended Multi-Seed Training Sweeps"
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Config path"
    )
    parser.add_argument("--episodes", type=int, default=150, help="Episodes per seed")
    parser.add_argument(
        "--run-csi-and-generalization",
        action="store_true",
        help=(
            "Also run CSI-robustness and generalization evaluation per seed, "
            "reusing that seed's just-trained checkpoint"
        ),
    )
    args = parser.parse_args()

    run_extended_sweeps(
        config_path=args.config,
        episodes=args.episodes,
        run_csi_and_generalization=args.run_csi_and_generalization,
    )
