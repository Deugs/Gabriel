"""Extended Multi-Seed Training Sweeps for 5G C-RAN Energy Optimization."""

import argparse
import time

from evaluation.convergence import analyze_convergence
from training.train_baselines import run_baseline_benchmarks
from training.train_hybrid import train_hybrid_agent


def run_extended_sweeps(
    config_path: str = "config/default.yaml",
    episodes: int = 150,
    seeds: list = [42, 123, 456, 789, 1011, 1337, 2024, 2718, 3141, 4242],
    results_dir: str = "data/results",
):
    """Execute extended training sweeps across all evaluation seeds."""
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
    print("\n>>> Step 2/3: Training Proposed Branching MP-DQN + TD3 Agent Across Seeds...")
    for s in seeds:
        print(f"\n--- Training Seed {s} ({episodes} Episodes) ---")
        train_hybrid_agent(
            config_path=config_path,
            seed=s,
            episodes=episodes,
            eval_freq=10,
            save_dir=results_dir,
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
    args = parser.parse_args()

    run_extended_sweeps(config_path=args.config, episodes=args.episodes)
