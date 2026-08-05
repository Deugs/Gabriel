"""Scalability sweep across the five RRH counts Concept Note v4.0 mandates
(Section 12.2/15): R = {5, 12, 20, 35, 50}. R=50 is a stretch goal per the
concept note's revised timeline, but is still included here since this
function's caller controls episode budget via `episodes`.

Each size has its own committed, independently-runnable config file
(config/small_network.yaml, config/default.yaml, config/rrh20_network.yaml,
config/rrh35_network.yaml, config/large_network.yaml) rather than a
tempfile+deepcopy override -- R is a small, fixed, concept-note-mandated set,
not an open-ended exploratory sweep, so each point deserves the same
first-class, git-committed status as the two sizes that already had files.

For each size: train, save a checkpoint, reload it via
training.checkpoint_utils.load_checkpoint, and benchmark pure inference
latency on the *loaded* (not in-memory) agent -- this both answers the
concept note's latency requirement and exercises the checkpoint round-trip as
a regression check on every scalability run.
"""

from pathlib import Path
from typing import Dict

import yaml  # type: ignore[import-untyped]

from agents import BranchingMPDQN
from evaluation.inference_latency import benchmark_inference_latency
from evaluation.plot_utils import plot_scalability_analysis
from training.checkpoint_utils import load_checkpoint
from training.train_hybrid import train_hybrid_agent

# Matches the "algorithm" label training.train_hybrid.train_hybrid_agent
# actually writes for the proposed method (BranchingMPDQN).
_PROPOSED_ALGO_LABEL = "Branching_MP_DQN"

SCALABILITY_CONFIGS = {
    "R5": "config/small_network.yaml",
    "R12": "config/default.yaml",
    "R20": "config/rrh20_network.yaml",
    "R35": "config/rrh35_network.yaml",
    "R50": "config/large_network.yaml",
}


def analyze_scalability(
    episodes: int = 20,
    save_dir: str = "thesis/figures",
    configs: Dict[str, str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Train + benchmark BranchingMPDQN across the scalability-sweep sizes.

    `configs` defaults to SCALABILITY_CONFIGS (all five R={5,12,20,35,50}
    points); pass a subset for a faster smoke test.
    """
    if configs is None:
        configs = SCALABILITY_CONFIGS

    scalability_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("--- Starting Scalability Analysis ---")

    for scale_name, cfg_path in configs.items():
        print(f"\nEvaluating Scale: {scale_name} ({cfg_path})")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        ckpt_dir = save_path / "checkpoints" / scale_name
        res = train_hybrid_agent(
            config_path=cfg_path,
            seed=42,
            episodes=episodes,
            eval_freq=episodes,
            save_dir=str(ckpt_dir),
        )

        n_rrh = int(cfg.get("network", {}).get("n_rrh", 0))
        checkpoint_path = ckpt_dir / "branching_mp_dqn_seed42" / "final_model.pt"
        agent = load_checkpoint(BranchingMPDQN, checkpoint_path)
        latency = benchmark_inference_latency(agent, cfg, n_trials=200)

        scalability_results[scale_name] = {
            _PROPOSED_ALGO_LABEL: {
                "power": float(res["final_eval_power_w"]),
                "time": float(latency["mean_latency_ms"]),
                "n_rrh": n_rrh,
                "p95_latency_ms": float(latency["p95_latency_ms"]),
            }
        }

    plot_scalability_analysis(
        scalability_results, save_path=str(save_path / "scalability_analysis.pdf")
    )
    print(
        f"Saved scalability analysis figure to {save_path / 'scalability_analysis.pdf'}"
    )

    return scalability_results


if __name__ == "__main__":
    analyze_scalability(episodes=10)
