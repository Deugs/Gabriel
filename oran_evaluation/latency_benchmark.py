"""Inference-Time Latency Benchmark for the O-RAN track.

Concept Note Section 6.1/7.1: this track's scope is a single, focused
scenario (single-gNB, not a scalability sweep like the C-RAN track's
R=5..50) -- so this benchmark measures per-decision forward-pass latency
for BMPP-DQN and the 3 baselines (DQN, DDPG, MP-DQN) at the one default
scenario, not across a range of network sizes.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml  # type: ignore[import-untyped]

from oran_agents import BMPPDQNAgent, ORANDDPGAgent, ORANDQNAgent, ORANMPDQNAgent
from oran_env import ORANEnv
from oran_evaluation.plot_utils import plot_bar_comparison

_AGENT_FACTORIES = {
    "bmpp_dqn": lambda env, cfg: BMPPDQNAgent(
        state_dim=env.state_dim,
        n_ru=env.n_ru,
        n_splits=env.n_splits,
        p_max_w=env.p_max_w,
        config=cfg,
    ),
    "dqn": lambda env, cfg: ORANDQNAgent(
        state_dim=env.state_dim,
        n_ru=env.n_ru,
        n_splits=env.n_splits,
        p_max_w=env.p_max_w,
        config=cfg,
    ),
    "ddpg": lambda env, cfg: ORANDDPGAgent(
        state_dim=env.state_dim,
        n_ru=env.n_ru,
        n_splits=env.n_splits,
        p_max_w=env.p_max_w,
        config=cfg,
    ),
    "mpdqn": lambda env, cfg: ORANMPDQNAgent(
        state_dim=env.state_dim,
        n_ru=env.n_ru,
        n_splits=env.n_splits,
        p_max_w=env.p_max_w,
        config=cfg,
    ),
}


def _measure_forward_pass_ms(
    agent: Any, obs: np.ndarray, n_repeats: int = 50, n_warmup: int = 5
) -> float:
    for _ in range(n_warmup):
        agent.select_action(obs, evaluate=True)

    start = time.perf_counter()
    for _ in range(n_repeats):
        agent.select_action(obs, evaluate=True)
    elapsed_s = time.perf_counter() - start
    return (elapsed_s / n_repeats) * 1000.0


def run_latency_benchmark(
    config_path: str = "config/oran_default.yaml",
    methods: Optional[List[str]] = None,
    n_repeats: int = 50,
    save_dir: str = "thesis/figures_oran",
) -> Dict[str, Optional[float]]:
    """Measure per-decision forward-pass latency (ms) for each method at
    this track's single default scenario."""
    if methods is None:
        methods = ["bmpp_dqn", "dqn", "ddpg", "mpdqn"]

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    env = ORANEnv(cfg)
    obs, _ = env.reset(seed=42)

    results: Dict[str, Optional[float]] = {}
    for method in methods:
        try:
            agent = _AGENT_FACTORIES[method](env, cfg)
        except ValueError as exc:
            print(f"  n_ru={env.n_ru:3d} | {method:10s} | SKIPPED (intractable): {exc}")
            results[method] = None
            continue

        latency_ms = _measure_forward_pass_ms(agent, obs, n_repeats=n_repeats)
        results[method] = latency_ms
        print(f"  n_ru={env.n_ru:3d} | {method:10s} | {latency_ms:8.3f} ms/decision")

    bar_data = {m: {"mean": v} for m, v in results.items() if v is not None}
    save_path = Path(save_dir)
    plot_bar_comparison(
        bar_data,
        ylabel="Forward-pass latency (ms/decision)",
        title="O-RAN Track: Inference-Time Latency Comparison",
        save_path=str(save_path / "latency_benchmark_oran.pdf"),
    )

    return results


if __name__ == "__main__":
    run_latency_benchmark(n_repeats=10)
