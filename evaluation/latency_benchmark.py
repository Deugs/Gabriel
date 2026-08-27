"""Inference-Time Latency Benchmark (Concept Note v3.0/v4.0 Section 12.3, A3/G14).

Measures forward-pass (policy decision) latency in isolation from training or
full environment-step cost, at the scalability-sweep sizes R = 5, 12, 20, 35, 50
(Section 12.2's table), which bracket the reviewer's requested 5/10/25/50
range. Reported against Fathy et al. (2021, Table II)'s ~24-minute heuristic
and ~11-minute ANN-assisted benchmarks (for the *entire* offline solve on
their hardware, not a per-decision cost — this benchmark instead reports a
per-decision forward-pass time in milliseconds, the relevant quantity for a
real-time control loop).

P-DQN/MP-DQN are included where tractable (R<=12, Section 10.3.1/B3) and
skipped with an explicit note above that size — the resulting gap in the
comparison is itself part of the documented case for branching, not missing
data.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml  # type: ignore[import-untyped]

from agents import BranchingMPDQN, DDQNAgent, MPDQNAgent, PDQNAgent
from cran_env import CRANEnv
from evaluation.plot_utils import plot_degradation_curve

SCALABILITY_SWEEP_N_RRH = (5, 12, 20, 35, 50)

# Same R->U pairing as evaluation/scalability.py's `scales` table, so latency
# results at a given R are measured under the identical n_ue that method's
# scalability-sweep result at that R used (comparable state/action dims).
_SCALABILITY_SWEEP_N_UE_BY_N_RRH = {5: 2, 12: 10, 20: 20, 35: 25, 50: 30}

_AGENT_FACTORIES = {
    "branching_mp_dqn": lambda env, cfg: BranchingMPDQN(
        state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
    ),
    "ddqn": lambda env, cfg: DDQNAgent(
        state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w
    ),
    "pdqn": lambda env, cfg: PDQNAgent(
        state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
    ),
    "mpdqn": lambda env, cfg: MPDQNAgent(
        state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
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
    config_path: str = "config/default.yaml",
    methods: Optional[List[str]] = None,
    n_rrh_values: Optional[List[int]] = None,
    n_repeats: int = 50,
    save_dir: str = "thesis/figures",
) -> Dict[str, Dict[int, Optional[float]]]:
    """Measure per-decision forward-pass latency (ms) for each method at each R."""
    if methods is None:
        methods = ["branching_mp_dqn", "ddqn", "pdqn", "mpdqn"]
    if n_rrh_values is None:
        n_rrh_values = list(SCALABILITY_SWEEP_N_RRH)

    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    results: Dict[str, Dict[int, Optional[float]]] = {m: {} for m in methods}

    for n_rrh in n_rrh_values:
        cfg = dict(base_cfg)
        cfg["network"] = dict(base_cfg.get("network", {}))
        cfg["network"]["n_rrh"] = n_rrh
        # Same n_ue as evaluation/scalability.py's sweep at this R, falling
        # back to max(2, n_rrh) only for an R outside that table's five sizes.
        cfg["network"]["n_ue"] = _SCALABILITY_SWEEP_N_UE_BY_N_RRH.get(
            n_rrh, max(2, n_rrh)
        )

        env = CRANEnv(cfg)
        obs, _ = env.reset(seed=42)

        for method in methods:
            try:
                agent = _AGENT_FACTORIES[method](env, cfg)
            except ValueError as exc:
                print(f"  R={n_rrh:3d} | {method:16s} | SKIPPED (intractable): {exc}")
                results[method][n_rrh] = None
                continue

            latency_ms = _measure_forward_pass_ms(agent, obs, n_repeats=n_repeats)
            results[method][n_rrh] = latency_ms
            print(f"  R={n_rrh:3d} | {method:16s} | {latency_ms:8.3f} ms/decision")

    curve = {m: {r: v for r, v in results[m].items() if v is not None} for m in methods}
    save_path = Path(save_dir)
    plot_degradation_curve(
        curve,
        xlabel="Number of RRHs (R)",
        ylabel="Forward-pass latency (ms/decision)",
        title="Inference-Time Latency vs. Network Scale",
        save_path=str(save_path / "latency_benchmark.pdf"),
    )

    return results


if __name__ == "__main__":
    run_latency_benchmark(n_repeats=10)
