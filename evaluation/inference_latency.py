"""Pure forward-pass inference-latency benchmarking (Concept Note v4.0 Section
12.3 / A3): times ONLY `agent.select_action(obs, evaluate=True)`, never
`agent.update()` and never the environment's own `step()` cost, so the result
reflects deployment-time decision latency, not training or simulation cost.
"""

import time
from typing import Any, Dict, Union

import numpy as np

from cran_env import CRANEnv


def benchmark_inference_latency(
    agent: Any,
    env_config: Union[dict, Any],
    n_warmup: int = 10,
    n_trials: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """Benchmark the wall-clock cost of one `agent.select_action` call.

    Observations are realistic, in-distribution states: the environment is
    actually stepped with the agent's own action between timed calls (that
    step's cost is excluded from the timer), rather than using a fixed or
    synthetic observation.
    """
    env = CRANEnv(env_config)
    obs, _ = env.reset(seed=seed)

    for _ in range(n_warmup):
        action = agent.select_action(obs, evaluate=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    latencies_ms = []
    for _ in range(n_trials):
        start = time.perf_counter()
        action = agent.select_action(obs, evaluate=True)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    latencies_ms = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "n_rrh": env.n_rrh,
        "n_trials": n_trials,
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "std_latency_ms": float(np.std(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
    }
