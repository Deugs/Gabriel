"""CSI-robustness evaluation (Concept Note v4.0 Section 12.5 / S3).

Trains happen under perfect CSI (unchanged, Section 8's Scope Boundary Rule);
this module only *evaluates* an already-trained, frozen policy with its
observed channel perturbed by additive Gaussian noise, to test how gracefully
it degrades -- it never retrains and never touches the environment's reward/
SINR computation, which always reflects the true channel
(see CRANEnv.observation_noise_std).
"""

from typing import Any, Dict, Sequence, Union

from cran_env import CRANEnv
from training.eval_utils import run_eval_episodes


def evaluate_csi_robustness(
    agent: Any,
    env_config: Union[dict, Any],
    sigmas: Sequence[float] = (0.0, 0.01, 0.05, 0.1),
    episodes: int = 10,
    seed: int = 42,
) -> Dict[float, Dict[str, float]]:
    """Evaluate a frozen `agent` under increasing CSI observation noise.

    For each sigma in `sigmas`, builds a fresh `CRANEnv(env_config,
    observation_noise_std=sigma)` and runs `run_eval_episodes` with the SAME
    seed for every sigma, so the underlying trajectory (positions, true
    channel path, traffic draws) is identical across sigma -- only the
    policy's *observed* channel differs. sigma=0.0 is a structural no-op in
    CRANEnv (see its `_get_obs`), so it reproduces a plain, non-perturbed
    evaluation exactly.

    Returns a dict keyed by sigma, each value the same metrics dict
    `run_eval_episodes` returns.
    """
    results: Dict[float, Dict[str, float]] = {}
    for sigma in sigmas:
        env = CRANEnv(env_config, observation_noise_std=sigma)
        results[float(sigma)] = run_eval_episodes(
            env, agent, episodes=episodes, seed=seed
        )
    return results
