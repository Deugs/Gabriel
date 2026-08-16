"""CSI Robustness Evaluation (Concept Note v3.0/v4.0 Section 12.5, S3).

Addresses the thesis's single most significant acknowledged limitation
(perfect CSI at training time) with a bounded, evaluation-only experiment:

1. Train each method (hybrid agent, DDQN, pure-DDPG) under perfect CSI,
   exactly as elsewhere in this codebase.
2. At evaluation time only, perturb every channel-gain magnitude fed to the
   *frozen* trained policy with additive Gaussian noise, ghat = g + n,
   n ~ N(0, sigma^2), for sigma in {0, 0.01, 0.05, 0.1} (sigma=0 reproduces
   the perfect-CSI point on the curve). The environment's own physics
   (SINR, reward) still use the true channel gains — only the observation
   fed to the policy is perturbed, isolating the policy's *sensitivity* to
   CSI error rather than retraining under noise.
3. Report EE (Mbit/Joule) and QoS-violation rate as a function of sigma for
   each method, producing a degradation curve. No retraining occurs at any
   sigma.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml  # type: ignore[import-untyped]

from agents import BranchingMPDQN, DDPGAgent, DDQNAgent
from cran_env import CRANEnv
from evaluation.plot_utils import plot_degradation_curve

DEFAULT_SIGMAS = (0.0, 0.01, 0.05, 0.1)


def _perturb_channel_obs(
    obs: np.ndarray, n_channel: int, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Perturb only the channel-gain-magnitude slice of the observation vector."""
    if sigma <= 0.0:
        return obs
    noisy = obs.copy()
    noisy[:n_channel] = np.maximum(
        0.0, noisy[:n_channel] + rng.normal(0.0, sigma, size=n_channel)
    )
    return noisy


def _train_branching_mp_dqn(
    env: CRANEnv,
    cfg: Dict[str, Any],
    episodes: int,
    batch_size: int,
    checkpoint_path: Optional[str] = None,
) -> BranchingMPDQN:
    agent = BranchingMPDQN(
        state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
    )
    if checkpoint_path is not None:
        # Reuse an already-trained checkpoint (Concept Note v4.0 Section 14)
        # instead of training a fresh agent from scratch.
        agent.load_checkpoint(checkpoint_path)
        return agent
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            cont_action = action.get(
                "continuous",
                np.stack([action["power"] / env.p_max_w, action["bandwidth"]], axis=-1),
            )
            agent.memory.push(
                obs, action["rrh_on"], cont_action, reward, next_obs, terminated
            )
            agent.update(batch_size=batch_size)
            obs = next_obs
            done = terminated or truncated
        agent.decay_exploration()
    return agent


def _train_ddqn(
    env: CRANEnv,
    cfg: Dict[str, Any],
    episodes: int,
    batch_size: int,
    checkpoint_path: Optional[str] = None,
) -> DDQNAgent:
    if checkpoint_path is not None:
        raise NotImplementedError(
            "Checkpoint reuse is only implemented for branching_mp_dqn — "
            "training/train_baselines.py does not save DDQN's model weights "
            "anywhere, so there is no checkpoint format for this agent to "
            "load from."
        )
    agent = DDQNAgent(
        state_dim=env.state_dim,
        n_rrh=env.n_rrh,
        p_max_w=env.p_max_w,
        batch_size=batch_size,
    )
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.memory.push(obs, action["rrh_on"], reward, next_obs, terminated)
            agent.update()
            obs = next_obs
            done = terminated or truncated
        agent.decay_exploration()
    return agent


def _train_ddpg(
    env: CRANEnv,
    cfg: Dict[str, Any],
    episodes: int,
    batch_size: int,
    checkpoint_path: Optional[str] = None,
) -> DDPGAgent:
    if checkpoint_path is not None:
        raise NotImplementedError(
            "Checkpoint reuse is only implemented for branching_mp_dqn — "
            "training/train_baselines.py does not save DDPG's model weights "
            "anywhere, so there is no checkpoint format for this agent to "
            "load from."
        )
    agent = DDPGAgent(
        state_dim=env.state_dim, n_rrh=env.n_rrh, p_max_w=env.p_max_w, config=cfg
    )
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.memory.push(
                obs, action["continuous_action"], reward, next_obs, terminated
            )
            agent.update(batch_size=batch_size)
            obs = next_obs
            done = terminated or truncated
    return agent


_TRAINERS = {
    "branching_mp_dqn": _train_branching_mp_dqn,
    "ddqn": _train_ddqn,
    "ddpg": _train_ddpg,
}


def _evaluate_under_csi_noise(
    env: CRANEnv,
    agent: Any,
    sigma: float,
    eval_episodes: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    n_channel = env.n_rrh * env.n_ue
    ee_values: List[float] = []
    qos_violation_flags: List[float] = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        done = False
        while not done:
            noisy_obs = _perturb_channel_obs(obs, n_channel, sigma, rng)
            action = agent.select_action(noisy_obs, evaluate=True)
            obs, _, terminated, truncated, info = env.step(action)
            ee_values.append(float(info.get("ee_mbit_per_joule", 0.0)))
            qos_violation_flags.append(
                1.0 if info.get("qos_violations_count", 0) > 0 else 0.0
            )
            done = terminated or truncated

    return {
        "ee_mbit_per_joule": float(np.mean(ee_values)) if ee_values else 0.0,
        "qos_violation_rate": (
            float(np.mean(qos_violation_flags)) if qos_violation_flags else 0.0
        ),
    }


def run_csi_robustness_evaluation(
    config_path: str = "config/default.yaml",
    methods: Optional[List[str]] = None,
    sigmas: Optional[List[float]] = None,
    train_episodes: int = 30,
    eval_episodes: int = 5,
    batch_size: int = 64,
    seed: int = 42,
    save_dir: str = "thesis/figures",
    checkpoint_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[float, Dict[str, float]]]:
    """Train each method under perfect CSI, then evaluate under CSI noise sweep.

    checkpoint_paths, when given, maps a method name to an already-trained
    checkpoint file to load instead of training that method from scratch
    (Concept Note v4.0 Section 14's "reuse already-trained checkpoints"
    mitigation) — e.g. {"branching_mp_dqn": "data/results/branching_mp_dqn_seed42/final_model.pt"}.
    Only "branching_mp_dqn" currently has a checkpoint format to load from;
    passing a path for "ddqn"/"ddpg" raises NotImplementedError rather than
    silently ignoring it and training fresh anyway.
    """
    if methods is None:
        methods = ["branching_mp_dqn", "ddqn", "ddpg"]
    if sigmas is None:
        sigmas = list(DEFAULT_SIGMAS)
    if checkpoint_paths is None:
        checkpoint_paths = {}

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(seed)
    results: Dict[str, Dict[float, Dict[str, float]]] = {}

    for method in methods:
        if method not in _TRAINERS:
            raise ValueError(
                f"Unknown method '{method}'; expected one of {list(_TRAINERS)}"
            )

        checkpoint_path = checkpoint_paths.get(method)
        if checkpoint_path is not None:
            print(f"\n--- CSI Robustness: loading {method} from {checkpoint_path} ---")
        else:
            print(f"\n--- CSI Robustness: training {method} under perfect CSI ---")
        env = CRANEnv(deepcopy(cfg))
        agent = _TRAINERS[method](
            env, cfg, train_episodes, batch_size, checkpoint_path=checkpoint_path
        )

        results[method] = {}
        for sigma in sigmas:
            metrics = _evaluate_under_csi_noise(env, agent, sigma, eval_episodes, rng)
            results[method][sigma] = metrics
            print(
                f"  sigma={sigma:.3f} | EE={metrics['ee_mbit_per_joule']:.3f} Mbit/J | "
                f"QoS violation rate={metrics['qos_violation_rate']*100:.1f}%"
            )

    ee_curve = {
        m: {s: results[m][s]["ee_mbit_per_joule"] for s in sigmas} for m in methods
    }
    qos_curve = {
        m: {s: results[m][s]["qos_violation_rate"] for s in sigmas} for m in methods
    }

    save_path = Path(save_dir)
    plot_degradation_curve(
        ee_curve,
        xlabel="CSI perturbation sigma",
        ylabel="Energy Efficiency (Mbit/Joule)",
        title="CSI Robustness: Energy Efficiency vs. Channel Estimation Error",
        save_path=str(save_path / "csi_robustness_ee.pdf"),
    )
    plot_degradation_curve(
        qos_curve,
        xlabel="CSI perturbation sigma",
        ylabel="QoS Violation Rate",
        title="CSI Robustness: QoS Violation Rate vs. Channel Estimation Error",
        save_path=str(save_path / "csi_robustness_qos.pdf"),
    )

    return results


if __name__ == "__main__":
    run_csi_robustness_evaluation(train_episodes=10, eval_episodes=2)
