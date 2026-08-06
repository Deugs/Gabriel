"""DRL Agents Package for C-RAN Simulation."""

from agents.branching_mp_dqn import BranchingMPDQN
from agents.ddpg_agent import DDPGAgent
from agents.ddqn_agent import DDQNAgent
from agents.mpdqn_agent import MPDQNAgent
from agents.pdqn_agent import PDQNAgent

__all__ = [
    "BranchingMPDQN",
    "DDQNAgent",
    "DDPGAgent",
    "PDQNAgent",
    "MPDQNAgent",
]

try:
    # Superseded by BranchingMPDQN (see agents/branching_mp_dqn.py); kept only
    # for tests/test_hybrid_agent.py and historical reference. Imported
    # defensively so a future break in this no-longer-used-in-production
    # module can't take down imports of the active agents above.
    from agents.hybrid_sac_dqn import (
        ContinuousActor,
        DiscreteActor,
        HybridCritic,
        HybridReplayBuffer,
        HybridSACDDQN,
    )

    __all__ += [
        "HybridSACDDQN",
        "DiscreteActor",
        "ContinuousActor",
        "HybridCritic",
        "HybridReplayBuffer",
    ]
except Exception as e:  # noqa: BLE001 - deliberately broad, see comment above
    import warnings

    warnings.warn(
        f"agents.hybrid_sac_dqn (superseded Hybrid SAC-DDQN) failed to import: "
        f"{e}. The active agents (BranchingMPDQN, DDQNAgent, DDPGAgent, "
        f"PDQNAgent, MPDQNAgent) are unaffected.",
        stacklevel=2,
    )
