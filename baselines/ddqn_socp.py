"""Double Deep Q-Network + Convex SOCP Power Allocation Baseline (Iqbal et al., 2021).

Two-stage architecture:
- Stage 1: DDQN policy selects discrete RRH activation vector k_r.
- Stage 2: Convex SOCP / SciPy solver optimizes continuous transmit power given active RRHs.
"""

from typing import Any, Dict, Optional, Union
import numpy as np

from agents.ddqn_agent import DDQNAgent
from baselines.convex_power import ConvexPowerBaseline


class DDQNSOCPBaseline:
    """Two-Stage DDQN + Convex Power Allocation Baseline (Iqbal et al., 2021)."""

    def __init__(
        self,
        state_dim: int,
        n_rrh: int,
        n_ue: int,
        p_max_w: float = 1.0,
        config: Optional[Union[dict, Any]] = None,
    ):
        self.n_rrh = n_rrh
        self.n_ue = n_ue
        self.p_max_w = p_max_w
        self.ddqn = DDQNAgent(state_dim=state_dim, n_rrh=n_rrh, p_max_w=p_max_w)
        self.convex_solver = ConvexPowerBaseline(
            n_rrh=n_rrh, n_ue=n_ue, p_max_w=p_max_w
        )

    def select_action(
        self, obs: np.ndarray, evaluate: bool = False
    ) -> Dict[str, np.ndarray]:
        """Select joint action: Stage 1 DDQN discrete activation + Stage 2 Convex power."""
        # Stage 1: DDQN discrete selection
        ddqn_action = self.ddqn.select_action(obs, evaluate=evaluate)
        rrh_on = ddqn_action["rrh_on"]

        # Stage 2: Convex power allocation for selected active RRHs
        gains_flat = obs[: self.n_rrh * self.n_ue]
        gains_matrix = gains_flat.reshape(self.n_rrh, self.n_ue)
        power = self.convex_solver.solve_power_allocation(rrh_on, gains_matrix)
        bandwidth = np.ones(self.n_rrh, dtype=np.float32) / max(1, np.sum(rrh_on))

        return {"rrh_on": rrh_on, "power": power, "bandwidth": bandwidth}

    def update(self) -> Dict[str, float]:
        """Update discrete DDQN policy stage."""
        return self.ddqn.update()
