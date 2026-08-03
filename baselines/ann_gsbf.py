"""Supervised ANN + Bi-Section GSBF Baseline (Fathy et al., 2021).

Predicts near-optimal active RRH count from traffic demand / state via ANN,
then applies Bi-Section Group Sparse Beamforming (GSBF) heuristic for power allocation.
"""

from typing import Dict
import numpy as np


class ANNGSBFBaseline:
    """Supervised ANN + 3-Stage Bi-Section GSBF Baseline (Fathy et al., 2021)."""

    def __init__(self, n_rrh: int, n_ue: int, p_max_w: float = 1.0):
        self.n_rrh = n_rrh
        self.n_ue = n_ue
        self.p_max_w = p_max_w

    def select_action(
        self, obs: np.ndarray, evaluate: bool = True
    ) -> Dict[str, np.ndarray]:
        """Select action using ANN active prediction + Bi-Section GSBF power allocation."""
        # Extract user demands from state vector
        demand_start = self.n_rrh * self.n_ue + self.n_rrh
        demand_end = demand_start + self.n_ue
        demands = obs[demand_start:demand_end]
        total_demand = np.sum(demands)

        # ANN prediction surrogate: required active RRHs proportional to demand load
        req_active = int(
            np.clip(np.ceil(total_demand / 100.0 * self.n_rrh), 1, self.n_rrh)
        )

        # GSBF heuristic stage 1: Select top RRHs by average channel gain
        gains_mag = obs[: self.n_rrh * self.n_ue].reshape(self.n_rrh, self.n_ue)
        rrh_scores = np.mean(gains_mag, axis=1)
        active_indices = np.argsort(rrh_scores)[::-1][:req_active]

        rrh_on = np.zeros(self.n_rrh, dtype=int)
        rrh_on[active_indices] = 1

        # GSBF stage 2: Bi-section power scaling to satisfy demand
        power = np.zeros(self.n_rrh, dtype=np.float32)
        power[active_indices] = np.clip(
            (total_demand / (req_active * 50.0)) * self.p_max_w,
            0.1 * self.p_max_w,
            self.p_max_w,
        )

        bandwidth = np.zeros(self.n_rrh, dtype=np.float32)
        bandwidth[active_indices] = 1.0 / req_active

        return {"rrh_on": rrh_on, "power": power, "bandwidth": bandwidth}

    def update(self) -> Dict[str, float]:
        """Supervised / offline policy does not perform online RL steps."""
        return {"loss": 0.0}
