"""Greedy Channel-Aware Heuristic Baseline for C-RAN Simulation.

Greedily selects minimum RRHs with strongest channel gains to satisfy current traffic demands.
"""

from typing import Dict
import numpy as np


class GreedyHeuristicBaseline:
    """Greedy Channel-Quality RRH Selection Baseline.

    Attributes:
        n_rrh (int): Number of Remote Radio Heads.
        n_ue (int): Number of User Equipments.
        p_max_w (float): Maximum transmit power per RRH in Watts.
    """

    def __init__(self, n_rrh: int, n_ue: int, p_max_w: float = 1.0):
        self.n_rrh = n_rrh
        self.n_ue = n_ue
        self.p_max_w = p_max_w

    def select_action(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Select action based on greedy channel quality and traffic demand heuristic.

        Args:
            obs (np.ndarray): State observation containing channel gains magnitude and demands.

        Returns:
            Dict[str, np.ndarray]: Action dict with 'rrh_on' and 'power'.
        """
        # Extract channel gains magnitude (R*U) and user demands (U) from observation vector
        gains_flat = obs[: self.n_rrh * self.n_ue]
        gains_matrix = gains_flat.reshape(self.n_rrh, self.n_ue)

        demands_start = self.n_rrh * self.n_ue + self.n_rrh
        demands_end = demands_start + self.n_ue
        demands_mbps = obs[demands_start:demands_end]

        total_demand_mbps = float(np.sum(demands_mbps))

        # Compute aggregate channel quality for each RRH
        rrh_scores = np.sum(gains_matrix**2, axis=1)
        sorted_rrh_indices = np.argsort(-rrh_scores)

        # Estimate number of RRHs needed (scale with traffic demand)
        max_demand_threshold = 300.0  # Mbps reference
        frac_needed = min(1.0, max(0.2, total_demand_mbps / max_demand_threshold))
        num_to_activate = max(1, int(np.ceil(frac_needed * self.n_rrh)))

        rrh_on = np.zeros(self.n_rrh, dtype=int)
        active_indices = sorted_rrh_indices[:num_to_activate]
        rrh_on[active_indices] = 1

        # Power allocation: scale transmit power with demand
        power = np.zeros(self.n_rrh, dtype=np.float32)
        power_level = float(
            min(1.0, max(0.1, total_demand_mbps / max_demand_threshold))
        )
        power[active_indices] = power_level * self.p_max_w

        return {"rrh_on": rrh_on, "power": power}
