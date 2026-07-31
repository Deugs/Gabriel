"""Normalized Minimum Bin Slot (NMBS) Bin-Packing Baseline for C-RAN Simulation.

Implements Al-Zubaedi (2019) NMBS bin-packing heuristic that packs user traffic demands
into minimum number of RRHs/BBUs using First-Fit Decreasing (FFD).
"""

from typing import Dict
import numpy as np


class NMBSBinPackingBaseline:
    """NMBS Bin-Packing Baseline (Al-Zubaedi 2019).

    Attributes:
        n_rrh (int): Number of Remote Radio Heads.
        n_ue (int): Number of User Equipments.
        p_max_w (float): Maximum transmit power per RRH in Watts.
        bin_capacity_mbps (float): Maximum traffic capacity per RRH bin in Mbps.
    """

    def __init__(
        self,
        n_rrh: int,
        n_ue: int,
        p_max_w: float = 1.0,
        bin_capacity_mbps: float = 100.0,
    ):
        self.n_rrh = n_rrh
        self.n_ue = n_ue
        self.p_max_w = p_max_w
        self.bin_capacity_mbps = bin_capacity_mbps

    def select_action(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Select action using First-Fit Decreasing (FFD) bin-packing algorithm.

        Args:
            obs (np.ndarray): State observation.

        Returns:
            Dict[str, np.ndarray]: Action dict with 'rrh_on' and 'power'.
        """
        demands_start = self.n_rrh * self.n_ue + self.n_rrh
        demands_end = demands_start + self.n_ue
        demands_mbps = obs[demands_start:demands_end]

        # Sort user demands in decreasing order (First-Fit Decreasing)
        sorted_demand_indices = np.argsort(-demands_mbps)

        bin_loads = np.zeros(self.n_rrh, dtype=float)
        rrh_on = np.zeros(self.n_rrh, dtype=int)

        for u_idx in sorted_demand_indices:
            demand = float(demands_mbps[u_idx])
            placed = False

            # Try to place demand in an existing open bin
            for r in range(self.n_rrh):
                if rrh_on[r] == 1 and (bin_loads[r] + demand <= self.bin_capacity_mbps):
                    bin_loads[r] += demand
                    placed = True
                    break

            # If no open bin has space, open a new bin
            if not placed:
                for r in range(self.n_rrh):
                    if rrh_on[r] == 0:
                        rrh_on[r] = 1
                        bin_loads[r] += demand
                        placed = True
                        break

        # Ensure at least one RRH is active
        if np.sum(rrh_on) == 0:
            rrh_on[0] = 1

        # Power allocation: scale transmit power with bin load utilization ratio
        power = np.zeros(self.n_rrh, dtype=np.float32)
        active_indices = np.where(rrh_on == 1)[0]

        for r in active_indices:
            utilization = float(
                min(1.0, max(0.15, bin_loads[r] / self.bin_capacity_mbps))
            )
            power[r] = utilization * self.p_max_w

        return {"rrh_on": rrh_on, "power": power}
