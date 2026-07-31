"""All-ON Uniform Power Baseline for C-RAN Simulation.

Baseline policy that keeps all RRHs active and allocates uniform maximum transmit power.
Used to establish energy upper bound.
"""

from typing import Dict
import numpy as np


class AllOnUniformBaseline:
    """All-ON Uniform Power Allocation Baseline.

    Attributes:
        n_rrh (int): Number of Remote Radio Heads.
        p_max_w (float): Maximum transmit power per RRH in Watts.
    """

    def __init__(self, n_rrh: int, p_max_w: float = 1.0):
        self.n_rrh = n_rrh
        self.p_max_w = p_max_w

    def select_action(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Select action for current observation.

        Args:
            obs (np.ndarray): Environment state observation.

        Returns:
            Dict[str, np.ndarray]: Action dict with 'rrh_on' and 'power'.
        """
        rrh_on = np.ones(self.n_rrh, dtype=int)
        power = np.full(self.n_rrh, self.p_max_w, dtype=np.float32)

        return {"rrh_on": rrh_on, "power": power}
