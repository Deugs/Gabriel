"""Traffic Model for O-RAN Simulation.

Implements a deterministic trapezoidal daily arrival-rate envelope with a
stochastic per-UE Poisson arrival count on top -- the traffic model this
track's concept note specifies ("time-varying Poisson arrival with a daily
trapezoidal pattern", ORAN_BMPP_DQN_Concept_Note_v1.md Section 5.1),
structurally analogous to (deterministic diurnal shape x stochastic per-UE
draw) but with a different shape (trapezoidal, not dual-Gaussian) and a
different stochastic law (Poisson arrival count, not log-normal
burstiness) than cran_env/traffic_model.py. Zero imports from cran_env/.

All numeric breakpoints/rate constants are needs-validation placeholders
per the concept note's Section 10 implementation addendum.
"""

import numpy as np


class ORANTrafficModel:
    """Trapezoidal-Poisson Traffic Demand Model for O-RAN UEs.

    Attributes:
        n_ue (int): Number of User Equipments.
        lambda_peak (float): Peak Poisson arrival rate (packets/step) per UE.
        floor_ratio (float): Off-peak floor as a fraction of lambda_peak.
        packet_size_bits (float): Size of one arrival "packet" in bits.
        step_duration_s (float): Wall-clock duration of one env step, in
            seconds -- used to convert an arrival count into a bps rate.
        t1, t2, t3, t4 (float): Trapezoid breakpoints (hour of day, 0-24):
            rise starts at t1, plateau from t2 to t3, falls to the floor by
            t4; floor holds for the remaining hours (wrapping past midnight).
    """

    def __init__(
        self,
        n_ue: int,
        lambda_peak: float = 5.0,
        floor_ratio: float = 0.2,
        packet_size_bits: float = 1.0e6,
        step_duration_s: float = 0.1,
        t1: float = 7.0,
        t2: float = 10.0,
        t3: float = 20.0,
        t4: float = 23.0,
    ):
        self.n_ue = n_ue
        self.lambda_peak = lambda_peak
        self.lambda_floor = floor_ratio * lambda_peak
        self.packet_size_bits = packet_size_bits
        self.step_duration_s = step_duration_s
        self.t1, self.t2, self.t3, self.t4 = t1, t2, t3, t4

    def _envelope(self, hour: float) -> float:
        """Deterministic trapezoidal arrival-rate envelope at a given hour."""
        t = hour % 24.0
        if self.t2 <= t < self.t3:
            return self.lambda_peak
        if self.t1 <= t < self.t2:
            frac = (t - self.t1) / (self.t2 - self.t1)
            return self.lambda_floor + (self.lambda_peak - self.lambda_floor) * frac
        if self.t3 <= t < self.t4:
            frac = (t - self.t3) / (self.t4 - self.t3)
            return self.lambda_peak - (self.lambda_peak - self.lambda_floor) * frac
        return self.lambda_floor

    def get_demands(self, hour: float, rng: np.random.Generator) -> np.ndarray:
        """Compute user data rate demands in bps for a given hour of day.

        Args:
            hour (float): Hour of the day (0 to 24, wraps via modulo).
            rng (np.random.Generator): NumPy random number generator.

        Returns:
            np.ndarray: Vector of user data rate demands in bps (n_ue,).
        """
        lam = self._envelope(hour)
        counts = rng.poisson(lam=lam, size=self.n_ue)
        return counts * self.packet_size_bits / self.step_duration_s
