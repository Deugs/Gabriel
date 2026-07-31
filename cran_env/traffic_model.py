"""Traffic Model for 5G C-RAN Simulation.

Implements a 24-hour tidal traffic demand model with business/residential peaks
and log-normal demand burstiness.
"""

import numpy as np


class TrafficModel:
    """Tidal Traffic Demand Model for C-RAN Users.

    Attributes:
        n_ue (int): Number of User Equipments.
        base_rate_bps (float): Base traffic rate in bits per second (default: 50 Mbps).
        peak_multiplier (float): Peak traffic multiplier (default: 3.0).
        burstiness_sigma (float): Log-normal burstiness standard deviation (default: 0.2).
    """

    def __init__(
        self,
        n_ue: int,
        base_rate_mbps: float = 50.0,
        peak_multiplier: float = 3.0,
        burstiness_sigma: float = 0.2,
    ):
        self.n_ue = n_ue
        self.base_rate_bps = base_rate_mbps * 1e6
        self.peak_multiplier = peak_multiplier
        self.burstiness_sigma = burstiness_sigma

    def get_demands(self, hour: int, rng: np.random.Generator) -> np.ndarray:
        """Compute user data rate demands in bps for a given hour of the day (0-23).

        Args:
            hour (int): Hour of the day (0 to 23).
            rng (np.random.Generator): NumPy random number generator.

        Returns:
            np.ndarray: Vector of user data rate demands in bps (n_ue,).
        """
        t = float(int(hour) % 24)

        # Dual Gaussian peaks: Business peak at 11:00, Residential peak at 20:00
        business_peak = np.exp(-((t - 11.0) ** 2) / 18.0)
        residential_peak = np.exp(-((t - 20.0) ** 2) / 18.0)

        # Diurnal factor normalized to [0.15, 1.0]
        diurnal_factor = 0.15 + 0.85 * (0.6 * business_peak + 0.4 * residential_peak)

        # Base rate scaled by diurnal profile and peak multiplier
        effective_base = (
            self.base_rate_bps * diurnal_factor * (self.peak_multiplier / 3.0)
        )

        # Log-normal random fluctuation (burstiness) per user
        burstiness = rng.lognormal(
            mean=0.0, sigma=self.burstiness_sigma, size=self.n_ue
        )

        demands_bps = effective_base * burstiness
        return demands_bps
