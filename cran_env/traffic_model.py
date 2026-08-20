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
        profile (str): Traffic profile — "weekday_urban" (default, business peak at
            11:00 + residential peak at 20:00) or "weekend_suburban" (Concept Note
            v4.0 Section 12.8's cross-profile generalization variant: flatter
            daytime demand, later and lower residential peak, no business peak).
    """

    def __init__(
        self,
        n_ue: int,
        base_rate_mbps: float = 50.0,
        peak_multiplier: float = 3.0,
        burstiness_sigma: float = 0.2,
        profile: str = "weekday_urban",
    ):
        self.n_ue = n_ue
        self.base_rate_bps = base_rate_mbps * 1e6
        self.peak_multiplier = peak_multiplier
        self.burstiness_sigma = burstiness_sigma
        if profile not in ("weekday_urban", "weekend_suburban"):
            raise ValueError(
                f"Unknown traffic profile '{profile}'; expected 'weekday_urban' or "
                "'weekend_suburban' (Concept Note v4.0 Section 12.8)."
            )
        self.profile = profile

    def get_demands(self, hour: int, rng: np.random.Generator) -> np.ndarray:
        """Compute user data rate demands in bps for a given hour of the day (0-23).

        Args:
            hour (int): Hour of the day (0 to 23).
            rng (np.random.Generator): NumPy random number generator.

        Returns:
            np.ndarray: Vector of user data rate demands in bps (n_ue,).
        """
        t = float(int(hour) % 24)

        if self.profile == "weekend_suburban":
            # Flatter daytime profile (no distinct business peak), later and
            # lower residential peak (23:00 instead of 20:00). Baseline 0.25
            # is above weekday_urban's 0.15 floor (flatter), but the 0.20
            # swing keeps the peak (0.45) below weekday_urban's residential
            # peak alone (~0.50) and well below its overall business-hour
            # peak (~0.66) -- previously 0.25 + 0.45 * residential_peak
            # gave a peak of 0.70, higher than weekday_urban's peak, which
            # contradicted the "lower residential peak" description above.
            business_peak = 0.0
            residential_peak = np.exp(-((t - 23.0) ** 2) / 30.0)
            diurnal_factor = 0.25 + 0.20 * residential_peak
        else:
            # Dual Gaussian peaks: Business peak at 11:00, Residential peak at 20:00
            business_peak = np.exp(-((t - 11.0) ** 2) / 18.0)
            residential_peak = np.exp(-((t - 20.0) ** 2) / 18.0)

            # Diurnal factor: 0.15 floor + a weighted sum of the two Gaussian
            # peaks, each scaled by 0.85. Since the peaks are centered at
            # different hours, they never both reach 1 simultaneously, so the
            # achievable range is [0.15, ~0.664] (at t=11: 0.15 + 0.85*(0.6*1
            # + 0.4*exp(-81/18)) ~= 0.664), not the full [0.15, 1.0] the
            # weights alone might suggest.
            diurnal_factor = 0.15 + 0.85 * (
                0.6 * business_peak + 0.4 * residential_peak
            )

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
