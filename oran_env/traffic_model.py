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

**2026-08-30 literature check (see docs/daily_log.md's 2026-08-30 entry;
Concept Note Section 10.6)**: re-examined all 8 O-RAN-context sources
already supplied for the power-model checks (power_model.py's own
docstring) for traffic-shape content specifically. No source gives a
matching lambda_peak/floor_ratio/packet_size_bits or exact t1-t4 for a
5G/O-RAN scenario -- that stays genuinely open. One useful, order-of-
magnitude finding: Lassoued & Boujnah 2026 (Computers)'s Figure 7 ("Daily
traffic load variations during a 24 h weekday") shows the same qualitative
diurnal shape this module assumes -- near-zero floor overnight, a morning
rise, a sustained (if noisy) daytime peak, and an evening decline -- with
breakpoint timing roughly consistent with this module's own t1=7 and
t4=23, though its decline looks closer to ~18:00 than this module's t3=20
and its "peak" is noisy/bimodal rather than flat. That figure reports a
generic macro-cellular network's relative occupation rate (%), not a
Poisson arrival rate or bps demand for a 5G/O-RAN small cell, so it
corroborates only the general shape and rough timing, not any of this
module's actual numeric constants. The temporal-Poisson-arrival design
itself is not directly precedented either: OREO's own traffic model uses a
Poisson *point process* for UE spatial positions (not per-step arrival
counts), and a cited work in the MEC/Open RAN survey uses an
"inhomogeneous Poisson point process (without temporal variability)" --
i.e. spatial-only, the opposite structural choice from this module's
temporal/diurnal design. Poisson-based traffic modeling is precedented in
this literature broadly, just not in this specific temporal-arrival form.
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
