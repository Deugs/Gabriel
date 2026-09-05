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
module's actual numeric constants.

**2026-08-30 literature check, part 2 -- 3GPP TR 38.864 ("Study on network
energy savings for NR", the actual source document behind the 3GPP
TR 38.864/ETSI TS 103 786 citations noted in power_model.py's own
docstring)**: Annex A defines the real, standard 3GPP traffic models used
for NR system-level simulation -- FTP Model 3 (0.5 MB packet/file size,
200 ms mean inter-arrival time), FTP Model 3 IM (0.1 MB, 2 s mean
inter-arrival), and VoIP. FTP Model 3's file arrivals are themselves a
per-UE Poisson process (a standard 3GPP reading-time model) -- this
directly precedents this module's temporal-Poisson-arrival design (the
2026-08-29 finding above, that no source precedents this, is corrected by
this later, more relevant source: OREO's and the MEC-survey citation's
Poisson processes were spatial-only, but 3GPP's own FTP Model 3 is exactly
the temporal form used here). `lambda_peak` and `packet_size_bits` below
are now set directly from FTP Model 3's own numbers (200 ms mean
inter-arrival = 5 arrivals/s = 0.5 per this module's 0.1 s step; 0.5 MB =
4e6 bits) -- a real primary-source match, not a guess. `floor_ratio` and
`t1`-`t4` remain unvalidated: TR 38.864 Annex A's own "load (L)%" scenarios
(Table A-1: idle=0%, low<=15%, light<=30%, medium<=50%) are instantaneous
PRB-utilization snapshots with no time-of-day association, and the TR's own
scope explicitly stops at "medium load" ("The study prioritizes idle/empty
and low/medium load scenarios") -- it never defines a busy-hour/full-load
reference point, so it gives no floor:peak ratio or diurnal timing to
derive those from. Two things worth flagging for future refinement, not
acted on here: (1) FTP Model 3 IM's lighter traffic (0.1 MB, 2 s
inter-arrival) could plausibly represent off-peak/floor behavior better
than a flat `floor_ratio` scaling of the same `packet_size_bits`, but this
module's structure only supports one packet size for the whole day --
changing that is a design change, not a citation fix, so it's left as a
noted option rather than implemented; (2) the 3-way choice of FTP3 vs.
FTP3-IM vs. VoIP as "the" traffic class for a small-cell O-RAN UE is itself
a modeling decision 3GPP leaves to the evaluating company, so adopting
FTP3 specifically (as the most commonly used baseline in 3GPP energy-
saving evaluations) is a defensible choice, not the only possible one.

**2026-08-30 literature check, part 2**: a 2025 MASc thesis (SK Razib
Ahmed, UBC) cites an ETSI standard (ETSI TR 103 737) for 24-hour power
averaging using three weighted load periods: Busy=6h, Medium=10h, Low=8h
(summing to 24h). This module's own t1=7/t4=23 imply a floor duration of
exactly 24-(23-7)=8 hours and an active (rise+plateau+fall) duration of
exactly 23-7=16 hours -- an exact match to the ETSI standard's Low
duration (8h) and combined Medium+Busy duration (16h). This is a genuine
confirmation of the *aggregate* floor-vs-active day-fraction split these
breakpoints imply, not a coincidence to dismiss -- but it does not fully
validate the four individual breakpoints, since three aggregate durations
underdetermine four specific t-values (many rise/plateau/fall splits
within the same 16-hour window would give the same aggregate durations),
and it says nothing about lambda_peak, floor_ratio, or packet_size_bits.
"""

import numpy as np


class ORANTrafficModel:
    """Trapezoidal-Poisson Traffic Demand Model for O-RAN UEs.

    Attributes:
        n_ue (int): Number of User Equipments.
        lambda_peak (float): Peak Poisson arrival rate (packets/step) per
            UE. Default derived from 3GPP TR 38.864 Annex A's FTP Model 3
            (200 ms mean inter-arrival time = 5 arrivals/s = 0.5 per this
            module's default 0.1 s step) -- see this module's docstring.
        floor_ratio (float): Off-peak floor as a fraction of lambda_peak.
            Still an unvalidated placeholder -- see this module's docstring.
        packet_size_bits (float): Size of one arrival "packet" in bits.
            Default derived from 3GPP TR 38.864 Annex A's FTP Model 3 (0.5
            MB packet/file size = 4e6 bits) -- see this module's docstring.
        step_duration_s (float): Wall-clock duration of one env step, in
            seconds -- used to convert an arrival count into a bps rate.
        t1, t2, t3, t4 (float): Trapezoid breakpoints (hour of day, 0-24):
            rise starts at t1, plateau from t2 to t3, falls to the floor by
            t4; floor holds for the remaining hours (wrapping past midnight).
            Still unvalidated placeholders -- see this module's docstring.
    """

    def __init__(
        self,
        n_ue: int,
        lambda_peak: float = 0.5,
        floor_ratio: float = 0.2,
        packet_size_bits: float = 4.0e6,
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
