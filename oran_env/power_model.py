"""Power Consumption Model for O-RAN Simulation.

Implements a per-RU/DU/CU/fronthaul power model, monotonic in the functional
split's centralization level (docs/skills/skill_oran_env.md; Concept Note
ORAN_BMPP_DQN_Concept_Note_v1.md Section 10.2/10.5). All numeric constants
are literature-style placeholders explicitly flagged "needs validation" in
the concept note -- they are chosen only to preserve a monotonic energy
trade-off across splits, not asserted as physically validated figures.

**2026-08-29 literature check (see docs/daily_log.md's 2026-08-29 entry for
the full writeup)**: no supplied source gives a split-level RU/DU/CU/
fronthaul wattage breakdown matching this model's exact parameterization --
that remains a genuine gap. What partial support does exist:
- The RU active-vs-sleep-power structure here mirrors the general EARTH
  linear power model (Auer et al. 2011; independently reproduced in
  Lassoued & Boujnah 2026, Computers 15(50) Table 1) already validated for
  the C-RAN track's BBU model, though that table gives no O-RAN-specific
  RU/DU/CU split figures.
- Eskandarinia et al.'s DQRL clustered-RAN paper models a comparable
  small-cell/mmWave RU active power scale (tens of Watts, scaled down from
  macro-cell figures) and an explicit per-RU activation cost concept
  (P_newRU), broadly consistent with this model's own small p_ru_proc_by_split
  scale and p_switch_ru_w -- order-of-magnitude corroboration, not a
  matching numeric table.
- Qazzaz et al. 2026 (OREO)'s own O-RAN RL energy model explicitly scopes
  itself to RF-only RU power and excludes "auxiliary site power consumption
  such as baseband processing, associated cooling, backhaul equipment"
  (their footnote 1) -- i.e. the DU/CU/fronthaul costs this model includes
  are outside the scope even of directly-comparable recent O-RAN RL
  literature, which is itself evidence that a validated source for exactly
  those numbers is unlikely to already exist and may need to come from
  O-RAN Alliance/vendor hardware measurements instead.

**2026-08-30 literature check (3 more sources; see docs/daily_log.md's
2026-08-30 entry)**: still no matching wattage table, but one genuinely new
result and one important disclosable limitation:
- Rony et al. 2021 (IEEE Access, PHY-layer fronthaul functional-split cost
  analysis) independently confirms the *qualitative direction* this model
  assumes: their Split-A (near-full centralization, "almost no processing at
  RRH") requires the most fronthaul capacity, while their Split-D (least
  centralization, "all PHY layer processing... performed at RRH") requires
  the least -- the same RU-processing-vs-fronthaul-cost trade-off direction
  as this model's c=0..2. Their numbers are CAPEX/OPEX cost percentages
  (e.g. Split-A=40%, Split-D=39% of a per-RU cost-weighting function),
  not power/energy Watts, so this corroborates the trade-off's direction,
  not any of this model's actual wattage constants.
- The Open RAN Handbook 2nd Edition (Vodafone + Keysight, Feb 2025) and the
  Hoffmann/Dryjanski/Kliks (Rimedo Labs/i4y Lab) E2E energy-testing
  presentation both report real measured hardware power figures: Fujitsu
  macro-cell O-RU static power of roughly 200-550 W depending on band/load
  (Handbook Section 6.1.5), and a Dell R750 enterprise server drawing
  roughly 625-780 W across its CPU load range (used in the presentation as
  a stand-in for O-DU/O-CU compute-host power). Both are 20-100x larger
  than this model's own RU/DU/CU placeholder scale (~3-10 W RU processing,
  ~50-70 W DU, ~30-34 W CU). Neither source states what power scale a
  small, n_ru=4-style testbed/simulation scenario like this one should use,
  so no constant has been rescaled from this finding -- it is disclosed
  here as a genuine scale mismatch between this model's placeholders and
  real macro-cell/enterprise-hardware O-RAN deployments, not silently
  patched with a guessed scaling factor.
- Both sources also reconfirm, independently of the OREO footnote above,
  that no O-RAN Alliance normative power-measurement framework yet exists
  (the Hoffmann presentation quotes an O-RAN SuFG technical report stating
  exactly this), and point to real standardized test methodologies this
  model's own linear/step form only loosely resembles: ETSI ES 202 706
  (static base-station power) and ETSI TS 103 786 (dynamic power/energy
  efficiency), with measurements fitted to the 3GPP TR 38.864 static +
  dynamic power-consumption model (scaled by antenna elements, occupied
  RBs/CCs, TRPs, transmit PSD, and occupied symbols per slot).

**2026-08-30 literature check, part 2**: obtained 3GPP TR 38.864 itself
("Study on network energy savings for NR") rather than only secondary
citations of it. Its §5.1 defines the real 3GPP NR BS power model:
`P_DL = P_static,DL + P_dynamic,DL` (and analogously for UL), with the
dynamic term scaled by the fraction of active TRX/RUs, the RF-to-system
bandwidth ratio, and the transmit PSD ratio -- independent confirmation,
from the actual governing 3GPP source rather than a secondary citation,
that a static-plus-scaled-dynamic linear-style structure (the same family
as this model's own form and the EARTH model already used for the C-RAN
track) is the right family of model. Its Table 5.1-3 gives concrete
sleep/active-state power ratios for two "BS Category" classes across three
reference configurations -- e.g. Category 2/Set 1: Deep sleep=1, Micro
sleep=5.5, Active DL=32 -- order-of-magnitude comparable to this model's
own active:sleep ratios, though not identical. This is **not** used to
change any constant here: TR 38.864's model is whole-BS, not disaggregated
into O-RAN's RU/DU/CU/fronthaul components, and Table 5.1-3's values are
relative units with no stated absolute-Watt anchor -- converting them to
Watts would require inventing a scale factor, which was not done. (Compare
oran_env/traffic_model.py's docstring, where the same TR *did* directly
inform two constants -- its Annex A traffic-model definitions are in
absolute, directly-usable units, unlike this power-model section.)

**2026-08-30 literature check, part 3**: obtained a real small-cell O-RU
vendor datasheet -- Benetel's RAN550 (n78/n79, Split 7.2x indoor O-RU) --
plus a HUBER+SUHNER/CubeOptics infographic reproducing 3GPP TR 38.801's
per-split fronthaul bandwidth table. Two results:
- The RAN550 datasheet gives a real *small-cell-class* (not macro, not
  enterprise-server) O-RU total power figure: "Typical power consumption:
  40 W". This is 5-14x this model's own composite active-RU power estimate
  (processing + max-TX/eta, roughly 3-18 W across split levels using this
  model's own constants), narrowing -- not resolving -- the 20-100x scale
  mismatch found against macro-cell/enterprise-hardware figures in the
  2026-08-29/2026-08-30 checks above. It does not decompose the 40 W into
  processing vs. RF vs. fronthaul-interface shares, so it still cannot set
  p_ru_proc_by_split, p_du_per_ru_by_split, or p_fh_per_ru_by_split without
  guessing that decomposition -- not done.
- The RAN550's "Maximum TX output power (total EIRP): 2 W" (33 dBm) *is* a
  clean, same-quantity, same-units match for oran_env/oran_env.py's
  p_max_dbm (the RU transmit-power action-space ceiling, read by this
  model's compute_ru_power() as transmit_power_w) -- updated from 30 dBm
  (1 W, an unvalidated placeholder) to 33 dBm (2 W), citing this datasheet
  directly. See oran_env/oran_env.py and config/oran_default.yaml's own
  comments for this specific change.
- The HUBER+SUHNER infographic reproduces 3GPP TR 38.801's real per-split
  fronthaul bandwidth requirements for exactly the three options this
  model's split-centralization mapping (Concept Note Section 10.2) uses:
  Option 2 = 3/4 Gbps (UL/DL), Option 6 = 7.1/5.6 Gbps, Option 8 = 157.3/
  157.3 Gbps (100 MHz reference; Option 2's figure is on an 8-layer/256QAM
  basis, Option 6/8's on a 32-antenna basis -- the two reference conditions
  differ, a caveat on direct comparison). This quantitatively confirms the
  monotonic direction this model assumes (least-to-most fronthaul need as
  centralization increases), and also reveals that this model's own
  p_fh_per_ru_by_split ratio ([3.0, 6.0, 15.0], a 1:2:5 ratio) is far
  shallower than the real bandwidth-requirement ratio implied by these
  figures (roughly 1:1.4-2.4:39-52, depending on UL/DL). Fronthaul *power*
  does not necessarily scale linearly with fronthaul *bandwidth*
  requirement, and no source states that relationship, so this was not
  used to rescale p_fh_per_ru_by_split -- it is disclosed as a further,
  more precise version of the "monotonic shape confirmed, magnitude
  relationship unconfirmed" finding already on record for this constant.
  See ORAN_BMPP_DQN_Concept_Note_v1.md Section 10.2's own note for the full
  bandwidth table and this citation.

**2026-08-30 literature check, part 4**: obtained Al-Tahmeesschi et al.
2025 ("Enhancing Open RAN Digital Twin Through Power Consumption
Measurement," arXiv:2507.00928) and 3GPP TR 38.801 itself (V0.4.0,
2016-08 -- the primary document behind the Option 2/6/8 definitions and
bandwidth table cited above, not a secondary reproduction).
- TR 38.801's own Annex A Table A-1 gives exact, non-rounded bandwidth
  figures that **exactly cross-validate** the pixel-verified HUBER+SUHNER
  reading above: Option 2 = 4016/3024 Mb/s (DL/UL), Option 6 = 5626.7/7140
  Mb/s, Option 8 = 157.3/157.3 Gb/s. See Concept Note Section 10.2's own
  note for the full table (including sub-options 7a/7b/7c and per-option
  latency, not previously available from a primary source).
- Al-Tahmeesschi et al. 2025 is the first source in either literature-
  check round giving real, *component-decomposed* (RU vs. DU vs. CU) O-RAN
  power measurements. Their Split 8 testbed is an exact match to this
  model's c=2 (both are literally Option 8/PHY-RF split): measured RU
  power (a USRP) is ~43-45 W, essentially load-independent across 0-100%
  PRB utilization; combined DU+CU power (one shared server) is ~119.5-
  141.6 W. This model's own composite c=2 RU estimate (~11 W) is now only
  ~4x below this real measurement -- the closest gap found in either
  round (was 20-100x against macro-cell figures, 5-14x against the RAN550
  datasheet's own "typical" figure in part 3 above). Still not decomposable
  into this model's separate processing/RF terms without guessing that
  split -- not done.
- Their Split 7.2b testbed (not one of this model's three mapped options)
  used *separate* dedicated servers for DU and CU (~187-194 W and ~189.6-
  192.7 W respectively), unlike Split 8's single shared server. This is a
  genuine hardware-choice confound: comparing Split 7.2b's DU/CU figures
  against Split 8's combined DU+CU figure to infer a split-dependent power
  difference would be misleading, since most of the apparent gap reflects
  which server class was used, not split-driven processing cost -- flagged
  explicitly, not used for any inference here.
- The paper's own conclusion -- "power consumption does not scale
  significantly with network load... a large portion of energy consumption
  remains constant regardless of traffic demand" -- independently
  corroborates this model's existing design: compute_du_power() and
  compute_cu_power() already depend on active-RU-count and split choice,
  not on instantaneous PRB/throughput load. No design change needed; cited
  as validation of an existing choice.
- The RAN550's *measured* Split-7.2b RU power here (~28.3-30.1 W) is
  somewhat lower than its own datasheet's "typical power consumption: 40 W"
  claim used for the p_max_dbm fix in part 3 above -- a real-vs-nominal
  discrepancy worth noting; it does not affect that fix (a different
  physical quantity, max TX power vs. typical total consumption).

**2026-08-30 literature check, part 5**: obtained Abubakar et al. 2023
("Energy Efficiency of Open Radio Access Network: A Survey," IEEE
VTC2023-Spring) and a 2025 MASc thesis (SK Razib Ahmed, UBC, on CF-mMIMO
under O-RAN Split 7.2/8).
- Abubakar et al.'s own survey conclusion states that RU-specific and
  transport/fronthaul-specific O-RAN power modeling remains an open
  research gap in the literature at large as of 2023 -- a survey-level
  confirmation that this model's own "still open" status reflects a
  genuine field-wide gap, not a failure of this literature search.
- The same survey cites a real, quantified, split-dependent fronthaul
  *power* percentage (not bandwidth) from Lopez-Perez et al.: for a C-RAN
  with split options 6/7/8, "the transport network contributes about
  2%, 30%, and 60% respectively" of total power. This model's own implied
  fronthaul fraction of total RU+DU+CU+fronthaul power is roughly 11% at
  c=0 and 18% at c=2 (using this model's own default constants) -- well
  below the cited 60% for the most-centralized option, suggesting this
  model likely under-weights fronthaul's power share at high
  centralization. Not used to rescale p_fh_per_ru_by_split (their
  percentage is for a differently-scoped "total power" basis, so
  converting it into this model's absolute Watt terms would need
  additional unstated assumptions), but a further quantification of the
  bandwidth-vs-power-ratio gap already on record in part 3 above.
- The MASc thesis's own CF-mMIMO power model is structurally
  `P_total = P_fixed + P_load` (static + load-dependent) -- the same
  family already cited from 3GPP TR 38.864 and EARTH, further structural
  corroboration, no new numbers. Its closed-form fronthaul data-rate
  formulas for Split 7.2/8 are another independent confirmation of the
  bandwidth-monotonicity direction (in formula form, not fixed numbers).
- Caveat: the two supplied PDF parts of this thesis have a gap (pages
  ~23-50 not included) and stop at page 74, before the thesis's own
  referenced Appendix A ("Table A.1: Maximum Supported APs per DU under
  20 Gbps Fronthaul Budget for Split 7.2 and Split 8," page 115) --
  exactly the kind of concrete numeric table this flag still needs.
  Nothing was inferred to fill this gap; those pages remain the most
  promising lead if supplied later.

This module is fully decoupled from cran_env/power_model.py: no shared code,
no shared imports.
"""

from typing import Optional
import numpy as np


class ORANPowerModel:
    """RU/DU/CU/Fronthaul power model for the O-RAN track.

    Attributes:
        n_ru (int): Number of Radio Units.
        n_splits (int): Number of representative functional split options.
        p_ru_proc_by_split (np.ndarray): Active RU processing power per split
            centralization level c (n_splits,), in Watts.
        p_ru_sleep_w (float): RU sleep-mode power in Watts.
        pa_efficiency (float): Power amplifier drain efficiency.
        p_du_static_w (float): DU static (always-on) power in Watts.
        p_du_per_ru_by_split (np.ndarray): DU processing power contributed per
            active RU at split level c (n_splits,), in Watts.
        p_cu_static_w (float): CU static power in Watts.
        p_cu_dyn_per_ru_w (float): CU dynamic power per active RU in Watts.
        p_fh_common_w (float): Fronthaul/midhaul common power (present
            whenever any RU is active) in Watts.
        p_fh_per_ru_by_split (np.ndarray): Fronthaul power contributed per
            active RU at split level c (n_splits,), in Watts.
        p_switch_ru_w (float): Switching cost per RU activation-state flip.
        p_switch_split_w (float): Switching cost per RU split-choice change.
    """

    def __init__(
        self,
        n_ru: int,
        n_splits: int = 3,
        p_ru_proc_by_split: Optional[np.ndarray] = None,
        p_ru_sleep_w: float = 2.0,
        pa_efficiency: float = 0.25,
        p_du_static_w: float = 50.0,
        p_du_per_ru_by_split: Optional[np.ndarray] = None,
        p_cu_static_w: float = 30.0,
        p_cu_dyn_per_ru_w: float = 1.0,
        p_fh_common_w: float = 10.0,
        p_fh_per_ru_by_split: Optional[np.ndarray] = None,
        p_switch_ru_w: float = 2.0,
        p_switch_split_w: float = 1.0,
    ):
        self.n_ru = n_ru
        self.n_splits = n_splits

        # c=0 (Option 2, most RU-side processing) -> c=n_splits-1 (Option 8,
        # least RU-side processing, most fronthaul/DU cost); see Concept
        # Note Section 10.2 for the centralization-level mapping rationale.
        self.p_ru_proc_by_split = (
            np.array(p_ru_proc_by_split, dtype=np.float64)
            if p_ru_proc_by_split is not None
            else np.array([10.0, 6.0, 3.0], dtype=np.float64)
        )
        self.p_ru_sleep_w = p_ru_sleep_w
        self.eta = pa_efficiency

        self.p_du_static_w = p_du_static_w
        self.p_du_per_ru_by_split = (
            np.array(p_du_per_ru_by_split, dtype=np.float64)
            if p_du_per_ru_by_split is not None
            else np.array([5.0, 10.0, 20.0], dtype=np.float64)
        )

        self.p_cu_static_w = p_cu_static_w
        self.p_cu_dyn_per_ru_w = p_cu_dyn_per_ru_w

        self.p_fh_common_w = p_fh_common_w
        self.p_fh_per_ru_by_split = (
            np.array(p_fh_per_ru_by_split, dtype=np.float64)
            if p_fh_per_ru_by_split is not None
            else np.array([3.0, 6.0, 15.0], dtype=np.float64)
        )

        self.p_switch_ru_w = p_switch_ru_w
        self.p_switch_split_w = p_switch_split_w

    def compute_ru_power(
        self,
        active_mask: np.ndarray,
        split_idx: np.ndarray,
        transmit_power_w: np.ndarray,
    ) -> float:
        """Compute total RU power (processing + RF transmit + sleep).

        Args:
            active_mask (np.ndarray): Binary active mask (n_ru,).
            split_idx (np.ndarray): Per-RU split centralization level in
                [0, n_splits) (n_ru,). Inactive RUs' entries are ignored.
            transmit_power_w (np.ndarray): Per-RU transmit power in Watts
                (n_ru,). Inactive RUs must already be zeroed by the caller.

        Returns:
            float: Total RU power consumption in Watts.
        """
        active_bool = active_mask.astype(bool)
        proc_power = self.p_ru_proc_by_split[split_idx[active_bool]]
        p_active = float(np.sum(proc_power))
        p_tx = float(np.sum(transmit_power_w[active_bool]) / self.eta)
        p_sleep = float(np.sum(~active_bool) * self.p_ru_sleep_w)
        return p_active + p_tx + p_sleep

    def compute_du_power(self, active_mask: np.ndarray, split_idx: np.ndarray) -> float:
        """Compute DU power (static + per-active-RU, split-dependent).

        Args:
            active_mask (np.ndarray): Binary active mask (n_ru,).
            split_idx (np.ndarray): Per-RU split centralization level (n_ru,).

        Returns:
            float: Total DU power consumption in Watts.
        """
        active_bool = active_mask.astype(bool)
        per_ru = self.p_du_per_ru_by_split[split_idx[active_bool]]
        return float(self.p_du_static_w + np.sum(per_ru))

    def compute_cu_power(self, active_mask: np.ndarray) -> float:
        """Compute CU power (static + dynamic per active RU).

        Args:
            active_mask (np.ndarray): Binary active mask (n_ru,).

        Returns:
            float: Total CU power consumption in Watts.
        """
        n_active = int(np.sum(active_mask.astype(bool)))
        return float(self.p_cu_static_w + self.p_cu_dyn_per_ru_w * n_active)

    def compute_fronthaul_power(
        self, active_mask: np.ndarray, split_idx: np.ndarray
    ) -> float:
        """Compute fronthaul/midhaul power (common + per-RU, split-dependent).

        Args:
            active_mask (np.ndarray): Binary active mask (n_ru,).
            split_idx (np.ndarray): Per-RU split centralization level (n_ru,).

        Returns:
            float: Total fronthaul power consumption in Watts.
        """
        active_bool = active_mask.astype(bool)
        if not np.any(active_bool):
            return 0.0
        per_ru = self.p_fh_per_ru_by_split[split_idx[active_bool]]
        return float(self.p_fh_common_w + np.sum(per_ru))

    def compute_switching_cost(
        self,
        prev_active_mask: np.ndarray,
        current_active_mask: np.ndarray,
        prev_split_idx: np.ndarray,
        current_split_idx: np.ndarray,
    ) -> float:
        """Compute switching power penalty for RU activation flips and split changes.

        Args:
            prev_active_mask (np.ndarray): Previous active mask (n_ru,).
            current_active_mask (np.ndarray): Current active mask (n_ru,).
            prev_split_idx (np.ndarray): Previous per-RU split choice (n_ru,).
            current_split_idx (np.ndarray): Current per-RU split choice (n_ru,).

        Returns:
            float: Switching power cost in Watts.
        """
        ru_flips = np.sum(
            np.abs(current_active_mask.astype(int) - prev_active_mask.astype(int))
        )
        # A split change only matters for RUs active in both slots -- an RU
        # that just switched on/off didn't "change" its split in a way that
        # costs anything beyond the activation flip already counted above.
        both_active = current_active_mask.astype(bool) & prev_active_mask.astype(bool)
        split_changes = np.sum((current_split_idx != prev_split_idx) & both_active)
        return float(
            ru_flips * self.p_switch_ru_w + split_changes * self.p_switch_split_w
        )

    def compute_total_power(
        self,
        active_mask: np.ndarray,
        split_idx: np.ndarray,
        transmit_power_w: np.ndarray,
        prev_active_mask: Optional[np.ndarray] = None,
        prev_split_idx: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute complete system power breakdown (RU + DU + CU + Fronthaul + Switching).

        Args:
            active_mask (np.ndarray): Binary active mask (n_ru,).
            split_idx (np.ndarray): Per-RU split centralization level (n_ru,).
            transmit_power_w (np.ndarray): Per-RU transmit power in Watts (n_ru,).
            prev_active_mask (np.ndarray, optional): Previous active mask.
            prev_split_idx (np.ndarray, optional): Previous per-RU split choice.

        Returns:
            dict: Dictionary with keys 'ru', 'du', 'cu', 'fronthaul',
                'switching', 'total'.
        """
        p_ru = self.compute_ru_power(active_mask, split_idx, transmit_power_w)
        p_du = self.compute_du_power(active_mask, split_idx)
        p_cu = self.compute_cu_power(active_mask)
        p_fh = self.compute_fronthaul_power(active_mask, split_idx)

        p_sw = 0.0
        if prev_active_mask is not None and prev_split_idx is not None:
            p_sw = self.compute_switching_cost(
                prev_active_mask, active_mask, prev_split_idx, split_idx
            )

        total_power = p_ru + p_du + p_cu + p_fh + p_sw
        return {
            "ru": p_ru,
            "du": p_du,
            "cu": p_cu,
            "fronthaul": p_fh,
            "switching": p_sw,
            "total": total_power,
        }
