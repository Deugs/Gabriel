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
