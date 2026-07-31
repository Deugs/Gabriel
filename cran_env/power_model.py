"""Power Consumption Model for 5G C-RAN Simulation.

Implements EARTH-validated power models for RRHs, BBUs, and TWDM-PON fronthaul,
including transmit power amplifier efficiency, switching costs, and load-dependent BBU consumption.
"""

from typing import Optional
import numpy as np


class PowerModel:
    """EARTH-Validated Power Consumption Model (Al-Zubaedi 2019 / Fathy et al. 2021).

    Attributes:
        n_rrh (int): Number of Remote Radio Heads.
        n_bbu (int): Number of Baseband Units in the centralized pool.
        p_active_w (float): RRH active baseline power in Watts (default: 6.8 W).
        p_sleep_w (float): RRH sleep mode power in Watts (default: 4.3 W).
        p_switch_w (float): RRH mode transition cost in Watts (default: 3.0 W).
        pa_efficiency (float): Power amplifier drain efficiency (default: 0.25).
        p_stat_w (float): BBU static power per BBU in Watts (default: 175.0 W - EARTH model).
        p_dyn_w (float): BBU total dynamic power range in Watts (default: 250.0 W).
        delta_p (float): BBU load-power slope (default: 0.44).
        p_olt_w (float): Fronthaul Optical Line Terminal power in Watts (default: 20.0 W).
        p_onu_active_w (float): Optical Network Unit active power in Watts (default: 5.0 W).
        p_onu_sleep_w (float): Optical Network Unit sleep power in Watts (default: 0.5 W).
    """

    def __init__(
        self,
        n_rrh: int,
        n_bbu: int,
        p_active_w: float = 6.8,
        p_sleep_w: float = 4.3,
        p_switch_w: float = 3.0,
        pa_efficiency: float = 0.25,
        p_stat_w: float = 175.0,
        p_dyn_w: float = 250.0,
        delta_p: float = 0.44,
        p_olt_w: float = 20.0,
        p_onu_active_w: float = 5.0,
        p_onu_sleep_w: float = 0.5,
    ):
        self.n_rrh = n_rrh
        self.n_bbu = n_bbu

        # RRH parameters
        self.p_active = p_active_w
        self.p_sleep = p_sleep_w
        self.p_switch = p_switch_w
        self.eta = pa_efficiency

        # BBU parameters (EARTH model)
        self.p_stat = p_stat_w
        self.p_dyn = p_dyn_w
        self.delta_p = delta_p

        # Fronthaul parameters (TWDM-PON)
        self.p_olt = p_olt_w
        self.p_onu_active = p_onu_active_w
        self.p_onu_sleep = p_onu_sleep_w

    def compute_rrh_power(
        self, active_mask: np.ndarray, transmit_power: np.ndarray
    ) -> float:
        """Compute total power consumed by all RRHs (RF transmit + active + sleep).

        Args:
            active_mask (np.ndarray): Binary vector indicating active RRHs (n_rrh,).
            transmit_power (np.ndarray): Continuous transmit power per RRH in Watts (n_rrh,).

        Returns:
            float: Total RRH power consumption in Watts.
        """
        active_bool = active_mask.astype(bool)
        # Power amplifier drain power = P_tx / eta for active RRHs
        p_tx = np.sum(transmit_power[active_bool]) / self.eta
        p_active_circuit = np.sum(active_bool) * self.p_active
        p_sleep_circuit = np.sum(~active_bool) * self.p_sleep
        return float(p_tx + p_active_circuit + p_sleep_circuit)

    def compute_bbu_power(self, loads: np.ndarray) -> float:
        """Compute BBU pool power consumption based on compute load allocation.

        Args:
            loads (np.ndarray): Normalized compute load per BBU in [0, 1] (n_bbu,).

        Returns:
            float: Total BBU pool power consumption in Watts.
        """
        loads_clamped = np.clip(loads, 0.0, 1.0)
        active_bbus = np.sum(loads_clamped > 0.0)

        p_static = active_bbus * self.p_stat
        p_dynamic = self.delta_p * self.p_dyn * np.sum(loads_clamped)
        return float(p_static + p_dynamic)

    def compute_fronthaul_power(self, active_mask: np.ndarray) -> float:
        """Compute TWDM-PON fronthaul power consumption.

        Args:
            active_mask (np.ndarray): Binary vector indicating active RRHs (n_rrh,).

        Returns:
            float: Total fronthaul power consumption in Watts.
        """
        active_bool = active_mask.astype(bool)
        p_onu_active = np.sum(active_bool) * self.p_onu_active
        p_onu_sleep = np.sum(~active_bool) * self.p_onu_sleep
        return float(self.p_olt + p_onu_active + p_onu_sleep)

    def compute_switching_cost(
        self, prev_active_mask: np.ndarray, current_active_mask: np.ndarray
    ) -> float:
        """Compute total switching power penalty for RRH mode transitions.

        Args:
            prev_active_mask (np.ndarray): Previous RRH active mask.
            current_active_mask (np.ndarray): Current RRH active mask.

        Returns:
            float: Switching power cost in Watts.
        """
        transitions = np.sum(
            np.abs(current_active_mask.astype(int) - prev_active_mask.astype(int))
        )
        return float(transitions * self.p_switch)

    def compute_total_power(
        self,
        active_mask: np.ndarray,
        transmit_power: np.ndarray,
        bbu_loads: np.ndarray,
        prev_active_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute complete system power breakdown (RRH + BBU + Fronthaul + Switching).

        Args:
            active_mask (np.ndarray): Binary active mask (n_rrh,).
            transmit_power (np.ndarray): Continuous transmit power array (n_rrh,).
            bbu_loads (np.ndarray): BBU load array (n_bbu,).
            prev_active_mask (np.ndarray, optional): Previous active mask.

        Returns:
            dict: Dictionary with keys 'rrh', 'bbu', 'fronthaul', 'switching', 'total'.
        """
        p_rrh = self.compute_rrh_power(active_mask, transmit_power)
        p_bbu = self.compute_bbu_power(bbu_loads)
        p_fh = self.compute_fronthaul_power(active_mask)

        p_sw = 0.0
        if prev_active_mask is not None:
            p_sw = self.compute_switching_cost(prev_active_mask, active_mask)

        total_power = p_rrh + p_bbu + p_fh + p_sw
        return {
            "rrh": p_rrh,
            "bbu": p_bbu,
            "fronthaul": p_fh,
            "switching": p_sw,
            "total": total_power,
        }
