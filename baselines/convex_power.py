"""Convex Power Allocation Baseline for C-RAN Simulation.

Uses convex optimization (SciPy/CVXPY) to minimize transmit power subject to SINR constraints.
"""

from typing import Dict
import numpy as np
from scipy.optimize import linprog

try:
    import cvxpy as cp

    HAS_CVXPY = True
except (ImportError, ModuleNotFoundError, Exception):
    cp = None  # type: ignore[assignment]
    HAS_CVXPY = False


class ConvexPowerBaseline:
    """Convex Power Optimization Baseline using SciPy/CVXPY.

    Attributes:
        n_rrh (int): Number of Remote Radio Heads.
        n_ue (int): Number of User Equipments.
        p_max_w (float): Maximum transmit power per RRH in Watts.
        target_sinr_db (float): Target SINR in dB (default: 0 dB).
        noise_power_w (float): Noise power in Watts (default: 6.309573e-14 W,
            i.e. -102 dBm, matching Iqbal et al. (2021), Table 2).
        csi_uncertainty (float): Bounded per-user channel-gain estimation-error
            radius, as a fraction of the estimated gain vector's norm (default:
            0.0, the plain nominal/deterministic problem). When > 0, the SINR
            constraint is solved as a robust worst-case second-order cone
            program (SOCP) instead of a linear program — see
            `solve_power_allocation()`.
    """

    def __init__(
        self,
        n_rrh: int,
        n_ue: int,
        p_max_w: float = 1.0,
        target_sinr_db: float = 0.0,
        noise_power_w: float = 6.309573e-14,  # -102 dBm in Watts
        csi_uncertainty: float = 0.0,
    ):
        self.n_rrh = n_rrh
        self.n_ue = n_ue
        self.p_max_w = p_max_w
        self.target_sinr_linear = 10.0 ** (target_sinr_db / 10.0)
        self.noise_power_w = noise_power_w
        self.csi_uncertainty = csi_uncertainty

    def solve_power_allocation(
        self, active_mask: np.ndarray, channel_gains: np.ndarray
    ) -> np.ndarray:
        """Solve convex power optimization problem: min sum(p_r) s.t. SINR_u >= target_sinr.

        When `self.csi_uncertainty` is 0 (default), this is the nominal,
        deterministic-channel problem, which is a genuine linear program (the
        SINR constraint is linear in scalar per-RRH power; there's no cone
        structure to introduce without also modeling per-RRH beamforming
        vectors, out of scope for this baseline). When `csi_uncertainty` > 0,
        the problem is instead solved as a robust worst-case second-order
        cone program (SOCP) — see the branch below for the derivation.

        Args:
            active_mask (np.ndarray): Binary active RRH vector (n_rrh,).
            channel_gains (np.ndarray): Complex channel gain matrix (n_rrh, n_ue).

        Returns:
            np.ndarray: Optimal power vector (n_rrh,).
        """
        active_indices = np.where(active_mask)[0]
        power = np.zeros(self.n_rrh, dtype=np.float32)

        if len(active_indices) == 0:
            return power

        # Channel power gain matrix |H|^2 (n_active, n_ue)
        gains_sq = np.abs(channel_gains[active_indices, :]) ** 2
        n_active = len(active_indices)
        sinr_threshold = self.target_sinr_linear * self.noise_power_w

        if self.csi_uncertainty > 0.0:
            # Robust worst-case SOCP: the true per-user channel-gain vector
            # a_u = gains_sq[:, u] is assumed to lie within a bounded ball
            # around the estimated value, ||delta_u||_2 <= eps_u. Requiring
            # the SINR target a_u^T p >= sinr_threshold to hold for the
            # worst-case delta_u within that ball —
            #   min_{||delta_u||<=eps_u} (a_u + delta_u)^T p
            #     = a_u^T p - eps_u * ||p||_2   (Cauchy-Schwarz)
            # — turns the nominal linear constraint into a genuine
            # second-order cone constraint: eps_u*||p||_2 <= a_u^T p -
            # sinr_threshold. This isn't solvable via linprog; it requires
            # CVXPY's SOC support.
            if not (HAS_CVXPY and cp is not None):
                power[active_indices] = self.p_max_w
                return power
            try:
                p_var = cp.Variable(n_active, nonneg=True)
                objective = cp.Minimize(cp.sum(p_var))
                constraints = [p_var <= self.p_max_w]
                for u in range(self.n_ue):
                    a_u = gains_sq[:, u]
                    eps_u = self.csi_uncertainty * float(np.linalg.norm(a_u))
                    constraints.append(
                        cp.SOC(a_u @ p_var - sinr_threshold, eps_u * p_var)
                    )

                prob = cp.Problem(objective, constraints)
                prob.solve()

                if (
                    prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
                    and p_var.value is not None
                ):
                    power[active_indices] = np.clip(p_var.value, 0.0, self.p_max_w)
                    return power
            except Exception:
                pass

            power[active_indices] = self.p_max_w
            return power

        # --- Nominal (deterministic-channel) problem ---

        # 1. Try SciPy linprog (fast, robust, compatible with NumPy 2.x)
        c = np.ones(n_active)  # Minimize sum(p_r)
        A_ub = -gains_sq.T  # -sum_r |h_{r,u}|^2 * p_r <= -target_sinr * noise
        b_ub = -sinr_threshold * np.ones(self.n_ue)
        bounds = [(0.0, self.p_max_w) for _ in range(n_active)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if res.success and res.x is not None:
            power[active_indices] = np.clip(res.x, 0.0, self.p_max_w)
            return power

        # 2. Try CVXPY if available and linprog didn't find optimal
        if HAS_CVXPY and cp is not None:
            try:
                p_var = cp.Variable(n_active, nonneg=True)
                objective = cp.Minimize(cp.sum(p_var))
                constraints = [p_var <= self.p_max_w]
                for u in range(self.n_ue):
                    rx_signal = cp.sum(cp.multiply(gains_sq[:, u], p_var))
                    constraints.append(rx_signal >= sinr_threshold)

                prob = cp.Problem(objective, constraints)
                prob.solve()

                if (
                    prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
                    and p_var.value is not None
                ):
                    power[active_indices] = np.clip(p_var.value, 0.0, self.p_max_w)
                    return power
            except Exception:
                pass

        # 3. Infeasible fallback: allocate max power to active RRHs
        power[active_indices] = self.p_max_w
        return power

    def select_action(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Select action using channel-aware active mask and convex power allocation.

        Args:
            obs (np.ndarray): State observation.

        Returns:
            Dict[str, np.ndarray]: Action dict with 'rrh_on' and 'power'.
        """
        gains_flat = obs[: self.n_rrh * self.n_ue]
        gains_matrix = gains_flat.reshape(self.n_rrh, self.n_ue)

        demands_start = self.n_rrh * self.n_ue + self.n_rrh
        demands_end = demands_start + self.n_ue
        demands_mbps = obs[demands_start:demands_end]

        # Activate RRHs with strongest aggregate channel quality
        rrh_scores = np.sum(gains_matrix**2, axis=1)
        sorted_indices = np.argsort(-rrh_scores)

        num_active = max(1, int(np.ceil((np.sum(demands_mbps) / 300.0) * self.n_rrh)))
        num_active = min(self.n_rrh, num_active)

        rrh_on = np.zeros(self.n_rrh, dtype=int)
        rrh_on[sorted_indices[:num_active]] = 1

        power = self.solve_power_allocation(rrh_on, gains_matrix)
        return {"rrh_on": rrh_on, "power": power}
