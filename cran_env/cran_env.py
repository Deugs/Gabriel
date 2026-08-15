"""Gymnasium C-RAN Environment.

Provides a Gymnasium-compliant 5G Cloud Radio Access Network (C-RAN) environment
with hybrid discrete (RRH activation) and continuous (transmit power) action spaces.
"""

from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from cran_env.channel_model import ChannelModel
from cran_env.power_model import PowerModel
from cran_env.traffic_model import TrafficModel


class DictConfig:
    """Helper wrapper to access dictionary keys as attributes."""

    def __init__(self, cfg_dict: dict):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                setattr(self, key, DictConfig(value))
            else:
                setattr(self, key, value)


class CRANEnv(gym.Env):
    """Gymnasium 5G C-RAN Simulation Environment for Energy Optimization.

    Observation Space:
        Box(shape=(R*U + R + U + 2,))
        - Channel gains magnitude |H| (R * U)
        - Current RRH active status (R)
        - User traffic demands in Mbps (U)
        - Previous step total power in kW (1)
        - Current hour normalized to [0, 1] (1)

    Action Space:
        Dict({
            "rrh_on": MultiBinary(n_rrh),
            "power": Box(low=0, high=p_max_w, shape=(n_rrh,), dtype=np.float32)
        })
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Union[dict, Any]):
        super().__init__()

        # Handle dictionary or object config
        if isinstance(config, dict):
            cfg = DictConfig(config)
        else:
            cfg = config

        self.cfg = cfg

        # Extract network topology params (with robust fallbacks)
        net_cfg = getattr(cfg, "network", cfg)
        self.n_rrh = int(getattr(net_cfg, "n_rrh", 12))
        self.n_ue = int(getattr(net_cfg, "n_ue", 10))
        self.n_bbu = int(getattr(net_cfg, "n_bbu", 3))
        self.area_size_m = float(getattr(net_cfg, "area_size_m", 1000.0))
        self.carrier_freq_ghz = float(getattr(net_cfg, "carrier_freq_ghz", 2.1))
        self.bandwidth_mhz = float(getattr(net_cfg, "bandwidth_mhz", 20.0))
        self.noise_power_dbm = float(getattr(net_cfg, "noise_power_dbm", -102.0))

        # Thermal noise power in Watts per subband/user
        self.noise_power_w = 10.0 ** ((self.noise_power_dbm - 30.0) / 10.0)

        # Extract power params
        power_cfg = getattr(cfg, "power", cfg)
        rrh_cfg = getattr(power_cfg, "rrh", power_cfg)
        self.p_max_dbm = float(getattr(rrh_cfg, "p_max_dbm", 30.0))
        self.p_max_w = 10.0 ** (
            (self.p_max_dbm - 30.0) / 10.0
        )  # e.g. 1.0 Watt for 30 dBm

        # Extract reward params
        reward_cfg = getattr(cfg, "reward", cfg)
        self.alpha_energy = float(getattr(reward_cfg, "alpha_energy", 1.0))
        self.beta_qos = float(getattr(reward_cfg, "beta_qos", 10.0))
        self.gamma_switch = float(getattr(reward_cfg, "gamma_switch", 0.5))

        # Instantiate physical models
        channel_cfg = getattr(cfg, "channel", cfg)
        self.channel = ChannelModel(
            n_rrh=self.n_rrh,
            n_ue=self.n_ue,
            carrier_freq_ghz=self.carrier_freq_ghz,
            bandwidth_mhz=self.bandwidth_mhz,
            path_loss_exponent=float(getattr(channel_cfg, "path_loss_exponent", 3.5)),
            shadowing_std_db=float(getattr(channel_cfg, "shadowing_std_db", 8.0)),
            correlation_coeff=float(
                getattr(channel_cfg, "correlation_coefficient", 0.9)
            ),
        )

        traffic_cfg = getattr(cfg, "traffic", cfg)
        self.traffic = TrafficModel(
            n_ue=self.n_ue,
            base_rate_mbps=float(getattr(traffic_cfg, "base_rate_mbps", 50.0)),
            peak_multiplier=float(getattr(traffic_cfg, "peak_multiplier", 3.0)),
            burstiness_sigma=float(getattr(traffic_cfg, "burstiness", 0.2)),
            profile=str(getattr(traffic_cfg, "profile", "weekday_urban")),
        )

        rrh_power_cfg = getattr(power_cfg, "rrh", power_cfg)
        bbu_cfg = getattr(power_cfg, "bbu", power_cfg)
        fronthaul_cfg = getattr(power_cfg, "fronthaul", power_cfg)
        self.power = PowerModel(
            n_rrh=self.n_rrh,
            n_bbu=self.n_bbu,
            p_active_w=float(getattr(rrh_power_cfg, "p_active_w", 6.8)),
            p_sleep_w=float(getattr(rrh_power_cfg, "p_sleep_w", 4.3)),
            p_switch_w=float(getattr(rrh_power_cfg, "p_switch_w", 3.0)),
            pa_efficiency=float(getattr(rrh_power_cfg, "pa_efficiency", 0.25)),
            p_stat_w=float(getattr(bbu_cfg, "p_stat_w", 175.0)),
            p_dyn_w=float(getattr(bbu_cfg, "p_dyn_w", 250.0)),
            delta_p=float(getattr(bbu_cfg, "delta_p", 0.44)),
            p_olt_w=float(getattr(fronthaul_cfg, "p_olt_w", 20.0)),
            p_onu_active_w=float(getattr(fronthaul_cfg, "p_onu_active_w", 5.0)),
            p_onu_sleep_w=float(getattr(fronthaul_cfg, "p_onu_sleep_w", 0.5)),
            p_lc_w=float(getattr(fronthaul_cfg, "p_lc_w", 10.0)),
            wavelength_capacity_gbps=float(
                getattr(fronthaul_cfg, "wavelength_capacity_gbps", 10.0)
            ),
        )

        # State dimensions: gains (R*U) + mask (R) + demands (U) + prev_power (1) + hour (1)
        self.state_dim = self.n_rrh * self.n_ue + self.n_rrh + self.n_ue + 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.action_space = spaces.Dict(
            {
                "rrh_on": spaces.MultiBinary(self.n_rrh),
                "power": spaces.Box(
                    low=0.0, high=self.p_max_w, shape=(self.n_rrh,), dtype=np.float32
                ),
                "bandwidth": spaces.Box(
                    low=0.0, high=1.0, shape=(self.n_rrh,), dtype=np.float32
                ),
            }
        )

        # Internal state variables
        self.rng: np.random.Generator = np.random.default_rng(0)
        self.rrh_pos: np.ndarray = np.zeros((self.n_rrh, 2), dtype=np.float32)
        self.ue_pos: np.ndarray = np.zeros((self.n_ue, 2), dtype=np.float32)
        self.distances: np.ndarray = np.zeros((self.n_rrh, self.n_ue), dtype=np.float32)
        self.channel_gains: np.ndarray = np.zeros(
            (self.n_rrh, self.n_ue), dtype=np.complex128
        )
        self.active_mask: np.ndarray = np.ones(self.n_rrh, dtype=bool)
        self.prev_power_w: float = 0.0
        self.hour: int = 0
        self.step_count: int = 0
        # cfg is always either a DictConfig (attribute-based) or a non-dict
        # config object here (see the isinstance(config, dict) check above),
        # never a plain dict, so getattr() correctly performs the lookup —
        # unlike agents/branching_mp_dqn.py and its siblings, which receive
        # the raw dict from yaml.safe_load() directly and therefore need
        # cfg.get(...) instead (see their extraction logic and its comment).
        algo_cfg = getattr(cfg, "algorithm", cfg)
        self.max_steps: int = int(getattr(algo_cfg, "max_steps_per_episode", 100))

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the C-RAN environment to an initial state.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Options dict.

        Returns:
            Tuple[np.ndarray, Dict]: Initial observation vector and info dict.
        """
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.hour = int(self.rng.integers(0, 24))
        self.step_count = 0

        # Uniformly place RRHs and UEs in area_size_m x area_size_m grid
        self.rrh_pos = self.rng.uniform(0.0, self.area_size_m, (self.n_rrh, 2))
        self.ue_pos = self.rng.uniform(0.0, self.area_size_m, (self.n_ue, 2))

        # Distance matrix (n_rrh, n_ue)
        self.distances = np.linalg.norm(
            self.rrh_pos[:, None, :] - self.ue_pos[None, :, :], axis=2
        )

        # Initial channel gains and active status (all ON initially)
        self.channel_gains = self.channel.generate_channel(self.distances, self.rng)
        self.active_mask = np.ones(self.n_rrh, dtype=bool)
        self.prev_power_w = 0.0

        obs = self._get_obs()
        info = {"hour": self.hour, "active_rrhs": int(np.sum(self.active_mask))}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        """Construct state observation vector."""
        demands_bps = self.traffic.get_demands(self.hour, self.rng)
        demands_mbps = demands_bps / 1e6

        # Channel magnitude |H|
        gains_mag = np.abs(self.channel_gains).flatten()

        obs = np.concatenate(
            [
                gains_mag,
                self.active_mask.astype(np.float32),
                demands_mbps.astype(np.float32),
                np.array([self.prev_power_w / 1000.0], dtype=np.float32),  # kW
                np.array([self.hour / 24.0], dtype=np.float32),
            ]
        ).astype(np.float32)

        return obs

    def step(
        self, action: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the C-RAN environment.

        Args:
            action (dict): Action dictionary containing 'rrh_on', 'power', and
                (optionally) 'bandwidth' — each active RRH's share of the total
                channel bandwidth B, beta_r in [0, 1]. If omitted, or if the
                active RRHs' shares sum to (near) zero, falls back to an equal
                split across active RRHs.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]: (obs, reward, terminated, truncated, info).
        """
        self.step_count += 1

        # Parse actions
        rrh_on = np.array(action["rrh_on"], dtype=bool)
        power = np.array(action["power"], dtype=np.float32)

        # Clamp power to valid range [0, P_max] and zero out inactive RRHs
        power = np.clip(power, 0.0, self.p_max_w)
        power[~rrh_on] = 0.0

        # Signal/interference per user, and which RRH serves each user
        signal, interference, serving_rrh = self._signal_interference(rrh_on, power)
        sinr = np.where(
            signal > 0.0, signal / (interference + self.noise_power_w), 0.0
        ).astype(np.float32)

        # Bandwidth share beta_r per RRH: only active RRHs consume bandwidth,
        # and their shares are renormalized to sum to 1 (an inactive RRH's
        # share is unused capacity, reallocated among the active RRHs).
        bandwidth_share_raw = np.array(
            action.get("bandwidth", np.ones(self.n_rrh, dtype=np.float32)),
            dtype=np.float32,
        )
        active_shares = bandwidth_share_raw * rrh_on.astype(np.float32)
        share_sum = np.sum(active_shares)
        if share_sum > 1e-12:
            bandwidth_share = active_shares / share_sum
        else:
            n_active = max(1, int(np.sum(rrh_on)))
            bandwidth_share = rrh_on.astype(np.float32) / n_active

        # Each user gets the bandwidth share of the RRH serving it (0 if unserved)
        user_bandwidth_hz = (
            np.where(
                serving_rrh >= 0,
                bandwidth_share[np.clip(serving_rrh, 0, self.n_rrh - 1)],
                0.0,
            )
            * self.channel.bandwidth
        )

        # Achievable capacity per user across its allocated share of bandwidth B
        achievable_capacity_bps = user_bandwidth_hz * np.log2(1.0 + sinr)
        total_throughput_mbps = np.sum(achievable_capacity_bps) / 1e6

        # Traffic demands
        demands_bps = self.traffic.get_demands(self.hour, self.rng)
        qos_violations_bps = np.maximum(0.0, demands_bps - achievable_capacity_bps)

        # Power breakdown
        bbu_loads = np.ones(self.n_bbu) * (np.sum(rrh_on) / max(1, self.n_bbu))
        power_dict = self.power.compute_total_power(
            active_mask=rrh_on,
            transmit_power=power,
            bbu_loads=bbu_loads,
            prev_active_mask=self.active_mask,
            total_throughput_mbps=total_throughput_mbps,
        )

        p_total = power_dict["total"]
        p_switching = power_dict["switching"]

        # Exact discrete switching count (Concept Note v2.0 Section 10.2)
        exact_switching_count = np.sum(rrh_on != self.active_mask)

        # Energy Efficiency EE(t) in Mbit / Joule
        ee_mbit_per_joule = total_throughput_mbps / (p_total + 1e-6)

        # Scalar Reward: r(t) = EE(t) - lambda1*QoS_penalty - lambda2*switch_penalty
        # (Concept Note v4.0 Section 10.2). alpha_energy scales the EE(t) term
        # itself (default 1.0 recovers the note's literal formula) rather than
        # a raw power penalty, since the objective is to maximize energy
        # efficiency (Mbit/Joule), not merely minimize power consumption.
        energy_efficiency_term = self.alpha_energy * ee_mbit_per_joule
        qos_penalty = self.beta_qos * (
            np.sum(qos_violations_bps) / 1e6
        )  # QoS shortfall in Mbps
        switch_penalty = self.gamma_switch * exact_switching_count

        reward = energy_efficiency_term - qos_penalty - switch_penalty

        # Update environment state for next step
        self.active_mask = rrh_on
        self.prev_power_w = p_total
        self.hour = (self.hour + 1) % 24

        # Channel temporal correlation update
        self.channel_gains = self.channel.step_channel(
            self.channel_gains, self.distances, self.rng
        )

        obs = self._get_obs()

        # Termination / Truncation logic
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "total_power_w": p_total,
            "rrh_power_w": power_dict["rrh"],
            "bbu_power_w": power_dict["bbu"],
            "fronthaul_power_w": power_dict["fronthaul"],
            "switching_power_w": p_switching,
            "switching_events": int(exact_switching_count),
            "ee_mbit_per_joule": float(ee_mbit_per_joule),
            "qos_violations_count": int(np.sum(qos_violations_bps > 0.0)),
            "qos_shortfall_mbps": float(np.sum(qos_violations_bps) / 1e6),
            "mean_sinr_db": float(10.0 * np.log10(np.mean(sinr) + 1e-12)),
            "active_rrhs": int(np.sum(rrh_on)),
        }

        return obs, float(reward), terminated, truncated, info

    def _signal_interference(
        self, active_mask: np.ndarray, power: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-user signal power, interference power, and serving RRH.

        User u is assigned to the best active serving RRH:
            r*(u) = argmax_{r in active} (p_r * |h_{r,u}|^2)
        Desired Signal: S_u = p_{r*(u)} * |h_{r*(u),u}|^2
        Interference: I_u = sum_{r in active, r != r*(u)} (p_r * |h_{r,u}|^2)

        Returns:
            Tuple of (signal_power, interference_power, serving_rrh_idx), each
            shape (n_ue,). serving_rrh_idx is -1 for a user when no RRH is active.
        """
        signal = np.zeros(self.n_ue, dtype=np.float64)
        interference = np.zeros(self.n_ue, dtype=np.float64)
        serving_rrh = np.full(self.n_ue, -1, dtype=np.int64)
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            return signal, interference, serving_rrh  # All RRHs OFF -> zero signal

        for u in range(self.n_ue):
            # Calculate received power from each active RRH to user u
            channel_mag_sq = np.abs(self.channel_gains[active_indices, u]) ** 2
            rx_powers = channel_mag_sq * power[active_indices]

            best_local_idx = np.argmax(rx_powers)
            signal[u] = rx_powers[best_local_idx]
            interference[u] = np.sum(rx_powers) - signal[u]
            serving_rrh[u] = active_indices[best_local_idx]

        return signal, interference, serving_rrh

    def _compute_sinr(self, active_mask: np.ndarray, power: np.ndarray) -> np.ndarray:
        """Compute Signal-to-Interference-plus-Noise Ratio (SINR) for each user.

        SINR_u = S_u / (I_u + noise_power); see _signal_interference for how the
        serving RRH, signal, and interference terms are derived.
        """
        signal, interference, _ = self._signal_interference(active_mask, power)
        sinr = np.where(signal > 0.0, signal / (interference + self.noise_power_w), 0.0)
        return sinr.astype(np.float32)
