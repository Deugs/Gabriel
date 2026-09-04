"""Gymnasium O-RAN Environment.

Provides a Gymnasium-compliant O-RAN (disaggregated RU/DU/CU) environment
with a 4-branch hybrid discrete (RU activation, functional split) and
continuous (transmit power, PRB allocation) action space
(docs/skills/skill_oran_env.md; ORAN_BMPP_DQN_Concept_Note_v1.md Section
10.1). Fully decoupled from cran_env/ -- no shared imports.
"""

from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from oran_env.channel_model import ORANChannelModel
from oran_env.power_model import ORANPowerModel
from oran_env.traffic_model import ORANTrafficModel


class DictConfig:
    """Helper wrapper to access dictionary keys as attributes.

    A local copy of cran_env.cran_env.DictConfig's pattern -- duplicated,
    not imported, to keep this package's zero-dependency-on-cran_env
    guarantee structural rather than a convention to remember.
    """

    def __init__(self, cfg_dict: dict):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                setattr(self, key, DictConfig(value))
            else:
                setattr(self, key, value)


class ORANEnv(gym.Env):
    """Gymnasium O-RAN Simulation Environment for Energy Optimization.

    Observation Space:
        Box(shape=(n_ru*n_ue + n_ru + n_ru*n_splits + n_ue + 4,))
        - Channel gains magnitude |H| (n_ru * n_ue)
        - Current RU active status (n_ru)
        - Current per-RU split choice, one-hot (n_ru * n_splits)
        - User traffic demands in Mbps (n_ue)
        - Previous step total power in kW (1)
        - Current hour normalized to [0, 1] (1)
        - Rolling-window mean throughput in Mbps (1) -- lower-level metric
          propagated into the state for the upper-level decision, per
          Concept Note Section 5.2
        - Rolling-window mean power in kW (1) -- same propagation channel

    Action Space:
        Dict({
            "ru_on": MultiBinary(n_ru),
            "split": MultiDiscrete([n_splits] * n_ru),
            "power": Box(low=0, high=p_max_w, shape=(n_ru,)),
            "prb": Box(low=0, high=1, shape=(n_ru,)),
        })

    The environment itself is timescale-agnostic: step() always accepts the
    full 4-key action dict every call. The two-timescale (upper/lower)
    decision cadence is entirely the agent's responsibility
    (oran_agents/bmpp_dqn.py) -- this env only maintains the rolling window
    used to populate the propagated state fields above.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Union[dict, Any]):
        super().__init__()

        if isinstance(config, dict):
            cfg = DictConfig(config)
        else:
            cfg = config
        self.cfg = cfg

        net_cfg = getattr(cfg, "network", cfg)
        self.n_ru = int(getattr(net_cfg, "n_ru", 4))
        self.n_du = int(getattr(net_cfg, "n_du", 1))
        self.n_cu = int(getattr(net_cfg, "n_cu", 1))
        self.n_ue = int(getattr(net_cfg, "n_ue", 8))
        self.n_splits = int(getattr(net_cfg, "n_splits", 3))
        self.area_size_m = float(getattr(net_cfg, "area_size_m", 500.0))
        self.carrier_freq_ghz = float(getattr(net_cfg, "carrier_freq_ghz", 3.5))
        self.bandwidth_mhz = float(getattr(net_cfg, "bandwidth_mhz", 20.0))
        self.noise_power_dbm = float(getattr(net_cfg, "noise_power_dbm", -102.0))
        self.noise_power_w = 10.0 ** ((self.noise_power_dbm - 30.0) / 10.0)

        power_cfg = getattr(cfg, "power", cfg)
        ru_cfg = getattr(power_cfg, "ru", power_cfg)
        self.p_max_dbm = float(getattr(ru_cfg, "p_max_dbm", 30.0))
        self.p_max_w = 10.0 ** ((self.p_max_dbm - 30.0) / 10.0)

        reward_cfg = getattr(cfg, "reward", cfg)
        self.alpha_energy = float(getattr(reward_cfg, "alpha_energy", 1.0))
        self.beta_qos = float(getattr(reward_cfg, "beta_qos", 10.0))
        self.gamma_switch = float(getattr(reward_cfg, "gamma_switch", 0.5))

        channel_cfg = getattr(cfg, "channel", cfg)
        self.channel = ORANChannelModel(
            n_ru=self.n_ru,
            n_ue=self.n_ue,
            carrier_freq_ghz=self.carrier_freq_ghz,
            bandwidth_mhz=self.bandwidth_mhz,
            path_loss_exponent=float(getattr(channel_cfg, "path_loss_exponent", 3.5)),
        )

        traffic_cfg = getattr(cfg, "traffic", cfg)
        self.traffic = ORANTrafficModel(
            n_ue=self.n_ue,
            lambda_peak=float(getattr(traffic_cfg, "lambda_peak", 0.5)),
            floor_ratio=float(getattr(traffic_cfg, "floor_ratio", 0.2)),
            packet_size_bits=float(getattr(traffic_cfg, "packet_size_bits", 4.0e6)),
            step_duration_s=float(getattr(traffic_cfg, "step_duration_s", 0.1)),
            t1=float(getattr(traffic_cfg, "t1", 7.0)),
            t2=float(getattr(traffic_cfg, "t2", 10.0)),
            t3=float(getattr(traffic_cfg, "t3", 20.0)),
            t4=float(getattr(traffic_cfg, "t4", 23.0)),
        )

        du_cfg = getattr(power_cfg, "du", power_cfg)
        cu_cfg = getattr(power_cfg, "cu", power_cfg)
        fh_cfg = getattr(power_cfg, "fronthaul", power_cfg)
        self.power = ORANPowerModel(
            n_ru=self.n_ru,
            n_splits=self.n_splits,
            p_ru_proc_by_split=getattr(ru_cfg, "p_proc_by_split_w", None),
            p_ru_sleep_w=float(getattr(ru_cfg, "p_sleep_w", 2.0)),
            pa_efficiency=float(getattr(ru_cfg, "pa_efficiency", 0.25)),
            p_du_static_w=float(getattr(du_cfg, "p_static_w", 50.0)),
            p_du_per_ru_by_split=getattr(du_cfg, "p_per_ru_by_split_w", None),
            p_cu_static_w=float(getattr(cu_cfg, "p_static_w", 30.0)),
            p_cu_dyn_per_ru_w=float(getattr(cu_cfg, "p_dyn_per_ru_w", 1.0)),
            p_fh_common_w=float(getattr(fh_cfg, "p_common_w", 10.0)),
            p_fh_per_ru_by_split=getattr(fh_cfg, "p_per_ru_by_split_w", None),
            p_switch_ru_w=float(getattr(ru_cfg, "p_switch_w", 2.0)),
            p_switch_split_w=float(getattr(ru_cfg, "p_switch_split_w", 1.0)),
        )

        algo_cfg = getattr(cfg, "algorithm", cfg)
        self.max_steps: int = int(getattr(algo_cfg, "max_steps_per_episode", 100))
        self.upper_level_period_steps: int = int(
            getattr(algo_cfg, "upper_level_period_steps", 10)
        )

        # State: gains (n_ru*n_ue) + mask (n_ru) + split one-hot
        # (n_ru*n_splits) + demands (n_ue) + prev_power (1) + hour (1) +
        # rolling throughput/power (2) -- the last two realize the
        # lower-level -> upper-level propagation, Concept Note Section 5.2.
        self.state_dim = (
            self.n_ru * self.n_ue
            + self.n_ru
            + self.n_ru * self.n_splits
            + self.n_ue
            + 4
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Dict(
            {
                "ru_on": spaces.MultiBinary(self.n_ru),
                "split": spaces.MultiDiscrete([self.n_splits] * self.n_ru),
                "power": spaces.Box(
                    low=0.0, high=self.p_max_w, shape=(self.n_ru,), dtype=np.float32
                ),
                "prb": spaces.Box(
                    low=0.0, high=1.0, shape=(self.n_ru,), dtype=np.float32
                ),
            }
        )

        self.rng: np.random.Generator = np.random.default_rng(0)
        self.ru_pos: np.ndarray = np.zeros((self.n_ru, 2), dtype=np.float32)
        self.ue_pos: np.ndarray = np.zeros((self.n_ue, 2), dtype=np.float32)
        self.distances: np.ndarray = np.zeros((self.n_ru, self.n_ue), dtype=np.float32)
        self.channel_gains: np.ndarray = np.zeros(
            (self.n_ru, self.n_ue), dtype=np.complex128
        )
        self.active_mask: np.ndarray = np.ones(self.n_ru, dtype=bool)
        self.split_idx: np.ndarray = np.zeros(self.n_ru, dtype=np.int64)
        self.prev_power_w: float = 0.0
        self.hour: float = 0.0
        self.step_count: int = 0
        self.current_demands_bps: np.ndarray = np.zeros(self.n_ue, dtype=np.float64)
        self._throughput_window: list = []
        self._power_window: list = []

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the O-RAN environment to an initial state."""
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.hour = float(self.rng.integers(0, 24))
        self.step_count = 0

        self.ru_pos = self.rng.uniform(0.0, self.area_size_m, (self.n_ru, 2))
        self.ue_pos = self.rng.uniform(0.0, self.area_size_m, (self.n_ue, 2))
        self.distances = np.linalg.norm(
            self.ru_pos[:, None, :] - self.ue_pos[None, :, :], axis=2
        )

        self.channel_gains = self.channel.generate_channel(self.distances, self.rng)
        self.active_mask = np.ones(self.n_ru, dtype=bool)
        self.split_idx = np.zeros(self.n_ru, dtype=np.int64)
        self.prev_power_w = 0.0
        self._throughput_window = []
        self._power_window = []

        # Sample this hour's demand once; _get_obs() and the following
        # step() call's reward both reuse self.current_demands_bps (the
        # single-sample-per-hour pattern already established in
        # cran_env/cran_env.py to avoid the state/reward demand mismatch).
        self.current_demands_bps = self.traffic.get_demands(self.hour, self.rng)

        obs = self._get_obs()
        info = {"hour": self.hour, "active_rus": int(np.sum(self.active_mask))}
        return obs, info

    def _rolling_mean(self, window: list) -> float:
        return float(np.mean(window)) if window else 0.0

    def _get_obs(self) -> np.ndarray:
        """Construct state observation vector."""
        demands_mbps = self.current_demands_bps / 1e6
        gains_mag = np.abs(self.channel_gains).flatten()

        split_one_hot = np.zeros((self.n_ru, self.n_splits), dtype=np.float32)
        split_one_hot[np.arange(self.n_ru), self.split_idx] = 1.0

        obs = np.concatenate(
            [
                gains_mag,
                self.active_mask.astype(np.float32),
                split_one_hot.flatten(),
                demands_mbps.astype(np.float32),
                np.array([self.prev_power_w / 1000.0], dtype=np.float32),
                np.array([self.hour / 24.0], dtype=np.float32),
                np.array(
                    [self._rolling_mean(self._throughput_window)], dtype=np.float32
                ),
                np.array(
                    [self._rolling_mean(self._power_window) / 1000.0],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        return obs

    def _signal_interference(
        self, active_mask: np.ndarray, power_w: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-user signal power, interference power, and serving RU index.

        Each UE is served by its strongest active RU; every other active RU
        contributes co-channel interference (mirrors cran_env's convention).
        """
        gains_sq = np.abs(self.channel_gains) ** 2  # (n_ru, n_ue)
        rx_power = power_w[:, None] * gains_sq  # (n_ru, n_ue)
        rx_power_active = np.where(active_mask[:, None], rx_power, 0.0)

        signal = np.max(rx_power_active, axis=0)
        serving_ru = np.where(signal > 0.0, np.argmax(rx_power_active, axis=0), -1)

        total_active_power = np.sum(rx_power_active, axis=0)
        interference = total_active_power - signal
        return signal, interference, serving_ru

    def step(
        self, action: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the O-RAN environment.

        Args:
            action (dict): Action dict with 'ru_on', 'split', 'power', 'prb'
                (the 4 action branches, Concept Note Section 10.1).

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]: (obs, reward,
                terminated, truncated, info).
        """
        self.step_count += 1

        ru_on = np.array(action["ru_on"], dtype=bool)
        split_idx = np.array(action["split"], dtype=np.int64)
        power_w = np.clip(
            np.array(action["power"], dtype=np.float32), 0.0, self.p_max_w
        )
        power_w[~ru_on] = 0.0
        split_idx[~ru_on] = 0

        prb_raw = np.array(
            action.get("prb", np.ones(self.n_ru, dtype=np.float32)), dtype=np.float32
        )
        active_prb = prb_raw * ru_on.astype(np.float32)
        prb_sum = np.sum(active_prb)
        if prb_sum > 1e-12:
            prb_share = active_prb / prb_sum
        else:
            n_active = max(1, int(np.sum(ru_on)))
            prb_share = ru_on.astype(np.float32) / n_active

        signal, interference, serving_ru = self._signal_interference(ru_on, power_w)
        sinr = np.where(
            signal > 0.0, signal / (interference + self.noise_power_w), 0.0
        ).astype(np.float32)

        user_bandwidth_hz = (
            np.where(
                serving_ru >= 0,
                prb_share[np.clip(serving_ru, 0, self.n_ru - 1)],
                0.0,
            )
            * self.channel.bandwidth
        )
        achievable_capacity_bps = user_bandwidth_hz * np.log2(1.0 + sinr)
        total_throughput_mbps = float(np.sum(achievable_capacity_bps) / 1e6)

        # Reuse the exact demand realization already embedded in the
        # observation the agent acted on -- not a fresh independent sample
        # for this hour (see reset()'s comment / cran_env's fixed pattern).
        qos_violations_bps = np.maximum(
            0.0, self.current_demands_bps - achievable_capacity_bps
        )

        power_dict = self.power.compute_total_power(
            active_mask=ru_on,
            split_idx=split_idx,
            transmit_power_w=power_w,
            prev_active_mask=self.active_mask,
            prev_split_idx=self.split_idx,
        )
        p_total = power_dict["total"]

        exact_switching_count = int(
            np.sum(ru_on != self.active_mask)
            + np.sum((split_idx != self.split_idx) & (ru_on & self.active_mask))
        )

        ee_mbit_per_joule = total_throughput_mbps / (p_total + 1e-6)
        energy_efficiency_term = self.alpha_energy * ee_mbit_per_joule
        qos_penalty = self.beta_qos * (np.sum(qos_violations_bps) / 1e6)
        switch_penalty = self.gamma_switch * exact_switching_count
        reward = energy_efficiency_term - qos_penalty - switch_penalty

        self._throughput_window.append(total_throughput_mbps)
        self._power_window.append(p_total)
        if len(self._throughput_window) > self.upper_level_period_steps:
            self._throughput_window.pop(0)
            self._power_window.pop(0)

        self.active_mask = ru_on
        self.split_idx = split_idx
        self.prev_power_w = p_total
        self.hour = (self.hour + 1.0) % 24.0

        self.channel_gains = self.channel.generate_channel(self.distances, self.rng)

        # Sample the new hour's demand once; this is the realization
        # _get_obs() embeds below, and the *next* step() call reuses it.
        self.current_demands_bps = self.traffic.get_demands(self.hour, self.rng)

        obs = self._get_obs()

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "total_power_w": p_total,
            "ru_power_w": power_dict["ru"],
            "du_power_w": power_dict["du"],
            "cu_power_w": power_dict["cu"],
            "fronthaul_power_w": power_dict["fronthaul"],
            "switching_power_w": power_dict["switching"],
            "switching_events": exact_switching_count,
            "ee_mbit_per_joule": float(ee_mbit_per_joule),
            "throughput_mbps": total_throughput_mbps,
            "qos_violations_count": int(np.sum(qos_violations_bps > 0.0)),
            "qos_shortfall_mbps": float(np.sum(qos_violations_bps) / 1e6),
            "active_rus": int(np.sum(ru_on)),
        }
        return obs, float(reward), terminated, truncated, info
