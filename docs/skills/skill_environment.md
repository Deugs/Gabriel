# Skill: C-RAN Environment Design

> **Status**: Invokable as the Antigravity `build-environment` skill (`.agents/skills/build-environment/`), which points back at this file as the spec of record.
>
> **Note (tenth audit round)**: the code sketch below is illustrative, written before `cran_env/cran_env.py` existed, and has drifted from the real implementation in some details (e.g. `compute_fronthaul_power()` now also takes `total_throughput_mbps`; config access is `cfg["reward"]["..."]`-style, not `self.cfg.alpha_energy`-style attribute access). The reward formula specifically has been corrected below to include the `gamma_fronthaul` term added in round 9 (`config/default.yaml`'s `reward.gamma_fronthaul`) — read `cran_env/cran_env.py::step()` directly for the current, authoritative reward computation rather than this sketch.

## Purpose
Design, implement, and validate a Gymnasium-compatible C-RAN simulation environment for DRL training.

## Context
The environment models a 5G C-RAN with R Remote Radio Heads (RRHs), U User Equipments (UEs), and B Baseband Units (BBUs) in a centralized pool. The agent controls RRH activation (discrete) and transmit power (continuous) to maximize energy efficiency while meeting QoS.

## Rules
1. All physical models must be traceable to cited references (EARTH, 3GPP, Al-Zubaedi)
2. State space must include all information needed for optimal decision-making (Markov property)
3. Action space must exactly match the hybrid discrete-continuous formulation
4. Reward function must be differentiable with respect to continuous actions (for policy gradient)
5. Environment must be deterministic given a random seed (reproducibility)

## Components

### Channel Model
```python
class ChannelModel:
    def __init__(self, n_rrh, n_ue, carrier_freq_ghz=2.1, bandwidth_mhz=20):
        self.n_rrh = n_rrh
        self.n_ue = n_ue
        self.fc = carrier_freq_ghz * 1e9
        self.B = bandwidth_mhz * 1e6

    def compute_path_loss(self, distances):
        # Log-distance model with a COST-231 Hata-style intercept (not the
        # full COST-231 formula -- the 3.5 slope is a tunable exponent, not
        # COST-231's fixed height-derived slope, and a(hm)/C_m are omitted)
        PL0 = 46.3 + 33.9 * np.log10(self.fc / 1e6) - 13.82 * np.log10(30)
        PL = PL0 + 3.5 * np.log10(distances / 1000)  # d in km
        return PL

    def generate_channel(self, distances, rng):
        path_loss_db = self.compute_path_loss(distances)
        shadowing_db = rng.normal(0, 8, size=distances.shape)
        fading = (rng.standard_normal(distances.shape) + 
                  1j * rng.standard_normal(distances.shape)) / np.sqrt(2)

        h = 10**(-(path_loss_db + shadowing_db) / 20) * fading
        return h
```

### Traffic Model
```python
class TrafficModel:
    def __init__(self, n_ue, base_rate_mbps=50, peak_multiplier=3.0):
        self.n_ue = n_ue
        self.base_rate = base_rate_mbps * 1e6  # bps
        self.peak = peak_multiplier

    def get_demands(self, hour, rng):
        # Tidal traffic: business peaks at 10-12, 14-17; residential at 19-23
        business_factor = 0.5 + 0.5 * (
            np.sin(2 * np.pi * (hour - 9) / 24) + 
            np.sin(2 * np.pi * (hour - 17) / 24)
        )
        residential_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (hour - 21) / 24)

        base = self.base_rate * (0.3 + 0.7 * np.maximum(business_factor, residential_factor))
        noise = rng.lognormal(0, 0.2, self.n_ue)
        return base * noise
```

### Power Model
```python
class PowerModel:
    def __init__(self, n_rrh, n_bbu, pon_type="twdm"):
        self.n_rrh = n_rrh
        self.n_bbu = n_bbu

        # RRH parameters (Fathy et al.)
        self.p_active = 6.8  # W
        self.p_sleep = 4.3   # W
        self.p_switch = 3.0  # W per transition
        self.eta = 0.25    # PA efficiency

        # BBU parameters (EARTH model, Al-Zubaedi)
        self.p_stat = 175.0  # W
        self.p_dyn = 250.0   # W total dynamic
        self.delta_p = 0.44  # slope

        # Fronthaul parameters
        self.p_olt = 20.0
        self.p_onu_active = 5.0
        self.p_onu_sleep = 0.5

    def compute_rrh_power(self, active_mask, transmit_power):
        p_tx = np.sum(transmit_power[active_mask]) / self.eta
        p_static = np.sum(active_mask) * self.p_active
        p_sleep = np.sum(~active_mask) * self.p_sleep
        return p_tx + p_static + p_sleep

    def compute_bbu_power(self, loads):
        active_bbus = np.ceil(loads).astype(int)
        p_static = np.sum(active_bbus > 0) * self.p_stat
        p_dynamic = self.delta_p * self.p_dyn * np.sum(loads)
        return p_static + p_dynamic

    def compute_fronthaul_power(self, active_mask):
        p_onus = np.sum(active_mask) * self.p_onu_active
        p_onus += np.sum(~active_mask) * self.p_onu_sleep
        return self.p_olt + p_onus
```

### Main Environment
```python
import gymnasium as gym
from gymnasium import spaces

class CRANEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.n_rrh = config.n_rrh
        self.n_ue = config.n_ue
        self.n_bbu = config.n_bbu

        # Sub-models
        self.channel = ChannelModel(self.n_rrh, self.n_ue)
        self.traffic = TrafficModel(self.n_ue)
        self.power = PowerModel(self.n_rrh, self.n_bbu)

        # State: [channel_gains (R*U), active_mask (R), demands (U), prev_power (1), hour (1)]
        self.state_dim = self.n_rrh * self.n_ue + self.n_rrh + self.n_ue + 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Hybrid action space
        self.action_space = spaces.Dict({
            "rrh_on": spaces.MultiBinary(self.n_rrh),
            "power": spaces.Box(low=0, high=config.p_max, shape=(self.n_rrh,), dtype=np.float32)
        })

        self.rng = None
        self.state = None
        self.hour = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.hour = self.rng.integers(0, 24)

        # Initialize positions
        self.rrh_pos = self.rng.uniform(0, 1000, (self.n_rrh, 2))  # 1km x 1km area
        self.ue_pos = self.rng.uniform(0, 1000, (self.n_ue, 2))

        # Compute distances
        self.distances = np.linalg.norm(
            self.rrh_pos[:, None, :] - self.ue_pos[None, :, :], axis=2
        )

        self.channel_gains = self.channel.generate_channel(self.distances, self.rng)
        self.active_mask = np.ones(self.n_rrh, dtype=bool)
        self.prev_power = 0.0

        self.state = self._get_obs()
        return self.state, {}

    def _get_obs(self):
        demands = self.traffic.get_demands(self.hour, self.rng)
        obs = np.concatenate([
            self.channel_gains.flatten().real,  # Simplified: use magnitude
            self.active_mask.astype(float),
            demands / 1e6,  # Normalize to Mbps
            [self.prev_power / 1000],  # Normalize to kW
            [self.hour / 24]  # Normalize to [0,1]
        ]).astype(np.float32)
        return obs

    def step(self, action):
        rrh_on = action["rrh_on"].astype(bool)
        power = action["power"]

        # Ensure inactive RRHs have zero power
        power[~rrh_on] = 0.0

        # Compute SINR and capacity
        sinr = self._compute_sinr(rrh_on, power)
        capacity = self.B * np.log2(1 + sinr)

        # Compute power consumption
        p_rrh = self.power.compute_rrh_power(rrh_on, power)
        # Simplified: assume one BBU per active RRH for load
        loads = np.ones(self.n_bbu) * np.sum(rrh_on) / self.n_bbu
        p_bbu = self.power.compute_bbu_power(loads)
        p_fh = self.power.compute_fronthaul_power(rrh_on)
        p_total = p_rrh + p_bbu + p_fh

        # Compute reward
        demands = self.traffic.get_demands(self.hour, self.rng)
        qos_violations = np.maximum(0, demands - capacity)
        switching_cost = np.sum(np.abs(rrh_on.astype(int) - self.active_mask.astype(int))) * self.power.p_switch

        reward = -(self.cfg.alpha_energy * p_total / 1000 +
                   self.cfg.beta_qos * np.sum(qos_violations) / 1e6 +
                   self.cfg.gamma_switch * switching_cost / 10 +
                   self.cfg.gamma_fronthaul * p_fh / 1000)

        # Update state
        self.active_mask = rrh_on
        self.prev_power = p_total
        self.hour = (self.hour + 1) % 24

        # Update channel (Gauss-Markov)
        rho = 0.9
        new_fading = (self.rng.standard_normal(self.distances.shape) + 
                      1j * self.rng.standard_normal(self.distances.shape)) / np.sqrt(2)
        self.channel_gains = rho * self.channel_gains + np.sqrt(1 - rho**2) * new_fading

        self.state = self._get_obs()

        info = {
            "total_power": p_total,
            "qos_violations": np.sum(qos_violations > 0),
            "sinr_mean": np.mean(sinr),
            "active_rrhs": np.sum(rrh_on)
        }

        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, info

    def _compute_sinr(self, active_mask, power):
        # Simplified: single-user per RB, no CoMP
        sinr = np.zeros(self.n_ue)
        for u in range(self.n_ue):
            signal = np.sum(np.abs(self.channel_gains[active_mask, u])**2 * power[active_mask])
            interference = 0  # Simplified: no inter-cell interference in this model
            noise = self.cfg.noise_power
            sinr[u] = signal / (interference + noise)
        return sinr
```

## Validation Checklist
- [ ] Reset with same seed produces identical initial state
- [ ] Step with same action and seed produces identical next state
- [ ] Reward decreases monotonically with more active RRHs (all else equal)
- [ ] QoS violation penalty dominates when demands exceed capacity
- [ ] Power model values match reference (Al-Zubaedi) for known configurations
- [ ] Channel statistics match theoretical distributions (Rayleigh fading)
- [ ] Action space is valid for all algorithms (DDQN, SAC, Hybrid)
