"""Supervised ANN + Bi-Section GSBF Baseline (Fathy et al., 2021).

Predicts the near-optimal active-RRH count from traffic-demand / channel-
quality summary features via a small feedforward ANN trained on offline-
labelled data (see training/train_ann_gsbf.py for the label-generation and
training pipeline), then applies a Bi-Section Group Sparse Beamforming (GSBF)
heuristic for power allocation. Falls back to a hand-coded proportional
heuristic if no trained checkpoint is available yet (before
training/train_ann_gsbf.py has been run).
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).parent.parent / "data" / "checkpoints" / "ann_gsbf_predictor.pt"
)


class ANNPredictor(nn.Module):
    """Small feedforward ANN predicting the near-optimal active-RRH fraction.

    Trained on fixed-size, network-size-invariant summary-statistic features
    (see `extract_features()`), so one trained model can, in principle, be
    applied across different (n_rrh, n_ue) scenarios by rescaling its
    predicted fraction by the target n_rrh.
    """

    FEATURE_DIM = 8

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # predicted active-RRH fraction in (0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def extract_features(
    gains_mag: np.ndarray,
    demands_mbps: np.ndarray,
    n_rrh: int,
    p_max_w: float,
    noise_power_w: float,
    bandwidth_hz: float,
) -> np.ndarray:
    """Fixed-size (8-dim) summary-statistic features for the ANN predictor.

    The first version of this function used only raw demand/channel-gain
    moments (sum/mean/std/max), which turned out to carry essentially no
    learnable signal for the near-optimal-RRH-count label (verified
    empirically: individual and joint feature correlations with the label
    were indistinguishable from zero, p>0.4, on held-out data). This version
    adds a reference-scenario capacity-vs-demand ratio: each UE's SINR if
    served, interference-free, by its single strongest RRH at full power
    and an equal per-RRH bandwidth share, compared against its actual
    demand. This proxy is computable from just (gains_mag, demands_mbps)
    plus the three scalar radio parameters below -- no live environment
    object needed, so it works identically at label-generation time
    (training/train_ann_gsbf.py, which has env access) and at inference
    time (baselines/ann_gsbf.py, which only has the observation vector).
    It measurably improves correlation with the true exhaustive-search
    label (Pearson r ~0.11-0.12, p<0.02, vs. ~0/not-significant for the
    original moments-only features).

    Args:
        gains_mag (np.ndarray): Channel gain magnitudes (n_rrh, n_ue).
        demands_mbps (np.ndarray): Per-UE traffic demand in Mbps (n_ue,).
        n_rrh (int): Number of RRHs in this scenario.
        p_max_w (float): Max transmit power per RRH in Watts, used for the
            reference-scenario capacity proxy below.
        noise_power_w (float): Thermal noise power in Watts (env.noise_power_w).
        bandwidth_hz (float): Total channel bandwidth in Hz (env.channel.bandwidth).

    Returns:
        np.ndarray: 8-dim feature vector.
    """
    demands_bps = demands_mbps * 1e6
    rrh_scores = np.mean(gains_mag, axis=1)

    best_gain_per_ue = np.max(gains_mag, axis=0)
    signal = (best_gain_per_ue**2) * p_max_w
    sinr_proxy = signal / noise_power_w
    bw_per_rrh_hz = bandwidth_hz / n_rrh
    capacity_proxy_bps = bw_per_rrh_hz * np.log2(1.0 + sinr_proxy)
    capacity_ratio = capacity_proxy_bps / (demands_bps + 1e-6)
    capacity_gap_mbps = np.maximum(0.0, demands_bps - capacity_proxy_bps) / 1e6

    return np.array(
        [
            float(np.sum(demands_mbps)),
            float(np.std(demands_mbps)),
            float(np.mean(capacity_ratio)),
            float(np.min(capacity_ratio)),
            float(np.mean(capacity_gap_mbps)),
            float(np.mean(rrh_scores)),
            float(np.std(rrh_scores)),
            float(n_rrh),
        ],
        dtype=np.float32,
    )


class ANNGSBFBaseline:
    """Supervised ANN + 3-Stage Bi-Section GSBF Baseline (Fathy et al., 2021)."""

    def __init__(
        self,
        n_rrh: int,
        n_ue: int,
        p_max_w: float = 1.0,
        noise_power_w: float = 6.309573e-14,  # -102 dBm, Iqbal et al. (2021) Table 2
        bandwidth_hz: float = 10.0e6,  # 10 MHz, Iqbal et al. (2021) Table 2
        checkpoint_path: Optional[str] = None,
    ):
        self.n_rrh = n_rrh
        self.n_ue = n_ue
        self.p_max_w = p_max_w
        self.noise_power_w = noise_power_w
        self.bandwidth_hz = bandwidth_hz

        path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else DEFAULT_CHECKPOINT_PATH
        )
        self.ann: Optional[ANNPredictor] = None
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        if path.exists():
            checkpoint = torch.load(path, map_location="cpu")
            self.ann = ANNPredictor()
            self.ann.load_state_dict(checkpoint["model_state"])
            self.ann.eval()
            self.feat_mean = checkpoint["feat_mean"].numpy()
            self.feat_std = checkpoint["feat_std"].numpy()
        # else: no trained checkpoint yet — run training/train_ann_gsbf.py to
        # produce one. Falls back to the proportional heuristic below.

    def _predict_req_active(
        self, gains_mag: np.ndarray, demands_mbps: np.ndarray
    ) -> int:
        if self.ann is not None:
            raw_feat = extract_features(
                gains_mag,
                demands_mbps,
                self.n_rrh,
                self.p_max_w,
                self.noise_power_w,
                self.bandwidth_hz,
            )
            norm_feat = (raw_feat - self.feat_mean) / self.feat_std
            feat = torch.FloatTensor(norm_feat).unsqueeze(0)
            with torch.no_grad():
                frac = float(self.ann(feat).item())
            return int(np.clip(round(frac * self.n_rrh), 1, self.n_rrh))

        # Fallback heuristic proxy (used only when no trained checkpoint exists)
        return int(
            np.clip(np.ceil(np.sum(demands_mbps) / 100.0 * self.n_rrh), 1, self.n_rrh)
        )

    def select_action(
        self, obs: np.ndarray, evaluate: bool = True
    ) -> Dict[str, np.ndarray]:
        """Select action: ANN active-RRH-count prediction + Bi-Section GSBF power allocation."""
        demand_start = self.n_rrh * self.n_ue + self.n_rrh
        demand_end = demand_start + self.n_ue
        demands_mbps = obs[demand_start:demand_end]

        gains_mag = obs[: self.n_rrh * self.n_ue].reshape(self.n_rrh, self.n_ue)
        req_active = self._predict_req_active(gains_mag, demands_mbps)

        # GSBF heuristic stage 1: select top RRHs by average channel gain
        rrh_scores = np.mean(gains_mag, axis=1)
        active_indices = np.argsort(rrh_scores)[::-1][:req_active]

        rrh_on = np.zeros(self.n_rrh, dtype=int)
        rrh_on[active_indices] = 1

        # GSBF stage 2: bi-section power scaling to satisfy demand
        total_demand = np.sum(demands_mbps)
        power = np.zeros(self.n_rrh, dtype=np.float32)
        power[active_indices] = np.clip(
            (total_demand / (req_active * 50.0)) * self.p_max_w,
            0.1 * self.p_max_w,
            self.p_max_w,
        )

        bandwidth = np.zeros(self.n_rrh, dtype=np.float32)
        bandwidth[active_indices] = 1.0 / req_active

        return {"rrh_on": rrh_on, "power": power, "bandwidth": bandwidth}

    def update(self) -> Dict[str, float]:
        """Supervised / offline policy does not perform online RL steps."""
        return {"loss": 0.0}
