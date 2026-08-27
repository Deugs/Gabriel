"""Channel Model for O-RAN Simulation.

Implements a simplified log-distance path loss model with fresh,
independent-per-step Rayleigh small-scale fading -- no shadowing term, no
Gauss-Markov temporal correlation, matching the source concept note's own
"simplified SINR with path loss and Rayleigh fading" description
(ORAN_BMPP_DQN_Concept_Note_v1.md Section 5.1). Deliberately simpler than
cran_env/channel_model.py, not a reuse of it: this module has zero imports
from cran_env/.
"""

import numpy as np


class ORANChannelModel:
    """Simplified Channel Model for O-RAN Downlink.

    Attributes:
        n_ru (int): Number of Radio Units.
        n_ue (int): Number of User Equipments.
        fc (float): Carrier frequency in Hz.
        bandwidth (float): Channel bandwidth in Hz.
        path_loss_exponent (float): Log-distance path loss exponent.
    """

    def __init__(
        self,
        n_ru: int,
        n_ue: int,
        carrier_freq_ghz: float = 3.5,
        bandwidth_mhz: float = 20.0,
        path_loss_exponent: float = 3.5,
    ):
        self.n_ru = n_ru
        self.n_ue = n_ue
        self.fc = carrier_freq_ghz * 1e9
        self.bandwidth = bandwidth_mhz * 1e6
        self.path_loss_exponent = path_loss_exponent

    def compute_path_loss(self, distances: np.ndarray) -> np.ndarray:
        """Compute log-distance path loss in dB.

        Args:
            distances (np.ndarray): Distance matrix in meters (n_ru, n_ue),
                minimum distance clamped to 10m.

        Returns:
            np.ndarray: Path loss in dB (n_ru, n_ue).
        """
        d_km = np.maximum(distances, 10.0) / 1000.0
        fc_mhz = self.fc / 1e6
        pl0 = 46.3 + 33.9 * np.log10(fc_mhz) - 13.82 * np.log10(10.0)
        return pl0 + 10.0 * self.path_loss_exponent * np.log10(d_km)

    def generate_channel(
        self, distances: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate a fresh complex channel matrix H (path loss + Rayleigh fading).

        No shadowing, no temporal correlation: called once per step with a
        fresh Rayleigh draw -- the "simplified" channel this track's concept
        note specifies (Section 5.1).

        Args:
            distances (np.ndarray): Distance matrix in meters (n_ru, n_ue).
            rng (np.random.Generator): NumPy random number generator.

        Returns:
            np.ndarray: Complex channel gain matrix H (n_ru, n_ue).
        """
        path_loss_db = self.compute_path_loss(distances)

        fading = (
            rng.standard_normal(distances.shape)
            + 1j * rng.standard_normal(distances.shape)
        ) / np.sqrt(2.0)

        attenuation = 10.0 ** (-path_loss_db / 20.0)
        return attenuation * fading
