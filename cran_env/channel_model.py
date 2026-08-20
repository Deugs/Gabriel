"""Channel Model for 5G C-RAN Simulation.

Implements a log-distance path loss model (PL0 + 10*n*log10(d/d0), per
docs/thesis_guide.md Section 3.2) with a COST231-Hata-style intercept term,
log-normal shadowing, Rayleigh small-scale fading, and Gauss-Markov temporal
correlation. This is not the full COST231-Hata formula: the distance-decay
coefficient (`path_loss_exponent`) is a configurable generic exponent rather
than COST231's fixed height-derived slope, and COST231's mobile-antenna-height
correction a(hm) and city-size correction C_m terms are not implemented.
"""

import numpy as np


class ChannelModel:
    """Channel Model for C-RAN Downlink.

    Attributes:
        n_rrh (int): Number of Remote Radio Heads.
        n_ue (int): Number of User Equipments.
        fc (float): Carrier frequency in Hz (e.g. 2.1e9).
        bandwidth (float): Channel bandwidth in Hz (e.g. 20e6).
        path_loss_exponent (float): Path loss exponent (default: 3.5).
        shadowing_std_db (float): Shadowing standard deviation in dB (default: 8.0).
        correlation_coeff (float): Gauss-Markov temporal correlation factor rho (default: 0.9).
    """

    def __init__(
        self,
        n_rrh: int,
        n_ue: int,
        carrier_freq_ghz: float = 2.1,
        bandwidth_mhz: float = 20.0,
        path_loss_exponent: float = 3.5,
        shadowing_std_db: float = 8.0,
        correlation_coeff: float = 0.9,
    ):
        self.n_rrh = n_rrh
        self.n_ue = n_ue
        self.fc = carrier_freq_ghz * 1e9
        self.bandwidth = bandwidth_mhz * 1e6
        self.path_loss_exponent = path_loss_exponent
        self.shadowing_std_db = shadowing_std_db
        self.correlation_coeff = correlation_coeff

    def compute_path_loss(self, distances: np.ndarray) -> np.ndarray:
        """Compute log-distance path loss in dB (PL0 + 10*n*log10(d/d0)).

        PL0 uses a COST231-Hata-style intercept for a 30m base-station
        height, but the distance-decay term uses `path_loss_exponent` as a
        configurable generic exponent rather than COST231's fixed
        height-derived slope, and omits COST231's a(hm)/C_m correction
        terms -- so this is a log-distance model with a COST231-derived
        constant, not the full COST231-Hata formula.

        Args:
            distances (np.ndarray): Distance matrix in meters (n_rrh, n_ue),
                minimum distance clamped to 10m.

        Returns:
            np.ndarray: Path loss in dB (n_rrh, n_ue).
        """
        # Minimum distance 10m to avoid log(0)
        d_km = np.maximum(distances, 10.0) / 1000.0
        # COST231-Hata-style intercept (PL0) for a 30m base-station height
        fc_mhz = self.fc / 1e6
        pl0 = 46.3 + 33.9 * np.log10(fc_mhz) - 13.82 * np.log10(30.0)
        path_loss_db = pl0 + 10.0 * self.path_loss_exponent * np.log10(d_km)
        return path_loss_db

    def generate_channel(
        self, distances: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate initial complex channel matrix H with path loss, shadowing, and Rayleigh fading.

        Args:
            distances (np.ndarray): Distance matrix in meters (n_rrh, n_ue).
            rng (np.random.Generator): NumPy random number generator.

        Returns:
            np.ndarray: Complex channel gain matrix H (n_rrh, n_ue).
        """
        path_loss_db = self.compute_path_loss(distances)
        shadowing_db = rng.normal(0.0, self.shadowing_std_db, size=distances.shape)

        # Complex Rayleigh small-scale fading with unit variance E[|fading|^2] = 1
        fading = (
            rng.standard_normal(distances.shape)
            + 1j * rng.standard_normal(distances.shape)
        ) / np.sqrt(2.0)

        # Total channel power attenuation: PL_linear = 10^(-PL_dB / 10)
        # Channel coefficient: h = 10^(-(PL_dB + Shadowing_dB) / 20) * fading
        attenuation = 10.0 ** (-(path_loss_db + shadowing_db) / 20.0)
        channel_gains = attenuation * fading
        return channel_gains

    def step_channel(
        self,
        current_gains: np.ndarray,
        distances: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Update channel matrix using Gauss-Markov temporal correlation process.

        h_{t+1} = rho * h_t + sqrt(1 - rho^2) * e_{t+1}

        Args:
            current_gains (np.ndarray): Current complex channel gain matrix H_t.
            distances (np.ndarray): Distance matrix in meters.
            rng (np.random.Generator): NumPy random number generator.

        Returns:
            np.ndarray: Updated complex channel gain matrix H_{t+1}.
        """
        rho = self.correlation_coeff
        path_loss_db = self.compute_path_loss(distances)
        shadowing_db = rng.normal(0.0, self.shadowing_std_db, size=distances.shape)

        new_fading = (
            rng.standard_normal(distances.shape)
            + 1j * rng.standard_normal(distances.shape)
        ) / np.sqrt(2.0)
        new_gains = 10.0 ** (-(path_loss_db + shadowing_db) / 20.0) * new_fading

        updated_gains = rho * current_gains + np.sqrt(1.0 - rho**2) * new_gains
        return updated_gains
