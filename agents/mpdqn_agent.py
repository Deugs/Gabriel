"""Multi-Pass Deep Q-Network (MP-DQN) Baseline, Flat over the Joint RRH Activation Space.

Conforms to MPhil Thesis Concept Note v3.0/v4.0 Section 12.1 (S2): MP-DQN
(Bester et al., 2019) "without branching or twin critics" — same flat 2^R
joint discrete head and single critic as agents/pdqn_agent.py (S2), but
fixes P-DQN's false-gradient cross-talk via genuine multi-pass evaluation:
for each candidate joint action a, every RRH not active under a has its
continuous parameters masked to zero before that action's Q-value is
computed (Bester et al., 2019). This is strictly more expensive per step
than plain P-DQN (2^R masked passes vs. one), which is itself part of the
documented case for branching (Section 10.3.1/B3) — MP-DQN without
branching does not scale any better than P-DQN, it is only less biased.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from agents.pdqn_agent import JointDiscreteQNetwork, PDQNAgent


class MPDQNAgent(PDQNAgent):
    """MP-DQN Baseline: flat joint discrete head with true multi-pass masking."""

    def __init__(
        self,
        state_dim: int,
        n_rrh: int,
        p_max_w: float = 1.0,
        config: Optional[Union[dict, Any]] = None,
        device: str = "cpu",
    ):
        super().__init__(
            state_dim=state_dim,
            n_rrh=n_rrh,
            p_max_w=p_max_w,
            config=config,
            device=device,
        )
        # action_bits[a, r] = 1 if RRH r is ON under joint action index a, else 0.
        action_idx = torch.arange(self.n_joint_actions)
        rrh_idx = torch.arange(self.n_rrh)
        self.action_bits = (
            (action_idx.unsqueeze(1) >> rrh_idx.unsqueeze(0)) & 1
        ).float().to(self.device)  # (n_joint_actions, n_rrh)

    def _compute_q_all_actions(
        self,
        q_net: JointDiscreteQNetwork,
        features: torch.Tensor,
        continuous_params: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-pass (MP-DQN) evaluation: mask every RRH not active under the
        candidate action being scored, for every one of the 2^R candidates.
        """
        batch = features.shape[0]
        n_actions = self.n_joint_actions
        device = features.device

        bits = self.action_bits.to(device)  # (n_actions, n_rrh)

        # Expand state features across all candidate actions: (batch*n_actions, feat_dim)
        feat_exp = (
            features.unsqueeze(1)
            .expand(batch, n_actions, features.shape[-1])
            .reshape(batch * n_actions, -1)
        )

        # Expand + mask continuous params per candidate action:
        # (batch, n_actions, n_rrh, 2) -> (batch*n_actions, n_rrh, 2)
        params_exp = continuous_params.unsqueeze(1).expand(
            batch, n_actions, self.n_rrh, 2
        )
        mask = bits.unsqueeze(0).unsqueeze(-1)  # (1, n_actions, n_rrh, 1)
        masked_params = (params_exp * mask).reshape(batch * n_actions, self.n_rrh, 2)

        fused = q_net.fuse(feat_exp, masked_params)  # (batch*n_actions, hidden_dim)

        # Row i (i = b*n_actions + a) should read off Q at its OWN action index a.
        self_action_idx = (
            torch.arange(n_actions, device=device).unsqueeze(0).expand(batch, n_actions)
        ).reshape(-1)
        q_self = q_net.q_at_indices(fused, self_action_idx)  # (batch*n_actions,)

        return q_self.reshape(batch, n_actions)
