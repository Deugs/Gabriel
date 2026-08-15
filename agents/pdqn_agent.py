"""Parameterized Deep Q-Network (P-DQN) Baseline, Flat over the Joint RRH Activation Space.

Conforms to MPhil Thesis Concept Note v3.0/v4.0 Section 12.1 (S2): P-DQN
(Xiong et al., 2018) "without branching or twin critics" — the discrete side
is a single flat Q-head over the enumerated 2^R joint on/off combinations
(not R independent per-RRH branches, cf. agents/branching_mp_dqn.py), and the
critic is a single network (no TD3-style twin). This is deliberately
intractable beyond R~12 (Section 10.3.1/B3): it is a baseline chosen to
empirically demonstrate why branching was necessary, not a scaled-down
version of the proposed method.

P-DQN evaluates every joint action's Q-value from ONE forward pass fed the
full, unmasked continuous parameter vector for all R RRHs (Xiong et al.'s
original "concatenation" design) — this is the "false gradient" cross-talk
Bester et al. (2019) identify and MP-DQN (agents/mpdqn_agent.py) fixes via
multi-pass masking.
"""

from collections import deque
import copy
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.branching_mp_dqn import ContinuousParameterNetwork, SharedEncoder

# P-DQN/MP-DQN are only tractable up to R=12 (2^12=4,096 joint actions); beyond
# this the flat discrete head would need 2^R output units, which is why these
# baselines are capped here rather than left to fail via OOM at R=35/50.
MAX_N_RRH_FOR_FLAT_JOINT_ACTION = 12


class JointDiscreteQNetwork(nn.Module):
    """Flat Q-head Q(s, a) over all 2^R enumerated joint discrete actions a.

    `fuse()` and `q_at_indices()` are split out from `forward()` so that
    MP-DQN's multi-pass evaluation (agents/mpdqn_agent.py) can extract only
    the single "self" Q-value each masked pass needs via a weight-row gather,
    instead of materializing the full (batch, 2^R) head output per masked
    pass — which would cost O(batch * 2^(2R)) and is not tractable even at
    the R<=12 range these baselines are meant to support.
    """

    def __init__(self, feature_dim: int, n_rrh: int, hidden_dim: int = 128):
        super().__init__()
        self.n_rrh = n_rrh
        self.n_joint_actions = 2**n_rrh
        self.param_encoder = nn.Linear(2 * n_rrh, 64)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + 64, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, self.n_joint_actions)

    def fuse(
        self, features: torch.Tensor, continuous_params: torch.Tensor
    ) -> torch.Tensor:
        # features: (batch, feature_dim); continuous_params: (batch, n_rrh, 2) or (batch, 2*n_rrh)
        if continuous_params.dim() == 3:
            continuous_params = continuous_params.reshape(
                continuous_params.shape[0], -1
            )
        param_feat = F.relu(self.param_encoder(continuous_params))
        return self.trunk(
            torch.cat([features, param_feat], dim=-1)
        )  # (batch, hidden_dim)

    def forward(
        self, features: torch.Tensor, continuous_params: torch.Tensor
    ) -> torch.Tensor:
        return self.head(self.fuse(features, continuous_params))  # (batch, 2^n_rrh)

    def q_at_indices(
        self, fused: torch.Tensor, action_indices: torch.Tensor
    ) -> torch.Tensor:
        """Q-value at `action_indices` only, via a weight-row gather (no full head pass)."""
        selected_w = self.head.weight[action_indices]  # (batch, hidden_dim)
        selected_b = self.head.bias[action_indices]  # (batch,)
        return (fused * selected_w).sum(dim=-1) + selected_b  # (batch,)


class JointActionReplayBuffer:
    """Experience Replay Buffer for flat joint-action transitions (s, a_idx, x, r, s', done)."""

    def __init__(self, capacity: int = 100000):
        self.buffer: deque[
            Tuple[np.ndarray, int, np.ndarray, float, np.ndarray, bool]
        ] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        joint_action_idx: int,
        continuous_params: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append(
            (state, int(joint_action_idx), continuous_params, reward, next_state, done)
        )

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = random.sample(self.buffer, batch_size)
        states, action_idxs, cont_params, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(action_idxs)),
            torch.FloatTensor(np.array(cont_params)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class PDQNAgent:
    """P-DQN Baseline: flat joint discrete head + single-pass continuous coupling."""

    def __init__(
        self,
        state_dim: int,
        n_rrh: int,
        p_max_w: float = 1.0,
        config: Optional[Union[dict, Any]] = None,
        device: str = "cpu",
    ):
        if n_rrh > MAX_N_RRH_FOR_FLAT_JOINT_ACTION:
            raise ValueError(
                f"P-DQN/MP-DQN require a flat 2^R-sized discrete head (R={n_rrh} -> "
                f"2^{n_rrh} joint actions), which is intractable beyond R="
                f"{MAX_N_RRH_FOR_FLAT_JOINT_ACTION}. This is the documented scaling "
                "limitation of these baselines (Concept Note v3.0/v4.0 Section "
                "10.3.1/12.1), not a bug: use agents.branching_mp_dqn.BranchingMPDQN "
                "at larger R."
            )

        self.state_dim = state_dim
        self.n_rrh = n_rrh
        self.n_joint_actions = 2**n_rrh
        self.p_max_w = p_max_w
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        cfg = config if config is not None else {}
        # NOTE: getattr(cfg, "algorithm", cfg) does NOT perform dict key
        # lookup — for a plain dict config it silently returns the whole
        # `cfg` object instead of `cfg["algorithm"]`, so every get_val()
        # below would fall through to its Python-side default regardless of
        # the YAML. Dict configs must be indexed with cfg.get(...).
        if isinstance(cfg, dict):
            algo_cfg = cfg.get("algorithm", {})
        else:
            algo_cfg = getattr(cfg, "algorithm", cfg)

        def get_val(key, default):
            if isinstance(algo_cfg, dict):
                return algo_cfg.get(key, default)
            return getattr(algo_cfg, key, default)

        self.gamma = float(get_val("gamma", 0.99))
        self.tau = float(get_val("tau", 0.005))

        self.epsilon = float(get_val("epsilon_start", 1.0))
        self.epsilon_end = float(get_val("epsilon_end", 0.01))
        self.epsilon_decay = float(get_val("epsilon_decay", 0.995))

        lr_discrete = float(get_val("lr_discrete", 1e-3))
        lr_actor = float(get_val("lr_actor", 1e-4))
        buffer_size = int(get_val("buffer_size", 100000))

        # Same architecture keys as the proposed agent (agents/branching_mp_dqn.py)
        # — a baseline trained with a different encoder than the proposed
        # method under the same config would violate docs/rules.md's
        # "Forbidden: Training proposed method with different hyperparameters
        # than baselines."
        hidden_dims = get_val("hidden_dims", None)
        if hidden_dims is not None:
            hidden_dims = list(hidden_dims)
        activation = str(get_val("activation", "relu"))
        use_layer_norm = bool(get_val("use_layer_norm", True))

        self.encoder = SharedEncoder(
            state_dim, hidden_dims, activation, use_layer_norm
        ).to(self.device)
        self.param_net = ContinuousParameterNetwork(self.encoder.output_dim, n_rrh).to(
            self.device
        )
        self.q_net = JointDiscreteQNetwork(self.encoder.output_dim, n_rrh).to(
            self.device
        )

        self.encoder_target = copy.deepcopy(self.encoder).to(self.device)
        self.param_net_target = copy.deepcopy(self.param_net).to(self.device)
        self.q_net_target = copy.deepcopy(self.q_net).to(self.device)

        self.critic_opt = optim.Adam(self.q_net.parameters(), lr=lr_discrete)
        self.param_opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.param_net.parameters()),
            lr=lr_actor,
        )

        self.memory = JointActionReplayBuffer(buffer_size)

    def _decode_action(self, action_idx: int) -> np.ndarray:
        """Decode a flat joint-action index into a per-RRH on/off vector."""
        bits = ((action_idx >> np.arange(self.n_rrh)) & 1).astype(np.int64)
        return bits

    def _encode_features(
        self, encoder: SharedEncoder, state_t: torch.Tensor
    ) -> torch.Tensor:
        return encoder(state_t)

    def _compute_q_all_actions(
        self,
        q_net: JointDiscreteQNetwork,
        features: torch.Tensor,
        continuous_params: torch.Tensor,
    ) -> torch.Tensor:
        """Single-pass (P-DQN) evaluation: feed the full, unmasked param vector once."""
        return q_net(features, continuous_params)

    def select_action(
        self, obs: np.ndarray, evaluate: bool = False
    ) -> Dict[str, np.ndarray]:
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self._encode_features(self.encoder, state_t)
            p_ratio, bw_share = self.param_net(feat)
            cont_params = torch.stack([p_ratio, bw_share], dim=-1)  # (1, n_rrh, 2)

            if not evaluate and random.random() < self.epsilon:
                action_idx = random.randrange(self.n_joint_actions)
            else:
                q_vals = self._compute_q_all_actions(self.q_net, feat, cont_params)
                action_idx = int(q_vals[0].argmax().item())

            if not evaluate:
                noise = torch.randn_like(p_ratio) * 0.05
                p_ratio = torch.clamp(p_ratio + noise, 0.0, 1.0)

            rrh_on = self._decode_action(action_idx)
            p_np = p_ratio[0].cpu().numpy() * self.p_max_w
            bw_np = bw_share[0].cpu().numpy()
            cont_np = np.stack([p_ratio[0].cpu().numpy(), bw_np], axis=-1)

        return {
            "rrh_on": rrh_on,
            "power": p_np,
            "bandwidth": bw_np,
            "continuous": cont_np,
            "action_idx": action_idx,
        }

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        if len(self.memory) < batch_size:
            return {"critic_loss": 0.0, "param_loss": 0.0, "epsilon": self.epsilon}

        (
            states,
            action_idxs,
            cont_params,
            rewards,
            next_states,
            dones,
        ) = self.memory.sample(batch_size)

        states = states.to(self.device)
        action_idxs = action_idxs.to(self.device)
        cont_params = cont_params.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_feat = self.encoder_target(next_states)
            next_p_ratio, next_bw_share = self.param_net_target(next_feat)
            next_cont_params = torch.stack([next_p_ratio, next_bw_share], dim=-1)

            # Double-DQN-style target action selection (single critic, no twin)
            next_feat_online = self.encoder(next_states)
            next_q_online = self._compute_q_all_actions(
                self.q_net, next_feat_online, next_cont_params
            )
            next_action_idxs = next_q_online.argmax(dim=-1)

            next_q_target = self._compute_q_all_actions(
                self.q_net_target, next_feat, next_cont_params
            )
            next_q_sel = next_q_target.gather(
                -1, next_action_idxs.unsqueeze(-1)
            )  # (batch, 1)

            y_target = rewards + self.gamma * (1.0 - dones) * next_q_sel

        feat = self.encoder(states)
        q_all = self._compute_q_all_actions(self.q_net, feat, cont_params)
        q_sel = q_all.gather(-1, action_idxs.unsqueeze(-1))

        critic_loss = F.mse_loss(q_sel, y_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # Continuous parameter network: P-DQN's deterministic policy gradient
        # (Xiong et al., 2018) — differentiate the GREEDY discrete action's
        # Q-value w.r.t. the continuous parameters, not the replayed action.
        feat_for_param = self.encoder(states)
        p_ratio, bw_share = self.param_net(feat_for_param)
        pred_params = torch.stack([p_ratio, bw_share], dim=-1)
        q_pred = self._compute_q_all_actions(
            self.q_net, feat_for_param.detach(), pred_params
        )
        greedy_action_idx = q_pred.argmax(dim=-1, keepdim=True).detach()
        param_loss = -q_pred.gather(-1, greedy_action_idx).mean()

        self.param_opt.zero_grad()
        param_loss.backward()
        nn.utils.clip_grad_norm_(self.param_net.parameters(), max_norm=1.0)
        self.param_opt.step()

        self._soft_update(self.encoder_target, self.encoder)
        self._soft_update(self.param_net_target, self.param_net)
        self._soft_update(self.q_net_target, self.q_net)

        return {
            "critic_loss": float(critic_loss.item()),
            "param_loss": float(param_loss.item()),
            "epsilon": float(self.epsilon),
        }

    def decay_exploration(self):
        """Decay epsilon once per episode (config/default.yaml's
        epsilon_decay is a per-episode rate; calling this from update(),
        which runs once per environment step, decayed far faster than
        intended)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
