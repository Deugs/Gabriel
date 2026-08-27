"""MP-DQN Baseline, Flat over the Joint (RU activation x split) Action Space.

Per ORAN_BMPP_DQN_Concept_Note_v1.md Section 2.2/4.2: "MP-DQN fixes the
[P-DQN] over-parameterization but uses a flat architecture, leading to
action interference" -- this baseline enumerates the full joint discrete
action space (2^n_ru * n_splits^n_ru combinations of RU activation and
split choice) behind a single flat Q-head, with true multi-pass masking:
for each candidate joint action, every RU not active under that action has
its continuous parameters (power, PRB) masked to zero before that
action's Q-value is computed (Bester et al., 2019's fix for P-DQN's
cross-talk). Deliberately intractable beyond a small n_ru -- this is the
baseline the proposed method's branching decomposition is meant to
outperform on scalability, not a scaled-down version of it.

Structurally templated on agents/pdqn_agent.py's JointDiscreteQNetwork and
agents/mpdqn_agent.py's vectorized multi-pass masking, but reimplemented
locally with zero import from agents/.
"""

from collections import deque
import copy
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Joint action count is 2^n_ru * n_splits^n_ru; at n_ru=6, n_splits=3 this is
# already 2^6 * 3^6 = 46656 -- borderline but workable. Mirrors
# config/default.yaml's MAX_N_RRH_FOR_FLAT_JOINT_ACTION pattern.
MAX_N_RU_FOR_FLAT_JOINT_ORAN_ACTION = 6


class JointDiscreteQNetwork(nn.Module):
    """Flat Q-head Q(s, a) over all enumerated joint (ru_on, split) actions a."""

    def __init__(
        self, feature_dim: int, n_ru: int, n_joint_actions: int, hidden_dim: int = 128
    ):
        super().__init__()
        self.n_ru = n_ru
        self.n_joint_actions = n_joint_actions
        self.param_encoder = nn.Linear(2 * n_ru, 64)
        self.trunk = nn.Sequential(nn.Linear(feature_dim + 64, hidden_dim), nn.ReLU())
        self.head = nn.Linear(hidden_dim, n_joint_actions)

    def fuse(
        self, features: torch.Tensor, continuous_params: torch.Tensor
    ) -> torch.Tensor:
        if continuous_params.dim() == 3:
            continuous_params = continuous_params.reshape(
                continuous_params.shape[0], -1
            )
        param_feat = F.relu(self.param_encoder(continuous_params))
        return self.trunk(torch.cat([features, param_feat], dim=-1))

    def forward(
        self, features: torch.Tensor, continuous_params: torch.Tensor
    ) -> torch.Tensor:
        return self.head(self.fuse(features, continuous_params))

    def q_at_indices(
        self, fused: torch.Tensor, action_indices: torch.Tensor
    ) -> torch.Tensor:
        selected_w = self.head.weight[action_indices]
        selected_b = self.head.bias[action_indices]
        return (fused * selected_w).sum(dim=-1) + selected_b


class _Encoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ContinuousParameterNetwork(nn.Module):
    def __init__(self, feature_dim: int, n_ru: int):
        super().__init__()
        self.power_head = nn.Linear(feature_dim, n_ru)
        self.prb_head = nn.Linear(feature_dim, n_ru)

    def forward(self, features: torch.Tensor):
        power_ratio = torch.sigmoid(self.power_head(features))
        prb_share = F.softmax(self.prb_head(features), dim=-1)
        return power_ratio, prb_share


class JointActionReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action_idx, cont_params, reward, next_state, done):
        self.buffer.append((state, action_idx, cont_params, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, action_idx, cont_params, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(action_idx)),
            torch.FloatTensor(np.array(cont_params)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class ORANMPDQNAgent:
    """Flat, multi-pass-masked joint-action MP-DQN baseline for the O-RAN track."""

    def __init__(
        self,
        state_dim: int,
        n_ru: int,
        n_splits: int = 3,
        p_max_w: float = 1.0,
        config: Optional[Union[dict, Any]] = None,
        device: Optional[str] = None,
    ):
        if n_ru > MAX_N_RU_FOR_FLAT_JOINT_ORAN_ACTION:
            raise ValueError(
                f"n_ru={n_ru} exceeds MAX_N_RU_FOR_FLAT_JOINT_ORAN_ACTION="
                f"{MAX_N_RU_FOR_FLAT_JOINT_ORAN_ACTION}; the flat joint action "
                f"space (2^n_ru * n_splits^n_ru) is intractable beyond this."
            )

        self.state_dim = state_dim
        self.n_ru = n_ru
        self.n_splits = n_splits
        self.p_max_w = p_max_w
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.n_ru_combos = 2**n_ru
        self.n_split_combos = n_splits**n_ru
        self.n_joint_actions = self.n_ru_combos * self.n_split_combos

        cfg = config if config is not None else {}
        algo_cfg = cfg.get("algorithm", {}) if isinstance(cfg, dict) else {}

        def get_val(key: str, default: Any) -> Any:
            return algo_cfg.get(key, default) if isinstance(algo_cfg, dict) else default

        self.gamma = float(get_val("gamma", 0.99))
        self.tau = float(get_val("tau", 0.005))
        self.epsilon = float(get_val("epsilon_start", 1.0))
        self.epsilon_end = float(get_val("epsilon_end", 0.01))
        self.epsilon_decay = float(get_val("epsilon_decay", 0.995))
        self.gradient_clip_norm = float(get_val("gradient_clip_norm", 1.0))
        lr = float(get_val("lr_discrete", 1.0e-4))
        lr_actor = float(get_val("lr_actor", 3.0e-4))
        buffer_capacity = int(get_val("lower_buffer_size", 100000))

        self.encoder = _Encoder(state_dim).to(self.device)
        self.encoder_target = copy.deepcopy(self.encoder).to(self.device)
        self.q_net = JointDiscreteQNetwork(
            self.encoder.output_dim, n_ru, self.n_joint_actions
        ).to(self.device)
        self.q_net_target = copy.deepcopy(self.q_net).to(self.device)
        self.param_net = ContinuousParameterNetwork(self.encoder.output_dim, n_ru).to(
            self.device
        )

        self.critic_opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.q_net.parameters()), lr=lr
        )
        self.param_opt = optim.Adam(self.param_net.parameters(), lr=lr_actor)

        self.memory = JointActionReplayBuffer(buffer_capacity)

        # ru_bits[a, r] = 1 if RU r is ON under ru-combo index a.
        ru_combo_idx = torch.arange(self.n_ru_combos)
        ru_idx = torch.arange(n_ru)
        self.ru_bits = (
            ((ru_combo_idx.unsqueeze(1) >> ru_idx.unsqueeze(0)) & 1)
            .float()
            .to(self.device)
        )  # (n_ru_combos, n_ru)

        # split_digits[a, r] = split choice for RU r under split-combo index a
        # (base-n_splits digit extraction).
        split_combo_idx = torch.arange(self.n_split_combos)
        digits = torch.zeros(self.n_split_combos, n_ru, dtype=torch.long)
        tmp = split_combo_idx.clone()
        for r in range(n_ru):
            digits[:, r] = tmp % n_splits
            tmp = tmp // n_splits
        self.split_digits = digits.to(self.device)  # (n_split_combos, n_ru)

        # Full joint action's (ru_on, split) per index a = ru_idx * n_split_combos + split_idx
        self.joint_ru_bits = self.ru_bits.repeat_interleave(
            self.n_split_combos, dim=0
        )  # (n_joint_actions, n_ru)
        self.joint_split = self.split_digits.repeat(
            self.n_ru_combos, 1
        )  # (n_joint_actions, n_ru)

    def _compute_q_all_actions(
        self, features: torch.Tensor, continuous_params: torch.Tensor
    ) -> torch.Tensor:
        """Multi-pass evaluation: mask every RU not active under each
        candidate joint action, vectorized over batch*n_joint_actions."""
        batch = features.shape[0]
        n_actions = self.n_joint_actions
        device = features.device

        bits = self.joint_ru_bits.to(device)  # (n_actions, n_ru)

        feat_exp = (
            features.unsqueeze(1)
            .expand(batch, n_actions, features.shape[-1])
            .reshape(batch * n_actions, -1)
        )
        params_exp = continuous_params.unsqueeze(1).expand(
            batch, n_actions, self.n_ru, 2
        )
        mask = bits.unsqueeze(0).unsqueeze(-1)  # (1, n_actions, n_ru, 1)
        masked_params = (params_exp * mask).reshape(batch * n_actions, self.n_ru, 2)

        fused = self.q_net.fuse(feat_exp, masked_params)
        self_idx = (
            torch.arange(n_actions, device=device).unsqueeze(0).expand(batch, n_actions)
        ).reshape(-1)
        q_self = self.q_net.q_at_indices(fused, self_idx)
        return q_self.reshape(batch, n_actions)

    def select_action(
        self, obs: np.ndarray, evaluate: bool = False
    ) -> Dict[str, np.ndarray]:
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.encoder(state_t)
            power_ratio, prb_share = self.param_net(feat)
            if not evaluate:
                noise = torch.randn_like(power_ratio) * 0.1
                power_ratio = torch.clamp(power_ratio + noise, 0.0, 1.0)
            cont_params = torch.stack([power_ratio, prb_share], dim=-1)

            if not evaluate and random.random() < self.epsilon:
                action_idx = random.randrange(self.n_joint_actions)
            else:
                q_all = self._compute_q_all_actions(feat, cont_params)
                action_idx = int(q_all[0].argmax().item())

        ru_on = self.joint_ru_bits[action_idx].cpu().numpy().astype(np.int64)
        split = self.joint_split[action_idx].cpu().numpy().astype(np.int64)
        power = (power_ratio[0].cpu().numpy() * self.p_max_w).astype(np.float32)
        prb = prb_share[0].cpu().numpy().astype(np.float32)

        self._last_action_idx = action_idx
        return {"ru_on": ru_on, "split": split, "power": power, "prb": prb}

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        if len(self.memory) < batch_size:
            return {"critic_loss": 0.0, "param_loss": 0.0, "epsilon": self.epsilon}

        states, action_idx, cont_params, rewards, next_states, dones = (
            self.memory.sample(batch_size)
        )
        states = states.to(self.device)
        action_idx = action_idx.to(self.device)
        cont_params = cont_params.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_feat = self.encoder_target(next_states)
            next_power, next_prb = self.param_net(next_feat)
            next_cont = torch.stack([next_power, next_prb], dim=-1)
            next_q_all = self._compute_q_all_actions(next_feat, next_cont)
            next_q_max = next_q_all.max(dim=-1, keepdim=True).values
            target_q = rewards + self.gamma * (1.0 - dones) * next_q_max

        feat = self.encoder(states)
        fused = self.q_net.fuse(feat, cont_params)
        current_q = self.q_net.q_at_indices(fused, action_idx).unsqueeze(-1)

        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.q_net.parameters()),
            max_norm=self.gradient_clip_norm,
        )
        self.critic_opt.step()

        feat_for_param = self.encoder(states).detach()
        power_ratio, prb_share = self.param_net(feat_for_param)
        pred_params = torch.stack([power_ratio, prb_share], dim=-1)
        q_pred_all = self._compute_q_all_actions(feat_for_param, pred_params)
        greedy_idx = q_pred_all.argmax(dim=-1, keepdim=True).detach()
        param_loss = -q_pred_all.gather(-1, greedy_idx).mean()

        self.param_opt.zero_grad()
        param_loss.backward()
        nn.utils.clip_grad_norm_(
            self.param_net.parameters(), max_norm=self.gradient_clip_norm
        )
        self.param_opt.step()

        for t_param, s_param in zip(
            self.encoder_target.parameters(), self.encoder.parameters()
        ):
            t_param.data.copy_(
                self.tau * s_param.data + (1.0 - self.tau) * t_param.data
            )
        for t_param, s_param in zip(
            self.q_net_target.parameters(), self.q_net.parameters()
        ):
            t_param.data.copy_(
                self.tau * s_param.data + (1.0 - self.tau) * t_param.data
            )

        return {
            "critic_loss": float(critic_loss.item()),
            "param_loss": float(param_loss.item()),
            "epsilon": self.epsilon,
        }

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
