"""Branching Multi-Pass Parameterized Deep Q-Network with Twin Critics (Branching MP-DQN + TD3).

Conforms strictly to MPhil Thesis Concept Note v2.0 Section 10
(architecture unchanged through v3.0/v4.0, the current reference document):
- P-DQN (Xiong et al., 2018): Action coupling discrete RRH activation k_r with x_r = (p_r, beta_r).
- MP-DQN (Bester et al., 2019): Multi-pass masking to prevent parameter cross-talk.
- Branching DRL (Tavakoli et al., 2018): Factorized decision heads scaling linearly (2R).
- Twin Critics & Target Smoothing (Fujimoto et al. / TD3, 2018): Countering Q overestimation.
"""

from collections import deque
import copy
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_ACTIVATIONS: Dict[str, Any] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def _resolve_activation(name: str) -> Any:
    try:
        return _ACTIVATIONS[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm.activation '{name}'; supported: {sorted(_ACTIVATIONS)}"
        )


class SharedEncoder(nn.Module):
    """Shared state encoder h(s|theta_h) mapping state s(t) to feature representation.

    Widths (`hidden_dims`), the nonlinearity (`activation`), and whether each
    layer is followed by LayerNorm (`use_layer_norm`) are all config-driven via
    `config/default.yaml`'s `algorithm:` block (Section 10.3 of the concept
    note documents the [256, 128] spec value used when a config omits them).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        activation_cls = _resolve_activation(activation)

        layers: List[nn.Module] = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation_cls())
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            prev_dim = dim

        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class BranchingDiscreteHeads(nn.Module):
    """Factorized Dueling Discrete Branch Heads Q_r(s, k_r) for R RRHs."""

    def __init__(self, feature_dim: int, n_rrh: int):
        super().__init__()
        self.n_rrh = n_rrh
        # Value head V(s) and Advantage heads A_r(s, k_r) for k_r in {0, 1}
        self.value_head = nn.Linear(feature_dim, 1)
        self.adv_heads = nn.ModuleList(
            [nn.Linear(feature_dim, 2) for _ in range(n_rrh)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (batch_size, feature_dim)
        v = self.value_head(features).unsqueeze(1)  # (batch, 1, 1)
        advs = torch.stack(
            [head(features) for head in self.adv_heads], dim=1
        )  # (batch, n_rrh, 2)
        # Dueling aggregation: Q_r(s, a) = V(s) + (A_r(s, a) - mean(A_r(s, .)))
        q_vals = v + (advs - advs.mean(dim=-1, keepdim=True))
        return q_vals


class ContinuousParameterNetwork(nn.Module):
    """Deterministic continuous parameter network x(s|phi) producing (p_r, beta_r) per RRH."""

    def __init__(self, feature_dim: int, n_rrh: int):
        super().__init__()
        self.n_rrh = n_rrh
        # Transmit power ratio p_r in [0, 1] and bandwidth share beta_r in [0, 1]
        self.power_head = nn.Linear(feature_dim, n_rrh)
        self.bandwidth_head = nn.Linear(feature_dim, n_rrh)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        power_ratio = torch.sigmoid(self.power_head(features))
        bandwidth_share = F.softmax(self.bandwidth_head(features), dim=-1)
        return power_ratio, bandwidth_share


class SingleBranchCritic(nn.Module):
    """Single Multi-Pass Branch Critic Q_r(s, k_r, x_r).

    Takes the SharedEncoder's output feature directly rather than owning its
    own encoder copy (Section 10.3: theta_h is one set of weights feeding
    both the continuous parameter network and every branch critic, not a
    separately-trained copy per critic)."""

    def __init__(self, feature_dim: int, n_rrh: int):
        super().__init__()
        self.n_rrh = n_rrh
        self.discrete_heads = BranchingDiscreteHeads(feature_dim, n_rrh)
        # Parameter encoder: 2 continuous parameters (power, bandwidth) per RRH
        self.param_encoder = nn.Linear(2, 64)
        self.fusion = nn.Linear(feature_dim + 64, feature_dim)

    def forward(
        self,
        feat: torch.Tensor,
        continuous_params: torch.Tensor,
        branch_mask_idx: Optional[int] = None,
    ) -> torch.Tensor:
        # feat: (batch, feature_dim), already produced by the shared encoder
        # continuous_params: (batch, n_rrh, 2)

        # Ensure continuous_params is 3D (batch, n_rrh, 2)
        if continuous_params.dim() == 2:
            params_3d = continuous_params.view(-1, self.n_rrh, 2)
        else:
            params_3d = continuous_params

        if branch_mask_idx is not None:
            # Multi-Pass evaluation (MP-DQN): parameter for branch_mask_idx
            param_feat = F.relu(self.param_encoder(params_3d[:, branch_mask_idx, :]))
        else:
            # Mean parameter feature across branches (batch, 2)
            param_feat = F.relu(self.param_encoder(params_3d.mean(dim=1)))

        fused = F.relu(self.fusion(torch.cat([feat, param_feat], dim=-1)))
        q_vals = self.discrete_heads(fused)  # (batch, n_rrh, 2)
        return q_vals


class TwinBranchCritic(nn.Module):
    """Twin Critic Networks (Q^A, Q^B) for TD3 over-estimation mitigation."""

    def __init__(self, feature_dim: int, n_rrh: int):
        super().__init__()
        self.q_a = SingleBranchCritic(feature_dim, n_rrh)
        self.q_b = SingleBranchCritic(feature_dim, n_rrh)

    def forward(
        self,
        feat: torch.Tensor,
        continuous_params: torch.Tensor,
        branch_mask_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_a_vals = self.q_a(feat, continuous_params, branch_mask_idx)
        q_b_vals = self.q_b(feat, continuous_params, branch_mask_idx)
        return q_a_vals, q_b_vals


class ParameterizedReplayBuffer:
    """Experience Replay Buffer for Parameterized Transitions (s, k, x, r, s', done)."""

    def __init__(self, capacity: int = 100000):
        self.buffer: deque[
            Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, bool]
        ] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        discrete_action: np.ndarray,
        continuous_params: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append(
            (state, discrete_action, continuous_params, reward, next_state, done)
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
        states, disc_actions, cont_params, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(disc_actions)),
            torch.FloatTensor(np.array(cont_params)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class BranchingMPDQN:
    """Proposed Branching MP-DQN + TD3 Framework for 5G C-RAN Energy Optimization."""

    def __init__(
        self,
        state_dim: int,
        n_rrh: int,
        p_max_w: float = 1.0,
        config: Optional[Union[dict, Any]] = None,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.n_rrh = n_rrh
        self.p_max_w = p_max_w

        # Extract hyperparameters. NOTE: getattr(cfg, "algorithm", cfg) does
        # NOT perform dict key lookup — for a plain dict config (the only
        # kind this codebase ever actually constructs, via yaml.safe_load),
        # it silently returns the whole `cfg` object rather than
        # `cfg["algorithm"]`, so every get_val() below would fall through to
        # its Python-side default regardless of the YAML. Dict configs must
        # be indexed with cfg.get(...), not getattr(...).
        cfg = config if config is not None else {}
        if isinstance(cfg, dict):
            algo_cfg = cfg.get("algorithm", {})
            hardware_cfg = cfg.get("hardware", {})
        else:
            algo_cfg = getattr(cfg, "algorithm", cfg)
            hardware_cfg = getattr(cfg, "hardware", cfg)

        def get_val(key, default):
            if isinstance(algo_cfg, dict):
                return algo_cfg.get(key, default)
            return getattr(algo_cfg, key, default)

        def get_hw_val(key, default):
            if isinstance(hardware_cfg, dict):
                return hardware_cfg.get(key, default)
            return getattr(hardware_cfg, key, default)

        # `hardware.device` (config/default.yaml) only supplies a default —
        # an explicit `device=` argument from the caller always wins.
        if device is None:
            device = str(get_hw_val("device", "cpu"))
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.gamma = float(get_val("gamma", 0.99))
        self.tau = float(get_val("tau", 0.005))
        self.policy_delay = int(get_val("policy_delay", 2))
        self.target_noise_std = float(get_val("target_noise_std", 0.05))

        self.epsilon = float(get_val("epsilon_start", 1.0))
        self.epsilon_end = float(get_val("epsilon_end", 0.01))
        self.epsilon_decay = float(get_val("epsilon_decay", 0.995))

        # Continuous exploration noise (Section 12.2: "Gaussian, sigma=0.1*P_max
        # (decayed)"). p_ratio/bw_share are already normalized to [0,1] as a
        # fraction of P_max, so a std of 0.1 there is exactly "0.1*P_max" in
        # absolute power units. Decays alongside epsilon (same rate, once per
        # episode via decay_exploration()) down to a small floor rather than
        # vanishing entirely.
        self.continuous_noise_std = float(get_val("continuous_noise_std", 0.1))
        self.continuous_noise_std_end = float(get_val("continuous_noise_std_end", 0.01))

        lr_branch = float(get_val("lr_discrete", 1e-4))
        lr_param = float(get_val("lr_actor", 3e-4))
        buffer_size = int(get_val("buffer_size", 100000))
        # Replay-buffer warm-up: don't start training until at least this
        # many transitions are collected. Distinct from batch_size (how many
        # transitions a single update() samples) — min_buffer_size can, and
        # by default does, exceed batch_size for a longer warm-up.
        self.min_buffer_size = int(get_val("min_buffer_size", 10000))

        # Training stability (Section 10.3): gradient clipping applied to both
        # critic and parameter-network updates in update(); reward scaling
        # applied to sampled rewards before the Bellman target is computed.
        self.gradient_clip_norm = float(get_val("gradient_clip_norm", 1.0))
        self.reward_scale = float(get_val("reward_scale", 1.0))

        # Network architecture (Section 10.3): defaults match the [256, 128]
        # spec value when a config omits these keys.
        hidden_dims = get_val("hidden_dims", None)
        if hidden_dims is not None:
            hidden_dims = list(hidden_dims)
        activation = str(get_val("activation", "relu"))
        use_layer_norm = bool(get_val("use_layer_norm", True))

        # Shared encoder: ONE instance, feeding both the continuous parameter
        # network and every branch critic (Section 10.3) — not a separate
        # copy per consumer, which the pre-refactor code built by accident
        # (SingleBranchCritic used to construct its own SharedEncoder).
        self.encoder = SharedEncoder(
            state_dim, hidden_dims, activation, use_layer_norm
        ).to(self.device)
        self.param_net = ContinuousParameterNetwork(self.encoder.output_dim, n_rrh).to(
            self.device
        )

        # Twin branch critics (operate on the shared encoder's output feature)
        self.twin_critic = TwinBranchCritic(self.encoder.output_dim, n_rrh).to(
            self.device
        )

        # Targets
        self.encoder_target = copy.deepcopy(self.encoder).to(self.device)
        self.param_net_target = copy.deepcopy(self.param_net).to(self.device)
        self.twin_critic_target = copy.deepcopy(self.twin_critic).to(self.device)

        # Optimizers. The shared encoder is trained by BOTH signals: every
        # critic_opt.step() (via critic_loss) and every param_opt.step()
        # (via the delayed policy_delay-gated param_loss) update it, since
        # both losses' computation graphs pass through it.
        self.critic_opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.twin_critic.parameters()),
            lr=lr_branch,
        )
        self.param_opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.param_net.parameters()),
            lr=lr_param,
        )

        self.memory = ParameterizedReplayBuffer(buffer_size)
        self.update_counter = 0

    def load_checkpoint(self, path: str) -> None:
        """Load encoder/param_net/twin_critic weights from a checkpoint saved
        by training/train_hybrid.py (final_model.pt or checkpoint_ep*.pt),
        syncing the target networks to match. For evaluation-only reuse of
        an already-trained agent (Concept Note v4.0 Section 14) — e.g.
        evaluation/csi_robustness.py and evaluation/generalization.py — not
        for resuming training (the replay buffer and optimizer states are
        not part of this checkpoint format and are left at their fresh,
        empty/initialized values)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.param_net.load_state_dict(checkpoint["param_net"])
        self.twin_critic.load_state_dict(checkpoint["twin_critic"])
        self.encoder_target.load_state_dict(checkpoint["encoder"])
        self.param_net_target.load_state_dict(checkpoint["param_net"])
        self.twin_critic_target.load_state_dict(checkpoint["twin_critic"])

    def _multi_pass_q(
        self, critic: TwinBranchCritic, feat: torch.Tensor, cont_params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate every branch's Q-value via its own MP-DQN masked pass.

        `feat` is the shared encoder's output, computed once by the caller
        (not per branch) — the encoder is state-only and doesn't depend on
        branch_mask_idx, so hoisting it out of the per-branch loop is both a
        genuine efficiency win and what "shared" means here (Section 10.3):
        every branch's masked pass reuses the same state representation.

        One forward pass per branch (R total, per critic evaluation): each
        pass masks the critic to branch r's own (p_r, beta_r) only, and only
        that pass's branch-r slice of the output is kept. This is what
        removes the cross-talk P-DQN's single shared (or, as before this
        fix, mean-pooled) parameter feature otherwise introduces between
        unrelated RRHs (Bester et al., 2019) — Section 10.3/10.5 of the
        concept note. Compute cost scales with R, as documented in Section
        10.5/14 (the top flagged risk at large R).
        """
        batch_size = feat.shape[0]
        q_a = torch.zeros(batch_size, self.n_rrh, 2, device=self.device)
        q_b = torch.zeros(batch_size, self.n_rrh, 2, device=self.device)
        for r in range(self.n_rrh):
            q_a_r, q_b_r = critic(feat, cont_params, branch_mask_idx=r)
            q_a[:, r, :] = q_a_r[:, r, :]
            q_b[:, r, :] = q_b_r[:, r, :]
        return q_a, q_b

    def select_action(
        self, obs: np.ndarray, evaluate: bool = False
    ) -> Dict[str, np.ndarray]:
        """Select joint discrete activation (rrh_on), power allocation, and bandwidth share."""
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.encoder(state_t)
            p_ratio, bw_share = self.param_net(feat)
            cont_params = torch.stack([p_ratio, bw_share], dim=-1)  # (1, n_rrh, 2)

            if not evaluate and random.random() < self.epsilon:
                rrh_on = np.random.randint(0, 2, size=self.n_rrh)
            else:
                q_vals_a, _ = self._multi_pass_q(self.twin_critic, feat, cont_params)
                rrh_on = q_vals_a[0].argmax(dim=-1).cpu().numpy()

            # Continuous exploration noise (decayed; applies to both continuous
            # parameters, not just power)
            if not evaluate:
                p_noise = torch.randn_like(p_ratio) * self.continuous_noise_std
                p_ratio = torch.clamp(p_ratio + p_noise, 0.0, 1.0)
                bw_noise = torch.randn_like(bw_share) * self.continuous_noise_std
                bw_share = torch.clamp(bw_share + bw_noise, 0.0, 1.0)

            p_np = p_ratio[0].cpu().numpy() * self.p_max_w
            bw_np = bw_share[0].cpu().numpy()

            cont_np = np.stack([p_ratio[0].cpu().numpy(), bw_np], axis=-1)  # (n_rrh, 2)

        return {
            "rrh_on": rrh_on,
            "power": p_np,
            "bandwidth": bw_np,
            "continuous": cont_np,
        }

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Execute one Multi-Pass Branching MP-DQN + TD3 step."""
        if len(self.memory) < max(batch_size, self.min_buffer_size):
            return {
                "critic_loss": 0.0,
                "param_loss": 0.0,
                "epsilon": self.epsilon,
                "continuous_noise_std": self.continuous_noise_std,
            }

        self.update_counter += 1

        (
            states,
            disc_actions,
            cont_params,
            rewards,
            next_states,
            dones,
        ) = self.memory.sample(batch_size)

        states = states.to(self.device)
        disc_actions = disc_actions.to(self.device)
        cont_params = cont_params.to(self.device)
        # reward_scale (Section 10.3): scales the Bellman target, not the
        # stored transition, so replay-buffer contents stay in raw reward
        # units regardless of this hyperparameter.
        rewards = rewards.to(self.device) * self.reward_scale
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # --- 1. Target Q-Value Calculation ---
        with torch.no_grad():
            next_feat = self.encoder_target(next_states)
            next_p_ratio, next_bw_share = self.param_net_target(next_feat)
            next_cont_params = torch.stack([next_p_ratio, next_bw_share], dim=-1)

            # Target noise smoothing
            noise = (torch.randn_like(next_cont_params) * self.target_noise_std).clamp(
                -0.1, 0.1
            )
            next_cont_params = (next_cont_params + noise).clamp(0.0, 1.0)

            # Evaluate next target discrete actions (multi-pass, per branch)
            next_q1, next_q2 = self._multi_pass_q(
                self.twin_critic_target, next_feat, next_cont_params
            )
            next_disc_actions = next_q1.argmax(dim=-1)  # Double DQN target selection

            next_q_min = torch.min(
                next_q1.gather(-1, next_disc_actions.unsqueeze(-1)).squeeze(-1),
                next_q2.gather(-1, next_disc_actions.unsqueeze(-1)).squeeze(-1),
            )  # (batch, n_rrh)

            # Per-branch target y_{i,r} = r_i + gamma*(1-done)*min(Q_r^A',Q_r^B')
            # (docs/thesis_guide.md Section 3.7): each branch bootstraps off its
            # own next-state Q-value rather than one target shared/averaged
            # across all R branches, preserving the branching architecture's
            # per-RRH independence in the training signal, not just the
            # forward pass. rewards/dones are (batch, 1) and broadcast against
            # next_q_min's (batch, n_rrh) to give one target per branch.
            y_target = (
                rewards + self.gamma * (1.0 - dones) * next_q_min
            )  # (batch, n_rrh)

        # --- 2. Critic Loss Computation (multi-pass, per branch) ---
        feat = self.encoder(states)
        q1_curr, q2_curr = self._multi_pass_q(self.twin_critic, feat, cont_params)
        q1_sel = q1_curr.gather(-1, disc_actions.unsqueeze(-1)).squeeze(-1)
        q2_sel = q2_curr.gather(-1, disc_actions.unsqueeze(-1)).squeeze(-1)

        critic_loss = F.mse_loss(q1_sel, y_target) + F.mse_loss(q2_sel, y_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.twin_critic.parameters()),
            max_norm=self.gradient_clip_norm,
        )
        self.critic_opt.step()

        # --- 3. Delayed Policy & Parameter Network Update ---
        param_loss_val = 0.0
        if self.update_counter % self.policy_delay == 0:
            feat = self.encoder(states)
            p_ratio, bw_share = self.param_net(feat)
            pred_params = torch.stack([p_ratio, bw_share], dim=-1)

            q1_pred, _ = self._multi_pass_q(self.twin_critic, feat, pred_params)
            param_loss = -q1_pred.mean()

            self.param_opt.zero_grad()
            param_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.param_net.parameters()),
                max_norm=self.gradient_clip_norm,
            )
            self.param_opt.step()
            param_loss_val = float(param_loss.item())

            # Soft updates
            self._soft_update(self.encoder_target, self.encoder)
            self._soft_update(self.param_net_target, self.param_net)
            self._soft_update(self.twin_critic_target, self.twin_critic)

        return {
            "critic_loss": float(critic_loss.item()),
            "param_loss": param_loss_val,
            "epsilon": float(self.epsilon),
            "continuous_noise_std": float(self.continuous_noise_std),
        }

    def decay_exploration(self):
        """Decay epsilon and continuous-action exploration noise once per
        episode (config/default.yaml's epsilon_decay is documented as a
        per-episode rate; calling this from update(), which runs once per
        environment step, decayed ~100x faster than intended at
        max_steps_per_episode=100)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.continuous_noise_std = max(
            self.continuous_noise_std_end,
            self.continuous_noise_std * self.epsilon_decay,
        )

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
