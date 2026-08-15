"""Hybrid SAC-DDQN Agent for 5G C-RAN Energy Optimization.

Combines discrete action selection (DDQN) for RRH activation with continuous policy
optimization (SAC) for transmit power allocation via a shared twin critic.
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


class DiscreteActor(nn.Module):
    """Factorized Discrete Actor (DDQN-style Q-Network) per RRH.

    Outputs Q-values Q(s, v_r=0) and Q(s, v_r=1) for each RRH independently.
    """

    def __init__(
        self, state_dim: int, n_rrh: int, hidden_dims: Optional[List[int]] = None
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.n_rrh = n_rrh

        layers: List[nn.Module] = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.LayerNorm(dim)])
            prev_dim = dim

        self.backbone = nn.Sequential(*layers)
        # Factorized linear heads for binary decision (OFF=0, ON=1) per RRH
        self.heads = nn.ModuleList([nn.Linear(prev_dim, 2) for _ in range(n_rrh)])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.backbone(state)
        # Shape: (batch_size, n_rrh, 2)
        q_values = torch.stack([head(features) for head in self.heads], dim=1)
        return q_values

    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        if random.random() < epsilon:
            return torch.randint(0, 2, (self.n_rrh,), device=state.device)

        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))[0]  # (n_rrh, 2)
            actions = q_values.argmax(dim=-1)  # (n_rrh,)
        return actions


class ContinuousActor(nn.Module):
    """Continuous Actor (SAC Squashed Gaussian Policy) for transmit power allocation.

    Outputs normalized power ratios p_r in [0, 1] for each RRH using tanh squashing.
    """

    def __init__(
        self,
        state_dim: int,
        n_rrh: int,
        hidden_dims: Optional[List[int]] = None,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.n_rrh = n_rrh
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers: List[nn.Module] = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.LayerNorm(dim)])
            prev_dim = dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, n_rrh)
        self.log_std_head = nn.Linear(prev_dim, n_rrh)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        mean = self.mean_head(features)
        log_std = torch.clamp(
            self.log_std_head(features), self.log_std_min, self.log_std_max
        )
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick (mean + std * noise)
        y_t = torch.tanh(x_t)
        action = (y_t + 1.0) / 2.0  # Scale tanh output from [-1, 1] to [0, 1]

        # Log probability density with Jacobian correction for tanh squashing and [0,1] scaling
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0 - y_t.pow(2) + 1e-6) + np.log(0.5)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        mean, _ = self.forward(state)
        if deterministic:
            return (torch.tanh(mean) + 1.0) / 2.0
        action, _ = self.sample(state)
        return action


class SingleCritic(nn.Module):
    """Single Q-Network fusing state, discrete action embedding, and continuous action."""

    def __init__(
        self, state_dim: int, n_rrh: int, hidden_dims: Optional[List[int]] = None
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.n_rrh = n_rrh

        # State encoder
        state_layers: List[nn.Module] = []
        prev_dim = state_dim
        for dim in hidden_dims:
            state_layers.extend(
                [nn.Linear(prev_dim, dim), nn.ReLU(), nn.LayerNorm(dim)]
            )
            prev_dim = dim
        self.state_encoder = nn.Sequential(*state_layers)

        # Discrete action embedding: 2 states (0=OFF, 1=ON) -> 16-dim vector per RRH
        self.discrete_embed = nn.Embedding(2, 16)

        # Continuous action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(n_rrh, 128), nn.ReLU(), nn.LayerNorm(128)
        )

        # Fusion network
        fusion_dim = prev_dim + n_rrh * 16 + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        discrete_action: torch.Tensor,
        continuous_action: torch.Tensor,
    ) -> torch.Tensor:
        # state: (batch, state_dim)
        # discrete_action: (batch, n_rrh) in {0, 1}
        # continuous_action: (batch, n_rrh) in [0, 1]
        state_feat = self.state_encoder(state)

        disc_embed = self.discrete_embed(discrete_action.long())  # (batch, n_rrh, 16)
        disc_feat = disc_embed.view(disc_embed.size(0), -1)  # (batch, n_rrh * 16)

        cont_feat = self.action_encoder(continuous_action)

        fusion_input = torch.cat([state_feat, disc_feat, cont_feat], dim=-1)
        q_value = self.fusion(fusion_input)
        return q_value


class HybridCritic(nn.Module):
    """Twin Q-Network evaluating joint discrete and continuous actions Q(s, v, p)."""

    def __init__(
        self, state_dim: int, n_rrh: int, hidden_dims: Optional[List[int]] = None
    ):
        super().__init__()
        self.q1 = SingleCritic(state_dim, n_rrh, hidden_dims)
        self.q2 = SingleCritic(state_dim, n_rrh, hidden_dims)

    def forward(
        self,
        state: torch.Tensor,
        discrete_action: torch.Tensor,
        continuous_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1_val = self.q1(state, discrete_action, continuous_action)
        q2_val = self.q2(state, discrete_action, continuous_action)
        return q1_val, q2_val


class HybridReplayBuffer:
    """Experience Replay Buffer for Hybrid Transitions (s, v, p, r, s', done)."""

    def __init__(self, capacity: int = 100000):
        self.buffer: deque[
            Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, bool]
        ] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        discrete_action: np.ndarray,
        continuous_action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append(
            (state, discrete_action, continuous_action, reward, next_state, done)
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
        states, disc_actions, cont_actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(disc_actions)),
            torch.FloatTensor(np.array(cont_actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class HybridSACDDQN:
    """Proposed Hybrid SAC-DDQN Agent for Joint Discrete-Continuous C-RAN Control."""

    def __init__(
        self,
        state_dim: int,
        n_rrh: int,
        p_max_w: float = 1.0,
        config: Optional[Union[dict, Any]] = None,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.n_rrh = n_rrh
        self.p_max_w = p_max_w
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Extract hyperparameters with defaults
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
        self.alpha = float(get_val("alpha", 0.2))
        self.auto_tune_alpha = bool(get_val("auto_tune_alpha", True))

        self.epsilon = float(get_val("epsilon_start", 1.0))
        self.epsilon_end = float(get_val("epsilon_end", 0.01))
        self.epsilon_decay = float(get_val("epsilon_decay", 0.995))

        lr_disc = float(get_val("lr_discrete", 1e-4))
        lr_actor = float(get_val("lr_actor", 3e-4))
        lr_critic = float(get_val("lr_critic", 3e-4))
        lr_alpha = float(get_val("lr_alpha", 1e-4))
        buffer_size = int(get_val("buffer_size", 100000))

        # Core Neural Networks
        self.discrete_actor = DiscreteActor(state_dim, n_rrh).to(self.device)
        self.continuous_actor = ContinuousActor(state_dim, n_rrh).to(self.device)
        self.critic = HybridCritic(state_dim, n_rrh).to(self.device)

        # Target Networks
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.continuous_actor_target = copy.deepcopy(self.continuous_actor).to(
            self.device
        )
        self.discrete_actor_target = copy.deepcopy(self.discrete_actor).to(self.device)

        # Optimizers
        self.disc_opt = optim.Adam(self.discrete_actor.parameters(), lr=lr_disc)
        self.cont_opt = optim.Adam(self.continuous_actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Automatic Entropy Temperature Tuning
        if self.auto_tune_alpha:
            self.target_entropy = -float(n_rrh)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr_alpha)

        # Experience Replay Buffer
        self.memory = HybridReplayBuffer(buffer_size)

    def select_action(
        self, obs: np.ndarray, evaluate: bool = False
    ) -> Dict[str, np.ndarray]:
        """Select joint discrete (RRH activation) and continuous (power allocation) action."""
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # 1. Discrete action selection (DDQN factorized heads with epsilon-greedy)
        if not evaluate and random.random() < self.epsilon:
            rrh_on = np.random.randint(0, 2, size=self.n_rrh)
        else:
            with torch.no_grad():
                q_disc = self.discrete_actor(state_t)[0]  # (n_rrh, 2)
                rrh_on = q_disc.argmax(dim=-1).cpu().numpy()

        # 2. Continuous action selection (SAC Gaussian policy)
        with torch.no_grad():
            cont_ratio = (
                self.continuous_actor.get_action(state_t, deterministic=evaluate)[0]
                .cpu()
                .numpy()
            )

        power = cont_ratio * self.p_max_w
        power[rrh_on == 0] = 0.0

        return {"rrh_on": rrh_on, "power": power}

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Execute one joint SAC-DDQN optimization step on a mini-batch."""
        if len(self.memory) < batch_size:
            return {
                "critic_loss": 0.0,
                "disc_loss": 0.0,
                "actor_loss": 0.0,
                "alpha": self.alpha,
                "epsilon": self.epsilon,
            }

        (
            states,
            disc_actions,
            cont_actions,
            rewards,
            next_states,
            dones,
        ) = self.memory.sample(batch_size)

        states = states.to(self.device)
        disc_actions = disc_actions.to(self.device)
        cont_actions = cont_actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # --- 1. Critic Update ---
        with torch.no_grad():
            # Next discrete action using Double DQN selection
            # (online network selects, target network evaluates)
            next_q_disc_online = self.discrete_actor(next_states)
            next_disc_actions = next_q_disc_online.argmax(dim=-1)

            # Next continuous action (target continuous actor sample)
            next_cont_actions, next_log_prob = self.continuous_actor_target.sample(
                next_states
            )

            # Target Q-values
            q1_next, q2_next = self.critic_target(
                next_states, next_disc_actions, next_cont_actions
            )
            q_next_min = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            y = rewards + self.gamma * (1.0 - dones) * q_next_min

        # Current Q-values
        q1_curr, q2_curr = self.critic(states, disc_actions, cont_actions)
        critic_loss = F.mse_loss(q1_curr, y) + F.mse_loss(q2_curr, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # --- 2. Discrete Actor Update (DDQN loss against Bellman target y) ---
        q_disc_vals = self.discrete_actor(states)  # (batch, n_rrh, 2)
        disc_selected = q_disc_vals.gather(-1, disc_actions.unsqueeze(-1)).squeeze(
            -1
        )  # (batch, n_rrh)

        # Target Q-value for discrete action evaluation (expanded across RRHs)
        y_disc_target = y.expand(-1, self.n_rrh)
        disc_loss = F.mse_loss(disc_selected, y_disc_target.detach())

        self.disc_opt.zero_grad()
        disc_loss.backward()
        nn.utils.clip_grad_norm_(self.discrete_actor.parameters(), max_norm=1.0)
        self.disc_opt.step()

        # --- 3. Continuous Actor Update ---
        sampled_cont_actions, log_prob = self.continuous_actor.sample(states)
        with torch.no_grad():
            current_disc_actions = self.discrete_actor(states).argmax(dim=-1)

        # Freeze critic parameters to avoid computing critic gradients during actor step
        for param in self.critic.parameters():
            param.requires_grad = False

        q1_new, q2_new = self.critic(states, current_disc_actions, sampled_cont_actions)
        q_new_min = torch.min(q1_new, q2_new)

        actor_loss = -(q_new_min - self.alpha * log_prob).mean()

        self.cont_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.continuous_actor.parameters(), max_norm=1.0)
        self.cont_opt.step()

        # Unfreeze critic parameters
        for param in self.critic.parameters():
            param.requires_grad = True

        # --- 4. Entropy Temperature (Alpha) Update ---
        if self.auto_tune_alpha:
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = float(self.log_alpha.exp().item())

        # --- 5. Soft Update Target Networks ---
        self._soft_update(self.critic_target, self.critic)
        self._soft_update(self.continuous_actor_target, self.continuous_actor)
        self._soft_update(self.discrete_actor_target, self.discrete_actor)

        c_loss = float(critic_loss.item())
        d_loss = float(disc_loss.item())
        a_loss = float(actor_loss.item())

        del states, disc_actions, cont_actions, rewards, next_states, dones
        del q1_curr, q2_curr, y, sampled_cont_actions, log_prob, q1_new, q2_new

        return {
            "critic_loss": c_loss,
            "disc_loss": d_loss,
            "actor_loss": a_loss,
            "alpha": self.alpha,
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
