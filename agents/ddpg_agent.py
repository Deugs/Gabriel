"""Pure DDPG Agent with Continuous Relaxation, for C-RAN RRH Activation + Power Control.

Conforms to MPhil Thesis Concept Note v3.0/v4.0 Section 12.1 (RQ3 baseline):
the original v1.0 concept note's design (Lillicrap et al., 2016), kept
specifically to measure the hybrid agent's benefit over forcing a single
continuous-action policy to also decide RRH on/off. RRH activation is
represented as a continuous "soft" gate a_r in [0, 1] output by the actor
alongside the continuous power/bandwidth parameters, and is only
thresholded at a_r > 0.5 when the joint action is applied to the
environment (Section 3 of manuscript/concept_document.md: "forcing DDPG's
continuous output through a threshold destroys the gradient signal needed
to learn good switching behavior" — this baseline exists to demonstrate
that degradation empirically, not to work around it).
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


class DDPGActor(nn.Module):
    """Deterministic actor producing per-RRH (activation, power, bandwidth)."""

    def __init__(self, state_dim: int, n_rrh: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers: List[nn.Module] = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.LayerNorm(dim)])
            prev_dim = dim
        self.backbone = nn.Sequential(*layers)

        self.n_rrh = n_rrh
        self.activation_head = nn.Linear(prev_dim, n_rrh)
        self.power_head = nn.Linear(prev_dim, n_rrh)
        self.bandwidth_head = nn.Linear(prev_dim, n_rrh)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(state)
        activation_cont = torch.sigmoid(self.activation_head(feat))  # (batch, n_rrh)
        power_ratio = torch.sigmoid(self.power_head(feat))  # (batch, n_rrh)
        bandwidth_share = F.softmax(self.bandwidth_head(feat), dim=-1)  # (batch, n_rrh)
        return activation_cont, power_ratio, bandwidth_share


class DDPGCritic(nn.Module):
    """Q(s, a) over the full continuous-relaxed action (activation, power, bandwidth)."""

    def __init__(self, state_dim: int, n_rrh: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        action_dim = 3 * n_rrh
        layers: List[nn.Module] = []
        prev_dim = state_dim + action_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.LayerNorm(dim)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, action], dim=-1))


class ContinuousActionReplayBuffer:
    """Experience Replay Buffer for fully-continuous (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int = 100000):
        self.buffer: deque[
            Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
        ] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DDPGAgent:
    """Pure DDPG Agent for RRH Activation (continuous-relaxed) + Power Control."""

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

        cfg = config if config is not None else {}
        algo_cfg = (
            getattr(cfg, "algorithm", cfg)
            if hasattr(cfg, "algorithm") or isinstance(cfg, dict)
            else cfg
        )

        def get_val(key, default):
            if isinstance(algo_cfg, dict):
                return algo_cfg.get(key, default)
            return getattr(algo_cfg, key, default)

        self.gamma = float(get_val("gamma", 0.99))
        self.tau = float(get_val("tau", 0.005))
        self.exploration_sigma = float(get_val("target_noise_std", 0.1))

        lr_actor = float(get_val("lr_actor", 1e-4))
        lr_critic = float(get_val("lr_critic", 3.0e-4))
        buffer_size = int(get_val("buffer_size", 100000))

        self.actor = DDPGActor(state_dim, n_rrh).to(self.device)
        self.critic = DDPGCritic(state_dim, n_rrh).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.memory = ContinuousActionReplayBuffer(buffer_size)

    @staticmethod
    def _to_action_vec(
        activation: torch.Tensor, power: torch.Tensor, bandwidth: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([activation, power, bandwidth], dim=-1)

    def select_action(
        self, obs: np.ndarray, evaluate: bool = False
    ) -> Dict[str, np.ndarray]:
        """Select the continuous-relaxed action; RRH on/off is a > 0.5 threshold."""
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            activation, power_ratio, bandwidth = self.actor(state_t)

            if not evaluate:
                noise = torch.randn_like(activation) * self.exploration_sigma
                activation = torch.clamp(activation + noise, 0.0, 1.0)
                power_ratio = torch.clamp(
                    power_ratio + torch.randn_like(power_ratio) * self.exploration_sigma,
                    0.0,
                    1.0,
                )

            action_vec = self._to_action_vec(activation, power_ratio, bandwidth)[0]
            rrh_on = (activation[0] > 0.5).long().cpu().numpy()
            power_np = power_ratio[0].cpu().numpy() * self.p_max_w
            bandwidth_np = bandwidth[0].cpu().numpy()

        return {
            "rrh_on": rrh_on,
            "power": power_np,
            "bandwidth": bandwidth_np,
            "continuous_action": action_vec.cpu().numpy(),
        }

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        if len(self.memory) < batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_activation, next_power, next_bandwidth = self.actor_target(next_states)
            next_action = self._to_action_vec(next_activation, next_power, next_bandwidth)
            next_q = self.critic_target(next_states, next_action)
            y_target = rewards + self.gamma * (1.0 - dones) * next_q

        q_curr = self.critic(states, actions)
        critic_loss = F.mse_loss(q_curr, y_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        pred_activation, pred_power, pred_bandwidth = self.actor(states)
        pred_action = self._to_action_vec(pred_activation, pred_power, pred_bandwidth)
        actor_loss = -self.critic(states, pred_action).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
