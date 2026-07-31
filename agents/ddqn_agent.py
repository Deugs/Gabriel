"""Double Deep Q-Network (DDQN) Agent Baseline for C-RAN Simulation.

Implements discrete Double Q-Learning (Iqbal et al. 2021) for RRH activation decisions.
"""

from collections import deque
import copy
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Deep Q-Network with factorized heads per RRH."""

    def __init__(
        self, state_dim: int, n_rrh: int, hidden_dims: Optional[List[int]] = None
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.n_rrh = n_rrh

        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.LayerNorm(dim)])
            prev_dim = dim

        self.backbone = nn.Sequential(*layers)
        # Factorized output: 2 discrete actions (OFF=0, ON=1) per RRH
        self.heads = nn.ModuleList([nn.Linear(prev_dim, 2) for _ in range(n_rrh)])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.backbone(state)
        # Output shape: (batch_size, n_rrh, 2)
        q_values = torch.stack([head(features) for head in self.heads], dim=1)
        return q_values


class ReplayBuffer:
    """Experience Replay Buffer for Q-Learning."""

    def __init__(self, capacity: int = 100000):
        self.buffer: deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = (
            deque(maxlen=capacity)
        )

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
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DDQNAgent:
    """Double Deep Q-Network (DDQN) Agent for RRH Activation Control."""

    def __init__(
        self,
        state_dim: int,
        n_rrh: int,
        p_max_w: float = 1.0,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.n_rrh = n_rrh
        self.p_max_w = p_max_w
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_net = QNetwork(state_dim, n_rrh).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(
        self, obs: np.ndarray, evaluate: bool = False
    ) -> Dict[str, np.ndarray]:
        """Select action using epsilon-greedy policy."""
        if not evaluate and random.random() < self.epsilon:
            rrh_on = np.random.randint(0, 2, size=self.n_rrh)
        else:
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_vals = self.q_net(state_t)[0]  # (n_rrh, 2)
                rrh_on = q_vals.argmax(dim=-1).cpu().numpy()

        # Power allocation: uniform transmit power for active RRHs
        power = np.zeros(self.n_rrh, dtype=np.float32)
        power[rrh_on == 1] = self.p_max_w

        return {"rrh_on": rrh_on, "power": power}

    def update(self) -> Dict[str, float]:
        """Update Q-network using Double DQN loss."""
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0, "epsilon": self.epsilon}

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)  # (batch, n_rrh)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values for taken actions
        q_values = self.q_net(states)  # (batch, n_rrh, 2)
        q_selected = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(
            -1
        )  # (batch, n_rrh)
        q_current = q_selected.mean(dim=-1, keepdim=True)  # Average Q across RRHs

        # Double DQN Target: select best action using online net, evaluate using target net
        with torch.no_grad():
            next_q_online = self.q_net(next_states)  # (batch, n_rrh, 2)
            next_actions = next_q_online.argmax(dim=-1)  # (batch, n_rrh)

            next_q_target = self.target_q_net(next_states)
            next_q_eval = next_q_target.gather(-1, next_actions.unsqueeze(-1)).squeeze(
                -1
            )
            q_next = next_q_eval.mean(dim=-1, keepdim=True)

            target = rewards + self.gamma * (1.0 - dones) * q_next

        loss = nn.MSELoss()(q_current, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(
            self.target_q_net.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {"loss": float(loss.item()), "epsilon": float(self.epsilon)}
