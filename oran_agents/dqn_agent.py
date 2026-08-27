"""Plain DQN Baseline for O-RAN Energy Optimization.

Per ORAN_BMPP_DQN_Concept_Note_v1.md Section 2.2: "DQN handles discrete
actions only." This baseline learns the two discrete branch-groups
(RU activation, functional split) via per-branch heads and a *plain*
(not Double) DQN target -- select and evaluate off the target network
only, no online-net-argmax step -- and leaves the continuous decisions
(power, PRB) fixed/heuristic, since DQN has no mechanism to represent
continuous actions at all.

Structurally templated on agents/ddqn_agent.py's per-branch design, but
not imported from it (zero dependency on agents/), and with the
Double-DQN target replaced by a plain-DQN one.
"""

from collections import deque
import random
from typing import Any, Dict, List, Optional, Union

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
            f"Unknown activation '{name}'; supported: {sorted(_ACTIVATIONS)}"
        )


class _BranchQNetwork(nn.Module):
    """Per-RU factorized heads for both discrete decision types (no continuous fusion)."""

    def __init__(
        self,
        state_dim: int,
        n_ru: int,
        n_splits: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        activation_cls = _resolve_activation(activation)

        layers: List[nn.Module] = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation_cls())
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            prev_dim = dim
        self.backbone = nn.Sequential(*layers)

        self.activation_heads = nn.ModuleList(
            [nn.Linear(prev_dim, 2) for _ in range(n_ru)]
        )
        self.split_heads = nn.ModuleList(
            [nn.Linear(prev_dim, n_splits) for _ in range(n_ru)]
        )

    def forward(self, state: torch.Tensor):
        feat = self.backbone(state)
        activation_q = torch.stack(
            [head(feat) for head in self.activation_heads], dim=1
        )  # (batch, n_ru, 2)
        split_q = torch.stack(
            [head(feat) for head in self.split_heads], dim=1
        )  # (batch, n_ru, n_splits)
        return activation_q, split_q


class ReplayBuffer:
    """Experience Replay Buffer for (s, ru_on, split, r, s', done)."""

    def __init__(self, capacity: int = 100000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, ru_on, split, reward, next_state, done):
        self.buffer.append((state, ru_on, split, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, ru_on, split, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(ru_on)),
            torch.LongTensor(np.array(split)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class ORANDQNAgent:
    """Plain DQN baseline (discrete-only) for the O-RAN track."""

    def __init__(
        self,
        state_dim: int,
        n_ru: int,
        n_splits: int = 3,
        p_max_w: float = 1.0,
        config: Optional[Union[dict, Any]] = None,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.n_ru = n_ru
        self.n_splits = n_splits
        self.p_max_w = p_max_w

        cfg = config if config is not None else {}
        algo_cfg = cfg.get("algorithm", {}) if isinstance(cfg, dict) else {}
        hardware_cfg = cfg.get("hardware", {}) if isinstance(cfg, dict) else {}

        def get_val(key: str, default: Any) -> Any:
            return algo_cfg.get(key, default) if isinstance(algo_cfg, dict) else default

        # `hardware.device` (config/oran_default.yaml) only supplies a
        # default -- an explicit `device=` argument from the caller always
        # wins (mirrors agents/branching_mp_dqn.py's convention).
        if device is None:
            device = str(
                hardware_cfg.get("device", "cpu")
                if isinstance(hardware_cfg, dict)
                else "cpu"
            )
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.gamma = float(get_val("gamma", 0.99))
        self.tau = float(get_val("tau", 0.005))
        self.epsilon = float(get_val("epsilon_start", 1.0))
        self.epsilon_end = float(get_val("epsilon_end", 0.01))
        self.epsilon_decay = float(get_val("epsilon_decay", 0.995))
        self.gradient_clip_norm = float(get_val("gradient_clip_norm", 1.0))

        hidden_dims = get_val("hidden_dims", [128, 64])
        activation = get_val("activation", "relu")
        use_layer_norm = get_val("use_layer_norm", True)
        lr = float(get_val("lr_discrete", 1.0e-4))
        buffer_capacity = int(get_val("lower_buffer_size", 100000))

        self.q_net = _BranchQNetwork(
            state_dim, n_ru, n_splits, hidden_dims, activation, use_layer_norm
        ).to(self.device)
        self.target_q_net = _BranchQNetwork(
            state_dim, n_ru, n_splits, hidden_dims, activation, use_layer_norm
        ).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(
        self, obs: np.ndarray, evaluate: bool = False
    ) -> Dict[str, np.ndarray]:
        """Epsilon-greedy per-branch discrete selection; fixed heuristic continuous action."""
        if not evaluate and random.random() < self.epsilon:
            ru_on = np.random.randint(0, 2, size=self.n_ru)
            split = np.random.randint(0, self.n_splits, size=self.n_ru)
        else:
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                activation_q, split_q = self.q_net(state_t)
                ru_on = activation_q[0].argmax(dim=-1).cpu().numpy()
                split = split_q[0].argmax(dim=-1).cpu().numpy()

        # DQN has no continuous-control mechanism (Concept Note Section 2.2):
        # active RUs simply transmit at full power with an equal PRB split.
        power = np.where(ru_on == 1, self.p_max_w, 0.0).astype(np.float32)
        n_active = max(1, int(np.sum(ru_on)))
        prb = (ru_on.astype(np.float32) / n_active).astype(np.float32)

        return {"ru_on": ru_on, "split": split, "power": power, "prb": prb}

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """Plain-DQN update: target selects AND evaluates off the target
        network only (no online-net argmax step) -- per-branch targets for
        both discrete decision types, not one shared/averaged target
        (the fixed pattern from agents/branching_mp_dqn.py/ddqn_agent.py)."""
        if len(self.memory) < batch_size:
            return {"loss": 0.0, "epsilon": self.epsilon}

        states, ru_on, split, rewards, next_states, dones = self.memory.sample(
            batch_size
        )
        states = states.to(self.device)
        ru_on = ru_on.to(self.device)
        split = split.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        activation_q, split_q = self.q_net(states)
        act_sel = activation_q.gather(-1, ru_on.unsqueeze(-1)).squeeze(-1)
        split_sel = split_q.gather(-1, split.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_activation_q, next_split_q = self.target_q_net(next_states)
            next_act_max = next_activation_q.max(dim=-1).values
            next_split_max = next_split_q.max(dim=-1).values
            y_activation = rewards + self.gamma * (1.0 - dones) * next_act_max
            y_split = rewards + self.gamma * (1.0 - dones) * next_split_max

        loss = F.mse_loss(act_sel, y_activation) + F.mse_loss(split_sel, y_split)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.q_net.parameters(), max_norm=self.gradient_clip_norm
        )
        self.optimizer.step()

        for t_param, s_param in zip(
            self.target_q_net.parameters(), self.q_net.parameters()
        ):
            t_param.data.copy_(
                self.tau * s_param.data + (1.0 - self.tau) * t_param.data
            )

        return {"loss": float(loss.item()), "epsilon": self.epsilon}

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
