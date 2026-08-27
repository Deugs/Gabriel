"""Pure DDPG Baseline for O-RAN Energy Optimization.

Per ORAN_BMPP_DQN_Concept_Note_v1.md Section 2.2: "DDPG handles continuous
actions only." This baseline fixes RU activation (all RUs on) and
functional split (fixed at the middle centralization level, c=1 / Option
6) -- DDPG has no mechanism to represent the discrete decisions at all --
and learns only the continuous (power, PRB) allocation via a standard
DDPG actor-critic.

Structurally templated on agents/ddpg_agent.py's actor/critic head
pattern, but with 2 continuous heads instead of 3 (no activation head),
and zero import from agents/.
"""

from collections import deque
import copy
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


class _Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_ru: int,
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

        self.power_head = nn.Linear(prev_dim, n_ru)
        self.prb_head = nn.Linear(prev_dim, n_ru)

    def forward(self, state: torch.Tensor):
        feat = self.backbone(state)
        power_ratio = torch.sigmoid(self.power_head(feat))
        prb_share = F.softmax(self.prb_head(feat), dim=-1)
        return power_ratio, prb_share


class _Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_ru: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        activation_cls = _resolve_activation(activation)

        layers: List[nn.Module] = []
        prev_dim = state_dim + 2 * n_ru
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation_cls())
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            prev_dim = dim
        self.backbone = nn.Sequential(*layers)
        self.q_head = nn.Linear(prev_dim, 1)

    def forward(
        self, state: torch.Tensor, continuous_action: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([state, continuous_action], dim=-1)
        return self.q_head(self.backbone(x))


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, continuous_action, reward, next_state, done):
        self.buffer.append((state, continuous_action, reward, next_state, done))

    def sample(self, batch_size: int):
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


class ORANDDPGAgent:
    """Pure DDPG baseline (continuous-only) for the O-RAN track.

    RU activation and functional split are fixed, not learned:
    - ru_on: all RUs active.
    - split: fixed at c=1 (Option 6), the middle centralization level.
    """

    FIXED_SPLIT_LEVEL = 1

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
        self.exploration_sigma = float(get_val("continuous_noise_std", 0.1))
        self.gradient_clip_norm = float(get_val("gradient_clip_norm", 1.0))

        hidden_dims = get_val("hidden_dims", [128, 64])
        activation = get_val("activation", "relu")
        use_layer_norm = get_val("use_layer_norm", True)
        lr_actor = float(get_val("lr_actor", 3.0e-4))
        lr_critic = float(get_val("lr_critic", 3.0e-4))
        buffer_capacity = int(get_val("lower_buffer_size", 100000))

        self.actor = _Actor(
            state_dim, n_ru, hidden_dims, activation, use_layer_norm
        ).to(self.device)
        self.critic = _Critic(
            state_dim, n_ru, hidden_dims, activation, use_layer_norm
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(
        self, obs: np.ndarray, evaluate: bool = False
    ) -> Dict[str, np.ndarray]:
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            power_ratio, prb_share = self.actor(state_t)
            if not evaluate:
                noise = torch.randn_like(power_ratio) * self.exploration_sigma
                power_ratio = torch.clamp(power_ratio + noise, 0.0, 1.0)

        power = (power_ratio[0].cpu().numpy() * self.p_max_w).astype(np.float32)
        prb = prb_share[0].cpu().numpy().astype(np.float32)
        ru_on = np.ones(self.n_ru, dtype=np.int64)
        split = np.full(self.n_ru, self.FIXED_SPLIT_LEVEL, dtype=np.int64)

        return {"ru_on": ru_on, "split": split, "power": power, "prb": prb}

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        if len(self.memory) < batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        states, cont_actions, rewards, next_states, dones = self.memory.sample(
            batch_size
        )
        states = states.to(self.device)
        cont_actions = cont_actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_power, next_prb = self.actor_target(next_states)
            next_cont = torch.cat([next_power, next_prb], dim=-1)
            next_q = self.critic_target(next_states, next_cont)
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        current_q = self.critic(states, cont_actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(), max_norm=self.gradient_clip_norm
        )
        self.critic_opt.step()

        power, prb = self.actor(states)
        actor_cont = torch.cat([power, prb], dim=-1)
        actor_loss = -self.critic(states, actor_cont).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=self.gradient_clip_norm
        )
        self.actor_opt.step()

        for t_param, s_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(
                self.tau * s_param.data + (1.0 - self.tau) * t_param.data
            )
        for t_param, s_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(
                self.tau * s_param.data + (1.0 - self.tau) * t_param.data
            )

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }
