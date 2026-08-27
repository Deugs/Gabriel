"""Branching Multi-Pass Parameterized DQN (BMPP-DQN) for O-RAN Energy Optimization.

Conforms to ORAN_BMPP_DQN_Concept_Note_v1.md Section 5.2/10.4 and
docs/skills/skill_oran_bmpp_dqn.md:
- Branching decomposition: one independent decision branch per RU, per
  discrete decision type (activation, split).
- Multi-pass parameterized processing: each branch evaluates its own
  Q-value using only its own RU's continuous parameters (power, PRB),
  masking out other RUs' -- the MP-DQN mechanism, reimplemented locally
  (zero import from agents/mpdqn_agent.py or agents/branching_mp_dqn.py).
- Two independent encoders (upper/lower), not one shared encoder.
- Two-timescale cadence: discrete decisions held for
  `upper_level_period_steps` env steps; continuous outputs recomputed
  every step.
- Explicitly NO TD3/twin-critic machinery: a single critic, standard
  Double-DQN target computation, no target-policy-smoothing noise, no
  policy-delay gating.

This module is fully decoupled from agents/: no shared code, no shared
imports.
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


class _Encoder(nn.Module):
    """Feature encoder mapping state s(t) to a feature representation."""

    def __init__(
        self,
        state_dim: int,
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

        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class _BranchingHeads(nn.Module):
    """Factorized dueling per-RU heads for ONE discrete decision type.

    Generalizes agents/branching_mp_dqn.py's BranchingDiscreteHeads (fixed
    at n_actions=2) to an arbitrary action count, so the same class serves
    both the activation branch-group (n_actions=2) and the split
    branch-group (n_actions=n_splits).
    """

    def __init__(self, feature_dim: int, n_ru: int, n_actions: int):
        super().__init__()
        self.n_ru = n_ru
        self.n_actions = n_actions
        self.value_head = nn.Linear(feature_dim, 1)
        self.adv_heads = nn.ModuleList(
            [nn.Linear(feature_dim, n_actions) for _ in range(n_ru)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        v = self.value_head(features).unsqueeze(1)  # (batch, 1, 1)
        advs = torch.stack(
            [head(features) for head in self.adv_heads], dim=1
        )  # (batch, n_ru, n_actions)
        return v + (advs - advs.mean(dim=-1, keepdim=True))


class ContinuousParameterNetwork(nn.Module):
    """Deterministic continuous parameter network producing (power_ratio, prb_share) per RU."""

    def __init__(self, feature_dim: int, n_ru: int):
        super().__init__()
        self.n_ru = n_ru
        self.power_head = nn.Linear(feature_dim, n_ru)
        self.prb_head = nn.Linear(feature_dim, n_ru)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        power_ratio = torch.sigmoid(self.power_head(features))
        prb_share = F.softmax(self.prb_head(features), dim=-1)
        return power_ratio, prb_share


class BranchingCritic(nn.Module):
    """Single (no twin) multi-pass branching critic for both discrete decision types.

    Fuses the upper encoder's shared feature with the continuous
    parameters (power, prb), masked to one RU at a time (MP-DQN's
    multi-pass mechanism), and produces per-RU Q-values for both the
    activation branch-group and the split branch-group.
    """

    def __init__(self, feature_dim: int, n_ru: int, n_splits: int):
        super().__init__()
        self.n_ru = n_ru
        self.param_encoder = nn.Linear(2, 64)
        self.fusion = nn.Linear(feature_dim + 64, feature_dim)
        self.activation_heads = _BranchingHeads(feature_dim, n_ru, n_actions=2)
        self.split_heads = _BranchingHeads(feature_dim, n_ru, n_actions=n_splits)

    def forward(
        self,
        feat: torch.Tensor,
        continuous_params: torch.Tensor,
        branch_mask_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (batch, feature_dim); continuous_params: (batch, n_ru, 2)
        if branch_mask_idx is not None:
            param_feat = F.relu(
                self.param_encoder(continuous_params[:, branch_mask_idx, :])
            )
        else:
            param_feat = F.relu(self.param_encoder(continuous_params.mean(dim=1)))

        fused = F.relu(self.fusion(torch.cat([feat, param_feat], dim=-1)))
        activation_q = self.activation_heads(fused)  # (batch, n_ru, 2)
        split_q = self.split_heads(fused)  # (batch, n_ru, n_splits)
        return activation_q, split_q


class LowerReplayBuffer:
    """Fast-cadence buffer: one transition per environment step."""

    def __init__(self, capacity: int = 1000000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        continuous_params: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, continuous_params, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, cont_params, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(cont_params)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class UpperReplayBuffer:
    """Slow-cadence buffer: one transition per upper_level_period_steps env steps."""

    def __init__(self, capacity: int = 200000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        ru_on: np.ndarray,
        split: np.ndarray,
        aggregated_reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, ru_on, split, aggregated_reward, next_state, done))

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


class BMPPDQNAgent:
    """Branching Multi-Pass Parameterized DQN agent (two-timescale, no TD3)."""

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
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # config is the raw dict from yaml.safe_load() (this package never
        # wraps it in an attribute-accessor helper, unlike oran_env.ORANEnv
        # -- agent config access here mirrors agents/branching_mp_dqn.py's
        # own dict.get()-based convention).
        cfg = config if config is not None else {}
        algo_cfg = cfg.get("algorithm", {}) if isinstance(cfg, dict) else {}

        def get_val(key: str, default: Any) -> Any:
            return algo_cfg.get(key, default) if isinstance(algo_cfg, dict) else default

        hidden_dims = get_val("hidden_dims", [128, 64])
        activation = get_val("activation", "relu")
        use_layer_norm = get_val("use_layer_norm", True)

        self.gamma = float(get_val("gamma", 0.99))
        self.tau = float(get_val("tau", 0.005))
        self.epsilon = float(get_val("epsilon_start", 1.0))
        self.epsilon_end = float(get_val("epsilon_end", 0.01))
        self.epsilon_decay = float(get_val("epsilon_decay", 0.995))
        self.batch_size_default = int(get_val("batch_size", 128))
        self.min_buffer_size = int(get_val("min_buffer_size", 2000))
        self.gradient_clip_norm = float(get_val("gradient_clip_norm", 1.0))
        self.upper_level_period_steps = int(get_val("upper_level_period_steps", 10))

        lr_discrete = float(get_val("lr_discrete", 1.0e-4))
        lr_actor = float(get_val("lr_actor", 3.0e-4))
        upper_buffer_size = int(get_val("upper_buffer_size", 200000))
        lower_buffer_size = int(get_val("lower_buffer_size", 1000000))

        self.upper_encoder = _Encoder(
            state_dim, hidden_dims, activation, use_layer_norm
        ).to(self.device)
        self.lower_encoder = _Encoder(
            state_dim, hidden_dims, activation, use_layer_norm
        ).to(self.device)
        feat_dim = self.upper_encoder.output_dim

        self.critic = BranchingCritic(feat_dim, n_ru, n_splits).to(self.device)
        self.param_net = ContinuousParameterNetwork(feat_dim, n_ru).to(self.device)

        self.upper_encoder_target = copy.deepcopy(self.upper_encoder).to(self.device)
        self.lower_encoder_target = copy.deepcopy(self.lower_encoder).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.param_net_target = copy.deepcopy(self.param_net).to(self.device)

        self.critic_opt = optim.Adam(
            list(self.upper_encoder.parameters()) + list(self.critic.parameters()),
            lr=lr_discrete,
        )
        self.param_opt = optim.Adam(
            list(self.lower_encoder.parameters()) + list(self.param_net.parameters()),
            lr=lr_actor,
        )

        self.lower_memory = LowerReplayBuffer(lower_buffer_size)
        self.upper_memory = UpperReplayBuffer(upper_buffer_size)

        # Two-timescale cadence bookkeeping (agent-side; the env itself is
        # timescale-agnostic, docs/skills/skill_oran_env.md Rule 4).
        self._steps_since_decision = 0
        self._cached_ru_on = np.ones(n_ru, dtype=np.int64)
        self._cached_split = np.zeros(n_ru, dtype=np.int64)
        # Set by select_action() on every call so remember() knows, without
        # the training loop having to duplicate this bookkeeping itself,
        # whether the action it's about to log was a freshly-made discrete
        # decision or a replayed one.
        self.last_action_was_decision = True
        # Pending upper-level transition accumulator (state at decision time,
        # discrete action taken, summed reward over the window).
        self._pending_upper_state: Optional[np.ndarray] = None
        self._pending_upper_reward_sum = 0.0
        self._pending_ru_on = self._cached_ru_on.copy()
        self._pending_split = self._cached_split.copy()

    def _multi_pass_q(
        self, feat: torch.Tensor, cont_params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate every branch's Q-value via its own MP-DQN masked pass.

        One forward pass per RU (n_ru total): each pass masks the critic to
        that RU's own (power, prb) only, and only that pass's own-RU slice
        of the output is kept for both branch-groups (activation, split).
        """
        batch_size = feat.shape[0]
        activation_q = torch.zeros(batch_size, self.n_ru, 2, device=self.device)
        split_q = torch.zeros(batch_size, self.n_ru, self.n_splits, device=self.device)
        for r in range(self.n_ru):
            act_r, split_r = self.critic(feat, cont_params, branch_mask_idx=r)
            activation_q[:, r, :] = act_r[:, r, :]
            split_q[:, r, :] = split_r[:, r, :]
        return activation_q, split_q

    def select_action(
        self, obs: np.ndarray, evaluate: bool = False
    ) -> Dict[str, np.ndarray]:
        """Select an action, honoring the two-timescale decision cadence."""
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            lower_feat = self.lower_encoder(state_t)
            power_ratio, prb_share = self.param_net(lower_feat)

            if not evaluate:
                p_noise = torch.randn_like(power_ratio) * 0.1
                power_ratio = torch.clamp(power_ratio + p_noise, 0.0, 1.0)

            self.last_action_was_decision = self._steps_since_decision == 0
            if self.last_action_was_decision:
                if not evaluate and random.random() < self.epsilon:
                    self._cached_ru_on = np.random.randint(0, 2, size=self.n_ru)
                    self._cached_split = np.random.randint(
                        0, self.n_splits, size=self.n_ru
                    )
                else:
                    upper_feat = self.upper_encoder(state_t)
                    cont_params = torch.stack([power_ratio, prb_share], dim=-1)
                    activation_q, split_q = self._multi_pass_q(upper_feat, cont_params)
                    self._cached_ru_on = activation_q[0].argmax(dim=-1).cpu().numpy()
                    self._cached_split = split_q[0].argmax(dim=-1).cpu().numpy()

        self._steps_since_decision = (
            self._steps_since_decision + 1
        ) % self.upper_level_period_steps

        power_np = power_ratio[0].cpu().numpy() * self.p_max_w
        prb_np = prb_share[0].cpu().numpy()

        return {
            "ru_on": self._cached_ru_on.copy(),
            "split": self._cached_split.copy(),
            "power": power_np.astype(np.float32),
            "prb": prb_np.astype(np.float32),
        }

    def remember(
        self,
        state: np.ndarray,
        action: Dict[str, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Push one env-step transition into both buffers as appropriate.

        Reads `self.last_action_was_decision`, set by the immediately
        preceding `select_action()` call, to know whether this step's
        discrete (ru_on, split) choice was freshly made or replayed --
        callers must call select_action() then remember() in lockstep for
        this to stay accurate (the standard select -> step -> remember
        training-loop order already used throughout this codebase).
        """
        cont_params = np.stack([action["power"] / self.p_max_w, action["prb"]], axis=-1)
        self.lower_memory.push(state, cont_params, reward, next_state, done)

        if self.last_action_was_decision:
            # Flush any previously pending window (shouldn't normally
            # happen mid-window, but guards against an odd episode
            # boundary) and start a new one.
            self._pending_upper_state = state
            self._pending_upper_reward_sum = reward
            self._pending_ru_on = action["ru_on"].copy()
            self._pending_split = action["split"].copy()
        else:
            self._pending_upper_reward_sum += reward

        window_complete = self._steps_since_decision == 0
        if window_complete and self._pending_upper_state is not None:
            self.upper_memory.push(
                self._pending_upper_state,
                self._pending_ru_on,
                self._pending_split,
                self._pending_upper_reward_sum,
                next_state,
                done,
            )
            self._pending_upper_state = None

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(
                self.tau * s_param.data + (1.0 - self.tau) * t_param.data
            )

    def update_lower(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Every-step update of the continuous parameter (lower-level) network.

        P-DQN-style deterministic policy gradient: gather each branch's
        greedy activation-Q value (the discrete decision type power/prb
        most directly interacts with) before averaging, not a raw mean
        over an un-gathered action dimension -- the exact class of bug
        already found and fixed in agents/branching_mp_dqn.py's actor
        update.
        """
        batch_size = batch_size or self.batch_size_default
        if len(self.lower_memory) < max(batch_size, self.min_buffer_size):
            return {"param_loss": 0.0, "epsilon": self.epsilon}

        states, _cont_params, _rewards, _next_states, _dones = self.lower_memory.sample(
            batch_size
        )
        states = states.to(self.device)

        lower_feat = self.lower_encoder(states)
        power_ratio, prb_share = self.param_net(lower_feat)
        pred_params = torch.stack([power_ratio, prb_share], dim=-1)

        upper_feat = self.upper_encoder(states).detach()
        activation_q, _split_q = self._multi_pass_q(upper_feat, pred_params)
        greedy_idx = activation_q.argmax(dim=-1, keepdim=True).detach()
        activation_q_greedy = activation_q.gather(-1, greedy_idx).squeeze(-1)
        param_loss = -activation_q_greedy.mean()

        self.param_opt.zero_grad()
        param_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.lower_encoder.parameters()) + list(self.param_net.parameters()),
            max_norm=self.gradient_clip_norm,
        )
        self.param_opt.step()

        self._soft_update(self.lower_encoder_target, self.lower_encoder)
        self._soft_update(self.param_net_target, self.param_net)

        return {"param_loss": float(param_loss.item()), "epsilon": self.epsilon}

    def update_upper(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Every-N-steps update of the branching critic (upper-level network).

        Standard Double-DQN target (online net selects, target net
        evaluates) for both branch-groups -- no twin critic, no
        target-policy-smoothing noise, per the explicit no-TD3 decision
        (Concept Note Section 10.4).
        """
        batch_size = batch_size or self.batch_size_default
        if len(self.upper_memory) < max(
            min(batch_size, 32), self.min_buffer_size // 10
        ):
            return {"critic_loss": 0.0}

        states, ru_on, split, rewards, next_states, dones = self.upper_memory.sample(
            min(batch_size, len(self.upper_memory))
        )
        states = states.to(self.device)
        ru_on = ru_on.to(self.device)
        split = split.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_lower_feat = self.lower_encoder_target(next_states)
            next_power_ratio, next_prb_share = self.param_net_target(next_lower_feat)
            next_cont_params = torch.stack([next_power_ratio, next_prb_share], dim=-1)

            next_upper_feat_online = self.upper_encoder(next_states)
            next_act_online, next_split_online = self._multi_pass_q(
                next_upper_feat_online, next_cont_params
            )
            next_act_actions = next_act_online.argmax(dim=-1)
            next_split_actions = next_split_online.argmax(dim=-1)

            next_upper_feat_target = self.upper_encoder_target(next_states)
            next_act_target, next_split_target = self._multi_pass_q(
                next_upper_feat_target, next_cont_params
            )
            next_act_eval = next_act_target.gather(
                -1, next_act_actions.unsqueeze(-1)
            ).squeeze(-1)
            next_split_eval = next_split_target.gather(
                -1, next_split_actions.unsqueeze(-1)
            ).squeeze(-1)

            # Per-branch targets (not one shared/averaged target across
            # branches -- the fixed pattern from agents/branching_mp_dqn.py
            # and agents/ddqn_agent.py).
            y_activation = rewards + self.gamma * (1.0 - dones) * next_act_eval
            y_split = rewards + self.gamma * (1.0 - dones) * next_split_eval

        upper_feat = self.upper_encoder(states)
        # Continuous params for the *current* multi-pass pass: the critic
        # still needs some (power, prb) to fuse per branch; using the
        # current param net's own prediction at these states keeps this
        # self-consistent without needing to store/replay historical
        # continuous params in the upper buffer.
        with torch.no_grad():
            lower_feat = self.lower_encoder(states)
            power_ratio, prb_share = self.param_net(lower_feat)
            cont_params = torch.stack([power_ratio, prb_share], dim=-1)

        activation_q, split_q = self._multi_pass_q(upper_feat, cont_params)
        act_sel = activation_q.gather(-1, ru_on.unsqueeze(-1)).squeeze(-1)
        split_sel = split_q.gather(-1, split.unsqueeze(-1)).squeeze(-1)

        critic_loss = F.mse_loss(act_sel, y_activation) + F.mse_loss(split_sel, y_split)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.upper_encoder.parameters()) + list(self.critic.parameters()),
            max_norm=self.gradient_clip_norm,
        )
        self.critic_opt.step()

        self._soft_update(self.upper_encoder_target, self.upper_encoder)
        self._soft_update(self.critic_target, self.critic)

        return {"critic_loss": float(critic_loss.item())}

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset_decision_cadence(self):
        """Force the next select_action() call to make a fresh discrete
        decision. Call this at the start of each evaluation episode so
        eval runs don't inherit leftover cadence state from wherever
        training happened to leave it."""
        self._steps_since_decision = 0
        self._pending_upper_state = None
