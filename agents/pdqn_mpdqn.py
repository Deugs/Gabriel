"""P-DQN and MP-DQN baselines: non-branching parameterized (discrete-continuous) DQNs.

Per MPhil Thesis Concept Note v4.0 Section 12.1/10.3.1: unlike BranchingMPDQN's
per-RRH factorized (2R) discrete heads, these two baselines enumerate the FULL
joint discrete action space of 2^R RRH on/off configurations in one un-factorized
Q-network, conditioned on one shared continuous parameter vector
x(s) = (p_1..p_R, beta_1..beta_R). This is deliberately intractable beyond R~12
(2^R blows up combinatorially) -- that intractability is itself the empirical
evidence Concept Note v4.0 Section 10.3.1 cites for why branching is necessary,
so these baselines are hard-capped at MAX_SAFE_N_RRH and must not be run at the
larger scalability-sweep sizes (R=20/35/50).

- P-DQN (Xiong et al., 2018), mode="single_pass": every one of the 2^R
  configurations is evaluated against the SAME, unmasked continuous parameter
  vector.
- MP-DQN (Bester et al., 2019), mode="multi_pass": fixes the resulting
  false-gradient/cross-talk problem by masking (zeroing) the power/bandwidth of
  any RRH that is OFF under a given configuration before evaluating that
  configuration's Q-value -- one "pass" per configuration, hence the name.

Neither baseline uses twin critics or TD3-style target-policy smoothing/policy
delay (unlike BranchingMPDQN) -- Concept Note v4.0 Section 12.1 is explicit that
these two run "without branching or twin critics". The critic here is therefore a
single Double-DQN-style network (mirroring agents/ddqn_agent.py's DDQNAgent), and
the continuous side is a plain DDPG-style deterministic update with no policy
delay (mirroring the param-net step of BranchingMPDQN, minus the TD3 machinery).
"""

import copy
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.branching_mp_dqn import (
    ContinuousParameterNetwork,
    ParameterizedReplayBuffer,
    SharedEncoder,
)

MAX_SAFE_N_RRH = 12  # 2**12 = 4096 joint configurations; hard ceiling.


class JointActionSpace:
    """Bidirectional mapping between an integer 0..2^R-1 and an R-length on/off vector.

    `table[k, r] == 1.0` iff RRH r is ON under joint configuration k. Built once at
    construction via pure bitwise/broadcast ops (no Python loop over configurations).
    """

    def __init__(self, n_rrh: int, device: torch.device):
        if n_rrh > MAX_SAFE_N_RRH:
            raise ValueError(
                f"JointActionSpace: n_rrh={n_rrh} exceeds MAX_SAFE_N_RRH="
                f"{MAX_SAFE_N_RRH}. P-DQN/MP-DQN enumerate 2**n_rrh joint discrete "
                "actions and are only defined at R<=12 per Concept Note v4.0 "
                "Section 12.1/10.3.1 -- use BranchingMPDQN for larger R."
            )
        self.n_rrh = n_rrh
        self.n_configs = 2**n_rrh
        self.device = device
        bit_positions = torch.arange(n_rrh, device=device)
        idx = torch.arange(self.n_configs, device=device)
        self.table = ((idx.unsqueeze(-1) >> bit_positions) & 1).float()  # (K, R)

    def decode(self, index: torch.Tensor) -> torch.Tensor:
        """index: (...,) long -> (..., n_rrh) binary float on/off vector."""
        return self.table[index]

    def encode(self, binary: torch.Tensor) -> torch.Tensor:
        """binary: (..., n_rrh) in {0,1} -> (...,) long joint configuration index."""
        weights = (2 ** torch.arange(self.n_rrh, device=binary.device)).float()
        return (binary.float() * weights).sum(dim=-1).long()


class JointQNetwork(nn.Module):
    """Un-factorized Q-network over the 2^R joint discrete action space.

    Has its own internal state encoder (mirroring BranchingMPDQN's
    SingleBranchCritic, which likewise owns a private SharedEncoder rather than
    sharing the agent's outer encoder) so the critic optimizer never needs to
    touch the continuous-parameter-network's encoder.
    """

    def __init__(
        self, state_dim: int, n_rrh: int, action_space: JointActionSpace, mode: str
    ):
        super().__init__()
        if mode not in ("single_pass", "multi_pass"):
            raise ValueError(f"Unknown JointQNetwork mode: {mode!r}")
        self.n_rrh = n_rrh
        self.mode = mode
        self.n_configs = action_space.n_configs

        self.encoder = SharedEncoder(state_dim, [256, 128])
        feat_dim = self.encoder.output_dim
        self.param_encoder = nn.Linear(2 * n_rrh, feat_dim)
        self.fusion = nn.Linear(feat_dim * 2, feat_dim)
        self.adv_head = nn.Linear(feat_dim, self.n_configs)

        # (K, 2R): each RRH's on/off bit duplicated for its (power, bandwidth) pair.
        self.register_buffer(
            "config_mask", action_space.table.repeat_interleave(2, dim=1)
        )

    def forward(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """state: (batch, state_dim). x: (batch, n_rrh, 2) continuous params
        (power_ratio, bandwidth_share). Returns (batch, K) -- Q for every config."""
        batch = state.shape[0]
        feat = self.encoder(state)
        x_flat = x.reshape(batch, -1)  # (batch, 2R)

        if self.mode == "single_pass":
            pf = F.relu(self.param_encoder(x_flat))  # (batch, H)
            fused = F.relu(self.fusion(torch.cat([feat, pf], dim=-1)))  # (batch, H)
            return self.adv_head(fused)  # (batch, K)

        # multi_pass: mask x per-configuration (zero the power/bandwidth of any RRH
        # that would be OFF under that configuration) before evaluating it. Folding
        # the config axis into a wide batch axis (rather than looping) keeps this a
        # handful of large matmuls; the final contraction is a per-row dot product
        # against adv_head's matching weight row (an einsum), NOT a Linear(H, K)
        # over the (batch, K, H) tensor -- that would materialize a (batch, K, K)
        # tensor (billions of floats at K=4096), which this avoids entirely.
        masked_x = x_flat.unsqueeze(1) * self.config_mask.unsqueeze(0)  # (batch,K,2R)
        feat_tiled = feat.unsqueeze(1).expand(-1, self.n_configs, -1)  # (batch,K,H)
        pf = F.relu(self.param_encoder(masked_x))  # (batch, K, H)
        fused = F.relu(self.fusion(torch.cat([feat_tiled, pf], dim=-1)))  # (batch,K,H)
        return (
            torch.einsum("bkh,kh->bk", fused, self.adv_head.weight)
            + self.adv_head.bias
        )

    def forward_single(
        self, state: torch.Tensor, x: torch.Tensor, config_idx: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate Q for exactly ONE configuration per batch row: O(batch*H), never
        enumerates all K. Used for the replayed-action critic loss and for the
        continuous-parameter actor update, both of which only need one config."""
        batch = state.shape[0]
        feat = self.encoder(state)
        x_flat = x.reshape(batch, -1)  # (batch, 2R)

        if self.mode == "multi_pass":
            x_flat = x_flat * self.config_mask[config_idx]  # (batch, 2R)

        pf = F.relu(self.param_encoder(x_flat))  # (batch, H)
        fused = F.relu(self.fusion(torch.cat([feat, pf], dim=-1)))  # (batch, H)
        w = self.adv_head.weight[config_idx]  # (batch, H)
        b = self.adv_head.bias[config_idx]  # (batch,)
        return (fused * w).sum(dim=-1) + b  # (batch,)


class PDQNMPDQNBase:
    """Shared implementation for PDQNAgent/MPDQNAgent -- they differ ONLY in
    JointQNetwork's masking behavior (set via the MODE class attribute), so every
    other line (hyperparameter parsing, select_action, update, checkpointing)
    lives here once. This is deliberate: docs/rules.md's Baseline Fairness Rule
    requires these two to differ *only* in the one thing they're meant to isolate.
    """

    MODE: str = ""

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

        self.epsilon = float(get_val("epsilon_start", 1.0))
        self.epsilon_end = float(get_val("epsilon_end", 0.01))
        self.epsilon_decay = float(get_val("epsilon_decay", 0.995))

        lr_branch = float(get_val("lr_discrete", 1e-3))
        lr_param = float(get_val("lr_actor", 1e-4))
        buffer_size = int(get_val("buffer_size", 100000))

        # Raises ValueError here if n_rrh > MAX_SAFE_N_RRH -- the hard,
        # defense-in-depth guard that protects every caller, not just
        # training/train_baselines.py's own soft skip-guard.
        self.action_space = JointActionSpace(n_rrh, self.device)

        self.encoder = SharedEncoder(state_dim, [256, 128]).to(self.device)
        self.param_net = ContinuousParameterNetwork(
            self.encoder.output_dim, n_rrh
        ).to(self.device)
        self.q_net = JointQNetwork(state_dim, n_rrh, self.action_space, self.MODE).to(
            self.device
        )

        self.encoder_target = copy.deepcopy(self.encoder).to(self.device)
        self.param_net_target = copy.deepcopy(self.param_net).to(self.device)
        self.q_net_target = copy.deepcopy(self.q_net).to(self.device)

        self.critic_opt = optim.Adam(self.q_net.parameters(), lr=lr_branch)
        self.param_opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.param_net.parameters()),
            lr=lr_param,
        )

        self.memory = ParameterizedReplayBuffer(buffer_size)

    def select_action(
        self, obs: np.ndarray, evaluate: bool = False
    ) -> Dict[str, np.ndarray]:
        """Select a joint RRH on/off configuration plus shared power/bandwidth."""
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.encoder(state_t)
            p_ratio, bw_share = self.param_net(feat)
            cont_params = torch.stack([p_ratio, bw_share], dim=-1)  # (1, n_rrh, 2)

            if not evaluate and random.random() < self.epsilon:
                config_idx = torch.randint(
                    0, self.action_space.n_configs, (1,), device=self.device
                )
            else:
                q_all = self.q_net(state_t, cont_params)  # (1, K)
                config_idx = q_all.argmax(dim=-1)  # (1,)

            rrh_on = self.action_space.decode(config_idx)[0].cpu().numpy()

            if not evaluate:
                noise = torch.randn_like(p_ratio) * 0.05
                p_ratio = torch.clamp(p_ratio + noise, 0.0, 1.0)

            p_np = p_ratio[0].cpu().numpy() * self.p_max_w
            # `bandwidth` is produced (and masked exactly like power for MP-DQN)
            # so the system model's "power/bandwidth are 0 when the RRH is off"
            # convention has something to act on -- matching the shape every
            # other agent/baseline already returns. Note CRANEnv.step() does not
            # currently read this key at all (a pre-existing, out-of-scope
            # environment gap that predates and is not fixed by this baseline).
            bw_np = bw_share[0].cpu().numpy()

        return {
            "rrh_on": rrh_on.astype(np.int64),
            "power": p_np,
            "bandwidth": bw_np,
            "config_idx": int(config_idx.item()),
        }

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """One plain Double-DQN critic step + one plain DDPG-style param step."""
        if len(self.memory) < batch_size:
            return {"critic_loss": 0.0, "param_loss": 0.0, "epsilon": self.epsilon}

        (
            states,
            disc_actions,
            cont_params,
            rewards,
            next_states,
            dones,
        ) = self.memory.sample(batch_size)

        states = states.to(self.device)
        disc_actions = disc_actions.to(self.device)  # (batch,) joint config indices
        cont_params = cont_params.to(self.device)
        rewards = rewards.to(self.device).squeeze(-1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).squeeze(-1)

        # --- Critic update: classic Double DQN (select via online, evaluate via
        # target) -- there is no twin critic here to fall back on the TD3-style
        # min(Q^A,Q^B) trick, so this mirrors agents/ddqn_agent.py's DDQNAgent. ---
        with torch.no_grad():
            next_feat_online = self.encoder(next_states)
            next_p_online, next_bw_online = self.param_net(next_feat_online)
            next_x_online = torch.stack([next_p_online, next_bw_online], dim=-1)
            next_config = self.q_net(next_states, next_x_online).argmax(dim=-1)

            next_feat_target = self.encoder_target(next_states)
            next_p_target, next_bw_target = self.param_net_target(next_feat_target)
            next_x_target = torch.stack([next_p_target, next_bw_target], dim=-1)
            next_q = self.q_net_target.forward_single(
                next_states, next_x_target, next_config
            )
            y = rewards + self.gamma * (1.0 - dones) * next_q

        q_curr = self.q_net.forward_single(states, cont_params, disc_actions)
        critic_loss = F.mse_loss(q_curr, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # --- Continuous parameter update: plain DDPG-style, no policy delay.
        # Select the config the current policy currently prefers (no_grad, since
        # this selection step itself needs no gradient), then take the actor
        # gradient only through the cheap forward_single at that one config --
        # never through a full (batch, K) forward, which would be needlessly
        # expensive here (unlike the critic path, which needs the full forward
        # once regardless, to select next_config for the target). ---
        feat_actor = self.encoder(states)
        pred_p, pred_bw = self.param_net(feat_actor)
        pred_x = torch.stack([pred_p, pred_bw], dim=-1)

        with torch.no_grad():
            best_config = self.q_net(states, pred_x).argmax(dim=-1)

        q_best = self.q_net.forward_single(states, pred_x, best_config)
        param_loss = -q_best.mean()

        self.param_opt.zero_grad()
        param_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.param_net.parameters()),
            max_norm=1.0,
        )
        self.param_opt.step()

        self._soft_update(self.encoder_target, self.encoder)
        self._soft_update(self.param_net_target, self.param_net)
        self._soft_update(self.q_net_target, self.q_net)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            "critic_loss": float(critic_loss.item()),
            "param_loss": float(param_loss.item()),
            "epsilon": float(self.epsilon),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "param_net": self.param_net.state_dict(),
            "q_net": self.q_net.state_dict(),
            "encoder_target": self.encoder_target.state_dict(),
            "param_net_target": self.param_net_target.state_dict(),
            "q_net_target": self.q_net_target.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "param_opt": self.param_opt.state_dict(),
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.param_net.load_state_dict(state["param_net"])
        self.q_net.load_state_dict(state["q_net"])
        self.encoder_target.load_state_dict(state["encoder_target"])
        self.param_net_target.load_state_dict(state["param_net_target"])
        self.q_net_target.load_state_dict(state["q_net_target"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.param_opt.load_state_dict(state["param_opt"])
        self.epsilon = state.get("epsilon", self.epsilon)


class PDQNAgent(PDQNMPDQNBase):
    """P-DQN (Xiong et al., 2018): single-pass -- all 2^R configs share one
    unmasked continuous parameter vector. Kept as a baseline specifically to
    demonstrate the parameter cross-talk MP-DQN fixes (Bester et al., 2019)."""

    MODE = "single_pass"


class MPDQNAgent(PDQNMPDQNBase):
    """MP-DQN (Bester et al., 2019): multi-pass -- each of the 2^R configs is
    evaluated against its own masked continuous parameter vector (masking the
    power/bandwidth of any RRH that would be OFF under that configuration)."""

    MODE = "multi_pass"
