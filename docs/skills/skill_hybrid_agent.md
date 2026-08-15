# Skill: Branching MP-DQN + TD3 Agent Design

> **Status**: Invokable as the Antigravity `build-hybrid-agent` skill (`.agents/skills/build-hybrid-agent/`), which points back at this file as the spec of record.
>
> **v4.0 correction**: this file previously described the superseded v1.0 "Hybrid SAC-DDQN" design (a discrete DDQN actor, a continuous SAC Gaussian actor, and a single shared twin critic). That architecture was abandoned in favor of the branching, multi-pass, twin-critic parameterized DQN below (Concept Note v2.0/v3.0/v4.0 Section 10) well before this file was corrected to match — `agents/hybrid_sac_dqn.py` still exists only as the superseded alternative, kept for comparison, not as the proposed method. Everything below describes the actual implementation, `agents/branching_mp_dqn.py`.

## Purpose

Implement a hybrid Deep Reinforcement Learning agent that couples a true discrete decision per RRH (on/off, via R independent branching heads) with a continuous parameter per RRH (transmit power ratio and bandwidth share), through **one** coupled Q-network family — not two separate actor networks arbitrated by a shared critic. The coupling mechanism is P-DQN (Xiong et al., 2018), corrected for parameter cross-talk by MP-DQN's multi-pass masking (Bester et al., 2019), scaled to R RRHs by branching (Tavakoli et al., 2018), and stabilized by TD3-style twin critics and target-policy smoothing (Fujimoto et al., 2018).

## Architecture Overview

```
                    State s(t)
                         |
                  Shared Encoder h(s|theta_h)
                  FC(256)-ReLU-LayerNorm
                  FC(128)-ReLU-LayerNorm
                         |
          +--------------+--------------+
          |                             |
   Continuous Parameter Net      R Branch Heads
   x(s|phi): all R RRHs'         (BranchingDiscreteHeads,
   (p_r, beta_r)                  dueling: V(s)+A_r(s,k_r))
          |                             |
   Multi-pass mask (MP-DQN):     Q_r(s, k_r, x_r) for
   keep x_r, zero x_j (j!=r)     k_r in {0,1}, r=1..R
   before each branch's                 |
   Q-value pass                  argmax_k_r per branch
          |                      (independent, epsilon-greedy)
          +--------------+--------------+
                         |
              Twin Critics (Q^A, Q^B)
              -- two independent copies of the
                 branch-heads pipeline, both fed by
                 the ONE shared encoder above (not
                 a separate encoder copy each)
                         |
              min(Q^A, Q^B) -> TD3 Bellman target
```

Concretely, per RRH branch r (Concept Note Section 10.3):

- **Shared encoder** `h(s|theta_h)`: two FC layers (256, 128 units), ReLU + LayerNorm each — `SharedEncoder`.
- **Continuous parameter network** `x(s|phi)`: a deterministic sub-network producing `x_r(s) = (p_r, beta_r)` for **all** R RRHs from the shared representation (the P-DQN mechanism) — `ContinuousParameterNetwork`. `p_r` (power ratio) is sigmoid-activated in [0,1]; `beta_r` (bandwidth share) is softmax-normalized across RRHs so shares sum to 1.
- **Multi-pass mask** (MP-DQN): before branch r's Q-value is computed, only `x_r` is fed into that pass — every other RRH's continuous parameters are excluded from that pass's computation graph, removing the false-gradient cross-talk P-DQN's single-pass design would otherwise introduce (Bester et al., 2019).
- **R discrete branches** (Tavakoli et al., 2018): each RRH gets a dueling-style head producing `Q_r(s, k_r)` for `k_r in {0,1}` off the shared representation plus its own (masked) `x_r` — `BranchingDiscreteHeads`, output grows as 2R, not 2^R.
- **Twin critics** (Fujimoto et al., 2018): two independent copies of the branch/critic network (`Q^A`, `Q^B` — `TwinBranchCritic`), each with its own target network; the Bellman target uses `min(Q^A, Q^B)` to counter overestimation bias, with delayed, less-frequent updates to `phi` (`policy_delay`, default 2) and target-policy smoothing noise on `x'` at the target networks.

## Network Specifications

The actual module classes (`agents/branching_mp_dqn.py`) — read that file directly for the current, authoritative implementation; the excerpts below exist to keep this spec in sync with it, not to duplicate it as a separate source of truth.

### Shared Encoder

```python
class SharedEncoder(nn.Module):
    """Shared state encoder h(s|theta_h) mapping state s(t) to feature representation."""

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
```

`hidden_dims`, `activation`, and `use_layer_norm` are genuinely config-driven (`algorithm.hidden_dims`/`activation`/`use_layer_norm` in `config/default.yaml`) — `BranchingMPDQN.__init__` reads all three and constructs exactly **one** `SharedEncoder` instance (`self.encoder`), so a config that omits them gets the `[256, 128]`/ReLU/LayerNorm spec defaults shown above. Unlike an earlier version of this design, `SingleBranchCritic`/`TwinBranchCritic` no longer construct their own encoder copies — see below.

### R Branching Discrete Heads (Dueling)

```python
class BranchingDiscreteHeads(nn.Module):
    """Factorized Dueling Discrete Branch Heads Q_r(s, k_r) for R RRHs."""

    def __init__(self, feature_dim: int, n_rrh: int):
        super().__init__()
        self.value_head = nn.Linear(feature_dim, 1)
        self.adv_heads = nn.ModuleList(
            [nn.Linear(feature_dim, 2) for _ in range(n_rrh)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        v = self.value_head(features).unsqueeze(1)          # (batch, 1, 1)
        advs = torch.stack([head(features) for head in self.adv_heads], dim=1)
        # Dueling aggregation: Q_r(s, a) = V(s) + (A_r(s, a) - mean(A_r(s, .)))
        return v + (advs - advs.mean(dim=-1, keepdim=True))
```

### Continuous Parameter Network (P-DQN)

```python
class ContinuousParameterNetwork(nn.Module):
    """Deterministic continuous parameter network x(s|phi) producing (p_r, beta_r) per RRH."""

    def __init__(self, feature_dim: int, n_rrh: int):
        super().__init__()
        self.power_head = nn.Linear(feature_dim, n_rrh)
        self.bandwidth_head = nn.Linear(feature_dim, n_rrh)

    def forward(self, features):
        power_ratio = torch.sigmoid(self.power_head(features))       # [0,1] per RRH
        bandwidth_share = F.softmax(self.bandwidth_head(features), dim=-1)  # sums to 1
        return power_ratio, bandwidth_share
```

### Single-Branch Critic with MP-DQN Multi-Pass Masking

`SingleBranchCritic`/`TwinBranchCritic` take the shared encoder's output feature directly (`feat`) rather than a raw `state` and their own encoder copy — an earlier version of this file (and of the code) had each twin critic construct a private `SharedEncoder`, which meant `theta_h` was silently duplicated three ways (agent's own + one per twin critic) instead of genuinely shared, contradicting the "one coupled network" framing below. `BranchingMPDQN` now computes `feat = self.encoder(state)` once per critic evaluation and passes it in; see `_multi_pass_q()`.

```python
class SingleBranchCritic(nn.Module):
    """Single Multi-Pass Branch Critic Q_r(s, k_r, x_r)."""

    def __init__(self, feature_dim: int, n_rrh: int):
        super().__init__()
        self.n_rrh = n_rrh
        self.discrete_heads = BranchingDiscreteHeads(feature_dim, n_rrh)
        self.param_encoder = nn.Linear(2, 64)  # 2 continuous params (power, bandwidth)
        self.fusion = nn.Linear(feature_dim + 64, feature_dim)

    def forward(self, feat, continuous_params, branch_mask_idx: Optional[int] = None):
        # feat: (batch, feature_dim), already produced by the shared encoder
        params_3d = continuous_params.view(-1, self.n_rrh, 2)

        if branch_mask_idx is not None:
            # Multi-pass (MP-DQN): only branch_mask_idx's own (p_r, beta_r)
            # enters this pass's computation graph -- every other RRH's
            # parameters are excluded, not merely zeroed after the fact.
            param_feat = F.relu(self.param_encoder(params_3d[:, branch_mask_idx, :]))
        else:
            param_feat = F.relu(self.param_encoder(params_3d.mean(dim=1)))

        fused = F.relu(self.fusion(torch.cat([feat, param_feat], dim=-1)))
        return self.discrete_heads(fused)  # (batch, n_rrh, 2)
```

The `_multi_pass_q()` driver (on `BranchingMPDQN`, not the critic itself) runs this forward pass once **per branch** (R passes total per critic evaluation) and keeps only that pass's branch-r slice of the output — this is what actually implements "multi-pass": R independent, cross-talk-free evaluations, not R simultaneous branch outputs from one pass. The encoder computation itself is hoisted outside this per-branch loop (computed once, reused across all R passes), since it depends only on `state`, not on `branch_mask_idx`.

### Twin Critics (TD3)

```python
class TwinBranchCritic(nn.Module):
    """Twin Critic Networks (Q^A, Q^B) for TD3 over-estimation mitigation."""

    def __init__(self, feature_dim: int, n_rrh: int):
        super().__init__()
        self.q_a = SingleBranchCritic(feature_dim, n_rrh)
        self.q_b = SingleBranchCritic(feature_dim, n_rrh)

    def forward(self, feat, continuous_params, branch_mask_idx=None):
        return self.q_a(feat, continuous_params, branch_mask_idx), \
               self.q_b(feat, continuous_params, branch_mask_idx)
```

Both twin critics, and the continuous parameter network, are trained through this single shared `self.encoder` — `BranchingMPDQN`'s `critic_opt` includes `list(self.encoder.parameters()) + list(self.twin_critic.parameters())`, and `param_opt` includes `list(self.encoder.parameters()) + list(self.param_net.parameters())`, so the encoder receives a gradient contribution from both the critic loss (every `update()` call) and the delayed parameter-network loss (every `policy_delay`-th call).

## Training Algorithm (`BranchingMPDQN`)

Key points from `agents/branching_mp_dqn.py::update()` (read the actual file for the full, current implementation):

1. **Target computation**: the target encoder/parameter-net produce `next_cont_params`; **target-policy smoothing noise** (`N(0, target_noise_std)`, clamped to `[-0.1, 0.1]`) is added before clamping to `[0,1]` — this is genuinely applied, not omitted.
2. **Multi-pass target Q-values**: `_multi_pass_q()` evaluates both target critics per branch; **Double-DQN-style action selection** uses `Q^A`'s argmax, then **`torch.min(Q^A, Q^B)` at that selected action** forms the TD3 Bellman target — a true element-wise min of both twin critics, not just one.
3. **Critic loss**: MSE between each twin critic's Q-value at the *taken* discrete action (from the replay buffer) and the shared target, summed across both critics and all R branches.
4. **Delayed policy update**: every `policy_delay` (default 2) critic updates, the continuous parameter network (and encoder) are updated to maximize `Q^A`'s multi-pass value, and all three target networks (encoder, param net, twin critic) receive a soft (Polyak) update — TD3's delayed-actor-update pattern, factored here around the shared continuous parameter net rather than a discrete actor.
5. **Exploration**: independent epsilon-greedy per discrete branch (`self.epsilon`) and additive Gaussian noise on the continuous parameters (`self.continuous_noise_std`) — both mechanisms present, matching Concept Note Section 10.5. Both decay once per **episode**, via `agent.decay_exploration()` (called by the training loop after each episode's `while not done:` block ends) — not inside `update()` itself, which runs once per environment step; `config/default.yaml`'s `epsilon_decay` is documented as a per-episode rate, and decaying it once per step instead would hit the exploration floor roughly `max_steps_per_episode` times faster than intended.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Branching (R independent heads) instead of a joint 2^R head | Avoids exponential action-space growth (Section 10.3.1); output scales as 2R |
| MP-DQN multi-pass masking, not P-DQN's single shared pass | Removes false-gradient cross-talk between unrelated RRHs (Bester et al., 2019); a structural requirement of this design, not optional (Section 10.5) |
| Twin critics + delayed policy update + target-policy smoothing | TD3's fix for DDPG-style overestimation bias, applied to the branch/critic pipeline |
| LayerNorm (not BatchNorm) after every FC layer | More stable with the varying/small batch sizes typical of RL |
| Sigmoid for power ratio, softmax for bandwidth share | Power ratio is an independent [0,1] fraction of P_max per RRH; bandwidth shares must sum to 1 across active RRHs |
| One shared encoder instance feeding both the continuous parameter net and every branch's critic (not a copy per consumer) | The actual coupling mechanism that makes this "one coupled network" rather than two separate actor networks arbitrated by a shared critic (the superseded v1.0 design this file used to describe) |
| Exploration decay (epsilon, continuous noise) once per episode via `decay_exploration()`, not once per `update()` call | Matches `config/default.yaml`'s documented per-episode `epsilon_decay` rate; decaying per environment step instead reaches the exploration floor far faster than intended |

## Hyperparameters

Read from `config/default.yaml`'s `algorithm:` block via `BranchingMPDQN`'s `get_val()` helper (falls back to the defaults below if a key is absent):

```yaml
lr_discrete: 1.0e-4    # Branch/critic learning rate
lr_actor: 3.0e-4       # Continuous parameter network learning rate
buffer_size: 1000000   # Replay buffer capacity
batch_size: 256        # Mini-batch size (passed to update(), not read from config)
hidden_dims: [256, 128]  # SharedEncoder layer widths (Section 10.3 spec)
activation: "relu"       # SharedEncoder nonlinearity: relu, leaky_relu, tanh, or gelu
use_layer_norm: true     # Whether SharedEncoder applies LayerNorm after each layer
gamma: 0.99            # Discount factor
tau: 0.005             # Soft update rate
policy_delay: 2        # TD3 actor/target-update delay (critic updates per policy update)
target_noise_std: 0.05 # TD3 target-policy smoothing noise std
epsilon_start: 1.0     # Initial discrete-branch exploration rate
epsilon_end: 0.01      # Final exploration rate
epsilon_decay: 0.995   # Decay per EPISODE (agent.decay_exploration(), not update())
continuous_noise_std: 0.1       # Initial continuous-parameter exploration noise std
continuous_noise_std_end: 0.01  # Final continuous-parameter noise std
gradient_clip_norm: 1.0  # max_norm for both twin-critic and param-net gradient clipping
reward_scale: 1.0        # Multiplicative scale applied to sampled rewards before the Bellman target
```

All of the above are genuinely read via `BranchingMPDQN`'s `get_val()` helper — none are dead, and `config/default.yaml`'s `hidden_dims` now matches this file's `[256, 128]` spec value exactly (it briefly diverged to `[256, 256]`; both are now reconciled to the validated spec). `hardware.device` (in `config/default.yaml`'s `hardware:` block, not `algorithm:`) supplies `BranchingMPDQN`'s device default when the constructor's `device` argument is left as `None`; an explicit `device=` argument always wins over it.

## Validation Checklist

- [ ] Critic loss decreases over the first ~1000 updates
- [ ] Q-values are finite (no NaN/Inf) across both twin critics
- [ ] Each discrete branch explores initially (epsilon starts at 1.0) and decays
- [ ] Continuous parameters stay in valid ranges (power ratio in [0,1], bandwidth shares summing to ~1)
- [ ] Target networks update slower than main networks (tau << 1); policy/target updates only occur every `policy_delay` critic updates
- [ ] `_multi_pass_q()` genuinely produces different Q-values for two candidate parameter vectors that differ only in an unrelated RRH's `(p_j, beta_j)` for `j != r` (i.e., cross-talk is actually absent) — see `tests/test_baselines_v2.py`/`agents/branching_mp_dqn.py`'s own tests for a concrete check of this
- [ ] Gradient norms are reasonable (clipped to `algorithm.gradient_clip_norm`, default 1.0)
- [ ] Action selection is faster than environment step time (see `evaluation/latency_benchmark.py`)
