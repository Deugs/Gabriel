# Skill: Hybrid SAC-DDQN Agent Design

> **Status**: Invokable as the Antigravity `build-hybrid-agent` skill (`.agents/skills/build-hybrid-agent/`), which points back at this file as the spec of record.

## Purpose
Implement a hybrid Deep Reinforcement Learning agent that combines discrete action selection (DDQN) for RRH on/off decisions with continuous policy optimization (SAC) for transmit power allocation, coordinated through a shared critic.

## Architecture Overview

```
                    State s(t)
                         |
          +--------------+--------------+
          |                             |
    Discrete Actor                Continuous Actor
    (DDQN-style)                  (SAC Gaussian)
          |                             |
    Q-values per                 Mean, Log-Std
    RRH binary                   per RRH power
          |                             |
    Epsilon-greedy               Reparameterization
    sampling                     trick
          |                             |
    v(t+1) in {0,1}^R          p(t) ~ N(mu, sigma)
          |                             |
          +--------------+--------------+
                         |
                  Shared Critic
                  (Twin Q-Networks)
                         |
              Q(s, v, p) -> scalar value
```

## Network Specifications

### Discrete Actor (Q-Network)
```python
class DiscreteActor(nn.Module):
    def __init__(self, state_dim, n_rrh, hidden_dims=[256, 256]):
        super().__init__()
        self.n_rrh = n_rrh

        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.LayerNorm(dim)
            ])
            prev_dim = dim

        self.backbone = nn.Sequential(*layers)
        # Factorized: independent binary decision per RRH
        self.heads = nn.ModuleList([
            nn.Linear(prev_dim, 2) for _ in range(n_rrh)
        ])

    def forward(self, state):
        features = self.backbone(state)
        q_values = torch.stack([head(features) for head in self.heads], dim=1)
        # q_values: (batch, n_rrh, 2) -> Q(s, v_r=0) and Q(s, v_r=1)
        return q_values

    def select_action(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            return torch.randint(0, 2, (self.n_rrh,))

        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))[0]  # (n_rrh, 2)
            actions = q_values.argmax(dim=-1)  # (n_rrh,)
        return actions
```

### Continuous Actor (Gaussian Policy)
```python
class ContinuousActor(nn.Module):
    def __init__(self, state_dim, n_rrh, hidden_dims=[256, 256], 
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.n_rrh = n_rrh
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.LayerNorm(dim)
            ])
            prev_dim = dim

        self.backbone = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_dim, n_rrh)
        self.log_std = nn.Linear(prev_dim, n_rrh)

    def forward(self, state):
        features = self.backbone(state)
        mean = torch.sigmoid(self.mean(features))  # [0, 1]
        log_std = torch.clamp(self.log_std(features), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        noise = torch.randn_like(mean)
        action = mean + std * noise
        action = torch.clamp(action, 0, 1)  # Ensure valid power ratio

        # Log probability
        log_prob = -0.5 * ((noise ** 2) + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state, deterministic=False):
        mean, _ = self.forward(state)
        if deterministic:
            return mean
        return self.sample(state)[0]
```

### Shared Critic (Twin Q-Networks)
```python
class HybridCritic(nn.Module):
    def __init__(self, state_dim, n_rrh, hidden_dims=[256, 256]):
        super().__init__()

        # State path
        state_layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            state_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.LayerNorm(dim)
            ])
            prev_dim = dim
        self.state_encoder = nn.Sequential(*state_layers)

        # Discrete action embedding
        self.discrete_embed = nn.Embedding(2, 16)  # 2 states (on/off) -> 16-dim

        # Continuous action path
        self.action_encoder = nn.Sequential(
            nn.Linear(n_rrh, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # Fusion layers
        fusion_dim = prev_dim + n_rrh * 16 + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        )

    def forward(self, state, discrete_action, continuous_action):
        # discrete_action: (batch, n_rrh) in {0, 1}
        # continuous_action: (batch, n_rrh) in [0, 1]

        state_feat = self.state_encoder(state)

        disc_embed = self.discrete_embed(discrete_action.long())  # (batch, n_rrh, 16)
        disc_feat = disc_embed.view(disc_embed.size(0), -1)  # (batch, n_rrh*16)

        cont_feat = self.action_encoder(continuous_action)

        fusion = torch.cat([state_feat, disc_feat, cont_feat], dim=-1)
        q_value = self.fusion(fusion)
        return q_value
```

## Training Algorithm

```python
class HybridSACDDQN:
    def __init__(self, state_dim, n_rrh, n_ue, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_rrh = n_rrh
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha  # SAC temperature
        self.epsilon = config.epsilon_start

        # Networks
        self.discrete_actor = DiscreteActor(state_dim, n_rrh).to(self.device)
        self.continuous_actor = ContinuousActor(state_dim, n_rrh).to(self.device)
        self.critic1 = HybridCritic(state_dim, n_rrh).to(self.device)
        self.critic2 = HybridCritic(state_dim, n_rrh).to(self.device)

        # Target networks
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.continuous_actor_target = copy.deepcopy(self.continuous_actor)

        # Optimizers
        self.disc_opt = optim.Adam(self.discrete_actor.parameters(), lr=config.lr_disc)
        self.cont_opt = optim.Adam(self.continuous_actor.parameters(), lr=config.lr_actor)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.lr_critic
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(batch.states).to(self.device)
        discrete_actions = torch.LongTensor(batch.discrete_actions).to(self.device)
        continuous_actions = torch.FloatTensor(batch.continuous_actions).to(self.device)
        rewards = torch.FloatTensor(batch.rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(batch.next_states).to(self.device)
        dones = torch.FloatTensor(batch.dones).to(self.device).unsqueeze(1)

        # --- Critic Update ---
        with torch.no_grad():
            # Next discrete action (target Q-network)
            next_q_disc = self.discrete_actor(next_states)  # (batch, n_rrh, 2)
            next_disc_actions = next_q_disc.argmax(dim=-1)  # (batch, n_rrh)

            # Next continuous action (target policy)
            next_cont_actions, next_log_prob = self.continuous_actor_target.sample(next_states)

            # Target Q-values
            q1_next = self.critic1_target(next_states, next_disc_actions, next_cont_actions)
            q2_next = self.critic2_target(next_states, next_disc_actions, next_cont_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob

            y = rewards + self.gamma * (1 - dones) * q_next

        # Current Q-values
        q1 = self.critic1(states, discrete_actions, continuous_actions)
        q2 = self.critic2(states, discrete_actions, continuous_actions)

        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0
        )
        self.critic_opt.step()

        # --- Discrete Actor Update ---
        q_values = self.discrete_actor(states)  # (batch, n_rrh, 2)
        # Maximize Q for selected actions
        disc_loss = -(q_values.gather(-1, discrete_actions.unsqueeze(-1)).squeeze(-1).mean())

        self.disc_opt.zero_grad()
        disc_loss.backward()
        self.disc_opt.step()

        # --- Continuous Actor Update ---
        cont_actions, log_prob = self.continuous_actor.sample(states)
        # Use current discrete actor's best action
        disc_actions = self.discrete_actor(states).argmax(dim=-1)

        q1_new = self.critic1(states, disc_actions, cont_actions)
        q2_new = self.critic2(states, disc_actions, cont_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = -(q_new - self.alpha * log_prob).mean()

        self.cont_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.continuous_actor.parameters(), 1.0)
        self.cont_opt.step()

        # --- Soft Update Target Networks ---
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)
        self._soft_update(self.continuous_actor_target, self.continuous_actor)

        # Decay epsilon
        self.epsilon = max(config.epsilon_end, self.epsilon * config.epsilon_decay)

        return {
            "critic_loss": critic_loss.item(),
            "disc_loss": disc_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_value": q1.mean().item(),
            "epsilon": self.epsilon
        }

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Factorized discrete actions | Independent per-RRH decisions avoid exponential action space (2^R) |
| Twin critics | Reduces overestimation bias (TD3/SAC principle) |
| LayerNorm vs BatchNorm | LayerNorm more stable with varying batch sizes in RL |
| Sigmoid output for continuous actor | Natural [0,1] scaling; multiply by P_max in environment |
| Shared state encoder | Could be added; currently separate for flexibility |

## Hyperparameters

```yaml
lr_disc: 1e-4        # Discrete actor learning rate
lr_actor: 3e-4        # Continuous actor learning rate
lr_critic: 3e-4       # Critic learning rate
buffer_size: 1000000  # Replay buffer capacity
batch_size: 256       # Mini-batch size
gamma: 0.99           # Discount factor
tau: 0.005            # Soft update rate
alpha: 0.2            # SAC entropy temperature (auto-tune recommended)
epsilon_start: 1.0    # Initial exploration rate
epsilon_end: 0.01     # Final exploration rate
epsilon_decay: 0.995  # Decay per episode
```

## Validation Checklist
- [ ] Critic loss decreases monotonically over first 1000 updates
- [ ] Q-values are finite (no NaN/Inf)
- [ ] Discrete actor explores initially (epsilon starts at 1.0)
- [ ] Continuous actor outputs in [0,1] range
- [ ] Target networks update slower than main networks (tau << 1)
- [ ] Gradient norms are reasonable (< 10 after clipping)
- [ ] Action selection is faster than environment step time
