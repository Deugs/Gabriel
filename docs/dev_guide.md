# C-RAN DRL Thesis — Development Guide

## Environment Setup

### Prerequisites
```bash
# Python 3.10+
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gymnasium
pip install numpy scipy pandas matplotlib seaborn
pip install cvxpy tensorboard wandb
pip install pytest black flake8 mypy

# Optional: Jupyter for prototyping
pip install jupyterlab ipywidgets
```

### Repository Structure
```bash
git init cran-drl-thesis
cd cran-drl-thesis
mkdir -p {cran_env,agents,baselines,training,evaluation,config,data/{traces,results},tests,thesis/{chapters,figures}}
touch {cran_env,agents,baselines,training,evaluation,config,tests}/__init__.py
```

---

## Development Workflow

### Phase 1: Environment (Week 1-2)
**Goal**: A fully functional, tested C-RAN Gymnasium environment

```python
# cran_env/cran_env.py — Core interface
gymnasium.Env
├── reset() → obs, info
├── step(action) → obs, reward, terminated, truncated, info
│   └── action: dict{"rrh_on": ndarray[bool], "power": ndarray[float]}
├── _get_obs() → state vector
├── _compute_reward() → scalar reward
└── render() → optional visualization
```

**Implementation Order**:
1. `channel_model.py` — Path loss, shadowing, small-scale fading
2. `traffic_model.py` — Tidal patterns, burstiness
3. `power_model.py` — EARTH-validated BBU/RRH/fronthaul power
4. `cran_env.py` — Integrate into Gymnasium Env
5. `tests/test_env.py` — Unit tests for each component

**Validation Checkpoints**:
- [ ] SINR calculation matches analytical closed-form for simple cases
- [ ] Power model sums to known values from Al-Zubaedi thesis
- [ ] Traffic model shows realistic 24-hour tidal pattern
- [ ] Environment passes Gymnasium API compliance check

---

### Phase 2: Baselines (Week 3)
**Goal**: Working non-DRL and simple-DRL baselines for comparison

| Baseline | File | Key Implementation Detail |
|----------|------|--------------------------|
| All ON + Uniform | `baselines/all_on_uniform.py` | Fixed action; compute reward |
| Greedy Heuristic | `baselines/greedy_heuristic.py` | Sort UEs by channel gain; activate minimum RRHs |
| NMBS Bin-Packing | `baselines/nmbs_binpack.py` | Modified first-fit decreasing; load-aware switching |
| Convex Power | `baselines/convex_power.py` | CVXPY: minimize Σp subject to SINR ≥ γ_target |
| DDQN (Iqbal) | `agents/ddqn_agent.py` | Hand-rolled PyTorch Double DQN; discrete action = RRH subset |

**Validation Checkpoints**:
- [ ] Convex baseline matches CVXPY reference solution
- [ ] DDQN reproduces Iqbal's ~22% power savings on comparable scenario
- [ ] All baselines use identical environment (fair comparison)

---

### Phase 3: Proposed Method (Week 4-6)
**Goal**: Hybrid SAC-DDQN agent

#### Architecture Specification

```python
# agents/hybrid_sac_dqn.py

class HybridSACDQNAgent:
    """
    Hybrid Actor-Critic for C-RAN Energy Optimization

    Discrete Actor (DDQN-style): Selects RRH activation pattern
    Continuous Actor (SAC): Allocates transmit power per active RRH
    Shared Critic: Evaluates joint (discrete, continuous) action quality
    """

    def __init__(self, state_dim, n_rrh, n_ue, config):
        # Discrete actor: outputs Q-values for each RRH subset
        self.discrete_actor = QNetwork(state_dim, 2**n_rrh)

        # Continuous actor (SAC): outputs power per RRH
        self.continuous_actor = GaussianPolicy(state_dim, n_rrh)

        # Shared critic: Q(s, v, p) where v is discrete, p is continuous
        self.critic1 = HybridQNetwork(state_dim, n_rrh)
        self.critic2 = HybridQNetwork(state_dim, n_rrh)

        # Target networks
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.continuous_actor_target = copy.deepcopy(self.continuous_actor)

    def select_action(self, state, evaluate=False):
        # Discrete: epsilon-greedy or argmax Q
        v = self.discrete_actor.select_action(state, epsilon=self.epsilon)
        # Continuous: sample from Gaussian policy
        p, log_prob = self.continuous_actor.sample(state)
        return {"rrh_on": v, "power": p}

    def update(self, batch):
        # Critic update: minimize Bellman error
        # Actor update: maximize Q (discrete) + maximize Q + entropy (continuous)
        pass
```

#### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Discrete action space | Factorized binary (per-RRH) vs. joint subset | Factorized: tractable for R>10; joint: optimal but exponential |
| Continuous action space | Per-RRH power [0, P_max] | Natural physical interpretation; matches SAC output |
| Critic architecture | Concatenate (state, discrete_action, continuous_action) → MLP | Standard hybrid Q-network; alternative: separate paths then fuse |
| Entropy target | Auto-tuned alpha (SAC) | Better exploration than fixed temperature |
| Experience replay | Shared buffer for hybrid transitions | (s, v, p, r, s') tuples |

**Validation Checkpoints**:
- [ ] Agent runs without errors for 100 episodes
- [ ] Critic loss decreases monotonically (check for bugs if not)
- [ ] Reward improves over random policy within 500 episodes
- [ ] Discrete and continuous components train at comparable rates

---

### Phase 4: Training Infrastructure (Week 6-7)

```python
# training/train_hybrid.py

def train(config):
    env = make_vec_env(lambda: CRANEnv(config), n_envs=config.n_envs)
    agent = HybridSACDQNAgent(env.observation_space.shape[0], 
                               config.n_rrh, config.n_ue, config)

    # W&B logging
    wandb.init(project="cran-drl-thesis", config=config)

    for episode in range(config.max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(config.max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)

            if len(agent.replay_buffer) > config.min_buffer_size:
                losses = agent.update(agent.replay_buffer.sample(config.batch_size))
                wandb.log(losses)

            state = next_state
            episode_reward += reward

            if done:
                break

        wandb.log({"episode_reward": episode_reward, "episode": episode})

        # Evaluation every N episodes
        if episode % config.eval_freq == 0:
            eval_reward = evaluate(agent, config)
            wandb.log({"eval_reward": eval_reward})
            save_checkpoint(agent, episode)
```

**Hyperparameter Configuration** (`config/default.yaml`):
```yaml
# Network
n_rrh: 12
n_ue: 10
n_bbu: 3
bandwidth_ghz: 0.18  # per RB
n_antennas: 4

# Channel
path_loss_exp: 3.5
shadowing_std_db: 8
fading_model: "rayleigh"

# Traffic
traffic_model: "tidal"
peak_hours: [9, 12, 18, 22]
base_demand_mbps: 50

# Power Model (EARTH-aligned)
rrh:
  p_active_w: 6.8
  p_sleep_w: 4.3
  p_switch_w: 3.0
  pa_efficiency: 0.25
bbu:
  p_stat_w: 175.0
  p_dyn_w: 250.0
  delta_p: 0.44
fronthaul:
  pon_type: "twdm"
  p_olt_w: 20.0
  p_onu_active_w: 5.0
  p_onu_sleep_w: 0.5

# DRL
algorithm: "hybrid_sac_dqn"
max_episodes: 5000
max_steps_per_episode: 100
replay_buffer_size: 1000000
batch_size: 256
gamma: 0.99
tau: 0.005
lr_actor: 1e-4
lr_critic: 3e-4
lr_alpha: 1e-4
hidden_dims: [256, 256]

# Exploration
epsilon_start: 1.0
epsilon_end: 0.01
epsilon_decay: 0.995

# Reward weights
alpha_energy: 1.0
beta_qos: 10.0
gamma_switch: 0.5

# Evaluation
eval_freq: 100
n_eval_episodes: 10
n_random_seeds: 5
```

---

### Phase 5: Evaluation and Analysis (Week 8-9)

```python
# evaluation/convergence.py

def plot_convergence(results_dir, algorithms):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for algo in algorithms:
        rewards = load_results(results_dir, algo)
        mean = rewards.mean(axis=0)
        std = rewards.std(axis=0)

        axes[0,0].plot(mean, label=algo)
        axes[0,0].fill_between(range(len(mean)), mean-std, mean+std, alpha=0.2)

    axes[0,0].set_xlabel("Episode")
    axes[0,0].set_ylabel("Episode Reward")
    axes[0,0].set_title("Convergence Comparison")
    axes[0,0].legend()
    axes[0,0].grid(True)

    plt.tight_layout()
    plt.savefig("thesis/figures/convergence.pdf", dpi=300)

# evaluation/ablation.py

def ablation_study(base_config):
    variants = [
        ("Full", base_config),
        ("No Switching Cost", {**base_config, "gamma_switch": 0}),
        ("No Fronthaul Power", {**base_config, "fronthaul_weight": 0}),
        ("No QoS Penalty", {**base_config, "beta_qos": 0}),
    ]

    results = {}
    for name, cfg in variants:
        results[name] = train_and_evaluate(cfg)

    plot_ablation(results)
    return results

# evaluation/scalability.py

def scalability_analysis():
    scenarios = [
        {"n_rrh": 5, "n_ue": 2, "label": "Small"},
        {"n_rrh": 12, "n_ue": 10, "label": "Medium"},
        {"n_rrh": 20, "n_ue": 20, "label": "Large"},
        {"n_rrh": 50, "n_ue": 50, "label": "Very Large"},
    ]

    for scenario in scenarios:
        config = load_config("default.yaml")
        config.update(scenario)
        results = train_and_evaluate(config)
        save_results(results, f"scalability_{scenario['label']}.pkl")
```

---

## Code Quality Standards

### Style
- **Black** for formatting: `black cran_env/ agents/ baselines/ training/ evaluation/`
- **Flake8** for linting: `flake8 --max-line-length=100`
- **Type hints** for all function signatures
- **Docstrings** for all public methods (Google style)

### Testing
```python
# tests/test_env.py
import pytest
import numpy as np
from cran_env import CRANEnv

class TestCRANEnv:
    def test_reset_returns_valid_obs(self):
        env = CRANEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (env.state_dim,)
        assert np.isfinite(obs).all()

    def test_power_model_matches_earth(self):
        env = CRANEnv()
        assert env.bbu.p_stat == pytest.approx(175.0, rel=1e-2)
        assert env.bbu.p_dyn == pytest.approx(250.0, rel=1e-2)

    def test_sinr_calculation(self):
        env = CRANEnv(n_rrh=1, n_ue=1)
        env.reset(seed=42)
        h = env.channel_gains[0, 0]
        p = 1.0
        sigma2 = env.noise_power
        expected_sinr = np.abs(h)**2 * p / sigma2
        assert env.compute_sinr(0, 0, np.array([p])) == pytest.approx(expected_sinr)
```

---

## Git Workflow

```bash
# Branch structure
main                    # Stable, thesis-ready code
develop                 # Integration branch
feature/env-channel     # Environment components
feature/baselines       # Baseline implementations  
feature/hybrid-agent    # Proposed method
feature/evaluation      # Analysis and plotting
experiment/scalability  # Long-running experiments

# Commit messages
feat(env): add Rayleigh fading channel model
fix(power): correct BBU static power to 175W per EARTH model
docs(thesis): add MDP formulation to Ch. 3.4
refactor(agent): extract shared critic base class
experiment: run scalability analysis for R=50, U=50

# Tags
thesis-v0.1-env-complete
thesis-v0.2-baselines-complete  
thesis-v0.3-hybrid-agent-stable
thesis-v1.0-final-results
```

---

## Experiment Tracking

### Weights and Biases Setup
```python
import wandb

def init_wandb(config, group=None):
    wandb.init(
        project="cran-drl-thesis",
        group=group or config.algorithm,
        config=config,
        tags=[config.algorithm, f"R{config.n_rrh}_U{config.n_ue}"],
    )
```

---

## Debugging Checklist

### Agent Not Learning
- [ ] Reward scale: Is reward magnitude reasonable (-100 to 0)?
- [ ] Observation normalization: Are states standardized?
- [ ] Gradient flow: Check for vanishing/exploding gradients
- [ ] Exploration: Is epsilon-decay too fast? Is noise scale appropriate?
- [ ] Replay buffer: Is it filling before training starts?
- [ ] Target network update: Is tau too large/small?

### Unstable Training
- [ ] Learning rate: Try 10x smaller for actor
- [ ] Batch size: Increase to 512+ for continuous actions
- [ ] Gradient clipping: Add max_norm=1.0 to critic updates
- [ ] Reward clipping: Clip to [-10, 0] to prevent extreme values
- [ ] Network initialization: Use orthogonal init for actor

### Baseline Beats Proposed Method
- [ ] Fair comparison: Same environment, same seeds, same evaluation protocol
- [ ] Training duration: Is proposed method still converging?
- [ ] Hyperparameters: Did you sweep LR, batch size, network size?
- [ ] Architecture bug: Is discrete head actually training? Check gradients.
- [ ] Reward design: Is the reward too sparse? Add shaping.

---

## Performance Benchmarks

Target metrics for final thesis:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Energy reduction vs. All ON | >=25% | Average over 24-hour traffic cycle |
| Energy reduction vs. Iqbal DDQN | >=5% | Same scenario, same traffic |
| QoS violation rate | <=5% | Fraction of UEs below SINR target |
| Convergence episodes | <=3000 | Reward within 5% of final value |
| Training time (Medium network) | <=48 hours | Single GPU (RTX 3090 or equivalent) |
| Inference latency | <=10ms | Per-decision time on CPU |
