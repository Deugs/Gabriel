# C-RAN DRL Thesis — Development Guide

> **Status note (sixth audit round)**: this file was written as a week-by-week development *roadmap* before implementation began. Phases 1–4 below are now complete (see `README.md`'s "Immediate Next Steps" for what's actually still outstanding), and Phase 3's architecture spec/Phase 4's config block described the superseded v1.0 "Hybrid SAC-DDQN" design and a flat config schema that never matched the real nested `config/default.yaml`. Both have been corrected below to point at the actual implementation rather than duplicate it (duplication is exactly how this drift happened) — for the authoritative, current architecture spec, read `docs/skills/skill_hybrid_agent.md` and `agents/branching_mp_dqn.py` directly.

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

# Required before running any of the commands below (or add to your shell
# profile / venv activate script): there's no setup.py/pyproject.toml, so
# `python training/train_hybrid.py` etc. won't find the top-level packages
# (agents/, cran_env/, ...) without the repo root on PYTHONPATH.
export PYTHONPATH="$(pwd):$PYTHONPATH"
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
- [ ] DDQN reproduces Iqbal et al.'s *qualitative* behavior (outperforming All-ON) at their exact studied scenarios — see `tests/test_baseline_paper_scenarios.py`. A specific reported power-savings percentage is deliberately NOT used as a pass/fail gate: no primary-source access to the paper is available in this environment to verify it, and treating an unverifiable number as a checkpoint would itself violate the Ethical AI Rule (`docs/rules.md` §10) this checklist is meant to uphold.
- [ ] All baselines use identical environment (fair comparison)

---

### Phase 3: Proposed Method (Week 4-6)
**Goal**: Branching MP-DQN + TD3 agent (`agents/branching_mp_dqn.py`) — the discrete-actor/continuous-SAC-actor/shared-critic design originally sketched here (`agents/hybrid_sac_dqn.py`) was superseded early in implementation by a branching, multi-pass, twin-critic parameterized DQN; see `docs/skills/skill_hybrid_agent.md` for the current, authoritative architecture spec (network diagrams, class-by-class breakdown, training algorithm, hyperparameters) rather than duplicating it here.

**Key architectural differences from the original sketch above**: one shared encoder feeding both a coupled continuous-parameter network (P-DQN) and R independent per-RRH dueling discrete branch heads (Tavakoli et al., 2018) — not a separate discrete actor / continuous SAC actor arbitrated by a shared critic; MP-DQN multi-pass masking (Bester et al., 2019) to prevent parameter cross-talk between branches; TD3 twin critics + delayed policy update + target-policy smoothing, not SAC's entropy-regularized objective.

**Validation Checkpoints**:
- [ ] Agent runs without errors for 100 episodes
- [ ] Critic loss decreases over the first ~1000 updates
- [ ] Reward improves over random policy within 500 episodes
- [ ] Multi-pass masking genuinely produces different Q-values for candidate parameter vectors differing only in an unrelated RRH's parameters (cross-talk absent) — see `docs/skills/skill_hybrid_agent.md`'s Validation Checklist for the complete list

---

### Phase 4: Training Infrastructure (Week 6-7)

The actual training entrypoint is `training/train_hybrid.py::train_hybrid_agent()` — read that file directly rather than this sketch, which described a `make_vec_env(..., n_envs=...)` vectorized-environment setup that was never built (every training loop in this codebase is a single synchronous `CRANEnv`; `hardware.n_envs`/`num_workers` in `config/default.yaml` are explicitly documented there as aspirational/reserved, not implemented). The real loop is a plain per-episode `for` loop over `env.reset()`/`env.step()`, with `agent.update()` called once per environment step and `agent.decay_exploration()` called once per episode (epsilon/continuous-noise decay is a per-episode rate, not per-step).

**Hyperparameter Configuration** (`config/default.yaml`): the real schema is nested by section (`network:`, `channel:`, `traffic:`, `power:` with `rrh:`/`bbu:`/`fronthaul:` sub-sections, `algorithm:`, `reward:`, `evaluation:`, `logging:`, `hardware:`) — read `config/default.yaml` directly rather than this guide's copy, which used a flat, unnested schema (`n_rrh:` at the top level, `bandwidth_ghz`, `n_antennas`, `traffic_model: "tidal"`, `algorithm: "hybrid_sac_dqn"`) that never matched any real config file and would silently fail to configure anything if used as written. Every key under `config/default.yaml`'s `algorithm:`/`evaluation:`/`logging:`/`hardware:` blocks is genuinely read by `BranchingMPDQN`/`training/train_hybrid.py` via a `get_val()`-style helper, except the explicitly aspirational ones noted in that file's own comments.

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

Target metrics for final thesis. Per Concept Note v4.0 §5.2/G10, the headline comparison is the margin over DDQN/P-DQN/MP-DQN (RQ4) — the All-ON figure below is a **sanity-check floor** any working method is expected to clear comfortably, not a contribution reported in its own right:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Energy reduction vs. All ON (sanity-check floor, not a headline result) | >=25% | Average over 24-hour traffic cycle |
| Energy reduction vs. Iqbal DDQN / P-DQN / MP-DQN (headline comparison) | >=5% | Same scenario, same traffic |
| QoS violation rate | <=5% | Fraction of UEs below SINR target |
| Convergence episodes | <=3000 | Reward within 5% of final value |
| Training time (Medium network) | <=48 hours | Single GPU (RTX 3090 or equivalent) |
| Inference latency | <=10ms | Per-decision time on CPU |
