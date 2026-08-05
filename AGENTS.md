# C-RAN Energy Optimization Thesis — Antigravity Agent Context

## Project Identity
**Title**: Optimization of Energy Efficient Cloud Radio Access Network for 5G Using Deep Deterministic Policy Gradient Algorithm  
**Candidate**: Gabriel Kwame Freeman  
**Degree**: MPhil (Master of Philosophy)  
**Institution**: [University Name]  
**Supervisor**: [Supervisor Name]  
**Version**: 1.0 (July 2026)

---

## Thesis Status

| Chapter | Status | Word Count | Completeness |
|---------|--------|------------|--------------|
| Ch. 1: Introduction | ✅ Drafted | ~800 | 80% |
| Ch. 2: Literature Review | ✅ Drafted | ~2,500 | 60% |
| Ch. 3: System Model & Problem Formulation | ⚠️ Partial | ~4,200 | 50% |
| Ch. 4: Simulation Results | ❌ Missing | 0 | 0% |
| Ch. 5: Conclusion & Future Work | ❌ Missing | 0 | 0% |

**Critical Gap**: Chapter 3.7 (DRL Framework) describes generic DDPG mechanics but lacks:
- Formal MDP definition (state, action, reward, transition)
- Hybrid discrete-continuous action space handling
- Validated power model parameters
- Baseline algorithm specifications

---

## Core Research Question

> How can Deep Reinforcement Learning be applied to optimize joint Remote Radio Head (RRH) activation and transmit power allocation in 5G C-RAN, balancing energy efficiency against Quality of Service constraints?

**Refined Sub-questions**:
1. What is the optimal DRL architecture for hybrid discrete (RRH on/off) and continuous (power) action spaces?
2. How should the reward function incorporate switching costs and fronthaul power?
3. How does the proposed method compare against two-stage (DDQN + convex) approaches?

---

## Contribution Claim (To Be Defended)

**Primary**: A hybrid Actor-Critic framework combining discrete action selection (DDQN-style) for RRH on/off decisions with continuous policy optimization (SAC) for transmit power allocation, with a fronthaul-aware reward function.

**Secondary**:
- Validated power model aligned with EARTH/GreenTouch standards
- Comprehensive baseline comparison (heuristic, convex, pure discrete RL, pure continuous RL)
- Scalability analysis across network sizes

---

## Foundational References

| Reference | Role | Key Insight |
|-----------|------|-------------|
| Fathy et al. (2021) | Primary | ANN pre-processing + Bi-Section GSBF convex optimization |
| Iqbal et al. (2021) | Primary | DDQN for RRH on/off + convex power allocation; 22% power savings |
| Al-Zubaedi (2019) | Primary | Comprehensive C-RAN power model; NMBS bin-packing; TWDM-PON |
| Bordin et al. (2025) | Recent | DRL (PPO/DQN) for Open RAN RF-frontend energy saving, via `ns-O-RAN`; arXiv:2410.14021 |
| Shengren et al. (2022) | Methodology | DRL benchmark: SAC > TD3 > DDPG in stability |
| Frontiers (2026) | Validation | Hybrid DDPG+DDQL for ABS-assisted B5G networks |
| Xiong et al. (2018) / Bester et al. (2019) | Core architecture | P-DQN / MP-DQN — parameterized discrete-continuous action coupling, corrected for cross-talk |
| Tavakoli et al. (2018) | Core architecture | Branching (BDQ) — 2R not 2^R discrete output growth |
| Li et al. (2022) | Considered, not adopted | HyAR — VAE-based hybrid action representation; arXiv:2109.05490 |
| HySoft (2025) | Considered, not adopted | Max-entropy P-DQN/MP-DQN extension; **authorship unverified**, see Concept Note v4.0 §4.2 |
| Lu, Yan & Zeng (2026) | Closest related work | EExApp — dual-actor-dual-critic PPO+GAT, RU sleep + DU slicing, real O-RAN testbed; arXiv:2602.09206. Explicitly differentiated (single coupled network vs. two actor-critic pairs), see Concept Note v4.0 §4.4 |
| Liang et al. (2026) | Recent (O-RAN) | Federated TD3, rApp aggregator + xApp agents; arXiv:2604.00201. Cited as the justification for this thesis's single-agent scope (§7.1) |
| Sthankiya et al. (2024) | Recent (O-RAN) | Survey: AI-driven energy optimization in NG-RAN; arXiv:2411.02164 |
| Sohaib et al. (2024) | Recent (O-RAN) | DRL transfer learning for cloud-native O-RAN, eMBB+URLLC; arXiv:2407.11563 |
| Chuang et al. (2025) | Recent (adjacent) | Hybrid A3C + Dueling DQN in a 5G C-RAN — different problem (industrial IoT scheduling), engaged per supervisor review B1 |

---

## Methodological Pivot (Post-Critique)

> **Superseded**: the SAC-DDQN design below was the first hybrid revision. It was itself superseded by the branching/multi-pass (MP-DQN)/twin-critic architecture in `manuscript/MPhil_Thesis_Concept_Note_v2.md` §10, which `manuscript/MPhil_Thesis_Concept_Note_v3.md` §10 keeps and further specifies (critic diagram, combinatorial-action-space handling) in response to supervisor review. This section is kept for history; `agents/hybrid_sac_dqn.py` is the "Alternative hybrid" per the Code Architecture list below, not the proposed method.

**Original Plan**: Vanilla DDPG for joint discrete-continuous control  
**Revised Plan (superseded, see note above)**: Hybrid SAC-DDQN with:
- Discrete actor (DDQN) → RRH activation vector `v ∈ {0,1}^R` trained via MSE loss against Bellman Q-targets derived from the shared twin critic and discrete target network (`discrete_actor_target`).
- Continuous actor (SAC) → Power allocation `p ∈ [0, P_max]^R` using `tanh` squashed Gaussian policy with exact Jacobian log-probability correction.
- Shared twin critic evaluating joint discrete-continuous policy $Q(s, v, p)$.
- Environment interference model → Multi-cell downlink user association where each UE is served by its strongest active RRH and uncoordinated active RRHs induce co-channel interference $I_u = \sum_{r \neq r^*(u)} P_r |h_{r,u}|^2$.
- Fronthaul power (TWDM-PON model) & switching costs integrated into reward.

**Justification**: DDPG alone cannot naturally handle binary decisions without thresholding (destroys gradients). SAC provides superior stability and sample efficiency for continuous variables, while Double-DQN with explicit Bellman targets prevents Q-value divergence on factorized discrete heads.

---

## Code Architecture

```
Gabriel/
├── .agents/               # Antigravity agent skills, agents & guidelines
│   ├── AGENTS.md
│   ├── skills/            # Antigravity skills
│   └── agents/            # Specialized subagent definitions
├── AGENTS.md              # Project identity & developer context
├── cran_env/              # C-RAN Gymnasium environment
│   ├── __init__.py
│   ├── cran_env.py        # Main MDP environment
│   ├── channel_model.py   # Rayleigh fading, path loss
│   ├── traffic_model.py   # Tidal traffic patterns
│   └── power_model.py     # EARTH-validated power consumption
├── agents/                # DRL algorithms
│   ├── branching_mp_dqn.py# Proposed: Branching MP-DQN + TD3 (Concept Note v2.0)
│   ├── hybrid_sac_dqn.py  # Alternative hybrid
│   ├── sac_agent.py       # Baseline: pure SAC
│   ├── td3_agent.py       # Baseline: TD3
│   ├── ddpg_agent.py      # Baseline: Pure DDPG continuous relaxation (RQ3 ablation)
│   └── ddqn_agent.py      # Baseline: discrete only
├── baselines/             # 7-Baseline comparison suite
│   ├── all_on_uniform.py  # Baseline 1: All-ON uniform power
│   ├── greedy_heuristic.py# Baseline 2: Greedy heuristic
│   ├── nmbs_binpack.py    # Baseline 3: Al-Zubaedi's NMBS bin-packing
│   ├── convex_power.py    # Baseline 4: CVXPY SOCP power allocation
│   ├── ddqn_socp.py       # Baseline 5: Two-stage DDQN + SOCP (Iqbal et al., 2021)
│   └── ann_gsbf.py        # Baseline 6: Supervised ANN + GSBF (Fathy et al., 2021)
├── training/              # Training loops
│   ├── train_hybrid.py
│   ├── train_baselines.py
│   ├── run_extended_sweeps.py
│   └── hyperparam_search.py
├── evaluation/            # Analysis and plotting
│   ├── convergence.py
│   ├── ablation.py        # RQ3 discrete vs continuous relaxation ablation
│   ├── scalability.py     # 5-to-50 RRH scalability sweep
│   └── plot_utils.py
├── config/                # Experiment configurations
│   ├── default.yaml
│   ├── small_network.yaml # 5 RRH scenario
│   └── large_network.yaml # 50 RRH scalability scenario
├── data/                  # Traffic traces, results
│   ├── traces/
│   └── results/
├── tests/                 # Unit tests
└── thesis/                # LaTeX source
    ├── chapters/
    ├── figures/
    └── main.tex
```

---

## Quality Gates

Before any chapter is considered complete, it must pass:

1. **Mathematical Consistency**: All equations numbered, all variables defined, dimensions consistent
2. **Reference Alignment**: Every claim backed by citation; no unsubstantiated novelty claims
3. **Reproducibility**: All simulation parameters in config files; random seeds fixed
4. **Baseline Comparison**: Every result compared against at least 2 baselines
5. **Statistical Significance**: All results averaged over ≥5 random seeds with confidence intervals
6. **Code-Text Consistency**: Equations in thesis match implementation exactly

---

## Development Philosophy

- **Evidence over intuition**: Every design choice (algorithm, parameter, architecture) must be justified by reference or ablation
- **Baseline first**: Implement baselines before proposed method to ensure fair comparison
- **Incremental validation**: Test each component (env, agent, reward) in isolation before integration
- **Version everything**: Git commits for every significant change; experiments logged with W&B
- **Write as you go**: Document methodology in thesis text concurrently with code implementation

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Hybrid algorithm instability | Medium | High | Start with proven SAC; add discrete head gradually |
| Insufficient training time | High | High | Use smaller networks first; leverage GPU cluster |
| Baseline implementation bugs | Medium | High | Unit test each baseline; validate against published results |
| Parameter validation rejection | Medium | Medium | Document all sources; include sensitivity analysis |
| Scope creep (multi-agent, etc.) | High | Medium | Strictly bound to single-agent, downlink, single pool |

---

## Communication Protocol

When working on this thesis:
1. Always reference specific chapter/section numbers
2. Distinguish between: (a) what's written, (b) what's implemented, (c) what's planned
3. Flag any deviation from the revised hybrid approach
4. Cite sources for all technical claims
5. Maintain traceability: code ↔ equations ↔ text
