# C-RAN DRL Thesis — Writing and Structure Guide

## Document Specification

- **Format**: LaTeX (IEEEtran or university template)
- **Citation Style**: IEEE numbered [1], [2], ...
- **Target Length**: 15,000-25,000 words (MPhil standard)
- **Figures**: Vector graphics (PDF) via Matplotlib; all fonts embedded
- **Tables**: Booktabs style; no vertical rules
- **Equations**: Numbered sequentially per chapter; all variables defined at first use

---

## Chapter-by-Chapter Writing Guide

### Chapter 1: Introduction (~1,500 words)

**Current Status**: Draft exists (~800 words) — needs expansion

**Required Sections**:
1. **Background** (400 words): 5G evolution, C-RAN architecture, energy challenge
2. **Problem Statement** (300 words): Static methods fail; need adaptive DRL
3. **Research Questions** (200 words): 3 specific, answerable questions
4. **Objectives** (200 words): Measurable targets
5. **Significance** (200 words): Why this matters (OPEX, sustainability, 6G extension)
6. **Scope and Limitations** (200 words): Downlink only, single pool, single agent

**Key Revision**: Replace vague DRL claims with specific hybrid approach. Current text says "DDPG" throughout — must be updated to "Hybrid SAC-DDQN" or similar.

**Opening Paragraph Template**:
> The exponential growth of mobile data traffic, projected to reach [X] exabytes monthly by 2026, has intensified the energy consumption crisis in wireless networks. Cloud Radio Access Networks (C-RAN), which centralize baseband processing in BBU pools while distributing low-cost Remote Radio Heads (RRHs), offer a promising architecture for 5G and beyond. However, the continuous operation of densely deployed RRHs and high-capacity fronthaul links significantly increases energy expenditure, with the RAN accounting for 57-80% of total network power consumption [cite]. This thesis proposes a hybrid Deep Reinforcement Learning framework that jointly optimizes discrete RRH activation decisions and continuous transmit power allocation to maximize energy efficiency while guaranteeing Quality of Service.

---

### Chapter 2: Literature Review (~4,000 words)

**Current Status**: Draft exists (~2,500 words) — needs significant expansion and restructuring

**Proposed Structure**:

#### 2.1 C-RAN Architecture and Energy Challenges (~800 words)
- Historical evolution: IBM WNC (2010) -> China Mobile C-RAN (2011) -> 5G NR
- Architecture types: Full, partial, hybrid centralization
- Fronthaul technologies: TWDM-PON, EPON, AON — power implications
- Energy breakdown: RRH (40-60%), BBU pool (20-30%), fronthaul (10-20%)

#### 2.2 Traditional Optimization Methods (~600 words)
- Convex optimization: GSBF, weighted MMSE
- Heuristics: Greedy, genetic algorithms
- Bin-packing: NMBS (Al-Zubaedi)
- Limitations: Require perfect CSI, high computational cost, non-adaptive

#### 2.3 Reinforcement Learning for Wireless Networks (~800 words)
- Q-learning -> DQN -> DDQN -> Dueling DQN (discrete actions)
- Policy gradients: REINFORCE, A2C, A3C
- Actor-Critic: DDPG, TD3, SAC (continuous actions)
- Multi-agent RL: MADDPG, QMIX
- Applications: BS sleeping, power control, resource allocation, caching

#### 2.4 DRL for C-RAN Energy Optimization (~1,200 words)
- **Fathy et al. (2021)**: ANN + Bi-Section GSBF — ANN reduces search space, convex solver handles power
- **Iqbal et al. (2021)**: DDQN for RRH on/off + convex power allocation — 22% power savings, switching costs
- **Al-Zubaedi (2019)**: Comprehensive power model, NMBS bin-packing, TWDM-PON analysis
- **Recent advances (2023-2026)**: 
  - Hybrid DRL for ABS-assisted networks (Frontiers 2026)
  - Open RAN energy saving with DRL (Bordin 2025)
  - SAC vs. DDPG benchmarks (Shengren 2022)
  - 5G-Advanced cell DTX/DRX features (5G Americas 2025)

#### 2.5 Research Gap and Motivation (~600 words)
- Gap 1: No hybrid discrete-continuous DRL for joint RRH-power optimization
- Gap 2: Fronthaul power often neglected in DRL reward functions
- Gap 3: Insufficient baseline comparison across algorithm families
- Gap 4: Scalability analysis missing in most works

**Critical Addition**: The critique document correctly notes your literature review lacks post-2021 references. Add at minimum:
- [5] Bordin et al., 2025 — Open RAN DRL
- [6] Shengren et al., 2022 — DRL benchmark study
- [7] Frontiers 2026 — Hybrid DDPG+DDQL for B5G
- [8] 5G Americas 2025 — 5G-Advanced energy features

---

### Chapter 3: System Model and Problem Formulation (~5,000 words)

**Current Status**: Partial (~4,200 words) — needs major restructuring

**Critical Reorganization**:

#### 3.1 Network Architecture (~600 words) — KEEP CURRENT
- RRH set R, UE set U, BBU set B
- Fronthaul: PON model (TWDM-PON preferred)
- Orchestra Server concept — CLARIFY: Is this distinct from BBU pool controller? If not, merge with BBU pool description.

#### 3.2 Channel Model (~500 words) — KEEP, ADD DETAIL
- Path loss: PL(d) = PL0 + 10n log10(d/d0)
- Shadowing: log-normal, sigma = 8 dB
- Small-scale fading: Rayleigh (no LoS) or Rician (LoS)
- Channel matrix H(t) in C^(R x U)
- SINR equation (keep current Eq. 3.2)
- Shannon capacity (keep current Eq. 3.3)

#### 3.3 Traffic Model (~400 words) — ADD
- Tidal traffic pattern: sinusoidal variation over 24 hours
- Peak hours: business (9-12, 14-17), residential (19-23)
- Burstiness: Poisson arrivals with time-varying rate
- Alternative: Real traces (China Mobile, 3GPP TR 38.913)

#### 3.4 Power Consumption Model (~800 words) — FIX PARAMETERS

**RRH Power** (Eq. 3.6-3.7 in current draft):
```
P_RRH(t) = sum_{r in M(t)} [P_active + (1/eta) sum_u |w_{r,u}|^2]
         + sum_{r in N(t)} P_sleep
         + sum_{r in S(t)} P_switch
```
Parameters MUST be:
- P_active = 6.8 W (matches Fathy)
- P_sleep = 4.3 W (matches Fathy)
- P_switch = 3.0 W (per transition)
- eta = 0.25 (PA efficiency)

**BBU Pool Power** (Eq. 3.9-3.12):
```
P_BBU(t) = sum_{j in B_active} [P_stat + DeltaP * (P_CPU + P_MEM + P_IO) * rho_j(t)]
```
Parameters MUST be (EARTH model, Al-Zubaedi):
- P_stat = 175 W (NOT 100 W)
- P_dyn = 250 W total (NOT decomposed arbitrarily)
- DeltaP = 0.44
- rho_j(t) = load fraction (0 to 1)

**Fronthaul Power** (Eq. 3.7-3.8):
```
P_FH(t) = P_OLT + sum_{k in K_active} P_LC,k + sum_{r in M(t)} P_ONU,active + sum_{r in N(t)} P_ONU,sleep
```
Parameters for TWDM-PON:
- P_OLT = 20 W
- P_LC = 10 W per active line card
- P_ONU,active = 5 W
- P_ONU,sleep = 0.5 W

**Total Power**:
```
P_total(t) = P_RRH(t) + P_BBU(t) + P_FH(t)
```

#### 3.5 MDP Formulation (~1,200 words) — CRITICAL ADDITION

This section is **entirely missing** from your draft. It is **mandatory** for any RL thesis.

**State Space**:
```
s_t = [h_{1,1}(t), ..., h_{R,U}(t),    # Channel gains (flattened)
       v_1(t), ..., v_R(t),             # RRH activation status
       D_1(t), ..., D_U(t),            # UE traffic demands
       P_total(t-1),                   # Previous total power
       t_hour] in R^{R*U + R + U + 2}   # Time of day (0-23)
```

**Action Space**:
```
a_t = (v(t+1), p(t))
where v(t+1) in {0,1}^R    # Discrete: RRH on/off decisions
      p(t) in [0, P_max]^R  # Continuous: transmit power per RRH
```

**Reward Function**:
```
r_t = -alpha * P_total(t) 
      - beta * sum_{u=1}^U max(0, D_u(t) - C_u(t)) 
      - gamma * sum_{r=1}^R |v_r(t) - v_r(t-1)| * P_switch
```

Where:
- alpha: energy weight (normalize by max power)
- beta: QoS violation penalty (must dominate if QoS is hard constraint)
- gamma: switching cost weight (prevents oscillation)

**Transition Dynamics**:
```
P(s_{t+1} | s_t, a_t) = P(H(t+1)|H(t)) * P(D(t+1)|D(t), t_hour)
```
- Channel: First-order Gauss-Markov: H(t+1) = rho*H(t) + sqrt(1-rho^2)*W(t)
- Traffic: Deterministic tidal pattern or stochastic process

**Discussion of Hybrid Action Space**:
> The joint optimization of RRH activation (discrete) and power allocation (continuous) constitutes a hybrid action space problem. Standard DRL algorithms are designed for either purely discrete (DQN family) or purely continuous (DDPG, SAC, TD3) spaces. This motivates our proposed hybrid architecture, which combines a discrete action selector for RRH states with a continuous policy optimizer for power levels, coordinated through a shared critic.

#### 3.6 Problem Formulation as Optimization (~600 words)

**Optimization Problem**:
```
maximize_{pi} E[sum_{t=0}^T gamma^t r_t]
subject to:
  C_u(t) >= D_u(t), for all u, t          # QoS constraint
  0 <= p_r(t) <= P_max, for all r, t      # Power constraint
  v_r(t) in {0,1}, for all r, t            # Binary constraint
  sum_{r in R} v_r(t) >= R_min, for all t  # Coverage constraint
```

**Why RL not Convex?**:
- Channel and traffic are stochastic and non-stationary
- Optimal policy is time-varying and history-dependent
- Model-free RL avoids need for explicit transition model

#### 3.7 Proposed Hybrid DRL Algorithm (~1,000 words) — COMPLETE REWRITE

**Replace generic DDPG description with**:

1. **Algorithm Selection Justification**:
   - DDPG: Poor stability, cannot handle discrete actions
   - TD3: Better than DDPG but still continuous-only
   - SAC: Superior sample efficiency, entropy regularization, continuous-only
   - **Our approach**: Hybrid SAC-DDQN with shared critic

2. **Architecture**:
   - Discrete Actor: DQN-style network outputting Q-values for each RRH binary decision
   - Continuous Actor: SAC Gaussian policy for power allocation
   - Shared Critic: Twin Q-networks evaluating joint (discrete, continuous) actions
   - Target networks for stability

3. **Training Procedure**:
   - Experience replay buffer stores (s, v, p, r, s') tuples
   - Critic minimizes Bellman error
   - Discrete actor maximizes Q via epsilon-greedy
   - Continuous actor maximizes Q + entropy via reparameterization trick

4. **Pseudocode** (formal algorithm box):
```
Algorithm 1: Hybrid SAC-DDQN for C-RAN Energy Optimization
Input: Initial network parameters theta_Q1, theta_Q2, theta_pi, theta_v
       Replay buffer D, target update rate tau

for episode = 1 to N_episodes do
    s_0 <- env.reset()
    for t = 0 to T-1 do
        # Discrete action selection
        v_t <- epsilon-greedy(Q_v(s_t, *)) 
        # Continuous action selection
        p_t ~ pi(*|s_t) + noise
        a_t <- (v_t, p_t)

        s_{t+1}, r_t, done <- env.step(a_t)
        D <- D union {(s_t, v_t, p_t, r_t, s_{t+1})}

        # Sample minibatch
        B ~ Uniform(D)

        # Critic update
        y_i <- r_i + gamma * min(Q'_1, Q'_2)(s_{i+1}, v_{i+1}, p_{i+1})
        Update theta_Q1, theta_Q2 to minimize sum (y_i - Q(s_i, v_i, p_i))^2

        # Discrete actor update
        Update theta_v to maximize Q(s_i, v_i, p_i)

        # Continuous actor update
        Update theta_pi to maximize Q(s_i, v_i, p_i) + alpha*H(pi(*|s_i))

        # Target network soft update
        theta' <- tau*theta + (1-tau)*theta'
    end for
end for
```

---

### Chapter 4: Simulation Results (~4,000 words) — WRITE ENTIRELY

**Section Structure**:

#### 4.1 Simulation Setup (~600 words)
- Hardware: GPU model, CPU, RAM
- Software: Python version, PyTorch version, key libraries
- Network scenarios: Small (R=5, U=2), Medium (R=12, U=10), Large (R=20, U=20)
- Traffic model parameters
- Hyperparameter table (all values used)

#### 4.2 Convergence Analysis (~800 words)
- Learning curves: reward vs. episodes for all algorithms
- Confidence intervals (shaded regions, n=5 seeds)
- Discussion: Why does SAC converge faster than DDPG? Why does hybrid outperform pure continuous?

#### 4.3 Energy Efficiency Comparison (~800 words)
- Bar chart: Total energy consumption (24-hour average) for all methods
- Table: Percentage savings vs. All ON baseline
- Discussion: Impact of switching costs, fronthaul power inclusion

#### 4.4 QoS Performance (~600 words)
- CDF of SINR per UE
- QoS violation rate over time
- Trade-off curve: energy vs. QoS violation rate

#### 4.5 Ablation Study (~600 words)
- Remove switching cost from reward
- Remove fronthaul power from reward
- Remove QoS penalty from reward
- Discussion: Each component's contribution

#### 4.6 Scalability Analysis (~600 words)
- Performance vs. network size (R = 5, 12, 20, 50)
- Training time vs. network size
- Discussion: Computational complexity, real-time feasibility

**Required Figures** (minimum):
1. Fig 4.1: Convergence curves (all algorithms, 5 seeds)
2. Fig 4.2: 24-hour energy profile comparison
3. Fig 4.3: CDF of per-UE SINR
4. Fig 4.4: Ablation study bar chart
5. Fig 4.5: Scalability: energy vs. network size
6. Fig 4.6: Scalability: training time vs. network size

**Required Tables**:
1. Table 4.1: Simulation parameters
2. Table 4.2: Energy savings comparison (all methods)
3. Table 4.3: QoS metrics comparison
4. Table 4.4: Ablation study results
5. Table 4.5: Scalability results

---

### Chapter 5: Conclusion and Future Work (~1,500 words) — WRITE ENTIRELY

**Structure**:
1. **Summary of Contributions** (400 words):
   - Hybrid SAC-DDQN architecture for discrete-continuous C-RAN control
   - Fronthaul-aware reward function
   - Comprehensive baseline comparison and scalability analysis

2. **Key Findings** (500 words):
   - Hybrid approach achieves X% energy savings vs. Y baseline
   - Switching costs account for Z% of total power — cannot be neglected
   - SAC stability superior to DDPG; converges in W episodes vs. DDPG's V
   - Scalable to R=50 RRHs with acceptable training time

3. **Limitations** (300 words):
   - Single-agent, single-pool assumption
   - Perfect CSI (no uncertainty quantification)
   - Simulation-only (no real-world validation)
   - Downlink only

4. **Future Work** (300 words):
   - Multi-agent DRL for distributed BBU pools
   - Imperfect CSI with Bayesian RL or robust optimization
   - Online learning for non-stationary traffic
   - Integration with 3GPP standards (cell DTX/DRX)
   - Real-world testbed validation

---

## Writing Quality Standards

### Equation Formatting
- All equations numbered: (3.1), (3.2), ...
- All variables defined at first use: "where P_total(t) is the total power consumption at time t"
- Consistent notation: vectors bold (v), matrices bold uppercase (H), scalars italic (p)
- Dimensions specified: "H(t) in C^{R x U} is the channel matrix"

### Figure Standards
- All figures in PDF format (vector graphics)
- Font size >= 8pt in figures
- Consistent color scheme across all figures
- Error bars or confidence intervals on all quantitative plots
- Captions: "Figure X.Y: [What is shown]. [Key observation]."

### Table Standards
- Booktabs style: top/mid/bottom rules
- Units in column headers, not body
- Bold best results
- Footnotes for parameter sources

### Citation Standards
- Every claim backed by citation
- No uncited assertions of novelty
- Distinguish between: "we propose" (our work) vs. "it has been shown" (prior work)
- Cross-reference related works: "Unlike Iqbal et al. [6], who use DDQN for discrete actions only, we..."

---

## Revision Checklist (Per Chapter)

Before declaring a chapter complete:

- [ ] All equations numbered and referenced
- [ ] All variables defined at first use
- [ ] All figures have captions and are referenced in text
- [ ] All tables have captions and are referenced in text
- [ ] All claims have citations
- [ ] Novelty claims are defensible and differentiated from prior work
- [ ] Code matches equations exactly (verified by inspection)
- [ ] Grammar and spelling checked
- [ ] Peer review by supervisor or colleague
- [ ] Plagiarism check passed
