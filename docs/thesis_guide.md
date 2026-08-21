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

**Key Revision**: Replace vague DRL claims with the specific proposed approach. Current text says "DDPG" throughout — must be updated to "Branching MP-DQN + TD3" (Concept Note v4.0 Section 10); "Hybrid SAC-DDQN" was an intermediate, now-superseded design and should not appear as the proposed method either.

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
- **Iqbal et al. (2021)**: DDQN for RRH on/off + convex power allocation, switching costs. (A specific reported power-savings % is deliberately not restated here — no primary-source access is available in this environment to verify it; see `tests/test_baseline_paper_scenarios.py`'s disclaimer and `docs/rules.md` §10's Ethical AI Rule.)
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
- Tidal traffic pattern: dual-Gaussian diurnal factor, not a sinusoid (`cran_env/traffic_model.py`) — business peak centered at 11:00, residential peak centered at 20:00, each a Gaussian bump in hour-of-day; the diurnal factor is `0.15 + 0.85 * (0.6*business_peak + 0.4*residential_peak)`. Since the two peaks are centered at different hours and never both reach 1 simultaneously, the achievable range is [0.15, ~0.664] (at t=11:00), not the full [0.15, 1.0] the 0.85 weight alone might suggest
- Peak hours: business (centered 11:00), residential (centered 20:00) — see the actual Gaussian centers/widths in `cran_env/traffic_model.py::get_demands`, not the 9-12/14-17/19-23 windows this section previously (incorrectly) described
- Burstiness: log-normal per-user multiplicative fluctuation (`rng.lognormal`), not Poisson arrivals
- Second profile: `weekend_suburban` (flatter daytime, no business peak, a later/lower residential peak at 23:00) — used by the cross-profile generalization evaluation, Concept Note v4.0 §12.3/A5
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
a_t = (v(t+1), p(t), beta(t))
where v(t+1)  in {0,1}^R    # Discrete: RRH on/off decisions
      p(t)    in [0, P_max]^R  # Continuous: transmit power per RRH
      beta(t) in [0, 1]^R      # Continuous: bandwidth share per RRH (sums to 1 over active RRHs)
```

**Reward Function** (Concept Note v4.0 Section 10.2 — energy-efficiency ratio, not a raw power penalty):
```
r_t = alpha * EE(t)
      - beta * sum_{u=1}^U max(0, D_u(t) - C_u(t))
      - gamma_switch * sum_{r=1}^R |v_r(t) - v_r(t-1)|
      - gamma_fronthaul * P_FH(t)

where EE(t) = C_total(t) / P_total(t)   # Mbit/Joule, C_total(t) = sum_u C_u(t)
```

Where:
- alpha: EE(t) weight (default 1.0 recovers the note's literal formula)
- beta: QoS violation penalty (must dominate if QoS is hard constraint)
- gamma_switch: switching cost weight (prevents oscillation)
- gamma_fronthaul: fronthaul-power penalty weight, in kW (default 1.0, chosen to keep the penalty on the same order of magnitude as EE(t) itself given P_FH's typical ~0.05-0.15 kW range at this thesis's scenario sizes — small and non-dominant relative to beta/gamma_switch by design, but not negligible; exists so fronthaul's reward contribution can be independently ablated in Section 4.5, since it also already reduces EE(t) implicitly via P_total(t))

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

#### 3.7 Proposed Branching MP-DQN + TD3 Algorithm (~1,000 words) — COMPLETE REWRITE

**Replace with Section 10 of `manuscript/MPhil_Thesis_Concept_Note_v4.md`** — this subsection previously described an intermediate "Hybrid SAC-DDQN" design (a separate DDQN discrete actor + SAC continuous actor arbitrated by a shared twin critic) that was itself superseded before implementation began; `agents/hybrid_sac_dqn.py` still exists as that superseded alternative, kept only for comparison. The actual proposed method, matching `agents/branching_mp_dqn.py` and `docs/skills/skill_hybrid_agent.md`, is:

1. **Algorithm Selection Justification**:
   - DDPG: Poor stability, cannot handle discrete actions without thresholding (destroys gradients)
   - A separate discrete-DDQN/continuous-SAC hybrid with a shared critic (the superseded v1.0 design): critic architecture for a mixed discrete-continuous action was never concretely specified, and SAC's entropy-regularized exploration and DDQN's epsilon-greedy exploration are driven by incompatible objectives
   - **Our approach**: a branching, multi-pass, twin-critic parameterized DQN (Branching MP-DQN + TD3) — one coupled Q-network family, not two arbitrated actor networks

2. **Architecture** (Concept Note Section 10.3):
   - Shared Encoder h(s|theta_h): two FC layers (256, 128 units), ReLU + LayerNorm each — one instance, feeding both of the below
   - Continuous Parameter Network x(s|phi): a deterministic sub-network producing (p_r, beta_r) for all R RRHs from the shared representation (P-DQN, Xiong et al. 2018)
   - R Branching Discrete Heads: dueling-style Q_r(s, k_r) for k_r in {0,1}, one independent head per RRH (Tavakoli et al. 2018) — output grows as 2R, not 2^R
   - MP-DQN Multi-Pass Masking (Bester et al. 2019): before branch r's Q-value is computed, only x_r enters that pass's computation graph — every other RRH's continuous parameters are excluded, removing the false-gradient cross-talk P-DQN's original single-pass design would introduce
   - Twin Critics (Q^A, Q^B) + target networks (Fujimoto et al. 2018, TD3): the Bellman target uses min(Q^A, Q^B) to counter overestimation bias

3. **Training Procedure**:
   - Experience replay buffer stores (s, k, x, r, s') tuples
   - Both twin critics minimize Bellman error against a Double-DQN-style target (argmax via Q^A, evaluated via min(Q^A, Q^B))
   - The continuous parameter network (and shared encoder) are updated every `policy_delay` critic updates to maximize Q^A's multi-pass value (TD3's delayed-actor-update pattern)
   - Exploration: independent epsilon-greedy per discrete branch, plus additive Gaussian noise on the continuous parameters — both decayed once per episode, not per gradient step

4. **Pseudocode** (formal algorithm box):
```
Algorithm 1: Branching MP-DQN + TD3 for C-RAN Energy Optimization
Input: Shared encoder theta_h, param net phi, twin critics theta_Q1A/theta_Q1B..theta_QRA/theta_QRB
       Replay buffer D, target update rate tau, policy delay d

for episode = 1 to N_episodes do
    s_0 <- env.reset()
    for t = 0 to T-1 do
        feat <- h(s_t | theta_h)
        x_t <- x(feat | phi)                       # (p_r, beta_r) for all R RRHs
        for r = 1 to R do
            k_{r,t} <- epsilon-greedy(Q_r(feat, x_t masked to branch r))
        end for
        a_t <- (k_t, x_t)

        s_{t+1}, r_t, done <- env.step(a_t)
        D <- D union {(s_t, k_t, x_t, r_t, s_{t+1})}

        # Sample minibatch, multi-pass per branch (R passes per critic)
        B ~ Uniform(D)
        for r = 1 to R do
            y_{i,r} <- r_i + gamma * min(Q_r^A, Q_r^B)(s_{i+1}, x'_{i+1} masked to branch r, k*_r)
            # k*_r selected via argmax of Q_r^A (Double DQN)
        end for
        Update theta_Q1A..theta_QRA, theta_Q1B..theta_QRB to minimize sum_r (y_{i,r} - Q_r(s_i, k_{i,r}, x_i))^2

        if update_counter mod d == 0 then
            Update theta_h, phi to maximize sum_r Q_r^A(s_i, k*_r, x_i masked to branch r),
                where k*_r = argmax_k Q_r^A(s_i, k, x_i) (greedy action per branch,
                not the replayed action -- standard P-DQN/MP-DQN practice)
            Soft update all target networks: theta' <- tau*theta + (1-tau)*theta'
        end if
    end for
end for
```

---

### Chapter 4: Simulation Results (~4,000 words) — WRITE ENTIRELY

**Section Structure**:

#### 4.1 Simulation Setup (~600 words)
- Hardware: GPU model, CPU, RAM
- Software: Python version, PyTorch version, key libraries
- Network scenarios: the R=5,12,20,35,50 scalability sweep (Concept Note Section 12.2/15; R=50 is a stretch goal), plus the R=12,U=10 primary scenario (`config/default.yaml`)
- Traffic model parameters (dual-Gaussian diurnal factor + log-normal burstiness, Section 12.8)
- Hyperparameter table (all values used)
- The 11-method roster: the proposed Branching MP-DQN + TD3 agent vs. 10 baselines — All-ON/FA, Greedy, NMBS, Convex, DDQN, DDQN+SOCP, ANN+GSBF, pure-DDPG, P-DQN, MP-DQN (Section 12.1)

#### 4.2 Convergence Analysis (~800 words)
- Learning curves: reward vs. episodes for all 11 methods
- Confidence intervals (shaded regions, n=10 seeds, per supervisor review S4)
- Discussion: why does the proposed method converge to a better operating point than DDQN/P-DQN/MP-DQN? What does the multi-pass masking buy over P-DQN's single-pass coupling?

#### 4.3 Energy Efficiency Comparison (~800 words)
- Bar chart: Total energy consumption (24-hour average) for all 11 methods
- Table: Percentage savings vs. DDQN/P-DQN/MP-DQN (the headline comparison, Section 5.2/G10) and, separately, vs. All-ON (reported only as a sanity-check floor any working method should clear, not a contribution in its own right)
- Discussion: Impact of switching costs, fronthaul power inclusion (Section 10.2's gamma_switch/gamma_fronthaul terms)

#### 4.4 QoS Performance (~600 words)
- CDF of SINR per UE
- QoS violation rate over time
- Trade-off curve: energy vs. QoS violation rate

#### 4.5 Ablation Study (~600 words)
- Remove switching cost from reward (gamma_switch=0)
- Remove fronthaul power from reward (gamma_fronthaul=0)
- Remove QoS penalty from reward (beta_qos=0)
- Discussion: Each component's contribution (`evaluation/ablation.py`)

#### 4.6 Scalability Analysis (~600 words)
- Performance vs. network size (R = 5, 12, 20, 35, 50; R=50 is a stretch goal)
- Training time vs. network size, and inference latency at each scale (Section 12.3)
- Discussion: Computational complexity of multi-pass masking (scales with R), real-time feasibility

#### 4.7 CSI-Robustness (~400 words, new per Section 12.5/S3)
- Degradation curve: EE and QoS-violation rate vs. channel-estimation-error sigma in {0, 0.01, 0.05, 0.1} (`evaluation/csi_robustness.py`)
- Discussion: sensitivity of the frozen trained policy to CSI error, isolated from retraining

#### 4.8 Cross-Profile Generalization (~400 words, new per Section 12.3/A5)
- Zero-shot evaluation on the weekend/suburban traffic profile after training only on weekday/urban (`evaluation/generalization.py`)
- Discussion: EE/QoS degradation relative to the matched (weekday-trained, weekday-evaluated) case

**Required Figures** (minimum):
1. Fig 4.1: Convergence curves (all 11 methods, 10 seeds)
2. Fig 4.2: 24-hour energy profile comparison
3. Fig 4.3: CDF of per-UE SINR
4. Fig 4.4: Ablation study bar chart
5. Fig 4.5: Scalability: energy vs. network size (R=5..50)
6. Fig 4.6: Scalability: training time and inference latency vs. network size
7. Fig 4.7: CSI-robustness degradation curve (EE and QoS-violation rate vs. sigma)
8. Fig 4.8: Cross-profile generalization bar chart (weekday-matched vs. weekend-generalization)

**Required Tables**:
1. Table 4.1: Simulation parameters
2. Table 4.2: Energy savings comparison (all 11 methods, headline margin over DDQN/P-DQN/MP-DQN plus the All-ON sanity-check figure)
3. Table 4.3: QoS metrics comparison
4. Table 4.4: Ablation study results
5. Table 4.5: Scalability results (R=5..50, including inference latency)

---

### Chapter 5: Conclusion and Future Work (~1,500 words) — WRITE ENTIRELY

**Structure**:
1. **Summary of Contributions** (400 words):
   - Branching, multi-pass, twin-critic parameterized DQN (Branching MP-DQN + TD3) for discrete-continuous C-RAN control — one coupled network, not two arbitrated actors
   - Fronthaul-aware reward function (explicit gamma_switch/gamma_fronthaul/beta_qos terms)
   - Comprehensive 11-method baseline comparison and scalability analysis (R=5..50)

2. **Key Findings** (500 words):
   - Proposed method achieves X% energy savings vs. DDQN/P-DQN/MP-DQN (the headline comparison, Section 5.2/G10), clearing the All-ON sanity-check floor by Y%
   - Switching costs account for Z% of total power — cannot be neglected
   - Branching + multi-pass masking scales linearly (2R) where P-DQN/MP-DQN's flat joint head does not, validated empirically up to R≈12-15 before it becomes intractable (Section 12.1)
   - Scalable to R=50 RRHs with acceptable training time (R=50 itself a stretch goal, Section 15)
   - CSI-robustness and cross-profile generalization degrade gracefully but measurably (Section 12.5/12.3) — see Limitations below

3. **Limitations** (300 words):
   - Single-agent, single-pool assumption
   - Perfect CSI at training time — the CSI-robustness evaluation (Section 12.5) stress-tests, but does not remove, this assumption; training under imperfect CSI remains future work
   - Simulation-only (no real-world validation); the O-RAN rApp framing (Section 11) is a simulation-based positioning, not a claim of real O1/E2 interface implementation
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
