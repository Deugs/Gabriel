MPHIL THESIS CONCEPT NOTE  —  v2.0

Optimization of Energy-Efficient Cloud Radio Access Networks for 5G Using a Hybrid Discrete-Continuous Deep Reinforcement Learning Framework

Candidate: Gabriel Kwame Freeman   (Index No. PG7373923)

Degree: MPhil  ·  Institution: KNUST

Supervisor: Prof. J. J. Kponyo

Document version: 2.0  (supersedes v1.0, the DDPG-only concept note)

Prepared: 26 July 2026  ·  Status: Draft for supervisor review

Purpose of this note

This version merges the original DDPG-based concept note with the subsequent methodological-revision proposal. It keeps the revision's core direction — moving from a single continuous-action agent to a hybrid framework that treats RRH activation as a genuine discrete decision — but grounds the architecture in established, peer-reviewed discrete-continuous RL methods rather than a bespoke design, and corrects two issues found in the revision proposal along the way.

1. Purpose of This Document

This concept note summarizes the research problem, the proposed hybrid deep-reinforcement-learning (DRL) methodology, and the evaluation and delivery plan, consolidating the working thesis draft (Chapters 1–3), the original DDPG-based concept note, and the intermediate methodological-revision proposal into a single current reference. It is a decision and planning document for supervisor review, not a substitute for the full literature review or the thesis chapters themselves, which remain the primary technical record.

2. Background and Problem Statement

Mobile data traffic continues to grow, and the Radio Access Network (RAN) is consistently reported as the dominant contributor to overall network energy consumption — GSMA benchmarking data and independent industry surveys converge on roughly 70–80% of total network power draw, commonly cited around 73%. Cloud Radio Access Network (C-RAN) architectures address part of this by centralizing baseband processing in a shared BBU pool while distributing low-cost Remote Radio Heads (RRHs), but the continuous operation of densely deployed RRHs and their fronthaul links still represents a large, largely static energy cost.

Existing mitigation strategies fall into two camps. Traditional optimization — convex relaxations, greedy heuristics, bin-packing — is computationally cheap but requires near-perfect channel state information and adapts poorly to non-stationary traffic. Deep reinforcement learning adapts to stochastic, time-varying conditions, but the published work closest to this thesis (Section 4) has generally handled RRH activation (a discrete on/off decision) and transmit power allocation (a continuous decision) as two separate, decoupled problems — for example, a discrete DQN/DDQN policy for RRH state feeding a convex solver for power. The gap this thesis targets: no existing C-RAN approach jointly learns discrete RRH activation and continuous power allocation within a single end-to-end DRL policy, and fronthaul power is frequently left out of the reward formulation despite representing a meaningful share of total consumption.

3. Methodological Note: Grounding the Hybrid Architecture

The project's methodology has evolved twice: from a single continuous-action DDPG agent (the original registration and the v1 concept note) toward a hybrid discrete-continuous framework. The underlying motivation for moving past plain DDPG is legitimate — DDPG is well documented to suffer from Q-value overestimation bias and training instability relative to more recent actor-critic variants; this is precisely the problem Twin Delayed DDPG, or TD3 (Fujimoto, van Hoof & Meger, 2018), was built to fix, via twin critics, delayed policy updates and target-policy smoothing.

What this version changes relative to the intermediate proposal is how the hybrid is built. Rather than an original, unpublished “DDQN discrete actor plus SAC continuous actor plus shared twin critic” architecture, the framework in Section 10 is assembled from four established, peer-reviewed components, each solving one specific piece of the discrete-continuous puzzle:

Parameterized action coupling — P-DQN (Xiong, Wang, Yang et al., 2018) couples a discrete decision with an associated continuous parameter through one Q-network: a DQN-style update for the discrete choice, and a DDPG-style deterministic policy gradient for the continuous parameter.

Correcting P-DQN's false gradients — MP-DQN (Bester, James & Konidaris, 2019) evaluates each discrete branch's Q-value using only its own continuous parameters (a “multi-pass” over the network), removing the cross-talk P-DQN otherwise introduces between unrelated RRHs.

Scaling to many independent decisions — the branching architecture (Tavakoli, Pardo & Kormushev, 2018) gives each RRH its own decision branch off a shared state representation, so the action output grows linearly (2R) rather than combinatorially (2^R) with the number of RRHs — directly relevant to scaling from 5 to about 50 RRHs.

Training stability — twin critics, delayed updates and target-policy smoothing (Fujimoto et al., 2018) address the overestimation-bias instability that motivated moving past plain DDPG in the first place.

Section 10 specifies the resulting framework in full. Section 4 also revisits the literature review with this framing in mind.

4. Review of Closely Related Work and the Research Gap

4.1  C-RAN energy-efficiency literature

Work

Technique

How activation & power are handled

Relation to the proposed hybrid

Iqbal, Tham & Chang (2021) — DQN/DDQN

Double Deep Q-Network + convex SOCP solver

DDQN picks one RRH's on/off status per slot; power/beamforming for the resulting set is solved as a separate SOCP every slot

Source of the system model and the discrete-RL baseline; the proposed hybrid keeps true discrete RRH decisions but couples them to power via one learned network instead of a per-slot solver

Fathy, Abood & Hamdi (2021) — ANN + Bi-Section GSBF

Supervised ANN + 3-stage GSBF heuristic

ANN predicts the near-optimal RRH count from offline-labelled data; GSBF heuristic then selects and beamforms

Supervised-learning baseline; needs labelled data from a slow heuristic and has no MDP or switching-cost notion

Xu, Wang, Tang, Wang & Gursoy (2017)

DNN value approximation + convex optimization

Same two-stage pattern as Iqbal et al. (its precursor)

Same structural limitation; earlier, simpler value network

Zhou et al. (2023) — Co-HDRL, RIS-aided RAN

Cooperative hierarchical DRL, two coordinated sub-controllers

One controller for discrete sleep, one for continuous RIS/power control, coordinated hierarchically

Closest in spirit — also couples discrete and continuous decisions — but via two separately-optimized hierarchical policies rather than one P-DQN-style coupled network, and for a different (RIS-aided) architecture

Al-Zubaedi (2019) — PhD thesis

Metaheuristics: Quasi-Newton Method, PSO, NMBS

Optimizes BBU-pool placement and RRH-to-BBU clustering (network planning)

Different timescale — deployment/planning, not the slot-by-slot EE resource-allocation problem this thesis targets

Proposed hybrid (this thesis)

Branching, multi-pass, twin-critic parameterized DQN

Each RRH gets its own discrete activation branch and continuous power/bandwidth parameters, coupled through one Q-network family

Extends the discrete formulation of Iqbal et al. with genuinely continuous power control, without a per-slot solver and without continuous-relaxing the discrete decision

4.2  Discrete-continuous reinforcement learning building blocks

These are not C-RAN papers; they are the general-purpose DRL components Section 10 combines for this problem.

Component

Source

Role in the proposed framework

Continuous relaxation (PA-DDPG)

Hausknecht & Stone (2016)

The approach used in the v1 concept note (RRH activation as a continuous variable, thresholded); kept as the “pure-DDPG” baseline in Section 11 to isolate the effect of true discrete actions

P-DQN

Xiong, Wang, Yang et al. (2018)

Core mechanism coupling each discrete RRH decision to its continuous power/bandwidth parameters through one Q-network

MP-DQN

Bester, James & Konidaris (2019)

Multi-pass fix for P-DQN's parameter cross-talk between unrelated RRHs

Branching / BDQ

Tavakoli, Pardo & Kormushev (2018)

Per-RRH decision branches off a shared encoder, avoiding 2^R combinatorial action growth

TD3 (twin critics)

Fujimoto, van Hoof & Meger (2018)

Twin critics, delayed updates, target-policy smoothing — the concrete fix for the DDPG instability that motivated this revision

4.3  Synthesis: the research gap

The C-RAN literature converges on the same structural limitation from three algorithmic directions — DDQN-plus-SOCP, ANN-plus-heuristic, DNN-plus-convex — and even the closest hybrid attempt (Zhou et al., 2023) uses two separately-optimized policies rather than one coupled network. Separately, the general DRL literature already has well-tested tools for exactly this discrete-plus-continuous-parameter structure (P-DQN, MP-DQN, branching), but they have not, to my knowledge, been applied to the C-RAN joint RRH-activation-and-power problem, nor benchmarked against the DDQN and ANN+GSBF baselines already established for it. That is the gap this thesis targets: a branching, multi-pass, twin-critic parameterized-DQN framework for C-RAN energy efficiency, evaluated head-to-head against the exact baselines from Iqbal et al. (2021) and Fathy et al. (2021), and against the simpler continuous-relaxation (pure-DDPG) alternative from the v1 concept note.

5. Aim and Objectives

5.1  Aim

To design, implement and evaluate a hybrid discrete-continuous DRL framework — combining branching Q-learning for RRH activation with parameterized continuous control for power and bandwidth — that maximizes long-term energy efficiency in a 5G C-RAN subject to QoS constraints, and that scales tractably from small to large RRH counts.

5.2  Specific objectives

Formulate the joint RRH-activation-and-power-control problem as a parameterized-action MDP (Section 10.2) compatible with branching P-DQN/MP-DQN.

Design and train the hybrid agent — branching discrete heads, a continuous parameter network, and twin critics — addressing the false-gradient and combinatorial-scaling issues documented in the P-DQN/branching literature from the outset rather than retrofitting fixes.

Re-implement Full Activation, a greedy/NMBS heuristic (Al-Zubaedi, 2019), convex-only power allocation, DDQN (Iqbal et al., 2021), ANN+GSBF (Fathy et al., 2021) and pure-DDPG (continuous relaxation) as baselines under identical simulation conditions to the proposed agent.

Evaluate energy efficiency, QoS-violation rate, RRH-switching frequency, training stability and convergence, and computational cost, benchmarking improvements against the margins reported by the closest published baselines rather than a fixed pass/fail target (Section 11 explains why).

Characterize how training time and performance scale from small (5 RRH) to large (≈50 RRH) network instances, exploiting the branching architecture's linear action-space growth.

6. Research Questions

How should the joint RRH-activation-and-power-control problem be formulated as a parameterized-action MDP so a branching P-DQN/MP-DQN agent can learn both decisions through one coupled network?

What architecture, reward design and training configuration (multi-pass evaluation, branching, twin critics) achieve stable convergence for this MDP as RRH count grows?

Does representing RRH activation as a true discrete decision (the hybrid framework) outperform the continuous-relaxation approach (pure DDPG), and by how much — holding the rest of the pipeline fixed?

How does the resulting policy's energy efficiency and QoS performance compare with DQN-, DDQN- and ANN+heuristic-based approaches under identical network conditions?

What is the trade-off between energy savings, QoS satisfaction and RRH-switching frequency, and how does it hold up as the network scales from 5 to ≈50 RRHs?

7. Significance of the Study

Extends the discrete-action DRL and supervised-learning literature for C-RAN energy efficiency (Iqbal et al., 2021; Fathy et al., 2021) with a hybrid framework that avoids both the per-slot solver of the former and the offline-labelling requirement of the latter.

Brings peer-reviewed discrete-continuous RL methods (P-DQN, MP-DQN, branching, TD3) into the C-RAN domain for the first time, to my knowledge, rather than a bespoke architecture.

Produces a reusable simulation and benchmarking harness (FA, heuristic, convex, DDQN, ANN+GSBF, pure-DDPG, hybrid) for future C-RAN DRL work.

The scalability characterization (5–50 RRH) and the energy/QoS/switching trade-off findings are relevant to green-communication planning for 5G and future 6G networks, including resource-constrained deployment settings.

8. Scope and Assumptions

Downlink transmission only; a single BBU pool; a single centralized DRL agent; uplink, multi-pool scenarios and multi-agent RL are out of scope.

Evaluation is simulation-based (MATLAB, consistent with the existing simulation work); no physical or SDR testbed is used.

Channel state information is assumed available to the BBU pool at each decision epoch, acknowledged as a limitation; imperfect or delayed CSI is left as future work.

The 5→50 RRH scalability sweep is the reason the branching architecture (Section 10) was chosen over a joint discrete action space, which would grow combinatorially and become intractable well before 50 RRHs.

Any expansion of this scope will be brought back to the supervisor as a further revision of this document.

9. System Model (Summary)

Chapter 3 of the thesis already derives the system model in detail — the RRH/UE/BBU network model, path-loss and SINR expressions, Shannon capacity, and the three-part power model (RRH, fronthaul, BBU pool) — following Iqbal et al. (2021) for the radio model and Al-Zubaedi (2019) for the BBU-pool and fronthaul power model. This does not change with the methodology and is not repeated here; it is summarized only to fix notation for Section 10.

The network comprises RRHs R = {1,…,R}, UEs U = {1,…,U} and BBUs B = {1,…,B}, connected by a fronthaul link. Each RRH r serves users through joint beamforming; UE u's achievable rate C_u(t) follows the Shannon capacity of its SINR. Total network power at slot t sums RRH power (active/sleep/switching), fronthaul power (OLT + ONU) and BBU-pool power (static + load-dependent). Energy efficiency is

EE(t) = Σu∈U Cu(t)  ⁄  [ B × Ptotal(t) ]

and the long-run objective is to choose, at every slot, which RRHs are active and at what power, to maximize Σᴛ EE(t) subject to each user's rate demand and each RRH's power ceiling — the same objective as Chapter 3, equation (9), and Iqbal et al. (2021), equation (9). What changes with this revision is only how the controller that makes those choices is built.

10. Proposed Hybrid DRL Framework

10.1  Design rationale

Section 3 explained why each component was chosen. In combination: the shared encoder and branching heads (Tavakoli et al., 2018) let R independent RRH on/off decisions scale linearly rather than combinatorially; the P-DQN coupling (Xiong et al., 2018), corrected by MP-DQN's multi-pass evaluation (Bester et al., 2019), lets each branch's decision carry its own continuous power and bandwidth parameters without cross-talk from other RRHs; and TD3-style twin critics (Fujimoto et al., 2018) address the overestimation bias that made plain DDPG the less stable choice. No part of this needs to be taken on faith — each piece has its own published ablation showing it does what it is used for here.

10.2  MDP formulation

State space

Unchanged in form from the v1 concept note, since the state does not depend on the control algorithm:

s(t) = [ D1(t),…,DU(t),  k1(t−1),…,kR(t−1),  g1,1(t),…,gR,U(t),  ρBBU(t),  E(t) ]T

with one substantive change: k_r(t−1) is now the RRH's true binary activation state (0/1) from the previous slot, not the continuous relaxation v_r(t−1) used in v1.

Action space — now a parameterized (hybrid) action

Each RRH r contributes one discrete choice and, when active, a pair of continuous parameters:

a(t) = { ( kr(t),  xr(t) )  :  r = 1,…,R },   xr(t) = ( pr(t), βr(t) )

k_r(t)∈{0,1} is RRH r's true discrete activation decision — no threshold or hysteresis band is needed here, unlike the v1 design. p_r(t)∈[0,P_r^max] is transmit power and β_r(t)∈[0,1] is bandwidth share (Σ_r β_r ≤ 1); both are only physically meaningful when k_r(t)=1, and are defined as 0 otherwise. This is exactly the parameterized action space 𝓜 = {(k,x_k)} of Xiong et al. (2018), applied independently across R RRHs via branching.

Reward function

Same functional form as v1, but the switching term is now an exact indicator rather than a continuous proxy:

r(t) = EE(t)  −  λ1 Σu∈U max(0, Du(t) − Cu(t))  −  λ2 Σr∈R |kr(t) − kr(t−1)|

With true discrete k_r, |k_r(t)−k_r(t−1)| ∈ {0,1} is an exact switching-event count rather than an approximation — a direct benefit of moving off continuous relaxation. λ₁=λ₂=0 again recovers a reward equivalent to Iqbal et al.'s EE(t), the same sanity check as before.

10.3  Network architecture

Shared encoder h(s|θ_h): state s(t) → shared representation; two fully-connected layers (256, 128 units, ReLU) — unchanged from v1.

R discrete branches (Tavakoli et al., 2018): each RRH gets a dueling-style head producing Q_r(s,k_r) for k_r∈{0,1} off the shared representation, so the output grows as 2R rather than 2^R.

Continuous parameter network x(s|φ): a DDPG-style deterministic sub-network producing x_r(s) = (p_r,β_r) for all R RRHs from the shared representation, following the P-DQN mechanism (Xiong et al., 2018).

Multi-pass evaluation (Bester et al., 2019): when branch r's Q-value is computed, only x_r is passed in and every other RRH's continuous parameters are masked to zero — this is the specific fix for the parameter cross-talk that plain P-DQN suffers from.

Twin critics (Fujimoto et al., 2018): two independent copies of the branch/critic network (Q^A, Q^B), each with its own target network; the Bellman target uses min(Q^A, Q^B) to counter overestimation bias, with delayed, less-frequent updates to φ and target-policy smoothing noise on x' at the target networks.

10.4  Training algorithm

Combining P-DQN's coupling mechanism, MP-DQN's multi-pass correction, branching's per-RRH decomposition and TD3's twin-critic stabilization:

1.  Initialize shared encoder h(s|theta_h), twin branch

networks Q^A, Q^B (R dueling branches each, params

theta_QA, theta_QB), and continuous parameter network

x(s|phi).

2.  Initialize targets: theta_h', theta_QA', theta_QB',

phi'  <-  theta_h, theta_QA, theta_QB, phi.

3.  Initialize an empty replay buffer (capacity N_D).

4.  for episode = 1 to M do

Observe initial state s_t from the environment (t=1).

for t = 1 to T do

Compute x_t = x(s_t|phi) + noise_t  (Gaussian, TD3-

style).

for r = 1 to R do  (multi-pass: mask other RRHs)

Compute Q_r^A(s_t,k_r,x_t | mask=r), k_r in {0,1}.

Select k_r,t by epsilon-greedy over Q_r^A(s_t,.,x_t).

end for

Apply a_t = {(k_r,t, x_r,t)} for all r; observe

reward r_t and next state s_t+1.

Store (s_t, {k_r,t}, x_t, r_t, s_t+1) in the buffer.

Sample a random mini-batch of N transitions.

For each i: k_r,i' = argmax_k Q_r^A(s_i+1,k,x'|phi')

per branch (multi-pass); x' smoothed with noise.

Set y_i = r_i + gamma * min(Q^A',Q^B')(s_i+1,

{k_r,i'}, x').

Update Q^A, Q^B by minimizing

L = (1/N) sum_i sum_r (y_i - Q_r(s_i,k_r,i,x_i))^2.

every d steps:

Update phi via the multi-pass policy gradient:

grad_phi J ~= (1/N) sum_i sum_r

  grad_x Q_r^A(s_i,k_r,i,x)|x=x(s_i|phi)

  . grad_phi x_r(s_i|phi).

Soft-update targets: theta' <- tau*theta +

(1-tau)*theta'  (for h, Q^A, Q^B, phi).

end for

5.  end for

10.5  Design notes

No more threshold/hysteresis workaround: because k_r(t) is a genuine discrete output, the design notes in the v1 concept note about thresholding continuous activation and adding a hysteresis band are no longer needed — one of the concrete simplifications this revision buys.

Multi-pass is not optional: using P-DQN's original single-pass evaluation (all R continuous parameters fed to every branch) is documented to invalidate the theoretical grounding of the discrete update (Bester et al., 2019); the multi-pass masking in Section 10.3 is part of the base design, not an optimization to add later.

Compute cost grows with R: R branches × 2 critics × multi-pass means R forward passes per critic evaluation per step. At R≈50 this is the most likely practical bottleneck in the whole plan — flagged again in Section 13 (Risks).

Exploration is two different mechanisms now: epsilon-greedy (decayed over training) for the R discrete branches, and additive Gaussian noise (decayed over training) for the continuous parameters — unlike v1, where a single Ornstein–Uhlenbeck process covered a fully continuous action.

11. Evaluation Plan

11.1  Baselines

Seven methods are compared under identical simulation conditions: Full Activation (FA), a greedy/NMBS bin-packing heuristic (Al-Zubaedi, 2019), convex-only power allocation with fixed RRH selection, DDQN (Iqbal et al., 2021), ANN + Bi-Section GSBF (Fathy et al., 2021), pure DDPG with continuous relaxation (the v1 concept note's design, kept specifically to answer RQ3), and the proposed hybrid agent. Pure-SAC and pure-TD3 are noted as optional stretch comparisons (Section 14) rather than core baselines: as continuous-only algorithms they face the same discrete-representation question as plain DDPG, so they mainly add algorithm-family coverage rather than new insight on the discrete-vs-continuous question this thesis is actually asking.

11.2  Simulation environment and parameters

The radio and power-model parameters are unchanged from Chapter 3 and trace back to Iqbal et al. (2021), Table 2. The hyperparameters below are specific to the hybrid agent and replace the DDPG-only table in v1.

Parameter

Value

Note

Noise power σ² / Bandwidth B

−102 dBm / 10 MHz

Unchanged (Iqbal et al., 2021, Table 2)

RRH active / sleep / switch power

6.8 W / 4.3 W / 3 W

Unchanged

RRHs (scalability sweep)

5, 12, 20, 35, 50

Branching keeps output size linear (2R) across this range

BBUs / Users (primary scenario)

B = 4 / U = 20

Re-run Iqbal's R=5,U=2 and R=12,U=4 scenarios too, for direct comparability

Replay buffer N_D

1×10⁵

As reconciled in v1; still appropriate at this training scale

Mini-batch / training episodes

64 / 1000

Unchanged

Discount factor γ

0.99

Standard DDPG/TD3 value (Lillicrap et al., 2016; Fujimoto et al., 2018)

Soft-update rate τ / actor delay d

0.005 / every 2 critic updates

TD3 defaults (Fujimoto et al., 2018)

Learning rate (branches / continuous net)

1×10⁻³ / 1×10⁻⁴

Branch (Q) network typically tolerates a higher rate than the continuous policy net

Discrete exploration

ε-greedy, 1.0→0.05 decayed over training

Genuinely applicable now that activation is a true discrete action

Continuous exploration

Gaussian, σ=0.1·P_max (decayed)

Replaces the v1 Ornstein–Uhlenbeck process; TD3-style

MATLAB's Reinforcement Learning Toolbox supports custom multi-headed agents, which would let the baselines and the proposed agent stay in one MATLAB-based environment; the branching/multi-pass/twin-critic combination is not a single built-in agent type, so the branch heads, multi-pass masking and twin-critic loss will need custom implementation regardless of language choice.

11.3  Performance metrics

Energy efficiency (Mbit/Joule) and average power (W) versus user demand — comparable to Iqbal et al.'s Figs. 3 and 5.

Power consumption versus time slot under dynamic demand — comparable to Iqbal et al.'s Fig. 4.

QoS-violation rate and exact RRH-switching frequency (now a true count, not a thresholded estimate).

Training convergence and stability — reward variance across seeds is the direct empirical test of whether the twin-critic design achieves the stability that motivated this revision.

Hybrid vs pure-DDPG (RQ3): energy efficiency, switching frequency and convergence speed, holding state space, reward and simulation conditions fixed, isolating the effect of true discrete representation.

Scalability: training time and converged EE as R runs from 5 to ≈50, testing whether the branching architecture's linear action growth translates into tractable training time in practice.

Inference-time cost per decision, following Fathy et al. (2021, Table II), who report roughly 24 minutes for their heuristic alone and about 11 minutes with an ANN pre-stage — a useful benchmark for how much a trained forward-pass policy can save at deployment.

11.4  Fair comparison and statistical reporting

Iqbal et al. (2021) and Fathy et al. (2021) report results under different network sizes and regions; none of those published numbers can be compared directly against a new result run under different conditions. All seven methods must be implemented in one shared environment and run under the same scenario(s), each averaged over 5 random seeds with 95% confidence intervals, consistent with standard practice in the DRL comparison literature (e.g. Shengren et al., 2022, who report seed-averaged confidence intervals for exactly this kind of multi-algorithm comparison).

12. Expected Contributions

A parameterized-action MDP formulation for joint RRH activation and power/bandwidth control in C-RAN energy-efficiency optimization, uniting the discrete formulation of Iqbal et al. (2021) with genuinely continuous power control — without continuous-relaxing the discrete decision.

A branching, multi-pass, twin-critic DRL architecture adapting P-DQN/MP-DQN (Xiong et al., 2018; Bester et al., 2019) and branching Q-networks (Tavakoli et al., 2018) to the C-RAN domain, not previously applied to this problem to my knowledge.

A direct empirical test of whether true discrete RRH representation outperforms continuous relaxation (RQ3) — a comparison the methodological-revision proposal assumed the answer to rather than testing.

A scalability characterization from 5 to ≈50 RRHs, made tractable by the branching architecture's linear action-space growth.

A head-to-head comparison against FA, heuristic, convex, DDQN and ANN+GSBF baselines re-implemented under identical conditions — a comparison that does not yet exist in the published literature.

13. Risks and Mitigations

Risk

Mitigation

Branching/multi-pass training instability or residual false gradients

Multi-pass masking is designed in from the start (Section 10.3), not retrofitted; validate on a small R=3–5 case before scaling up

Per-step compute cost at large R (up to 50 branches × multi-pass × twin critics)

Profile in the first 1–2 weeks of implementation; if R=50 is infeasible in the available time, cap the scalability sweep at a smaller maximum and report this explicitly as a limitation rather than silently dropping the objective

Reproduced baselines don't match published numbers

Unit-test each baseline independently against its source paper's reported operating point before using it comparatively

Scope creep

Enforced via Section 8; any expansion returns to this document for re-approval

14. Indicative Timeline

Revised from the methodological-revision proposal's 14-week estimate: baselines alone need more than the 1 week originally allotted (six methods, including faithfully reproducing a published DDQN pipeline, is not a one-week task), and the hybrid architecture is more involved to stabilize than a single off-the-shelf agent. The estimate below reflects that.

Phase

Duration

Deliverable

Environment & power model

Weeks 1–2

Validated C-RAN simulator, shared across all methods

Baselines

Weeks 2–5 (overlapping)

FA, heuristic, convex, DDQN, ANN+GSBF and pure-DDPG implemented and unit-tested

Hybrid agent

Weeks 4–8

Working, stable branching / multi-pass / twin-critic implementation, validated at small R first

Experiments

Weeks 8–12

Main comparison, RQ3 ablation, and 5→50 RRH scalability sweep, 5 seeds each

Thesis writing

Weeks 6–15 (parallel)

Chapters 1–5 drafted, starting once early baseline results are available

Revision & submission

Weeks 16–17

Final draft submitted

Total estimated duration: about 17 weeks from approval of this document — roughly three weeks longer than the original estimate, reflecting the added implementation complexity rather than a change in ambition.

15. Thesis Structure, Current Status and Recommended Next Steps

15.1  Chapter-by-chapter status

Ch.

Title

Status

Remaining work

1

Introduction

Drafted

Update framing from DDPG to the hybrid method; the core problem statement is unchanged

2

Literature Review

Substantially drafted

Unify the existing citation schemes into one bibliography; add §4.1–4.2 of this note (C-RAN table + DRL building-blocks table)

3

System Model & Problem Formulation

System model unchanged and reusable; DDPG methodology subsections (§3.7) need replacing

Replace §3.7 with Section 10 of this note; the previous algorithm listing (incomplete after step 1) is superseded by §10.4

4

Simulation Results, Performance Evaluation & Discussion

Not started

Implement the shared environment and all seven methods (Section 11); run scenarios; produce figures analogous to Iqbal et al.'s Figs. 3–7, plus the RQ3 ablation and scalability sweep

5

Conclusion & Future Work

Not started

Write after Chapter 4 results are available

—

References

Partial / inconsistent numbering

Consolidate into one bibliography including the new DRL references in Section 16

16. Key References

Iqbal, A., Tham, M.-L., & Chang, Y. C. (2021). Double Deep Q-Network-Based Energy-Efficient Resource Allocation in Cloud Radio Access Network. IEEE Access, 9, 20440–20449. https://doi.org/10.1109/ACCESS.2021.3054909

Fathy, M., Abood, M. S., & Hamdi, M. M. (2021). Optimization of Energy-Efficient Cloud Radio Access Networks for 5G using Neural Networks. 2021 International Conference on Intelligent Technology, System and Service for Internet of Everything (ITSS-IoE). https://doi.org/10.1109/ITSS-IoE53029.2021.9615290

Xu, Z., Wang, Y., Tang, J., Wang, J., & Gursoy, M. C. (2017). A deep reinforcement learning based framework for power-efficient resource allocation in cloud RANs. Proc. IEEE International Conference on Communications (ICC), 1–6.

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control with deep reinforcement learning. International Conference on Learning Representations (ICLR). arXiv:1509.02971

Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. Proc. AAAI Conference on Artificial Intelligence, 2094–2100.

Al-Zubaedi, W. H. A. (2019). Planning a C-RAN Deployment for the Next Generation Cellular Networks [Doctoral thesis, Brunel University London].

Zhou, H., Elsayed, M., Bavand, M., Gaigalas, R., Furr, S., & Erol-Kantarci, M. (2023). Cooperative Hierarchical Deep Reinforcement Learning based Joint Sleep and Power Control in RIS-aided Energy-Efficient RAN. arXiv:2304.13226.

Hausknecht, M., & Stone, P. (2016). Deep Reinforcement Learning in Parameterized Action Space. International Conference on Learning Representations (ICLR).

Xiong, J., Wang, Q., Yang, Z., et al. (2018). Parametrized Deep Q-Networks Learning: Reinforcement Learning with Discrete-Continuous Hybrid Action Space. arXiv:1810.06394

Bester, C. J., James, S. D., & Konidaris, G. D. (2019). Multi-Pass Q-Networks for Deep Reinforcement Learning with Parameterised Action Spaces. arXiv:1905.04388

Tavakoli, A., Pardo, F., & Kormushev, P. (2018). Action Branching Architectures for Deep Reinforcement Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), 4131–4138.

Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. Proc. International Conference on Machine Learning (ICML), 1587–1596.

Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. Proc. International Conference on Machine Learning (ICML).

Shengren, H., Salazar Duque, E. M., Vergara, P. P., & Palensky, P. (2022). Performance Comparison of Deep RL Algorithms for Energy Systems Optimal Scheduling. 2022 IEEE PES Innovative Smart Grid Technologies Conference Europe (ISGT-Europe), 1–6.