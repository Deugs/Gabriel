# Research Concept Document

**Optimization of Energy-Efficient Cloud Radio Access Networks (C-RAN) for 5G Using a Hybrid Deep Reinforcement Learning Framework**

| | |
|---|---|
| **Candidate** | Gabriel Kwame Freeman |
| **Degree** | MPhil (Master of Philosophy) |
| **Institution** | [University Name] |
| **Supervisor** | [Supervisor Name] |
| **Document version** | 1.0 |
| **Date** | 24 July 2026 |
| **Distribution** | Supervisor; Department Graduate Committee; Thesis Examination Board |

> **Note on title**: the working title inherited from the original proposal references "Deep Deterministic Policy Gradient" (DDPG). As explained in Section 3, the methodology has since been revised to a hybrid SAC-DDQN framework; the title above reflects that revision and supersedes the original for all purposes except the official registration record, which the candidate will update through the appropriate departmental process.

---

## 1. Purpose of This Document

This concept document summarizes the research problem, the proposed approach, and a revision to the originally registered methodology, for review and sign-off by the supervisor and any other stakeholders with an interest in the project's direction. It is intended as a decision document, not a full literature review or technical specification — those are maintained separately in the working thesis draft and development documentation.

## 2. Background and Problem Statement

Mobile data traffic continues to grow exponentially, and the Radio Access Network (RAN) is consistently reported as the dominant contributor to overall network energy consumption — commonly cited at 57–80% of total network power draw. Cloud Radio Access Network (C-RAN) architectures address part of this problem by centralizing baseband processing in a shared BBU pool while distributing low-cost Remote Radio Heads (RRHs), but the continuous operation of densely deployed RRHs and their fronthaul links still represents a large, largely static energy cost.

Existing mitigation strategies fall into two camps:

- **Traditional optimization** (convex relaxations, greedy heuristics, bin-packing) — computationally cheap but requires near-perfect channel state information and does not adapt well to non-stationary traffic.
- **Deep Reinforcement Learning (DRL)** — adapts to stochastic, time-varying traffic and channel conditions, but prior work has generally handled RRH activation (a discrete on/off decision) and transmit power allocation (a continuous decision) as two separate, decoupled problems — e.g. a discrete DDQN policy for RRH state feeding a convex solver for power allocation.

**The gap**: no existing approach jointly learns discrete RRH activation and continuous power allocation within a single end-to-end DRL policy, and fronthaul (PON) power is frequently omitted from the reward formulation despite representing a meaningful share of total consumption.

## 3. Methodological Revision (Why This Document Exists)

The project was originally scoped around a vanilla Deep Deterministic Policy Gradient (DDPG) agent controlling both RRH activation and power allocation. Design review surfaced two problems with that approach:

1. **DDPG cannot natively represent a binary decision.** RRH on/off is discrete; forcing DDPG's continuous output through a threshold destroys the gradient signal needed to learn good switching behavior.
2. **DDPG is known to be the least stable of the modern actor-critic family.** Recent benchmarking (Shengren et al., 2022) [1] ranks Soft Actor-Critic (SAC) above TD3 above DDPG on training stability and sample efficiency.

**Revised approach**: a hybrid actor-critic architecture that pairs a **DDQN-style discrete actor** (RRH activation) with a **SAC continuous actor** (power allocation), coordinated through a **shared twin critic** that evaluates the joint action. This is a direct methodological upgrade, not a change of research question — the core question and the C-RAN system model are unchanged; only the DRL algorithm changes, and the change is a formalization of the discrete-continuous split every reasonable design already required.

This revision is flagged explicitly per the project's own scope-boundary process, which requires supervisor awareness of any deviation from the originally registered method.

## 4. Research Question and Objectives

**Core research question**: How can Deep Reinforcement Learning be applied to optimize joint RRH activation and transmit power allocation in 5G C-RAN, balancing energy efficiency against Quality of Service constraints?

**Sub-questions**:
1. What is the optimal DRL architecture for a hybrid discrete (RRH on/off) and continuous (power) action space?
2. How should the reward function incorporate RRH switching costs and fronthaul power to avoid degenerate policies (e.g. oscillating RRHs on/off)?
3. How does the proposed joint-learning approach compare against the two-stage (discrete-selection + convex-power) approach used in prior work?

**Objectives** (measurable):
- Achieve ≥25% energy reduction versus an "all RRHs on, uniform power" baseline, averaged over a 24-hour traffic cycle.
- Achieve ≥5% energy reduction versus a reproduced two-stage DDQN + convex-optimization baseline (Iqbal et al., 2021) [2].
- Keep the QoS violation rate (fraction of UEs below their SINR target) at ≤5%.
- Demonstrate the approach scales from small (5 RRH) to large (50 RRH) network instances with characterized training-time growth.

## 5. Proposed Contribution

**Primary contribution**: a hybrid actor-critic DRL framework that jointly optimizes discrete RRH activation and continuous transmit power allocation through a shared critic, eliminating the need for an analytical sub-problem solver and enabling genuinely end-to-end policy learning — in contrast to prior two-stage approaches (Fathy et al., 2021 [3]; Iqbal et al., 2021 [2]) that decouple the discrete and continuous decisions.

**Secondary contributions**:
- A power-consumption model validated against the EARTH/GreenTouch standard parameters (via Al-Zubaedi, 2019) [4], correcting parameter values used in the original draft.
- A comprehensive baseline comparison spanning heuristic, convex-optimization, pure-discrete-RL, and pure-continuous-RL methods, all evaluated under an identical protocol.
- A scalability analysis characterizing performance and training cost as network size grows from 5 to 50 RRHs.

## 6. Methodology Summary

- **Environment**: a Gymnasium-compatible C-RAN simulator combining a Rayleigh-fading channel model, a tidal (time-of-day) traffic model, and an EARTH-aligned power model covering RRH, BBU pool, and fronthaul (TWDM-PON) consumption.
- **State, action, reward**: a formal MDP is defined — channel gains, RRH activation state, per-UE traffic demand, and time-of-day compose the state; the action is a joint (discrete RRH vector, continuous power vector); the reward penalizes total power draw, QoS violations, and RRH switching cost.
- **Agent**: the hybrid SAC-DDQN architecture described in Section 3, trained with a replay buffer over hybrid (state, discrete action, continuous action, reward, next state) transitions.
- **Baselines**: All-ON + uniform power, greedy heuristic, NMBS bin-packing (Al-Zubaedi, 2019) [4], convex power allocation (CVXPY), and reproduced DDQN (Iqbal et al., 2021) [2], plus pure-SAC, pure-TD3, and pure-DDPG agents for algorithm-family comparison.
- **Evaluation**: convergence, energy efficiency, QoS performance, an ablation study (removing switching cost / fronthaul power / QoS penalty individually), and a scalability sweep — all averaged over 5 fixed random seeds with reported confidence intervals and statistical significance tests.

## 7. Significance

Reducing RAN energy consumption directly reduces mobile operators' operating expenditure and carbon footprint, both increasingly material concerns as 5G densification continues and networks evolve toward 6G. A DRL controller that jointly reasons over discrete and continuous decisions — rather than decoupling them — is also a methodological contribution relevant beyond this specific application, to any control problem mixing on/off and continuous-value decisions under a shared objective.

## 8. Scope and Limitations

**In scope**: downlink transmission; a single BBU pool; a single, centralized DRL agent; simulation-based evaluation; the assumption of perfect channel state information (CSI), acknowledged as a limitation.

**Out of scope** (absent further approval): uplink traffic, multi-pool scenarios, multi-agent RL, imperfect CSI, hardware testbed validation, and 6G-specific features. Any expansion of scope will be brought back to the supervisor/committee as a further revision to this document.

## 9. Indicative Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Environment | Weeks 1–2 | Validated C-RAN simulator |
| Baselines | Week 3 | All comparison baselines implemented and validated |
| Hybrid agent | Weeks 4–6 | Working, stable hybrid SAC-DDQN implementation |
| Experiments | Weeks 6–9 | Convergence, energy, QoS, ablation, and scalability results |
| Thesis writing | Weeks 8–12 (parallel) | Chapters 1–5 drafted |
| Revision & submission | Weeks 13–14 | Final draft submitted |

Total estimated duration: 14 weeks from approval of this document.

## 10. Key Risks

| Risk | Mitigation |
|---|---|
| Hybrid agent training instability | Start from a proven pure-SAC baseline; add the discrete head incrementally |
| Insufficient GPU/compute time | Prioritize the medium network size (12 RRH); extrapolate scalability results |
| Reproduced baselines don't match published numbers | Unit-test each baseline independently; document any residual discrepancy transparently |
| Scope creep | Enforced via the scope boundary in Section 8; any deviation returns to this document for re-approval |

## 11. Request to Stakeholders

The candidate requests:
1. Acknowledgement of the methodological revision described in Section 3 (DDPG → hybrid SAC-DDQN).
2. Confirmation that the objectives and scope in Sections 4 and 8 remain acceptable.
3. Any feedback on the indicative timeline in Section 9 before implementation begins in earnest.

## 12. References

[1] Y. Shengren, et al., "Benchmarking deep reinforcement learning algorithms for [stability/sample-efficiency comparison of SAC, TD3, and DDPG]," 2022. **Citation incomplete** — full title, venue, and volume/page/DOI not yet verified against a source copy; complete before external distribution.

[2] A. Iqbal, M.-L. Tham, and Y. C. Chang, "Double Deep Q-Network-Based Energy-Efficient Resource Allocation in Cloud Radio Access Network," *IEEE Access*, vol. 9, pp. 20440–[end page not confirmed], 2021, doi: 10.1109/ACCESS.2021.3054909.

[3] M. Fathy, M. S. Abood, and M. M. Hamdi, "Optimization of Energy-Efficient Cloud Radio Access Networks for 5G using Neural Networks," in *2021 International Conference on Intelligent Technology, System and Service for Internet of Everything (ITSS-IoE)*, 2021, doi: 10.1109/ITSS-IoE53029.2021.9615290.

[4] Al-Zubaedi, "[Full title not yet verified]," 2019. **Citation incomplete** — referenced throughout the project's working documentation as the source for the EARTH-model power parameters and NMBS bin-packing baseline, but no source PDF is on file; the candidate should confirm the exact title, publication type (thesis/paper), and venue before this document or the thesis bibliography is finalized.

> References [2] and [3] were verified directly against the source PDFs held in `references/`. References [1] and [4] are cited in the project's working documentation but no source copy is on file — resolve these with the candidate's reference manager before this document is sent to stakeholders.

---

*Prepared from the project's working development documentation (`CLAUDE.md`, `docs/workflow.md`, `docs/thesis_guide.md`). For full technical detail — equations, network architectures, and hyperparameters — see the development guide and skill specifications under `docs/`.*
