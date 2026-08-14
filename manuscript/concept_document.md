# Research Concept Document

**Optimization of Energy-Efficient Cloud Radio Access Networks (C-RAN) for 5G Using a Hybrid Deep Reinforcement Learning Framework**

| | |
|---|---|
| **Candidate** | Gabriel Kwame Freeman |
| **Degree** | MPhil (Master of Philosophy) |
| **Institution** | KNUST |
| **Supervisor** | Prof. J. J. Kponyo |
| **Document version** | 1.0, with a status update appended 05 August 2026 |
| **Date** | 24 July 2026 (original); status update 05 August 2026 |
| **Distribution** | Supervisor; Department Graduate Committee; Thesis Examination Board |

> **Note on title**: the working title inherited from the original proposal references "Deep Deterministic Policy Gradient" (DDPG). As explained in Section 3, the methodology has since been revised — first to a hybrid SAC-DDQN framework, and then, per the supervisor's review of that revision, to a branching, multi-pass, twin-critic parameterized DQN. The title above still holds for all of these; the DRL algorithm identity is an implementation detail resolved in `manuscript/MPhil_Thesis_Concept_Note_v4.md`, not a change to the research question. It supersedes the original title for all purposes except the official registration record, which the candidate will update through the appropriate departmental process.

> **Status update (05 August 2026)**: this document is preserved close to its original 24 July 2026 form for provenance, since it is what the supervisor first reviewed. It is no longer the governing technical document — that is now `manuscript/MPhil_Thesis_Concept_Note_v4.md` (v4.0), which resolved two further rounds of supervisor review (blockers B1–B4, recommendations S1–S6, advisories A1–A6, and fourteen critical gaps G1–G14) and is itself now backed by tested code, not just a plan. Status-update notes are added inline below wherever this document's original content has since changed; sections with no such note are unchanged. See `manuscript/response_to_supervisor_review.md` for the candidate's consolidated reply to the supervisor's review, and `docs/supervisor_feedback_log.md` for the full review history.

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

**Revised approach**: a hybrid actor-critic architecture that pairs a **DDQN-style discrete actor** (RRH activation, updated via MSE loss against bootstrapped Bellman Q-targets) with a **SAC continuous actor** (power allocation, bounded via $\tanh$ squashing with Jacobian log-probability correction), coordinated through a **shared twin critic** evaluating joint actions $Q(s,v,p)$. The physical environment models downlink co-channel interference where uncoordinated active RRHs induce inter-cell interference $I_u = \sum_{r \neq r^*(u)} P_r |h_{r,u}|^2$. This is a direct methodological upgrade, not a change of research question — the core question and the C-RAN system model are unchanged; only the DRL algorithm changes, and the change is a formalization of the discrete-continuous split every reasonable design already required.

This revision is flagged explicitly per the project's own scope-boundary process, which requires supervisor awareness of any deviation from the originally registered method.

> **Status update (05 August 2026)**: the hybrid SAC-DDQN architecture above was itself superseded before implementation began. Supervisor review of this document raised two concrete objections the SAC-DDQN design did not resolve: (i) how a "shared twin critic" ingests a joint discrete-continuous action was never specified concretely (blocker B2), and (ii) SAC's entropy-regularized continuous exploration and DDQN's ε-greedy discrete exploration are driven by incompatible objectives with no stated reconciliation. Rather than patch SAC-DDQN, the candidate adopted a **branching, multi-pass, twin-critic parameterized DQN** (Concept Note v2.0 §10, unchanged through v4.0): R independent per-RRH dueling heads (2R outputs rather than a 2^R joint head — resolving blocker B3, the combinatorial action space, as a side effect of the same redesign), each coupled to its own continuous power/bandwidth parameters via MP-DQN's multi-pass masking (Bester et al., 2019) rather than P-DQN's original single-pass coupling (Xiong et al., 2018), with TD3-style twin critics (Fujimoto et al., 2018) for overestimation control. `agents/hybrid_sac_dqn.py` is retained in the codebase only as the superseded alternative for comparison; `agents/branching_mp_dqn.py` is the current proposed method. As with the DDPG→SAC-DDQN change above, this is a further algorithm-identity change, not a change to the research question, system model, or MDP.

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

> **Status update (05 August 2026)**: the supervisor's review of the objectives above (gaps G10–G11) noted that "≥25% vs. all-RRHs-on" is a weak floor most methods clear easily, and that a "≥5%" margin is a slim target given typical DRL seed-to-seed variance. The objectives are revised (Concept Note v4.0 §5.2) so the **headline comparison is now the margin over DDQN, P-DQN, and MP-DQN directly** — the all-ON baseline is relabeled a sanity floor rather than the primary claim — and the seed count doubled from 5 to 10 with Cohen's d reported alongside every p-value (§12.4) to give that margin real statistical power. The QoS target (≤5% violation rate) is unchanged but is now justified against ITU-R M.2410 / 3GPP TS 22.261 for eMBB traffic specifically (§12.7), not asserted without a standard to anchor it.

## 5. Proposed Contribution

**Primary contribution**: a hybrid actor-critic DRL framework that jointly optimizes discrete RRH activation and continuous transmit power allocation through a shared critic, eliminating the need for an analytical sub-problem solver and enabling genuinely end-to-end policy learning — in contrast to prior two-stage approaches (Fathy et al., 2021 [3]; Iqbal et al., 2021 [2]) that decouple the discrete and continuous decisions.

**Secondary contributions**:
- A power-consumption model validated against the EARTH/GreenTouch standard parameters (via Al-Zubaedi, 2019) [4], correcting parameter values used in the original draft.
- A comprehensive baseline comparison spanning heuristic, convex-optimization, pure-discrete-RL, and pure-continuous-RL methods, all evaluated under an identical protocol.
- A scalability analysis characterizing performance and training cost as network size grows from 5 to 50 RRHs.

> **Status update (05 August 2026)**: the literature review underlying this contribution claim has grown from the 4 references below to 30 (Concept Note v4.0 §4, §17), engaging directly with the parameterized-action-space DRL family (P-DQN through HySoft), the O-RAN DRL energy-optimization literature, and a 2025 hybrid A3C-Dueling-DQN C-RAN paper the original draft had not accounted for. The closest published related work located, EExApp (Lu, Yan & Zeng, 2026), is now discussed explicitly (§4.4) rather than only tabulated: it also couples a discrete and a continuous RAN-energy decision, but via two separate actor-critic pairs rather than this thesis's one coupled network, and is validated on a real O-RAN testbed rather than in simulation — a genuine point of comparison the contribution claim must now be defended against. The policy is also now explicitly framed as a candidate O-RAN rApp (discrete decisions via O1, continuous decisions via E2 — §11 of the concept note), which costs nothing in implementation but situates the work in the architecture the field has actually moved toward since 2022.

## 6. Methodology Summary

- **Environment**: a Gymnasium-compatible C-RAN simulator combining a Rayleigh-fading channel model, a tidal (time-of-day) traffic model, and an EARTH-aligned power model covering RRH, BBU pool, and fronthaul (TWDM-PON) consumption.
- **State, action, reward**: a formal MDP is defined — channel gains, RRH activation state, per-UE traffic demand, and time-of-day compose the state; the action is a joint (discrete RRH vector, continuous power vector); the reward penalizes total power draw, QoS violations, and RRH switching cost.
- **Agent**: the hybrid SAC-DDQN architecture described in Section 3, trained with a replay buffer over hybrid (state, discrete action, continuous action, reward, next state) transitions.
- **Baselines**: All-ON + uniform power, greedy heuristic, NMBS bin-packing (Al-Zubaedi, 2019) [4], convex power allocation (CVXPY), and reproduced DDQN (Iqbal et al., 2021) [2], plus pure-SAC, pure-TD3, and pure-DDPG agents for algorithm-family comparison.
- **Evaluation**: convergence, energy efficiency, QoS performance, an ablation study (removing switching cost / fronthaul power / QoS penalty individually), and a scalability sweep — all averaged over 5 fixed random seeds with reported confidence intervals and statistical significance tests.

> **Status update (05 August 2026)**: every bullet above has since expanded. **Agent**: the branching MP-DQN + TD3 architecture from Section 3's update note, not hybrid SAC-DDQN. **Baselines**: ten methods, not five — the five above plus reproduced ANN+GSBF (Fathy et al., 2021), a two-stage DDQN+SOCP reproduction of Iqbal et al. (2021) (distinct from the plain convex-only allocator already listed, which fixes RRH selection rather than coupling it to a learned discrete policy), pure-DDPG (kept specifically to answer RQ3, the discrete-vs-continuous-relaxation question), and **P-DQN and MP-DQN** (added per supervisor recommendation S2, specifically to isolate branching's own contribution — deliberately run only at R=5/R=12 where the flat 2^R joint action space they use without branching remains tractable, itself empirical evidence for why branching was necessary at scale). **Evaluation**: 10 seeds, not 5, with Cohen's d effect sizes reported alongside every significance test (S4); a CSI-robustness evaluation (perturbing the trained policy's observed channel gains at σ∈{0,0.01,0.05,0.1}, S3) and a cross-profile generalization evaluation (zero-shot transfer from a weekday/urban to a weekend/suburban traffic profile, A5) were added to address the perfect-CSI and single-traffic-pattern limitations without expanding training scope; an inference-latency benchmark at R=5,12,20,35,50 was added (A3) to report deployability, not just accuracy. All ten baselines and all three new evaluations now exist as tested code (`agents/`, `evaluation/`), not only as a plan — see Section 11 below for what has and has not yet been run at full scale.

## 7. Significance

Reducing RAN energy consumption directly reduces mobile operators' operating expenditure and carbon footprint, both increasingly material concerns as 5G densification continues and networks evolve toward 6G. A DRL controller that jointly reasons over discrete and continuous decisions — rather than decoupling them — is also a methodological contribution relevant beyond this specific application, to any control problem mixing on/off and continuous-value decisions under a shared objective.

## 8. Scope and Limitations

**In scope**: downlink transmission; a single BBU pool; a single, centralized DRL agent; simulation-based evaluation; the assumption of perfect channel state information (CSI), acknowledged as a limitation.

**Out of scope** (absent further approval): uplink traffic, multi-pool scenarios, multi-agent RL, imperfect CSI *training*, hardware testbed validation, and 6G-specific features. Any expansion of scope will be brought back to the supervisor/committee as a further revision to this document.

> **Status update (05 August 2026)**: the perfect-CSI assumption remains in scope for *training* — unchanged — but is no longer left as an unexamined limitation. A bounded, evaluation-only CSI-robustness experiment (Section 6's update note above; Concept Note v4.0 §12.5) now characterizes how much the trained policy's performance degrades under channel-estimation error, without requiring a CSI-robust training objective, which stays out of scope. Everything else in this section is unchanged, including the single-agent, simulation-only, no-hardware-testbed boundaries.

## 9. Indicative Timeline

> **Status update (05 August 2026)**: the 14-week timeline below was the original estimate before the methodology and evaluation plan grew per the supervisor's review (ten baselines instead of five, ten seeds instead of five, three additional evaluation protocols, and the architecture change in Section 3). It is superseded by the week-by-week Gantt chart in Concept Note v4.0 §15, now approximately **27 weeks** from approval, summarized here rather than reproduced in full:

| Phase | Duration | Status as of 05 August 2026 |
|---|---|---|
| Environment | Weeks 1–2 | Done — `cran_env/` implemented and tested |
| Baselines (10 methods) | Weeks 2–7 | Done — all 10 implemented and unit-tested (`agents/`, `baselines/`); no full-scale run yet |
| Proposed agent | Weeks 5–11 | Done — `agents/branching_mp_dqn.py` implemented and unit-tested |
| Experiments | Weeks 13–19 | **Not started** — convergence, energy, QoS, ablation, scalability, CSI-robustness, generalization, latency, and the §12.11 hyperparameter proxy sweep all have tested infrastructure but no full 10-seed run yet |
| Thesis writing (Ch. 1–5) | Weeks 17–25 (parallel) | Chapters 1–2 drafted from this document and its predecessors; Chapters 3–5 not started |
| Revision & submission | Weeks 26–27 | Not started |

The original 14-week total is retired; ~27 weeks from approval is the current estimate, with R=50 explicitly a stretch goal rather than a committed deliverable (Concept Note v4.0 §15).

## 10. Key Risks

| Risk | Mitigation |
|---|---|
| Hybrid agent training instability | Start from a proven pure-SAC baseline; add the discrete head incrementally |
| Insufficient GPU/compute time | Prioritize the medium network size (12 RRH); extrapolate scalability results |
| Reproduced baselines don't match published numbers | Unit-test each baseline independently; document any residual discrepancy transparently |
| Scope creep | Enforced via the scope boundary in Section 8; any deviation returns to this document for re-approval |

> **Status update (05 August 2026)**: the risk mitigation in row 1 above ("start from pure-SAC") no longer applies to the current architecture, since SAC itself was superseded (Section 3's update note). The current architecture's equivalent risks, per Concept Note v4.0 §14, are: per-step compute cost growing with R (branching means R forward passes per critic evaluation; profiling is scheduled early rather than discovered late), and P-DQN/MP-DQN's baseline being unable to scale past R≈12–15 without branching — reported as a documented finding, not treated as a bug, since it is itself evidence for why branching was necessary (Section 3's update note; Concept Note v4.0 §10.3.1).

## 11. Request to Stakeholders

The candidate's original three requests below (24 July 2026) have already been answered across two subsequent supervisor review rounds, both logged in `docs/supervisor_feedback_log.md` and resolved in Concept Note v4.0 — they are preserved for provenance, not because they remain open:

1. ~~Acknowledgement of the methodological revision described in Section 3 (DDPG → hybrid SAC-DDQN).~~ Superseded — the architecture has since moved again, to branching MP-DQN + TD3 (Section 3's update note).
2. ~~Confirmation that the objectives and scope in Sections 4 and 8 remain acceptable.~~ Revised per supervisor feedback (Sections 4, 8, and 6's update notes) and re-confirmed across the v2.0 and v3.0 review rounds.
3. ~~Any feedback on the indicative timeline in Section 9 before implementation begins in earnest.~~ Received and incorporated; see Section 9's update note.

**Current requests, as of 05 August 2026** (detailed in full in `manuscript/response_to_supervisor_review.md`):
1. **Sign-off on Concept Note v4.0** as the governing document, so the architecture, ten-method baseline suite, and evaluation plan can be treated as settled.
2. **Confirmation that the current single-agent, simulation-only, O-RAN-*framed*-but-not-O-RAN-*implemented* scope is still acceptable** — unchanged from what v3.0 already proposed.
3. Any remaining concerns **before the full 10-seed × 11-method experiment matrix is run**, since that is the next step and represents a non-trivial compute commitment.

## 12. References

[1] H. Shengren, E. M. Salazar Duque, P. P. Vergara, and P. Palensky, "Performance comparison of deep RL algorithms for energy systems optimal scheduling," in *2022 IEEE PES Innovative Smart Grid Technologies Conference Europe (ISGT-Europe)*, IEEE, 2022, pp. 1–6, doi: 10.1109/ISGT-Europe54678.2022.9960642.

[2] A. Iqbal, M.-L. Tham, and Y. C. Chang, "Double Deep Q-Network-Based Energy-Efficient Resource Allocation in Cloud Radio Access Network," *IEEE Access*, vol. 9, pp. 20440–20449, 2021, doi: 10.1109/ACCESS.2021.3054909.

[3] M. Fathy, M. S. Abood, and M. M. Hamdi, "Optimization of Energy-Efficient Cloud Radio Access Networks for 5G using Neural Networks," in *2021 International Conference on Intelligent Technology, System and Service for Internet of Everything (ITSS-IoE)*, 2021, doi: 10.1109/ITSS-IoE53029.2021.9615290.

[4] W. H. A. Al-Zubaedi, "Planning a C-RAN Deployment for the Next Generation Cellular Networks," Ph.D. dissertation, Dept. of Electronic and Computer Engineering, Brunel University London, London, U.K., 2019. Supervisors: H. Al-Raweshidy, A. Zobaa. [Online]. Available: <http://bura.brunel.ac.uk/handle/2438/17865>

> References [2] and [3] were verified directly against the source PDFs held in `references/`. References [1] and [4] were supplied directly by the candidate; no source PDF for either is on file in `references/` — the candidate should confirm both against a source copy before this document or the thesis bibliography is finalized.

> **Status update (05 August 2026)**: reference [2]'s end page, originally left unconfirmed here, is filled in above as 20449 to match the value used consistently in Concept Note v2.0/v3.0/v4.0 §17 — the candidate should still confirm this directly against the source PDF, since no record of an independent page-number verification for this specific reference was found in `docs/supervisor_feedback_log.md`. References [1] and [4] remain unverified as above. The architecture described in Section 3's update note is grounded in four further references, added to the bibliography in Concept Note v4.0 §17 along with 25 others (30 total):
>
> [5] W. Xiong, B. Wang, Z. Yang, et al., "Parametrized Deep Q-Networks Learning: Reinforcement Learning with Discrete-Continuous Hybrid Action Space," arXiv:1810.06394, 2018.
>
> [6] C. J. Bester, S. D. James, and G. D. Konidaris, "Multi-Pass Q-Networks for Deep Reinforcement Learning with Parameterised Action Spaces," arXiv:1905.04388, 2019.
>
> [7] A. Tavakoli, P. Pardo, and P. Kormushev, "Action Branching Architectures for Deep Reinforcement Learning," in *Proc. AAAI Conf. Artificial Intelligence*, vol. 32, no. 1, 2018.
>
> [8] S. Fujimoto, H. van Hoof, and D. Meger, "Addressing Function Approximation Error in Actor-Critic Methods," in *Proc. Int. Conf. Machine Learning (ICML)*, 2018, pp. 1587–1596.
>
> The full, independently-verified 30-reference bibliography — including the O-RAN DRL energy-optimization literature and the 2025 hybrid A3C-Dueling-DQN C-RAN paper cited in Section 5's update note — is maintained in Concept Note v4.0 §17, not duplicated here.

---

*Prepared from the project's working development documentation (`AGENTS.md`, `docs/workflow.md`, `docs/thesis_guide.md`). For full technical detail — equations, network architectures, and hyperparameters — see `manuscript/MPhil_Thesis_Concept_Note_v4.md`, the development guide, and the skill specifications under `docs/`.*
