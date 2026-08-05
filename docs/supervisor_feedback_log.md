# Supervisor Feedback Log

> Filled instances of `docs/supervisor_feedback_template.md`. Newest entry first.

## Review: 2026-08-05 — Detailed review (Overall Assessment, Methodology Assessment, G1-G14, Scientific Relevance)

### Agenda
1. Full detailed review underlying the earlier condensed B1-B4/S1-S6/A1-A6 review
2. Fourteen critical gaps (G1-G14) in literature, technical design, and evaluation
3. Scientific relevance and timeliness of the C-RAN/perfect-CSI/single-agent framing given where the field (O-RAN) has moved

### Feedback Received

| Section | Feedback | Priority | Action | Deadline |
|---|---|---|---|---|
| §2.2 weaknesses | Shared twin critic ambiguity; exploration-strategy conflict (SAC entropy vs. ε-greedy); combinatorial 2^N explosion | High | Already resolved by the v2.0/v3.0 branching/MP-DQN/twin-critic redesign, which predates this review reaching the candidate — confirmed still valid, no further change needed | v4.0 §0.1 |
| G1 | Parameterized-action lineage incomplete: add TS-MP-DQN, CP-DQN | Blocker-equivalent | Done — added, with TS-MP-DQN's primary source flagged unverified | v4.0 §4.2 |
| G2 | O-RAN DRL energy work missing named systems: OREO, ES-xApp, federated TD3, EExApp | Blocker-equivalent | Done — all four added; EExApp identified as the closest published related work and discussed explicitly, not just tabulated | v4.0 §4.3, §4.4 |
| G3 | 2025 hybrid A3C-Dueling DQN C-RAN paper | Blocker-equivalent | Already resolved in v3.0 (Chuang et al., 2025) | v4.0 §4.1 |
| G4 | Acknowledge multi-agent/federated RL; justify single-agent scope | Recommended | Done, citing Liang et al. (2026) directly | v4.0 §7.1 |
| G5 | Acknowledge foundation-model/LLM-driven network control | Recommended | Done | v4.0 §7.1 |
| G9 | No hyperparameter tuning protocol | Recommended | Done — new lightweight, timeline-bounded protocol | v4.0 §12.11 |
| G10 | Primary baseline (≥25% vs. All-ON) is a weak floor | Recommended | Done — re-ranked; DDQN/P-DQN/MP-DQN margins are now the headline comparison | v4.0 §5.2 |
| G6-G8, G11-G14 | State-space dimensionality, reward-weighting methodology, reproducibility, statistical power, QoS-target context, generalization, inference latency | — | All already resolved in v3.0; confirmed still valid | v4.0 §0.1 |
| §4 relevance | Novelty must be positioned precisely given mature hybrid-action-space solutions and O-RAN's dominance | High | Novelty synthesis rewritten around EExApp specifically, not just the general literature | v4.0 §4.4 |

### Decisions Made

| Decision | Rationale | Impact |
|---|---|---|
| Keep the branching/MP-DQN/twin-critic architecture (not EExApp's dual-actor-dual-critic pattern) as the proposed method | The single-coupled-network design remains the thesis's specific contribution claim; EExApp is named as the nearest comparable architecture to be argued against empirically (RQ3/RQ4), not adopted wholesale | The eventual Chapter 4 results must now explicitly discuss how the proposed design compares to a dual-actor-dual-critic alternative, even without re-implementing EExApp itself |
| Do not add a foundation-model/LLM baseline | No foundation-model approach was located targeting this thesis's specific joint RRH-activation-and-power problem, so no like-for-like comparison exists yet | Acknowledged in Chapter 2/5 as future work rather than benchmarked |

### Next Meeting
**Date**: To be scheduled once Concept Note v4.0 is reviewed
**Focus**: Sign-off on v4.0; confirm the EExApp differentiation is convincing; green-light Phase 1 implementation start

### Action Items
- [x] Draft Concept Note v4.0 resolving the detailed review's G1-G14 and relevance discussion
- [x] Sync `AGENTS.md`/`README.md` pointers to v4.0
- [x] Independently verify the OREO, TS-MP-DQN, and CP-DQN-journal references (Concept Note v4.0 §4.2-§4.3, §17) — TS-MP-DQN and CP-DQN's journal are now fully confirmed; OREO's authors are confirmed but its venue/year/DOI could not be verified against a primary source and remains flagged
- [ ] Supervisor sign-off on Concept Note v4.0
- [ ] Begin Phase 1 (environment & power model) per `docs/workflow.md`

---

## Review: 2026-08-05 — Review of Concept Note v2.0

### Agenda
1. Review of `manuscript/MPhil_Thesis_Concept_Note_v2.0.md` (methodological revision, objectives/scope, timeline)
2. Blockers and recommendations for the next revision
3. Decision on the DDPG → hybrid SAC-DDQN → branching/MP-DQN/twin-critic methodological revision

### Feedback Received

| Section | Feedback | Priority | Action | Deadline |
|---|---|---|---|---|
| Literature review | Expand bibliography from 4 to 20-30 refs; engage with P-DQN-through-HySoft lineage, O-RAN DRL energy work, and a 2025 hybrid A3C-Dueling DQN C-RAN paper; revise the novelty claim | Blocker (B1) | Done — Concept Note v3.0 §4.2-4.4, §16 (23 refs) | v3.0 |
| Hybrid critic architecture | Specify concretely with a diagram; state P-DQN vs MP-DQN vs novel and justify | Blocker (B2) | Done — Concept Note v3.0 §10.3 (diagram + explicit MP-DQN justification) | v3.0 |
| Combinatorial action space | Explain how the discrete head handles 2^N actions for N RRHs | Blocker (B3) | Done — Concept Note v3.0 §10.3.1 (worked example; no 2^N head exists) | v3.0 |
| Timeline | 6-week plan infeasible; reduce scope or extend, with a Gantt chart | Blocker (B4) | Done — Concept Note v3.0 §15 (20-week Gantt; R=50 demoted to stretch goal) | v3.0 |
| O-RAN framing | Position as an rApp; discrete via O1, continuous via E2 | Strongly recommended (S1) | Done — Concept Note v3.0 §11 | v3.0 |
| Baselines | Add P-DQN and/or MP-DQN | Strongly recommended (S2) | Done — Concept Note v3.0 §12.1 | v3.0 |
| CSI robustness | Train on perfect CSI, evaluate under N(0,σ²), σ∈{0.01,0.05,0.1} | Strongly recommended (S3) | Done — Concept Note v3.0 §12.5 | v3.0 |
| Statistical power | 5→10+ seeds; report effect sizes (Cohen's d) | Strongly recommended (S4) | Done — Concept Note v3.0 §12.4; `docs/rules.md` seed list updated | v3.0 |
| Reward weighting | Define the methodology for λ1/λ2 | Strongly recommended (S5) | Done — Concept Note v3.0 §12.6 (manual tuning + sensitivity sweep) | v3.0 |
| QoS target | Justify 5% against a 3GPP/ITU spec; state traffic class | Strongly recommended (S6) | Done — Concept Note v3.0 §12.7 (ITU-R M.2410 + 3GPP TS 22.261; eMBB, not URLLC) | v3.0 |
| Traffic model | Describe real trace vs synthetic model | Advisory (A1) | Done — Concept Note v3.0 §12.8 | v3.0 |
| State representation at scale | MLP vs GNN/attention for large N | Advisory (A2) | Done — Concept Note v3.0 §12.9 | v3.0 |
| Inference latency | Benchmark forward-pass time at N=5,10,25,50 | Advisory (A3) | Done — Concept Note v3.0 §12.3 (bracketed by the 5/12/20/35/50 sweep) | v3.0 |
| Reproducibility | Commit to releasing code/models/configs | Advisory (A4) | Done — Concept Note v3.0 §12.10 | v3.0 |
| Generalization | Evaluate on a different traffic pattern | Advisory (A5) | Done — Concept Note v3.0 §12.3 | v3.0 |
| Alternative architectures | Discuss why SAC-DDQN-lineage chosen over P-DQN/HyAR/HySoft/hierarchical | Advisory (A6) | Done — Concept Note v3.0 §10.1 | v3.0 |

### Decisions Made

| Decision | Rationale | Impact |
|---|---|---|
| Methodological revision (DDPG → hybrid SAC-DDQN → branching/MP-DQN/twin-critic) acknowledged and approved in principle, subject to B2/B3 | Direction correct; architecture needed full specification before implementation | Implementation may proceed once B2/B3 are resolved (now done in v3.0 §10.3-10.3.1) |
| Objectives and scope (Sections 4, 8 of v2.0 / Sections 5, 8 of v3.0) accepted with amendments | Baseline targets strengthened (S2); perfect-CSI limitation addressed via robustness evaluation (S3) rather than scope expansion | Multi-agent RL, imperfect-CSI *training*, and hardware validation remain out of scope |

### Next Meeting
**Date**: To be scheduled once Concept Note v3.0 is reviewed
**Focus**: Sign-off on v3.0; green-light for Phase 1 (environment) implementation start

### Action Items
- [x] Draft Concept Note v3.0 resolving B1-B4, S1-S6, A1-A6
- [x] Sync `docs/rules.md` (seed list) and `docs/workflow.md` (phases, experiment matrix, milestone timeline) to v3.0
- [x] Sync `AGENTS.md` / `README.md` status tables and key-decisions log to v3.0
- [ ] Independently verify the HySoft (2025) reference's author list/venue before it enters the thesis bibliography (Concept Note v3.0 §4.2)
- [ ] Supervisor sign-off on Concept Note v3.0
- [ ] Begin Phase 1 (environment & power model) per the revised `docs/workflow.md` timeline
