# Supervisor Feedback Log

> Filled instances of `docs/supervisor_feedback_template.md`. Newest entry first.

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
