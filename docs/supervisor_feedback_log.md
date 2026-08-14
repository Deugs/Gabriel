# Supervisor Feedback Log

> Filled instances of `docs/supervisor_feedback_template.md`. Newest entry first.

## Follow-up audit: 2026-08-05 — post-merge recheck + §12.11 proxy sweep implementation

**Requested**: independently re-verify (not just re-summarize) that the consolidated-letter fixes below actually match the Concept Note v4.0 spec and introduced no regressions, after merging them into `main` (PR #4).

**Result — all five implementations verified correct** by reading the actual masking/gradient/noise-isolation logic directly (not the prior session's own docstrings): P-DQN's single unmasked pass vs. MP-DQN's genuine per-action parameter masking, both single-critic; DDPG's continuous activation gradient flowing through the un-thresholded value; CSI noise touching only a copy of the observation, never the environment's true channel state; the `weekend_suburban` profile genuinely flatter/later/lower than `weekday_urban`; the latency benchmark's exact R=5/12/20/35/50 sweep. Full suite: 44/44 passed, no stale-reference regressions found repo-wide.

**One real gap found and closed in this follow-up**: Concept Note §12.11 specifies an exact hyperparameter-tuning proxy sweep (R=5, U=2, 100 episodes, 2 seeds, varying the branch/continuous-net learning-rate pair and τ by ~half an order of magnitude) that had no matching code — `training/hyperparam_search.py` only had a generic grid-search utility sweeping different parameters at the wrong network size. **Added** `run_proxy_sensitivity_sweep()` implementing the exact documented protocol (6 variants: lr-pair down/default/up, τ down/default/up), reporting a per-variant crash/stability check and a keep-vs-change decision per dimension, with the decision left for a human to log in `docs/daily_log_template.md` per item 3 rather than auto-writing to it. Also added a `docs/workflow.md` Phase 4 line item for this — the previous session's status trackers had no line for it at all, so it could have been silently lost.

**One disclosed-but-still-open item, not a regression**: `evaluation/scalability.py`'s training-time sweep still uses a different RRH-size set (6/12/24) than §12.2's table (5/12/20/35/50) and than the new `latency_benchmark.py`; this was already explicitly disclosed as the reason a separate latency module was added rather than fixing `scalability.py` itself, and remains open for whoever picks up Phase 4.

### Action Items
- [x] Independently re-verify the five consolidated-letter fixes against actual code (not summaries)
- [x] Implement `training/hyperparam_search.py::run_proxy_sensitivity_sweep` (§12.11/G9) with a short-run test
- [x] Add a `docs/workflow.md` Phase 4 tracking line for the §12.11 sweep
- [x] Run the §12.11 sweep at full scale (100 episodes, 2 seeds) and log its keep/change decision — done 2026-08-05, both the lr-pair and τ defaults kept unchanged (no variant crashed; the defaults outperformed both swept alternatives in each dimension); full results in `data/results/proxy_sweep/` and `docs/daily_log.md`
- [ ] Reconcile `evaluation/scalability.py`'s RRH-size set with §12.2's table (5/12/20/35/50), or explicitly document why it intentionally differs

---

## Review: 2026-08-05 — Consolidated letter (Overall/Methodology Assessment, Critical Gaps G1-G14, Scientific Relevance, Recommendations B1-B4/S1-S6/A1-A6)

### Agenda
1. A single consolidated review letter combining, verbatim, the same items already logged separately below as "Review of Concept Note v2.0" (B1-B4, S1-S6, A1-A6) and "Detailed review" (G1-G14) — cross-checked item-by-item to confirm nothing was missed, rather than re-answered from scratch.
2. Distinguishing what is genuinely *documented* (in the Concept Note v3.0/v4.0 lineage) from what is *implemented in code* — the two had drifted apart.

### Feedback Received

Every individual B/S/A/G item in this letter maps to a row already marked "Done" in the two entries below (v3.0 or v4.0 section references there). Re-verifying against the actual codebase (not just the docs) surfaced five places where the documentation's "Done" was accurate for the *concept note* but not yet true of the *code*:

| Section | Feedback | Priority | Action | Deadline |
|---|---|---|---|---|
| S2 | Add P-DQN and/or MP-DQN as baselines | Strongly recommended | `docs/workflow.md` and Concept Note §12.1 already claimed these as implemented, but `agents/pdqn_agent.py`/`agents/mpdqn_agent.py` did not exist. **Now implemented**: flat (non-branching) 2^R joint-action head, single critic, P-DQN single-pass vs. MP-DQN genuine multi-pass masking (verified by a dedicated unit test that perturbing a masked-out RRH's params does not change that action's Q-value); guarded against R>12 with an explicit error rather than a silent OOM (the guard was originally miswired to R>20; corrected in a later audit pass to match this R=12 cap) | This session |
| RQ3 (v1 concept note's design) | Pure-DDPG continuous-relaxation baseline | Strongly recommended | `agents/ddpg_agent.py` did not exist despite being listed as a `docs/workflow.md` Phase 2 deliverable. **Now implemented**: DDPG actor outputs a continuous "soft" RRH-activation gate thresholded only at execution time, per `manuscript/concept_document.md` §3's description of the original design | This session |
| S3 | CSI robustness evaluation (σ∈{0,0.01,0.05,0.1}) | Strongly recommended | Concept Note §12.5 fully specified this; no code existed. **Now implemented**: `evaluation/csi_robustness.py` perturbs only the channel-gain slice of the *observation* fed to the frozen policy, leaving the environment's true physics/reward unaffected, matching the spec's "evaluation-only, no retraining" design | This session |
| A5/G13 | Cross-profile generalization evaluation | Advisory | No second traffic profile existed anywhere in code. **Now implemented**: `cran_env/traffic_model.py` gained a `weekend_suburban` profile (flatter daytime, later/lower residential peak) alongside the existing default `weekday_urban`; `evaluation/generalization.py` trains once and evaluates zero-shot on both | This session |
| A3/G14 | Inference-latency benchmark at R=5,12,20,35,50 | Advisory | `evaluation/scalability.py` measured full env-step time at a different, mismatched size set (6/12/24). **Added** `evaluation/latency_benchmark.py`, measuring isolated forward-pass time at exactly the documented five sizes, explicitly skipping P-DQN/MP-DQN above their R=12 cap rather than crashing — itself further empirical evidence for B3 | This session |
| S4/G11 | Effect size (Cohen's d) alongside p-values | Strongly recommended | Concept Note §12.4 committed to this; `evaluation/convergence.py` had no such function. **Added** `compute_cohens_d()`. While fixing this, found and fixed a **real bug**: the module's `proposed_algo` lookup was still the hardcoded, superseded string `"Hybrid_SAC_DDQN"`, but `training/train_hybrid.py` actually saves `"Branching_MP_DQN"` — every paired comparison against the real proposed method was silently comparing against an empty `[0.0]` placeholder array. Same stale-name bug also existed in `evaluation/scalability.py`'s result dict key | This session |

### Decisions Made

| Decision | Rationale | Impact |
|---|---|---|
| No new Concept Note version (v5.0) drafted for this letter | Every substantive item was already resolved in the *documentation* by v3.0/v4.0; the gap this letter's re-verification found was purely between docs and code, not a new methodological question for the supervisor | Concept Note v4.0 remains the current reference document; only code, tests, and status trackers (`AGENTS.md`, `docs/workflow.md`) were updated |
| P-DQN/MP-DQN implemented as a flat 2^R-action head (no branching, single critic), reusing the proposed method's `SharedEncoder`/`ContinuousParameterNetwork` | Matches §12.1's explicit "without branching or twin critics" framing exactly, while keeping the continuous-parameterization mechanism identical across methods for a fair comparison | `agents/mpdqn_agent.py` subclasses `agents/pdqn_agent.py`, overriding only the Q-evaluation step, to avoid duplicating the shared machinery |
| CSI noise applied to the observation only, not the environment's internal channel state | The spec ("frozen trained policy", "no retraining", isolate policy *sensitivity*) requires the true physics/reward to stay computed from the true channel; only what the policy *sees* should be noisy | No changes needed to `cran_env/cran_env.py`'s core physics |

### Next Meeting
**Date**: N/A — this letter's items were already scheduled for v3.0/v4.0 sign-off (see the two entries below); no new meeting triggered
**Focus**: N/A

### Action Items
- [x] Implement `agents/pdqn_agent.py`, `agents/mpdqn_agent.py`, `agents/ddpg_agent.py` (S2, RQ3) with unit tests
- [x] Implement `evaluation/csi_robustness.py` (S3), `evaluation/generalization.py` (A5/G13), `evaluation/latency_benchmark.py` (A3/G14), each with a short-run test
- [x] Add `compute_cohens_d()` to `evaluation/convergence.py` (S4/G11); fix the stale `"Hybrid_SAC_DDQN"` proposed-method lookup bug there and in `evaluation/scalability.py`
- [x] Update `AGENTS.md`'s Code Architecture tree and `docs/workflow.md`'s Phase 2/3 checklists to match actual file existence (previously inconsistent in both directions per the earlier gap analysis)
- [ ] Run the full 10-seed × 9-method experiment matrix and the CSI-robustness/generalization/latency sweeps at thesis scale (the infrastructure above is tested at small scale only; no full-scale results have been generated yet)

---

## Verification follow-up: 2026-08-05 — HySoft authorship

Requested: independently verify HySoft's authorship (the last open flag from the reference-verification pass below). Same standard applied as the OREO check: require a located identifier or independent corroborating signal, not just a repeated search snippet.

**Result — corroborated, not primary-source-confirmed.** Authors: Lu, J., Jia, Y., & Görges, D. (2025), *IFAC-PapersOnLine*, ICONS 2025 (Padova, Italy). Unlike the retracted OREO citation, this PII (S2405896325027430) consistently resolves to the same title/venue/benchmarks across independently-phrased searches, and Daniel Görges has his own separately-verifiable ScienceDirect author profile (ID 23569037600) and academic identity (RPTU Kaiserslautern-Landau, control systems/ML) that surfaces specifically tied to this article — an independent corroborating signal the OREO case never had. Concept Note v4.0 §4.2/§17 updated accordingly.

**Follow-up (same day): asked to complete the institutional-access check directly.** Retried the ScienceDirect fetch (still HTTP 403), and additionally checked IFAC-PapersOnLine's own listing and RPTU's KLUEDO open-access repository for a mirror — neither had a matching entry. This environment has no real library/institutional login, so it cannot get past ScienceDirect's access control regardless of how many times it's retried. **This is now a closed-out action item for the candidate**, not something further automated search can resolve: open the ScienceDirect page via KNUST's (or any) library subscription and confirm title/authors/volume before the thesis bibliography is finalized.

---

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
- [x] Independently verify the OREO, TS-MP-DQN, and CP-DQN-journal references (Concept Note v4.0 §4.2-§4.3, §17) — TS-MP-DQN and CP-DQN's journal are now fully confirmed. **OREO was retracted, not just flagged**: a direct IEEE Xplore check found no real paper matching the previously-cited description (PPO rApp, Sionna ray-tracing, Qazzaz/Salama/Hafeez/Zaidi); the only confirmed "OREO" is Mungari et al. (2024, INFOCOM, arXiv:2405.18198), a different topic (xApp orchestration), now cited correctly in its place. Take-away: the earlier "authors confirmed" note on OREO was itself wrong — repeated agreement across independent search queries is not the same as a located primary source, and should not have been treated as verification.
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
- [x] Independently verify the HySoft (2025) reference's author list/venue (Concept Note v4.0 §4.2) — corroborated (Lu, Jia & Görges; IFAC-PapersOnLine, ICONS 2025) via an independent evidence chain (Görges' separately-verifiable ScienceDirect author profile and academic identity). Institutional-access check attempted directly (retried ScienceDirect, checked IFAC-PapersOnLine and RPTU's KLUEDO repository) — no route past the paywall exists without a real library login. **Closed out as a candidate action item**, not resolvable by further automated search: open the ScienceDirect page via KNUST's (or any) library subscription before citing definitively.
- [ ] Supervisor sign-off on Concept Note v3.0
- [ ] Begin Phase 1 (environment & power model) per the revised `docs/workflow.md` timeline
