# Workflow: C-RAN DRL Thesis Development

## Development Phases

### Phase 0: Setup (Week 0)
**Duration**: 3-5 days
**Deliverables**:
- [ ] Git repository initialized with branch structure
- [ ] Development environment (Python venv, dependencies installed)
- [ ] Project directory structure created
- [ ] W&B account configured
- [ ] Pre-commit hooks installed
- [ ] CI/CD pipeline (GitHub Actions) configured

**Entry Criteria**: None
**Exit Criteria**: `pytest tests/` passes on empty test suite

---

### Phase 1: Environment (Week 1-2)
**Duration**: 2 weeks
**Owner**: Code Reviewer + Thesis Architect
**Deliverables**:
- [ ] `cran_env/channel_model.py` — Path loss, shadowing, fading
- [ ] `cran_env/traffic_model.py` — Tidal patterns, burstiness
- [ ] `cran_env/power_model.py` — EARTH-validated power consumption
- [ ] `cran_env/cran_env.py` — Gymnasium-compatible environment
- [ ] `tests/test_env.py` — Unit tests (coverage >80%)
- [ ] `docs/equation_code_mapping.md` — Initial mapping

**Key Decisions**:
- Channel model: Rayleigh fading + log-normal shadowing
- Traffic model: Sinusoidal tidal + log-normal noise
- Power model: EARTH model parameters (Al-Zubaedi validated)

**Validation**:
- SINR matches analytical for single-RRH case
- Power model sums to known values
- Environment deterministic with fixed seed

**Entry Criteria**: Phase 0 complete
**Exit Criteria**: All tests pass; environment API stable

---

### Phase 2: Baselines (Week 2-7)
**Duration**: 5 weeks (revised from 1 week — 8 baselines, including two published reproductions and two parameterized-action-space methods, is not a one-week task; per supervisor review, Concept Note v3.0 §15/B4)
**Owner**: Baseline Implementer
**Deliverables**:
- [x] `baselines/all_on_uniform.py` — All RRHs ON, uniform power
- [x] `baselines/greedy_heuristic.py` — Greedy RRH selection
- [x] `baselines/nmbs_binpack.py` — Al-Zubaedi's NMBS
- [x] `baselines/convex_power.py` — CVXPY power allocation
- [x] `agents/ddqn_agent.py` — Iqbal's DDQN reproduction
- [x] `baselines/ann_gsbf.py` — Fathy's ANN + Bi-Section GSBF reproduction
- [x] `agents/ddpg_agent.py` — pure-DDPG continuous relaxation (RQ3 baseline)
- [x] `agents/pdqn_agent.py`, `agents/mpdqn_agent.py` — P-DQN and MP-DQN, **new per supervisor review S2** (Concept Note v3.0 §12.1); code enforces the R<=12-validated / untractable-at-R>=35 limitation with an explicit guard rather than a silent failure (§10.3.1/B3)
- [x] `tests/test_baselines.py`, `tests/test_new_baselines.py` — Validation tests (all 10 baselines + the proposed method)

**Key Decisions**:
- CVXPY for convex sub-problems (matches Fathy/Iqbal)
- DDQN from Stable-Baselines3 (proven implementation)
- P-DQN/MP-DQN capped at R≤12 by design, not by bug — the resulting scaling failure is itself evidence for the branching architecture (Concept Note v3.0 §12.1)
- Identical environment for all baselines

**Validation**:
- DDQN reproduces Iqbal's ~22% savings
- Convex baseline matches analytical solution
- All baselines run without errors

**Entry Criteria**: Phase 1 complete
**Exit Criteria**: Baseline results generated; comparison table drafted

---

### Phase 3: Proposed Method (Week 5-11)
**Duration**: 7 weeks (revised from 3 weeks; branching + multi-pass + twin-critic is more involved to stabilize than a single off-the-shelf agent, per Concept Note v3.0 §15/B4)
**Owner**: Methodology Validator + Code Reviewer
**Deliverables**:
- [x] `agents/branching_mp_dqn.py` — Branching, multi-pass, twin-critic parameterized DQN (the architecture specified in Concept Note v3.0 §10.3; `agents/hybrid_sac_dqn.py` remains only as the earlier, superseded alternative)
- [x] `training/train_hybrid.py` — Training loop
- [x] `config/default.yaml` — Default hyperparameters
- [x] `config/small_network.yaml` — Small scenario
- [x] `config/large_network.yaml` — Large scenario
- [x] `tests/test_hybrid_agent.py`, `tests/test_branching_mp_dqn.py` — Agent unit tests

**Key Decisions**:
- Factorized, branching discrete actions (per-RRH binary head; 2R outputs, not 2^R — Concept Note v3.0 §10.3.1)
- Multi-pass masking (MP-DQN) between the continuous parameter net and each branch — not optional (Concept Note v3.0 §10.3)
- Twin critics for reduced overestimation (TD3-style)
- LayerNorm for training stability

**Validation**:
- Agent runs 100 episodes without crash
- Critic loss decreases
- Reward improves over random policy
- Action space valid for environment; validated at R=3–5 before scaling up

**Entry Criteria**: Phase 2 complete
**Exit Criteria**: Training converges; hyperparameters stable

---

### Phase 4: Experiments (Week 13-19)
**Duration**: 7 weeks (revised from 4 weeks — 10 seeds instead of 5, plus the new CSI-robustness and cross-profile generalization evaluations, per supervisor review S3/S4; Concept Note v3.0 §15/B4)
**Owner**: Experiment Runner + Figure Designer
**Deliverables**:
- [ ] Convergence curves (all algorithms, **10 seeds**)
- [ ] Energy efficiency comparison (24-hour average)
- [ ] QoS performance analysis (SINR CDF)
- [ ] Ablation study: RQ3 (hybrid vs pure-DDPG) and RQ4 (hybrid vs P-DQN/MP-DQN)
- [ ] Scalability analysis (5 network sizes; R=50 is a stretch goal, not committed)
- [ ] **CSI-robustness curve** (σ ∈ {0, 0.01, 0.05, 0.1}, evaluation-only, no retraining — Concept Note v3.0 §12.5) — infrastructure done (`evaluation/csi_robustness.py`, tested), full-scale run not yet executed
- [ ] **Cross-profile generalization result** (weekday/urban-trained policy evaluated on weekend/suburban profile — Concept Note v3.0 §12.3) — infrastructure done (`evaluation/generalization.py`, `cran_env/traffic_model.py`'s `weekend_suburban` profile, tested), full-scale run not yet executed
- [ ] Inference-latency benchmark at R=5,12,20,35,50 — infrastructure done (`evaluation/latency_benchmark.py`, tested), full-scale run not yet executed
- [ ] Statistical reporting: paired t-tests + **Cohen's d** for every head-to-head comparison (Concept Note v3.0/v4.0 §12.4, S4/G11) — infrastructure done (`evaluation/convergence.py::compute_cohens_d`, tested; also fixed a bug where the proposed-method name check was still the superseded "Hybrid_SAC_DDQN", silently excluding it from every comparison)
- [ ] Statistical significance tests **and effect sizes (Cohen's d)** for every baseline comparison
- [x] **Hyperparameter-tuning proxy sweep** (R=5, U=2, 100 episodes, 2 seeds; lr-pair and τ each varied ~half an order of magnitude, per Concept Note v4.0 §12.11/G9) — run at full scale on 2026-08-05 (`data/results/proxy_sweep/`); no variant crashed, and the defaults tested (lr_discrete=1e-3, lr_actor=1e-4, τ=0.005) were in fact the best-performing of the three tested values in both dimensions, so both were **kept unchanged** per item 2. Decision logged in `docs/daily_log.md` (2026-08-05 entry) per item 3. **Caveat added later**: `config/default.yaml`'s actual lr_discrete/lr_actor (1e-4/3e-4) do not match the lr-pair this sweep tested/validated (1e-3/1e-4) — the sweep's "kept unchanged" conclusion was about that tested pair, not about today's actual config values, which were never themselves swept. Re-running the sweep centered on the real defaults is needed before treating them as validated. **Date note**: this ran on 2026-08-05, five days before Concept Note v4.0 §15's Gantt has Phase 1 (Environment) starting (2026-08-10). This is preliminary/exploratory validation of the Section 12.11 gate criterion carried out on early dev infrastructure ahead of the formally scheduled timeline, not evidence that Phase 4 itself started early — the full 10-seed × 11-method experiment matrix below remains unstarted, consistent with the Gantt.

**Experiment Matrix**:

| Algorithm | R=5, U=2 | R=12, U=10 | R=20, U=20 | R=35 | R=50 (stretch) |
|-----------|----------|------------|------------|------|----------------|
| All ON | 10 seeds | 10 seeds | 10 seeds | 10 seeds | — |
| Greedy | 10 seeds | 10 seeds | 10 seeds | 10 seeds | — |
| NMBS | 10 seeds | 10 seeds | — | — | — |
| Convex | 10 seeds | 10 seeds | — | — | — |
| DDQN | 10 seeds | 10 seeds | 10 seeds | 10 seeds | — |
| ANN+GSBF | 10 seeds | 10 seeds | — | — | — |
| DDPG (pure) | 10 seeds | 10 seeds | — | — | — |
| P-DQN | 10 seeds | 10 seeds | — | — | — |
| MP-DQN | 10 seeds | 10 seeds | — | — | — |
| Hybrid (proposed) | 10 seeds | 10 seeds | 10 seeds | 10 seeds | 10 seeds (stretch) |

**Total Runs**: ~440 training jobs (up from ~225 — reflects the two new baselines and the 5→10 seed increase; the P-DQN/MP-DQN rows are deliberately capped at R≤12 per §10.3.1/B3)

**Validation**:
- All results reproducible from config files
- Confidence intervals computed
- Statistical tests significant (p < 0.05), with Cohen's d reported alongside every test

**Entry Criteria**: Phase 3 complete
**Exit Criteria**: All figures and tables generated

---

### Phase 5: Thesis Writing (Week 9-24, parallel)
**Duration**: 16 weeks, parallel with Phases 3-4 (revised from 5 weeks to match the extended timeline in Concept Note v3.0 §15/B4)
**Owner**: Thesis Writer + Thesis Architect
**Deliverables**:
- [ ] Chapter 1: Introduction (revised — mention the O-RAN positioning, Concept Note v3.0 §11)
- [ ] Chapter 2: Literature Review (expanded to ~23 references per supervisor review B1; add the O-RAN and parameterized-action-space-lineage subsections, Concept Note v3.0 §4.2-4.4)
- [ ] Chapter 3: System Model & Problem Formulation (restructured; incorporate the critic-architecture diagram and the B3 combinatorial-action-space subsection, Concept Note v3.0 §10.3-10.3.1)
- [ ] Chapter 4: Simulation Results (written; include RQ4 ablation, CSI-robustness curve, generalization result, inference latency)
- [ ] Chapter 5: Conclusion & Future Work (written)
- [ ] Abstract
- [ ] List of Figures/Tables
- [ ] References (complete BibTeX; confirm the HySoft entry's authorship before finalizing, Concept Note v3.0 §4.2)

**Writing Schedule**:

| Week | Focus | Target Words |
|------|-------|-------------|
| 9-10 | Ch. 1 revision + Ch. 2 expansion (new lit.) | +2,000 |
| 11-14 | Ch. 3 restructuring (MDP, hybrid algorithm, diagram, B3 subsection) | +3,500 |
| 15-19 | Ch. 4 draft (using preliminary + main experiment results) | +3,500 |
| 20-22 | Ch. 5 + Abstract + integration | +2,000 |
| 23-24 | Full draft review + supervisor feedback | Revision |

**Validation**:
- Each chapter passes quality gates
- Cross-references verified
- Figures referenced in text
- All claims cited

**Entry Criteria**: Phase 3 underway
**Exit Criteria**: Complete draft submitted to supervisor

---

### Phase 6: Revision & Submission (Week 25-27)
**Duration**: 3 weeks (revised from 2 weeks)
**Owner**: All agents
**Deliverables**:
- [ ] Supervisor feedback incorporated
- [ ] Final proofreading
- [ ] Plagiarism check passed
- [ ] Formatting compliance verified
- [ ] PDF generated successfully
- [ ] Code repository tagged (thesis-v1.0-final)
- [ ] Reproducibility artifacts released (code, checkpoints, configs — Concept Note v3.0 §12.10/A4)

**Validation**:
- Pre-submission hook passes
- Word count: 15,000-25,000
- Figure count: >= 6
- Table count: >= 5
- Citation count: >= 50

**Entry Criteria**: Supervisor approves draft
**Exit Criteria**: Thesis submitted

---

## Milestone Timeline

Revised to ~27 weeks (from 14) per the supervisor review (Concept Note v3.0 §15/B4): two new baseline methods (P-DQN, MP-DQN), a 5→10 seed increase, and the CSI-robustness/generalization evaluations all needed time the original 14-week plan didn't have; R=50 is a stretch goal, not a committed deliverable, so the core plan does not depend on it landing. See Concept Note v3.0 §15 for the week-by-week Gantt chart; the summary below tracks phase-level milestones only.

```
Week 0:     [==] Setup
Week 1-2:   [====] Environment & power model
Week 2-7:   [========] Baselines (incl. P-DQN, MP-DQN — new)
Week 5-11:  [========] Hybrid agent (branching/multi-pass/twin-critic; build then scale)
Week 11-13: [====] CSI-robustness + reward-weight sensitivity harness (new)
Week 13-16: [========] Main experiments (10 seeds, RQ3/RQ4 ablations)
Week 16-19: [========] Extended experiments (scalability, CSI robustness, generalization, latency)
Week 9-24:  [================] Thesis writing (parallel, starts once early baselines land)
Week 25-26: [====] Full draft + supervisor review
Week 27:    [==] Final revision + submission
```

---

## Risk Mitigation

| Risk | Phase | Mitigation |
|------|-------|------------|
| Hybrid agent unstable | 3-4 | Fallback to pure SAC with thresholded actions; document as limitation |
| Insufficient GPU time | 4 | Prioritize medium network (R=12, U=10); extrapolate scalability |
| Baseline results don't match literature | 2 | Debug implementation; if unresolved, document discrepancy |
| Supervisor requests major restructuring | 5-6 | Build flexible LaTeX structure; maintain modular chapters |
| Word count exceeded | 5 | Move derivations to appendix; focus on key results |
| Code-text inconsistency discovered late | 5-6 | Automated consistency checker runs weekly |

---

## Communication Plan

| Stakeholder | Frequency | Channel | Content |
|-------------|-----------|---------|---------|
| Supervisor | Weekly | Email/Meeting | Progress update, blockers, decisions needed |
| Self (log) | Daily | `docs/daily_log.md` | What was done, what's next, blockers |
| Code repo | Per commit | Git + W&B | All code changes, experiment results |
| Thesis text | Per section | Git (LaTeX) | Draft sections for review |

---

## Definition of Done

The thesis is "done" when:
1. All chapters pass quality gates
2. All experiments are reproducible from config files
3. All figures are publication-quality PDFs
4. All baselines are fairly compared
5. Statistical significance is demonstrated
6. Supervisor has approved the final draft
7. Pre-submission hook passes without errors
8. Code repository is tagged and archived
