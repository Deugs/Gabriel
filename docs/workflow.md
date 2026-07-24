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

### Phase 2: Baselines (Week 3)
**Duration**: 1 week
**Owner**: Baseline Implementer
**Deliverables**:
- [ ] `baselines/all_on_uniform.py` — All RRHs ON, uniform power
- [ ] `baselines/greedy_heuristic.py` — Greedy RRH selection
- [ ] `baselines/nmbs_binpack.py` — Al-Zubaedi's NMBS
- [ ] `baselines/convex_power.py` — CVXPY power allocation
- [ ] `agents/ddqn_agent.py` — Iqbal's DDQN reproduction
- [ ] `tests/test_baselines.py` — Validation tests

**Key Decisions**:
- CVXPY for convex sub-problems (matches Fathy/Iqbal)
- DDQN from Stable-Baselines3 (proven implementation)
- Identical environment for all baselines

**Validation**:
- DDQN reproduces Iqbal's ~22% savings
- Convex baseline matches analytical solution
- All baselines run without errors

**Entry Criteria**: Phase 1 complete
**Exit Criteria**: Baseline results generated; comparison table drafted

---

### Phase 3: Proposed Method (Week 4-6)
**Duration**: 3 weeks
**Owner**: Methodology Validator + Code Reviewer
**Deliverables**:
- [ ] `agents/hybrid_sac_dqn.py` — Hybrid SAC-DDQN implementation
- [ ] `training/train_hybrid.py` — Training loop
- [ ] `config/default.yaml` — Default hyperparameters
- [ ] `config/small_network.yaml` — Small scenario
- [ ] `config/large_network.yaml` — Large scenario
- [ ] `tests/test_hybrid_agent.py` — Agent unit tests

**Key Decisions**:
- Factorized discrete actions (per-RRH binary)
- Twin critics for reduced overestimation
- LayerNorm for training stability
- Auto-tuned SAC temperature (optional)

**Validation**:
- Agent runs 100 episodes without crash
- Critic loss decreases
- Reward improves over random policy
- Action space valid for environment

**Entry Criteria**: Phase 2 complete
**Exit Criteria**: Training converges; hyperparameters stable

---

### Phase 4: Experiments (Week 6-9)
**Duration**: 4 weeks
**Owner**: Experiment Runner + Figure Designer
**Deliverables**:
- [ ] Convergence curves (all algorithms, 5 seeds)
- [ ] Energy efficiency comparison (24-hour average)
- [ ] QoS performance analysis (SINR CDF)
- [ ] Ablation study (4 variants)
- [ ] Scalability analysis (4 network sizes)
- [ ] Statistical significance tests

**Experiment Matrix**:

| Algorithm | R=5, U=2 | R=12, U=10 | R=20, U=20 | R=50, U=50 |
|-----------|----------|------------|------------|------------|
| All ON | 5 seeds | 5 seeds | 5 seeds | — |
| Greedy | 5 seeds | 5 seeds | 5 seeds | — |
| NMBS | 5 seeds | 5 seeds | — | — |
| Convex | 5 seeds | 5 seeds | — | — |
| DDQN | 5 seeds | 5 seeds | 5 seeds | — |
| DDPG | 5 seeds | 5 seeds | — | — |
| SAC | 5 seeds | 5 seeds | 5 seeds | — |
| TD3 | 5 seeds | 5 seeds | — | — |
| Hybrid (proposed) | 5 seeds | 5 seeds | 5 seeds | 5 seeds |

**Total Runs**: ~225 training jobs

**Validation**:
- All results reproducible from config files
- Confidence intervals computed
- Statistical tests significant (p < 0.05)

**Entry Criteria**: Phase 3 complete
**Exit Criteria**: All figures and tables generated

---

### Phase 5: Thesis Writing (Week 8-12)
**Duration**: 5 weeks (parallel with Phase 4)
**Owner**: Thesis Writer + Thesis Architect
**Deliverables**:
- [ ] Chapter 1: Introduction (revised)
- [ ] Chapter 2: Literature Review (expanded)
- [ ] Chapter 3: System Model & Problem Formulation (restructured)
- [ ] Chapter 4: Simulation Results (written)
- [ ] Chapter 5: Conclusion & Future Work (written)
- [ ] Abstract
- [ ] List of Figures/Tables
- [ ] References (complete BibTeX)

**Writing Schedule**:

| Week | Focus | Target Words |
|------|-------|-------------|
| 8 | Ch. 1 revision + Ch. 2 expansion | +1,500 |
| 9 | Ch. 3 restructuring (MDP, hybrid algorithm) | +3,000 |
| 10 | Ch. 4 draft (using preliminary results) | +3,000 |
| 11 | Ch. 5 + Abstract + integration | +2,000 |
| 12 | Full draft review + supervisor feedback | Revision |

**Validation**:
- Each chapter passes quality gates
- Cross-references verified
- Figures referenced in text
- All claims cited

**Entry Criteria**: Phase 3 underway
**Exit Criteria**: Complete draft submitted to supervisor

---

### Phase 6: Revision & Submission (Week 13-14)
**Duration**: 2 weeks
**Owner**: All agents
**Deliverables**:
- [ ] Supervisor feedback incorporated
- [ ] Final proofreading
- [ ] Plagiarism check passed
- [ ] Formatting compliance verified
- [ ] PDF generated successfully
- [ ] Code repository tagged (thesis-v1.0-final)

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

```
Week 0:  [====] Setup
Week 1:  [========] Environment
Week 2:  [========] Environment (cont.)
Week 3:  [====] Baselines
Week 4:  [========] Hybrid Agent (core)
Week 5:  [========] Hybrid Agent (training)
Week 6:  [====] Agent stabilization + Experiments start
Week 7:  [========] Experiments (convergence)
Week 8:  [========] Experiments (ablation) + Ch. 1-2 writing
Week 9:  [========] Experiments (scalability) + Ch. 3 writing
Week 10: [========] Ch. 4 writing
Week 11: [========] Ch. 5 + integration
Week 12: [========] Full draft + supervisor review
Week 13: [========] Revision round 1
Week 14: [====] Final revision + submission
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
