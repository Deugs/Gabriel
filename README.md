# DRL Energy-Optimization Research — Developer Context Index

This repository hosts **two independent, fully-decoupled DRL research
tracks** that share no code in either direction. Don't assume a section
below applies to "the whole repo" — check which track it's labeled under.

| Track | Governs | Codebase | Governing Document |
|-------|---------|----------|---------------------|
| **O-RAN / BMPP-DQN** (thesis) | The actual MPhil thesis submission | `oran_env/`, `oran_agents/`, `oran_training/`, `oran_evaluation/`, `config/oran_default.yaml` | `manuscript/ORAN_BMPP_DQN_Concept_Note_v1.md` (v1.0, supervisor-approved) |
| **C-RAN / Branching MP-DQN + TD3** (publications) | Ongoing publication work, not the thesis submission | `cran_env/`, `agents/`, `baselines/`, `training/`, `evaluation/`, `config/{default,small_network,large_network}.yaml` | `manuscript/MPhil_Thesis_Concept_Note_v4.md` (v4.0) |

Both tracks are unit-tested and developed in the same repo/branch for
convenience, but are evaluated, written up, and reasoned about
**independently** — a decision, bug, or config key in one track says
nothing about the other.

---

## Quick Navigation

**Shared / repo-wide**

| Document | Purpose | When to Read |
|----------|---------|-------------|
| [AGENTS.md](AGENTS.md) | High-level project context, identity, status (primarily the C-RAN track; see its own O-RAN section) | First time; when lost |
| [docs/rules.md](docs/rules.md) | Mandatory development rules | Before every commit |
| [docs/agents.md](docs/agents.md) | Agent role definitions | When delegating tasks |
| [docs/hooks.md](docs/hooks.md) | Lifecycle automation hooks | When setting up CI/CD |

**C-RAN Track (Publications)**

| Document | Purpose | When to Read |
|----------|---------|-------------|
| [manuscript/MPhil_Thesis_Concept_Note_v4.md](manuscript/MPhil_Thesis_Concept_Note_v4.md) | Governing research concept document | Before any C-RAN change |
| [docs/dev_guide.md](docs/dev_guide.md) | Detailed development guide, code architecture | When implementing |
| [docs/thesis_guide.md](docs/thesis_guide.md) | Thesis writing guide, chapter structure | When writing |
| [docs/workflow.md](docs/workflow.md) | Development phases and milestones | When planning |
| [docs/skills/skill_environment.md](docs/skills/skill_environment.md) | Environment design specification | When building env |
| [docs/skills/skill_hybrid_agent.md](docs/skills/skill_hybrid_agent.md) | Branching MP-DQN + TD3 implementation (the proposed agent) | When building agent |
| [docs/skills/skill_evaluation.md](docs/skills/skill_evaluation.md) | Evaluation and analysis protocols | When analyzing results |

**O-RAN / BMPP-DQN Track (Thesis)**

| Document | Purpose | When to Read |
|----------|---------|-------------|
| [manuscript/ORAN_BMPP_DQN_Concept_Note_v1.md](manuscript/ORAN_BMPP_DQN_Concept_Note_v1.md) | Governing research concept document | Before any O-RAN change |
| [docs/oran_thesis_guide.md](docs/oran_thesis_guide.md) | Thesis writing guide, chapter structure | When writing |
| [docs/dev_guide.md](docs/dev_guide.md) | Development guide (has its own O-RAN section) | When implementing |
| [docs/skills/skill_oran_env.md](docs/skills/skill_oran_env.md) | Environment design specification | When building env |
| [docs/skills/skill_oran_bmpp_dqn.md](docs/skills/skill_oran_bmpp_dqn.md) | BMPP-DQN implementation (the proposed agent) | When building agent |

---

## Project at a Glance

### C-RAN Track (Publications)

**Title**: Optimization of Energy Efficient Cloud Radio Access Network for 5G Using a Branching Multi-Pass Parameterized Deep Q-Network with Twin Critics (Branching MP-DQN + TD3)

**Status**: Incomplete draft — major restructuring required

**Critical Pivots Needed**:
1. **Algorithm**: Vanilla DDPG -> Hybrid SAC-DDQN -> **branching, multi-pass (MP-DQN), twin-critic parameterized DQN** (`agents/branching_mp_dqn.py`; `hybrid_sac_dqn.py` is now the superseded alternative) — see `manuscript/MPhil_Thesis_Concept_Note_v4.md` §10
2. **Chapter 3**: Add formal MDP formulation (currently missing) — Concept Note v3.0 §10.2, plus the new critic-architecture diagram and combinatorial-action-space subsection (§10.3-10.3.1)
3. **Chapter 4**: Write entirely (currently missing); the P-DQN/MP-DQN/pure-DDPG baselines and the CSI-robustness/generalization/inference-latency evaluation code now exist and are tested (`agents/pdqn_agent.py`, `agents/mpdqn_agent.py`, `agents/ddpg_agent.py`, `evaluation/csi_robustness.py`, `evaluation/generalization.py`, `evaluation/latency_benchmark.py`), but no full-scale (10-seed, thesis-scale) results have been generated yet — Chapter 4 still needs those runs (Concept Note v3.0 §12)
4. **Chapter 5**: Write entirely (currently missing)

**Completed** (formerly listed here as pending): the EARTH power-model parameters (P_stat=175W, not 100W) are already fixed in `config/default.yaml`.

**Estimated Timeline**: ~27 weeks (revised from 14; see `docs/workflow.md` and `manuscript/MPhil_Thesis_Concept_Note_v4.md` §15 for the week-by-week Gantt)

**Current reference document**: `manuscript/MPhil_Thesis_Concept_Note_v4.md` (v4.0, responds to two supervisor review rounds on v2.0/v3.0 — see its §0 and §0.1 for the full item-to-section mapping)

### O-RAN / BMPP-DQN Track (Thesis)

**Title**: BMPP-DQN Energy Optimization in Open Cloud-RAN with Hybrid Discrete-Continuous Control

**Status**: Environment, proposed agent (`oran_agents/bmpp_dqn.py`), all 3 baselines (`oran_agents/{dqn_agent,ddpg_agent,mpdqn_agent}.py`), training scripts, and evaluation modules are all implemented and unit-tested. No full-scale (3-seed, thesis-scale) experiment runs have been executed yet, and no thesis chapter text has been written for this track (`docs/oran_thesis_guide.md`'s Chapter Content Mapping is a scaffold, not yet filled in).

**Scope** (Concept Note §6.1): single-gNB, 3 functional split options (3GPP TR 38.801 Options 2/6/8), 4 action branches (RU activation, functional split, transmit power, PRB allocation), downlink only, comparison against 3 baselines (DQN, DDPG, MP-DQN), 3 random seeds.

**Target**: ≥15% energy savings vs. baseline DRL algorithms (Concept Note §4.2).

**Estimated Timeline**: 16 weeks (Concept Note §8's Gantt).

**Current reference document**: `manuscript/ORAN_BMPP_DQN_Concept_Note_v1.md` (v1.0, supervisor-approved; see its own status note for how it relates to the C-RAN track's Concept Note v4.0)

---

## Key Decisions Log (C-RAN / Publications Track)

The O-RAN track has no equivalent iterated log — it's governed by a single supervisor-approved concept document (`manuscript/ORAN_BMPP_DQN_Concept_Note_v1.md`) rather than a series of revised drafts; its scope/design decisions live in that document's own §4-§6, not here.

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| 2026-07-22 | Switch from DDPG to Hybrid SAC-DDQN | DDPG cannot handle discrete actions; SAC has better stability | Decided |
| 2026-07-22 | Use factorized discrete actions | Avoids exponential action space (2^R) | Decided |
| 2026-07-22 | Fix power model to EARTH standards | Al-Zubaedi validation required | Pending implementation |
| 2026-07-22 | Include fronthaul power in reward | TWDM-PON fronthaul is a material share of C-RAN power (Al-Zubaedi, 2019); a specific savings % is not restated here — no primary-source access is available in this environment to verify one (Ethical AI Rule, `docs/rules.md` §10) | Decided |
| 2026-07-22 | Add switching cost to reward | RRH mode transitions carry a real power cost (Iqbal et al., 2021); a specific reported % is not restated here for the same reason as above | Decided |
| 2026-08-05 | Add P-DQN and MP-DQN as baselines, capped at R≤12 | Supervisor review S2/B3 — isolates branching's contribution and empirically demonstrates why it's needed at scale | Decided |
| 2026-08-05 | Increase seeds 5→10; report Cohen's d alongside p-values | Supervisor review S4 — statistical power was a concern at the modest 5% target margin | Decided (docs/rules.md updated) |
| 2026-08-05 | Add CSI-robustness (σ∈{0.01,0.05,0.1}) and cross-profile generalization evaluations, evaluation-only | Supervisor review S3/A5 — addresses the perfect-CSI limitation without expanding scope | Decided |
| 2026-08-05 | Frame the policy as an O-RAN rApp (discrete via O1, continuous via E2) | Supervisor review S1 — costs nothing in implementation, increases relevance | Decided |
| 2026-08-05 | Extend timeline 17→~27 weeks; demote R=50 to a stretch goal | Supervisor review B4 — 10 baselines + 10 seeds + new evaluations don't fit the prior estimate | Decided |

---

## Immediate Next Steps

### C-RAN Track (Publications)

The environment (`cran_env/`), all 10 baselines (`agents/`, `baselines/`), and the proposed branching MP-DQN + TD3 agent are already implemented and unit-tested (Phases 1-3 of `docs/workflow.md` are complete). What actually remains:

1. **Run** the full 10-seed × 11-method experiment matrix at thesis scale (Phase 4, `docs/workflow.md`) — everything so far has been verified at small network sizes and short training runs only
2. **Run** the CSI-robustness, cross-profile generalization, and inference-latency evaluations at full scale (infrastructure exists and is tested; full-scale runs are not yet executed)
3. **Write** Chapter 3's formal MDP formulation, and Chapters 4-5 (Results and Conclusion), once the above results are available
4. **Obtain** supervisor sign-off on Concept Note v4.0 as the governing document before committing to the full experiment run (see `manuscript/response_to_supervisor_review.md`)

### O-RAN / BMPP-DQN Track (Thesis)

The environment (`oran_env/`), all 3 baselines (`oran_agents/{dqn_agent,ddpg_agent,mpdqn_agent}.py`), and the proposed BMPP-DQN agent (`oran_agents/bmpp_dqn.py`) are already implemented and unit-tested. What actually remains:

1. **Run** the full 3-seed × 4-method experiment matrix at thesis scale (Concept Note §5.3) — everything so far has been verified at short training runs only
2. **Resolve** the needs-validation placeholders flagged in `docs/oran_thesis_guide.md` (power-model constants, traffic breakpoints, split→centralization mapping, default scenario scale) before the thesis states any of them as fact
3. **Write** thesis chapter text per `docs/oran_thesis_guide.md`'s Chapter Content Mapping, once full-scale results are available

---

## File Structure

```
Gabriel/
├── README.md (this file)
├── AGENTS.md               # Project identity and high-level context
├── QUICK_REFERENCE.md      # One-page quick reference card
├── requirements.txt        # Python dependencies
├── docs/
│   ├── dev_guide.md            # Development guide (both tracks)
│   ├── thesis_guide.md         # C-RAN thesis writing and structure guide
│   ├── oran_thesis_guide.md    # O-RAN thesis writing and structure guide
│   ├── rules.md                # Mandatory development rules
│   ├── agents.md                # Agent role definitions
│   ├── hooks.md                 # Lifecycle automation hooks
│   ├── workflow.md              # Development phases and milestones (C-RAN track)
│   ├── equation_code_mapping.md
│   ├── ai_assistance_log.md
│   ├── daily_log_template.md
│   ├── supervisor_feedback_template.md
│   └── skills/
│       ├── skill_environment.md    # C-RAN environment design specification
│       ├── skill_hybrid_agent.md   # C-RAN hybrid agent implementation
│       ├── skill_evaluation.md     # C-RAN evaluation protocols
│       ├── skill_oran_env.md       # O-RAN environment design specification
│       └── skill_oran_bmpp_dqn.md  # O-RAN BMPP-DQN agent implementation
├── references/              # Cited papers and presentation materials
├── manuscript/              # Concept notes and thesis drafts (both tracks)
│
│   # --- C-RAN track (publications) ---
├── cran_env/                # C-RAN Gymnasium environment
├── agents/                  # DRL algorithms
├── baselines/                # Non-DRL and simple-DRL baselines
├── training/                 # Training loops
├── evaluation/                # Analysis and plotting
│
│   # --- O-RAN track (thesis) ---
├── oran_env/                   # O-RAN Gymnasium environment
├── oran_agents/                 # BMPP-DQN + DQN/DDPG/MP-DQN baselines
├── oran_training/                # Training loops
├── oran_evaluation/               # Analysis and plotting
│
│   # --- Shared infrastructure ---
├── config/                    # Experiment configurations (default*.yaml = C-RAN, oran_default.yaml = O-RAN)
├── data/                       # Traffic traces, results
├── tests/                      # Unit tests (both tracks)
└── thesis/                     # LaTeX source (future)
```

---

## Contact & Support

- **Supervisor**: [Name, Email]
- **University Template**: [Link to LaTeX template]
- **GPU Cluster**: [Access instructions]
- **W&B Project**: https://wandb.ai/[username]/cran-drl-thesis
- **Git Repository**: https://github.com/[username]/cran-drl-thesis

---

*Last Updated: 2026-08-28*
*Version: 1.0*
