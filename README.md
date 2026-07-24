# C-RAN DRL Thesis — Developer Context Index

## Quick Navigation

| Document | Purpose | When to Read |
|----------|---------|-------------|
| [CLAUDE.md](CLAUDE.md) | High-level project context, identity, status | First time; when lost |
| [docs/dev_guide.md](docs/dev_guide.md) | Detailed development guide, code architecture | When implementing |
| [docs/thesis_guide.md](docs/thesis_guide.md) | Thesis writing guide, chapter structure | When writing |
| [docs/skills/skill_environment.md](docs/skills/skill_environment.md) | Environment design specification | When building env |
| [docs/skills/skill_hybrid_agent.md](docs/skills/skill_hybrid_agent.md) | Hybrid SAC-DDQN implementation | When building agent |
| [docs/skills/skill_evaluation.md](docs/skills/skill_evaluation.md) | Evaluation and analysis protocols | When analyzing results |
| [docs/rules.md](docs/rules.md) | Mandatory development rules | Before every commit |
| [docs/agents.md](docs/agents.md) | Agent role definitions | When delegating tasks |
| [docs/hooks.md](docs/hooks.md) | Lifecycle automation hooks | When setting up CI/CD |
| [docs/workflow.md](docs/workflow.md) | Development phases and milestones | When planning |

---

## Project at a Glance

**Title**: Optimization of Energy Efficient Cloud Radio Access Network for 5G Using Deep Deterministic Policy Gradient Algorithm

**Status**: Incomplete draft — major restructuring required

**Critical Pivots Needed**:
1. **Algorithm**: Vanilla DDPG -> Hybrid SAC-DDQN (discrete + continuous)
2. **Chapter 3**: Add formal MDP formulation (currently missing)
3. **Power Model**: Fix parameters to match EARTH model (P_stat=175W, not 100W)
4. **Chapter 4**: Write entirely (currently missing)
5. **Chapter 5**: Write entirely (currently missing)

**Estimated Timeline**: 14 weeks (see workflow.md)

---

## Key Decisions Log

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| 2026-07-22 | Switch from DDPG to Hybrid SAC-DDQN | DDPG cannot handle discrete actions; SAC has better stability | Decided |
| 2026-07-22 | Use factorized discrete actions | Avoids exponential action space (2^R) | Decided |
| 2026-07-22 | Fix power model to EARTH standards | Al-Zubaedi validation required | Pending implementation |
| 2026-07-22 | Include fronthaul power in reward | 41% savings potential (TWDM-PON) | Decided |
| 2026-07-22 | Add switching cost to reward | 22% of savings (Iqbal) | Decided |

---

## Immediate Next Steps

1. **Read** claude.md and claude_dev.md completely
2. **Set up** development environment (Python, PyTorch, dependencies)
3. **Initialize** Git repository with branch structure
4. **Implement** C-RAN environment (Phase 1)
5. **Validate** environment against analytical test cases
6. **Implement** baselines (Phase 2)
7. **Design** hybrid SAC-DDQN architecture (Phase 3)

---

## File Structure

```
Gabriel/
├── README.md (this file)
├── CLAUDE.md               # Project identity and high-level context
├── QUICK_REFERENCE.md      # One-page quick reference card
├── requirements.txt        # Python dependencies
├── docs/
│   ├── dev_guide.md            # Development guide and code architecture
│   ├── thesis_guide.md         # Thesis writing and structure guide
│   ├── rules.md                # Mandatory development rules
│   ├── agents.md                # Agent role definitions
│   ├── hooks.md                 # Lifecycle automation hooks
│   ├── workflow.md              # Development phases and milestones
│   ├── equation_code_mapping.md
│   ├── ai_assistance_log.md
│   ├── daily_log_template.md
│   ├── supervisor_feedback_template.md
│   └── skills/
│       ├── skill_environment.md    # Environment design specification
│       ├── skill_hybrid_agent.md   # Hybrid agent implementation
│       └── skill_evaluation.md     # Evaluation protocols
├── references/              # Cited papers and presentation materials
├── manuscript/              # Current thesis draft (.docx)
├── cran_env/                # C-RAN Gymnasium environment
├── agents/                  # DRL algorithms
├── baselines/                # Non-DRL and simple-DRL baselines
├── training/                 # Training loops
├── evaluation/                # Analysis and plotting
├── config/                    # Experiment configurations
├── data/                       # Traffic traces, results
├── tests/                      # Unit tests
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

*Last Updated: 2026-07-22*
*Version: 1.0*
