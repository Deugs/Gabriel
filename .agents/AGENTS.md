# C-RAN Energy Optimization Thesis — Antigravity Workspace Guidelines

> **Status**: Workspace customization root for Antigravity agents. Refer to root `AGENTS.md` for full project status and thesis documentation.

## Agent System Overview

This workspace defines 3 Antigravity skills in `.agents/skills/`:
- `build-environment`: Build/extend C-RAN Gymnasium environment (`cran_env/`)
- `build-hybrid-agent`: Implement the proposed Branching MP-DQN + TD3 agent (`agents/branching_mp_dqn.py`); `agents/hybrid_sac_dqn.py` is the superseded v1.0 design, kept only for comparison
- `run-evaluation`: Run convergence, energy efficiency, ablation, and scalability analysis

It also defines 9 specialized agent roles in `.agents/agents/`:
- `thesis-architect.md`: Structural review & chapter consistency
- `methodology-validator.md`: Mathematical rigor & quality gate verification
- `literature-curator.md`: BibTeX database & citation tracking
- `cran-code-reviewer.md`: Code quality, style, & equation mapping review
- `figure-designer.md`: Publication-quality vector figures & LaTeX tables
- `baseline-implementer.md`: Non-DRL & simple-DRL baseline implementations
- `experiment-runner.md`: Training execution, seed sweeps, & run monitoring
- `thesis-writer.md`: Academic prose drafting & LaTeX source writing
- `gap-analyst.md`: Milestone tracking & MPhil requirement gap analysis

---

## Core Rules for All Agents

1. **Equation-to-Code Traceability**: Every mathematical equation implemented in code must be logged in `docs/equation_code_mapping.md`.
2. **Baseline Fairness**: All algorithms (proposed & baselines) must evaluate on identical environment seeds, traffic profiles, and network configurations.
3. **Reproducibility**: All experiments must be defined strictly in config files (`config/*.yaml`) with fixed seeds `[42, 123, 456, 789, 1011, 1337, 2024, 2718, 3141, 4242]` (10 seeds, per supervisor review S4).
4. **Statistical Significance**: All benchmark claims must be averaged over at least 10 seeds with 95% confidence intervals, paired t-tests, and Cohen's d.
5. **No Ad-Hoc Parameters**: Physical/hardware constants must reference valid sources (e.g. EARTH model parameters from Al-Zubaedi 2019).
