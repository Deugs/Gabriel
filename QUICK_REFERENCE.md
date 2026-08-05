# Quick Reference Card: C-RAN DRL Thesis

## One-Line Answers

| Question | Answer |
|----------|--------|
| What algorithm? | **Branching, multi-pass, twin-critic parameterized DQN (Branching MP-DQN + TD3)** — not vanilla DDPG, and not the superseded Hybrid SAC-DDQN (`agents/hybrid_sac_dqn.py`, kept only as the earlier alternative) |
| Why not DDPG? | Cannot handle discrete actions; unstable training |
| Discrete actions? | Per-RRH binary on/off (factorized) |
| Continuous actions? | Per-RRH transmit power [0, P_max] |
| Power model source? | EARTH model via Al-Zubaedi (2019) |
| BBU static power? | **175 W** (not 100 W) |
| BBU dynamic power? | **250 W** total |
| Switching cost? | **3 W** per transition; include in reward |
| Fronthaul in reward? | **Yes** — TWDM-PON model |
| Baselines required? | 9 methods (Concept Note v3.0/v4.0 §12.1): All-ON/FA, Greedy, NMBS, Convex, DDQN, ANN+GSBF, pure-DDPG, P-DQN, MP-DQN, plus the proposed hybrid agent. Pure-SAC/TD3 are optional stretch comparisons, not core baselines |
| Random seeds? | 10 seeds: [42, 123, 456, 789, 1011, 1337, 2024, 2718, 3141, 4242] |
| Convergence target? | <= 3000 episodes |
| Energy savings target? | >= 25% vs. All ON; >= 5% vs. Iqbal DDQN |
| QoS violation target? | <= 5% |

## Critical Commands

```bash
# Setup (local dev — full environment incl. notebooks/lint; not all of it
# is actually imported by the codebase, see requirements-runtime.txt)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Test environment
pytest tests/test_env.py -v

# Run the full test suite (45 tests) — the closest thing to a
# pre-submission check that currently exists; no scripts/pre_submission.sh
pytest tests/ -q

# Train proposed method directly
python training/train_hybrid.py --config config/default.yaml --seed 42

# Run all baselines directly
python training/train_baselines.py --config config/default.yaml

# Or run ANY experiment (training or evaluation) via the single reproducible
# entry point (docs/rules.md Rule 4) — see docs/deployment.md
python run_experiment.py --config experiments/hybrid_medium.yaml --seed 42
```

```bash
# Docker (see docs/deployment.md for the full cloud workflow)
docker build -t cran-drl:latest .
docker run --rm -v "$(pwd)/data:/app/data" cran-drl:latest \
    python run_experiment.py --config experiments/hybrid_small.yaml --seed 42
```

Not yet available: a `thesis/main.tex` to build (Chapters 3-5 aren't written
yet — see `README.md`), and `evaluation/generate_figures.py` as a single
all-figures script (each evaluation module under `evaluation/` saves its own
figures when run instead).

## File Quick Access

| Need | File |
|------|------|
| Understand the project | `AGENTS.md` |
| Start coding | `docs/dev_guide.md` |
| Write thesis text | `docs/thesis_guide.md` |
| Build environment | `docs/skills/skill_environment.md` |
| Build agent | `docs/skills/skill_hybrid_agent.md` |
| Analyze results | `docs/skills/skill_evaluation.md` |
| Check rules | `docs/rules.md` |
| Plan schedule | `docs/workflow.md` |
| Trace equations | `docs/equation_code_mapping.md` |
| Configure experiment | `config/default.yaml` |

## Emergency Contacts

| Issue | Who | How |
|-------|-----|-----|
| Algorithm not converging | Methodology Validator | Check hooks.md debugging checklist |
| Code-text mismatch | Code Reviewer | Run scripts/check_code_text_consistency.py |
| Missing references | Literature Curator | Check BibTeX; search recent papers |
| Scope creep | Thesis Architect | Review rules.md Scope Boundary Rule |
| Behind schedule | Gap Analyst | Re-prioritize per workflow.md risk table |

## Golden Rules (Memorize These)

1. **Never commit without tests passing** (Rule 8)
2. **Never change a parameter without citing a source** (Rule 2)
3. **Never evaluate without baselines** (Rule 3)
4. **Never claim novelty without comparison** (Rule 5)
5. **Always map equations to code** (Rule 1)
