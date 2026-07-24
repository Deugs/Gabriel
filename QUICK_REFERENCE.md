# Quick Reference Card: C-RAN DRL Thesis

## One-Line Answers

| Question | Answer |
|----------|--------|
| What algorithm? | **Hybrid SAC-DDQN** (not vanilla DDPG) |
| Why not DDPG? | Cannot handle discrete actions; unstable training |
| Discrete actions? | Per-RRH binary on/off (factorized) |
| Continuous actions? | Per-RRH transmit power [0, P_max] |
| Power model source? | EARTH model via Al-Zubaedi (2019) |
| BBU static power? | **175 W** (not 100 W) |
| BBU dynamic power? | **250 W** total |
| Switching cost? | **3 W** per transition; include in reward |
| Fronthaul in reward? | **Yes** — TWDM-PON model |
| Baselines required? | All ON, Greedy, NMBS, Convex, DDQN, DDPG, SAC, TD3 |
| Random seeds? | [42, 123, 456, 789, 1011] |
| Convergence target? | <= 3000 episodes |
| Energy savings target? | >= 25% vs. All ON; >= 5% vs. Iqbal DDQN |
| QoS violation target? | <= 5% |

## Critical Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pre-commit install

# Test environment
pytest tests/test_env.py -v

# Train proposed method
python training/train_hybrid.py --config config/default.yaml --seed 42

# Run all baselines
python training/train_baselines.py --config config/default.yaml

# Generate figures
python evaluation/generate_figures.py --results-dir data/results/

# Build thesis
cd thesis && pdflatex main.tex && bibtex main && pdflatex main.tex

# Pre-submission check
bash scripts/pre_submission.sh
```

## File Quick Access

| Need | File |
|------|------|
| Understand the project | `CLAUDE.md` |
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
