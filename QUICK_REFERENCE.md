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
| Baselines required? | 10 baselines (Concept Note v4.0 §12.1): All-ON/FA, Greedy, NMBS, Convex, DDQN, DDQN+SOCP, ANN+GSBF, pure-DDPG, P-DQN, MP-DQN, plus the proposed hybrid agent (11 methods total). Pure-SAC/TD3 are optional stretch comparisons, not core baselines |
| Random seeds? | 10 seeds: [42, 123, 456, 789, 1011, 1337, 2024, 2718, 3141, 4242] |
| Convergence target? | <= 3000 episodes |
| Energy savings target? | >= 5% vs. Iqbal DDQN/P-DQN/MP-DQN (headline comparison); >= 25% vs. All ON (sanity-check floor only, not a reported contribution — Concept Note v4.0 §5.2/G10) |
| QoS violation target? | <= 5% |

## Critical Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pre-commit install
export PYTHONPATH="$(pwd):$PYTHONPATH"  # no setup.py/pyproject.toml — needed for the commands below

# Test environment
pytest tests/test_env.py -v

# Train proposed method
python training/train_hybrid.py --config config/default.yaml --seed 42

# Run all baselines
python training/train_baselines.py --config config/default.yaml

# Generate figures (planned — evaluation/generate_figures.py not yet implemented;
# see evaluation/plot_utils.py's individual plotting functions for what exists today)
python evaluation/generate_figures.py --results-dir data/results/

# Build thesis (planned — thesis/main.tex not yet written; only thesis/figures/
# and thesis/tables/ output targets exist so far)
cd thesis && pdflatex main.tex && bibtex main && pdflatex main.tex

# Pre-submission check (planned — scripts/pre_submission.sh not yet implemented)
bash scripts/pre_submission.sh

# --- O-RAN / BMPP-DQN track (secondary, additive) ---
# Test environment
pytest tests/test_oran_env.py -v

# Train proposed method
python -c "from oran_training.train_bmpp_dqn import train_bmpp_dqn_agent; train_bmpp_dqn_agent(config_path='config/oran_default.yaml', seed=42)"

# Run all baselines
python -m oran_training.train_oran_baselines
```

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
| Build O-RAN env (secondary track) | `docs/skills/skill_oran_env.md` |
| Build BMPP-DQN agent (secondary track) | `docs/skills/skill_oran_bmpp_dqn.md` |
| O-RAN track overview | `docs/oran_thesis_guide.md` |

## Emergency Contacts

| Issue | Who | How |
|-------|-----|-----|
| Algorithm not converging | Methodology Validator | Check hooks.md debugging checklist |
| Code-text mismatch | Code Reviewer | Manually diff against docs/equation_code_mapping.md (scripts/check_code_text_consistency.py is planned — not yet implemented) |
| Missing references | Literature Curator | Check BibTeX; search recent papers |
| Scope creep | Thesis Architect | Review rules.md Scope Boundary Rule |
| Behind schedule | Gap Analyst | Re-prioritize per workflow.md risk table |

## Golden Rules (Memorize These)

1. **Never commit without tests passing** (Rule 8)
2. **Never change a parameter without citing a source** (Rule 2)
3. **Never evaluate without baselines** (Rule 3)
4. **Never claim novelty without comparison** (Rule 5)
5. **Always map equations to code** (Rule 1)
