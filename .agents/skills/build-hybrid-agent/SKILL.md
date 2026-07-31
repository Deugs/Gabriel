---
name: build-hybrid-agent
description: Implement the proposed Hybrid SAC-DDQN agent (agents/hybrid_sac_dqn.py) — discrete actor, continuous actor, shared twin critic, and training update. Use when building or modifying the hybrid agent, or when asked to implement the proposed DRL method.
---

Follow the full architecture spec in [docs/skills/skill_hybrid_agent.md](../../../docs/skills/skill_hybrid_agent.md) — it defines `DiscreteActor` (DDQN-style, factorized per-RRH), `ContinuousActor` (SAC Gaussian policy), `HybridCritic` (twin Q-networks fusing state + discrete + continuous action), the full `update()` training step, and hyperparameters.

Preconditions:
- `cran_env/` must already exist and pass its tests (see the `build-environment` skill) — the hybrid agent needs a working environment to train against.
- Baselines (`agents/ddqn_agent.py`, `baselines/*.py`) should exist first, per `docs/rules.md`'s Baseline-First development philosophy — implement/verify them before or alongside this if they're missing.

Steps:
1. Read `docs/skills/skill_hybrid_agent.md` in full, including the Key Design Decisions table (factorized discrete actions, twin critics, LayerNorm, sigmoid continuous-action output, shared/auto-tuned SAC temperature).
2. Implement the networks and training loop as specified; use the hyperparameters in `config/default.yaml`'s `algorithm:` block, not ad hoc values.
3. Work through the spec's Validation Checklist (critic loss decreasing, finite Q-values, epsilon starting at 1.0, continuous actions in [0,1], target networks lagging, bounded gradient norms) before declaring it stable.
4. Update `docs/equation_code_mapping.md` for equations (3.18)-(3.25) (Bellman equation, policy gradient, soft update, entropy term) as you implement each.
5. Write `tests/test_hybrid_agent.py` covering at minimum: agent survives 100 episodes without crashing, and reward improves over a random policy within 500 episodes.
