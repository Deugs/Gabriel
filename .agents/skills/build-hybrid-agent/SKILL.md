---
name: build-hybrid-agent
description: Implement the proposed Branching MP-DQN + TD3 agent (agents/branching_mp_dqn.py) — shared encoder, R branching discrete heads, MP-DQN multi-pass continuous parameter coupling, twin critics, and the training update. Use when building or modifying the hybrid agent, or when asked to implement the proposed DRL method.
---

Follow the full architecture spec in [docs/skills/skill_hybrid_agent.md](../../../docs/skills/skill_hybrid_agent.md) — it defines `SharedEncoder`, `BranchingDiscreteHeads` (R independent dueling branch heads, one per RRH), `ContinuousParameterNetwork` (P-DQN's coupled continuous output), `SingleBranchCritic`/`TwinBranchCritic` (MP-DQN multi-pass masking + TD3 twin critics), the full `update()` training step, and hyperparameters.

`agents/hybrid_sac_dqn.py` is the superseded v1.0 alternative (a separate discrete DDQN actor + continuous SAC actor arbitrated by a shared critic) — it is kept only for comparison, not as the proposed method. Do not use it as the reference architecture for new work.

Preconditions:
- `cran_env/` must already exist and pass its tests (see the `build-environment` skill) — the hybrid agent needs a working environment to train against.
- Baselines (`agents/ddqn_agent.py`, `baselines/*.py`) should exist first, per `docs/rules.md`'s Baseline-First development philosophy — implement/verify them before or alongside this if they're missing.

Steps:
1. Read `docs/skills/skill_hybrid_agent.md` in full, including the Key Design Decisions table (branching over a joint 2^R head, MP-DQN multi-pass masking, twin critics + delayed policy update + target-policy smoothing, LayerNorm, sigmoid power ratio / softmax bandwidth share).
2. Implement the networks and training loop as specified; use the hyperparameters in `config/default.yaml`'s `algorithm:` block, not ad hoc values (`hidden_dims`, `activation`, `use_layer_norm`, `gradient_clip_norm`, and `reward_scale` are all config-driven — see `BranchingMPDQN.__init__`'s `get_val()` calls).
3. Work through the spec's Validation Checklist (critic loss decreasing, finite Q-values across both twin critics, per-branch epsilon starting at 1.0, continuous parameters in valid ranges, target networks lagging via `tau`/`policy_delay`, multi-pass masking genuinely cross-talk-free, bounded gradient norms) before declaring it stable.
4. Update `docs/equation_code_mapping.md` for the MDP/reward/architecture equations in Concept Note Section 10 as you implement each.
5. Write/extend `tests/test_baselines_v2.py`-style coverage (see the existing multi-pass-masking test) covering at minimum: agent survives training without crashing, and the multi-pass mask genuinely zeroes out cross-talk from other RRHs' continuous parameters.
