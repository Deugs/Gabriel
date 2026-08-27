# Skill: BMPP-DQN Agent Design

> **Status**: Authoritative spec for `oran_agents/bmpp_dqn.py`, the O-RAN track's proposed method. Governed by `manuscript/ORAN_BMPP_DQN_Concept_Note_v1.md` §5.2 and §10.4. Fully separate from `agents/branching_mp_dqn.py` — no shared code, no shared imports.

## Purpose

Implement Branching Multi-Pass Parameterized DQN (BMPP-DQN): a two-timescale hybrid discrete-continuous DRL agent for O-RAN energy optimization.

## Core Design (Concept Note §5.2)

- **Branching decomposition** — one independent decision branch per RU per discrete decision type (activation, split), so the discrete output grows as `O(n_ru)` rather than combinatorially.
- **Multi-pass parameterized processing** — each branch evaluates its own Q-value using only its own RU's continuous parameters (power, PRB), masking out other RUs' parameters, removing the cross-talk a naive joint parameterization would introduce (mirrors the MP-DQN mechanism already validated in `agents/mpdqn_agent.py`, reimplemented locally with zero import from that file).
- **Two independent encoders**, not one shared encoder: `upper_encoder` feeds the two discrete branch-groups (activation, split); `lower_encoder` feeds the continuous parameter network and its critic. Rationale: the source document frames this as genuinely two Q-networks with different replay-buffer lengths and update cadences (§5.2); sharing one encoder trained at two very different frequencies risks representation drift between the two objectives, which multi-timescale separation is meant to avoid.
- **No TD3 / twin-critic machinery** (Concept Note §10.4) — a single critic per timescale level, standard Double-DQN target computation (online-net argmax, target-net evaluate), no target-policy-smoothing noise, no policy-delay gating. This is a deliberate divergence from `agents/branching_mp_dqn.py`'s TD3-based design: the source document's own "core innovation" text (§5.2) mentions only branching + multi-pass, never twin critics.

## Two-Timescale Mechanics

The environment (`oran_env/`) stays timescale-agnostic — `step()` always takes the full 4-key action dict every call. The multi-timescale behavior lives entirely in the agent:

- An internal step counter mod `algorithm.upper_level_period_steps` (config default 10, i.e. 1.0s at 100ms/step — the low end of the source document's 1-10s upper-level range) gates the discrete decision: on non-decision steps, the last `(ru_on, split)` choice is replayed unchanged; `(power, prb)` are recomputed from `lower_encoder` every call.
- **Two separate replay buffers**: the lower-level buffer stores a transition every step (fast updates, per §5.2's "faster updates" requirement); the upper-level buffer stores one transition every N steps, using the aggregated (summed) reward over the N-step window, `s` = the state at the decision point, `s'` = the state N steps later — the concrete realization of "longer replay buffers."
- The state-level propagation channel (rolling-window mean throughput/power, `oran_env/`'s `_get_obs()`) supplies the state-level half of "lower-level metrics feed into the upper-level state"; the aggregated-reward buffer construction supplies the reward-level half.

## Validation Checklist (mirrors `agents/branching_mp_dqn.py`'s established testing conventions)

1. Multi-pass cross-talk absence: two candidate continuous-parameter vectors differing only in an unrelated RU's `(power, prb)` must yield different masked Q-values only for that RU's own branch.
2. Discrete decision (activation, split) held constant across `upper_level_period_steps` consecutive `select_action` calls, while continuous outputs (power, PRB) vary every call.
3. No twin-critic attribute exists on the agent object (guards the explicit no-TD3 decision above from silently regressing).
4. `param_loss`/critic losses gather the correct action index before averaging (not `.mean()` over an un-gathered action dimension — the exact class of bug found and fixed in `agents/branching_mp_dqn.py`'s own actor update; this agent's code review should check for it from the start rather than discover it later).
