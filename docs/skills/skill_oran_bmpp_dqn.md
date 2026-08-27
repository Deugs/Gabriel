# Skill: BMPP-DQN Agent Design

> **Status**: Authoritative spec for `oran_agents/bmpp_dqn.py`, the O-RAN track's proposed method. Governed by `manuscript/ORAN_BMPP_DQN_Concept_Note_v1.md` §5.2 and §10.4. Fully separate from `agents/branching_mp_dqn.py` — no shared code, no shared imports.

## Purpose

Implement Branching Multi-Pass Parameterized DQN (BMPP-DQN): a two-timescale hybrid discrete-continuous DRL agent for O-RAN energy optimization.

## Core Design (Concept Note §5.2)

- **Branching decomposition** — one independent decision branch per RU per discrete decision type (activation, split), so the discrete output grows as `O(n_ru)` rather than combinatorially.
- **Multi-pass parameterized processing** — each branch evaluates its own Q-value using only its own RU's continuous parameters (power, PRB), masking out other RUs' parameters, removing the cross-talk a naive joint parameterization would introduce (mirrors the MP-DQN mechanism already validated in `agents/mpdqn_agent.py`, reimplemented locally with zero import from that file).
- **Two independent encoders**, not one shared encoder: `upper_encoder` feeds `BranchingCritic` (the two discrete branch-groups' Q-values, activation + split); `lower_encoder` feeds `ContinuousParameterNetwork` only. Rationale: sharing one encoder trained at two very different update cadences risks representation drift between the two objectives, which multi-timescale separation is meant to avoid.
- **Correction:** the source document's §5.2 describes "two Q-networks." The actual implementation has exactly **one** critic (`BranchingCritic`, fed by `upper_encoder`) — there is no second, independently-trained Q-network on the lower/continuous side. `ContinuousParameterNetwork` ("the lower-level network") is a deterministic-policy-gradient actor, trained by maximizing the *same* shared critic's output (P-DQN/DDPG-style), not a Q-network of its own. `update_lower()` and `update_upper()` are also both called every environment step in `oran_training/train_bmpp_dqn.py`'s training loop — the timescale separation is realized in *decision cadence* and *buffer refill rate* (the upper buffer only receives a new transition every `upper_level_period_steps` steps), not in differential gradient-step frequency. This is an honest description of what's implemented, not a claim that it matches §5.2's literal "two Q-networks, faster/slower updates" framing.
- **No TD3 / twin-critic machinery** (Concept Note §10.4) — a single critic per timescale level, standard Double-DQN target computation (online-net argmax, target-net evaluate), no target-policy-smoothing noise, no policy-delay gating. This is a deliberate divergence from `agents/branching_mp_dqn.py`'s TD3-based design: the source document's own "core innovation" text (§5.2) mentions only branching + multi-pass, never twin critics.

## Two-Timescale Mechanics

The environment (`oran_env/`) stays timescale-agnostic — `step()` always takes the full 4-key action dict every call. The multi-timescale behavior lives entirely in the agent:

- An internal step counter mod `algorithm.upper_level_period_steps` (config default 10, i.e. 1.0s at 100ms/step — the low end of the source document's 1-10s upper-level range) gates the discrete decision: on non-decision steps, the last `(ru_on, split)` choice is replayed unchanged; `(power, prb)` are recomputed from `lower_encoder` every call.
- **Two separate replay buffers**: the lower-level buffer stores a transition every step; the upper-level buffer stores one transition every N steps, using the aggregated (summed) reward over the N-step window, `s` = the state at the decision point, `s'` = the state N steps later — the concrete realization of "longer replay buffers." Both buffers' `update_*()` methods are called every step regardless (see the Correction note above) — the differential update *rate* claimed by §5.2 is not literally implemented; only the buffer refill rate differs.
- The state-level propagation channel (rolling-window mean throughput/power, `oran_env/`'s `_get_obs()`) supplies the state-level half of "lower-level metrics feed into the upper-level state"; the aggregated-reward buffer construction supplies the reward-level half.

## Validation Checklist (mirrors `agents/branching_mp_dqn.py`'s established testing conventions)

1. Multi-pass cross-talk absence: two candidate continuous-parameter vectors differing only in an unrelated RU's `(power, prb)` must yield different masked Q-values only for that RU's own branch.
2. Discrete decision (activation, split) held constant across `upper_level_period_steps` consecutive `select_action` calls, while continuous outputs (power, PRB) vary every call.
3. No twin-critic attribute exists on the agent object (guards the explicit no-TD3 decision above from silently regressing).
4. `param_loss`/critic losses gather the correct action index before averaging (not `.mean()` over an un-gathered action dimension — the exact class of bug found and fixed in `agents/branching_mp_dqn.py`'s own actor update; this agent's code review should check for it from the start rather than discover it later).
