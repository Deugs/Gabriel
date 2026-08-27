# Skill: O-RAN Environment Design

> **Status**: Authoritative spec for `oran_env/`, the O-RAN track's Gymnasium-compatible simulation environment. Governed by `manuscript/ORAN_BMPP_DQN_Concept_Note_v1.md` §5.1 and §10.1-10.3 (implementation addendum). This is a fully separate module from `cran_env/` — it does not import from, subclass, or otherwise depend on any C-RAN code, and none of that code imports from here.

## Purpose

Design, implement, and validate a Gymnasium-compatible O-RAN (disaggregated RU/DU/CU) simulation environment for the BMPP-DQN research track.

## Context

The environment models a single-gNB O-RAN system: `n_ru` Radio Units, one Distributed Unit (DU), one Central Unit (CU). The agent controls, per RU: activation (discrete on/off), functional split selection (discrete, one of 3 representative options), transmit power (continuous), and PRB allocation fraction (continuous) — the "4 action branches" of Concept Note §10.1.

## Rules

1. All physical models must be traceable to cited references or explicitly flagged "needs validation" placeholders (Concept Note §10).
2. State space must include the lower-level→upper-level propagation channel required by Concept Note §5.2 (rolling-window mean throughput/power).
3. Action space must exactly match the 4-branch hybrid discrete-continuous formulation (§10.1) — `MultiBinary(n_ru)` activation, `MultiDiscrete([3]*n_ru)` split, `Box(n_ru)` power, `Box(n_ru)` PRB share.
4. The environment itself is timescale-agnostic: `step()` always accepts the full 4-key action dict every call, regardless of the agent's internal upper/lower decision cadence — the two-timescale behavior lives entirely in the agent (see `skill_oran_bmpp_dqn.md`), not here.
5. Environment must be deterministic given a random seed (reproducibility, mirroring `cran_env`'s convention).
6. Zero imports from `cran_env/`, `agents/`, `baselines/`, `training/`, `evaluation/` — this module must remain fully decoupled from the C-RAN track.

## Components

### Channel Model (`oran_env/channel_model.py`)

Simplified relative to `cran_env/channel_model.py`: log-distance path loss (same functional form) + fresh, independent-per-step Rayleigh fading. No shadowing term, no Gauss-Markov temporal correlation — Concept Note §5.1 specifies "simplified SINR with path loss and Rayleigh fading" only, so this is a strict simplification, not a reuse, of the C-RAN channel model.

### Traffic Model (`oran_env/traffic_model.py`)

Deterministic trapezoidal daily envelope `λ(t)` (rise / plateau / fall / floor breakpoints) × Poisson per-UE arrival count — replacing the C-RAN model's dual-Gaussian × log-normal design with the shape/law Concept Note §5.1 specifies ("time-varying Poisson arrival with a daily trapezoidal pattern"). Numeric breakpoints/rate constants are needs-validation placeholders per Concept Note §10.

### Power Model (`oran_env/power_model.py`)

RU (active/sleep + per-split processing cost) + DU (static + per-active-RU, split-dependent) + CU (static + per-active-RU dynamic) + fronthaul (common + per-RU split-dependent) + switching (RU flips + split changes), monotonic in the split centralization level `c` defined in Concept Note §10.2. All numeric constants are needs-validation literature-style placeholders; the monotonicity itself is asserted by a regression test (`tests/test_oran_env.py`).

### State propagation (Concept Note §5.2)

The environment maintains a trailing window (length `algorithm.upper_level_period_steps`, config-driven, shared with the agent) of per-step total throughput and total power, and always embeds the window's mean into `_get_obs()`'s trailing two scalars — the concrete realization of "lower-level performance metrics ... feed into the upper-level state."
