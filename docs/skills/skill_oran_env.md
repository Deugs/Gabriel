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

### QoS Metrics (two, deliberately different strictness — Concept Note §6.5)

`step()`'s info dict exposes two independent QoS signals, not one, because they answer different questions and can diverge sharply:

- **`qos_violations_count`** (and the training/eval code's derived `qos_rate`): a step counts as satisfied only if *every* one of the `n_ue` UEs has zero shortfall simultaneously. Strict, and — because a single UE's occasional Poisson-driven demand spike fails the whole step regardless of how well the other UEs are served — partly a function of how often the traffic model itself produces a simultaneous all-UEs-idle moment, not purely of policy quality. A fully collapsed, zero-throughput policy and a genuinely good policy can both land on a similar-looking strict rate if the traffic model's own zero-demand steps dominate it.
- **`qos_ue_satisfaction_frac`** (and the derived `qos_per_ue_rate`): the fraction of the `n_ue` UEs satisfied *that step*, continuous in `[0, 1]`, averaged over the episode. Far less brutal — empirically, a policy with ~74% per-UE satisfaction can still have a strict rate near 20% (roughly `0.74^8 ≈ 0.10`-ish territory for 8 independent-ish UEs) — but still not a complete stand-in for throughput: even a fully collapsed (zero-transmit-power) policy scores nontrivially on this metric, since UEs with zero demand that step are trivially "satisfied" by zero capacity too. Neither metric alone should be read as a proxy for whether a policy is actually delivering service; `throughput_mbps` is the one signal that unambiguously drops to ~0 under a collapsed policy (see `docs/daily_log.md`'s 2026-09-24 entry for the empirical case this was found from).

Both are computed every step from the same underlying `qos_violations_bps = max(0, demand - achievable_capacity)` per-UE array; `qos_violations_count = sum(qos_violations_bps > 0)`, `qos_ue_satisfaction_frac = sum(qos_violations_bps <= 0) / n_ue`.

### Capacity/demand scale (needs-validation, partially resolved 2026-09-24)

`network.bandwidth_mhz` was raised from an original placeholder of 20 to 100 after a direct empirical check (a best-case all-RU/max-power/equal-PRB-share reference policy) found mean aggregate demand (~101 Mbps, driven by the traffic model's own Poisson burstiness) already exceeded mean achievable throughput (~77 Mbps) at 20 MHz — leaving no policy, however good, real headroom to satisfy QoS. 100 MHz is a standard, precedented channel bandwidth for this model's own 3.5 GHz (n78-band) carrier frequency, not an arbitrary tuning choice, and raises mean achievable throughput to ~384 Mbps under the same reference policy. This changes the environment's scenario (all results generated before this change reflect the old, capacity-starved 20 MHz configuration — see `data/results_oran_archive/` for the superseded runs) but does not change any code path, only a config default.
