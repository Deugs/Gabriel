# O-RAN / BMPP-DQN Thesis — Writing and Structure Guide

> **Status**: Secondary, additive track. Governs the actual MPhil thesis
> submission, per `manuscript/ORAN_BMPP_DQN_Concept_Note_v1.md` (the
> supervisor-approved concept document). Does not replace or modify
> `docs/thesis_guide.md`, which continues to guide the separate C-RAN
> track's publication-oriented writing.

Following this file's own established convention (and
`docs/thesis_guide.md`'s): this is a writing/formatting scaffold that
points at the concept note's section numbers as the source of truth,
rather than duplicating its content.

## Document Specification

Same conventions as `docs/thesis_guide.md`'s Document Specification
(LaTeX, IEEE numbered citations, vector-graphics figures, booktabs
tables, numbered equations).

## Chapter Content Mapping

| Thesis Chapter | Concept Note Section | Implementation |
|---|---|---|
| 1. Introduction | §2 (Background and Problem Statement), §3 (Gaps) | — |
| 2. Literature Review | §3 (Gaps in Existing Papers), §9 (References) | — |
| 3. System Model and Problem Formulation | §5.1 (System Modeling), §10 (Implementation Addendum) | `oran_env/` |
| 3.x Algorithm Design | §5.2 (Algorithm Design), §10.4 (No TD3) | `oran_agents/bmpp_dqn.py` |
| 4. Simulation Results | §5.3 (Implementation and Training), §6 (Scope and Limitations) | `oran_training/`, `oran_evaluation/` |
| 5. Conclusion and Future Work | §7 (Significance) | — |

## Key Figures/Tables (mirrors docs/thesis_guide.md's Chapter 4 convention)

1. Convergence curves (BMPP-DQN + 3 baselines, 3 seeds) — `oran_evaluation.convergence`
2. Energy savings comparison table (target: ≥15% vs. baselines, Concept Note §4.2) — `convergence_summary_oran.tex`
3. Inference-time latency comparison (single scenario, not a scalability sweep — Concept Note §6.1/7.1's focused single-gNB scope) — `oran_evaluation.latency_benchmark`
4. Multi-timescale convergence discussion (RQ3: does upper/lower branch separation affect convergence?) — `history["param_losses"]`/`history["critic_losses"]` in `oran_training.train_bmpp_dqn`'s summary output

## Writing Quality Standards

Same as `docs/thesis_guide.md`'s Writing Quality Standards section
(equation/figure/table/citation conventions) — not duplicated here.

## Needs-Validation Flags to Resolve Before Submission

Per Concept Note §10's implementation addendum, the following are
literature-style placeholders chosen for internal consistency (e.g.
monotonicity), not verified physical constants — resolve/cite before the
thesis states them as fact:
- `oran_env/power_model.py`'s RU/DU/CU/fronthaul power constants (§10.5) —
  **still open** after a 2026-08-29 check against 5 O-RAN-context sources
  (see §10.5's own note); some order-of-magnitude/qualitative support now
  exists, but no source gives a matching per-split numeric table
- `oran_env/traffic_model.py`'s trapezoidal breakpoints and Poisson rate (§10.3, via `config/oran_default.yaml`'s `traffic:` section) — still open, none of the sources checked so far address O-RAN/5G traffic-shape parameters specifically
- The 3GPP split → centralization-level mapping (§10.2) — **partially informed**: the O-RAN Alliance's own 2021 white paper confirms the real specified split is Option 7-2x, not literally Option 2/6/8 (see §10.2's own note); the 3-level abstraction itself is still a tractability simplification, not a literature-validated mapping
- Default scenario scale (`n_ru=4, n_ue=8`, §10.3) — still open
