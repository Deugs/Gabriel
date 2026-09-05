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
  **still open** after 2026-08-29 and 2026-08-30 checks against 13
  O-RAN-context sources total (see §10.5's own notes); some order-of-
  magnitude/qualitative support now exists, but no source gives a matching
  per-split numeric table. The 2026-08-30 passes additionally surfaced a
  genuine **scale mismatch** worth disclosing in the thesis: real measured
  macro-cell O-RU power (~200-550 W, Open RAN Handbook 2nd Ed.) and
  enterprise-server O-DU/O-CU host power (~625-780 W, Hoffmann et al.
  presentation) both run 20-100x above this model's own placeholder scale
  — neither source says what scale a small `n_ru=4` scenario should use,
  so nothing was rescaled from this finding alone. Obtaining 3GPP TR 38.864
  itself (the actual document, not a secondary citation) independently
  confirms the static+dynamic model *family* is right, but its own power
  table is in relative units with no absolute-Watt anchor, so it still
  couldn't be used to set any Watt-valued constant here. A real small-cell
  O-RU datasheet (Benetel RAN550, Split 7.2x) later narrowed the scale
  mismatch to 5-14x (from 20-100x) with its "typical power consumption:
  40 W" figure, still not decomposed into this model's RU/DU/CU/fronthaul
  shares — but its "max TX output power: 2 W (33 dBm)" *did* directly fix
  one constant: `power.ru.p_max_dbm` (30 → 33 dBm), the one exception to
  "still open" in this flag. A HUBER+SUHNER/CubeOptics infographic
  reproducing 3GPP TR 38.801's real per-split fronthaul bandwidth table
  also quantitatively confirmed this model's `p_fh_per_ru_by_split`
  monotonic *direction* for the three mapped options, while revealing its
  1:2:5 ratio is far shallower than the real bandwidth-ratio these figures
  imply (~1:1.4-2.4:39-52) — not used to rescale it, since fronthaul power
  isn't established to scale linearly with bandwidth (see §10.2). Most
  recently, Al-Tahmeesschi et al. 2025 gave the first real, RU/DU/CU-
  *decomposed* O-RAN power measurements found in either round, on a Split 8
  testbed that's an exact match to this model's `c=2`: measured RU power
  ~43-45 W (narrowing the scale mismatch to ~4x, the closest yet) and
  combined DU+CU power ~119.5-141.6 W — still not decomposable into this
  model's separate constants without guessing, and their separate Split
  7.2b DU/CU figures are confounded by a different-server-class testbed
  choice, so not directly comparable to Split 8's. Their finding that power
  barely scales with PRB load independently corroborates this model's
  existing load-independent DU/CU power design (no change needed). Two
  more sources supplied the same day: Abubakar et al. 2023's own survey
  conclusion states RU/fronthaul-specific O-RAN power modeling remains an
  open research gap in the literature at large (a survey-level
  confirmation this flag's "still open" status is a field-wide gap, not a
  search failure), and cites a real fronthaul-*power* percentage (not
  bandwidth) from Lopez-Perez et al. — 2%/30%/60% of total C-RAN power for
  split options 6/7/8 — that further quantifies (without rescaling) the
  bandwidth-vs-power gap already noted above, since this model's own
  implied fronthaul fraction (~11-18% across c=0..2) sits well below the
  cited 60% at the most-centralized option. A 2025 MASc thesis on CF-mMIMO
  under O-RAN Split 7.2/8 gave further structural corroboration (its own
  power model is the same static+load-dependent family) and closed-form
  fronthaul-rate formulas confirming the bandwidth-monotonicity direction.
  The candidate then supplied the thesis's remaining pages (Chapter 4 and
  Appendix A) the same day: Appendix A's own formula-derived fronthaul
  rates (Split 7.2≈2.764 Gbps, Split 8≈5.898 Gbps at N=8, a ~2.1x ratio,
  giving 7 vs. 3 max APs/DU under a 20 Gbps budget) *disagree* with the
  thesis's own Chapter 4 simulation assumptions (10/20 Gbps) and with
  3GPP TR 38.801's real Option 7-2-vs-8 ratio (~10-16x) — disclosed as a
  spread across (and within) sources rather than resolved by picking one.
  Neither the Chapter 4 assumptions nor Appendix A decompose power by
  RU/DU/CU component (both address fronthaul *bandwidth* only), so this
  flag's core RU/DU/CU wattage gap remains fully open after 6
  literature-check passes across two days — the most-open of the four
  O-RAN needs-validation flags
- `oran_env/traffic_model.py`'s trapezoidal breakpoints and Poisson rate
  (§10.6, via `config/oran_default.yaml`'s `traffic:` section) —
  **partially resolved** as of a 2026-08-30 check that obtained 3GPP
  TR 38.864 itself (see §10.6's own note): its Annex A's FTP Model 3 (0.5
  MB packet size, 200 ms mean inter-arrival time — a real, standard 3GPP
  Poisson traffic model) is a genuine primary-source match in the right
  units, so `lambda_peak` (5.0 → 0.5) and `packet_size_bits` (1.0e6 →
  4.0e6) have been updated to derive directly from it — not a guess. This
  also corrects the same day's earlier, more tentative finding that this
  module's temporal-Poisson-arrival design wasn't precedented; it is, by
  3GPP's own FTP Model 3. `floor_ratio` and `t1`-`t4` remain unvalidated:
  TR 38.864's own load scenarios are load-level snapshots with no
  time-of-day association, and its scope stops at "medium load," giving no
  floor:peak ratio or diurnal timing to derive those from. A same-day
  follow-up (a 2025 MASc thesis citing ETSI TR 103 737's 24-hour load
  weighting: Busy=6h/Medium=10h/Low=8h) gives a genuine, exact confirmation
  of the *aggregate* day-fraction split this model's `t1=7`/`t4=23` imply
  (floor=8h matches ETSI's Low exactly; active=16h matches ETSI's
  Medium+Busy exactly) — upgrading that aggregate split from unvalidated
  to ETSI-consistent, though the four individual breakpoints (and
  `floor_ratio` itself) remain underdetermined by this 3-bucket standard
- The 3GPP split → centralization-level mapping (§10.2) — **partially informed**: the O-RAN Alliance's own 2021 white paper confirms the real specified split is Option 7-2x, not literally Option 2/6/8 (see §10.2's own note); a 2026-08-30 check of Rony et al. 2021 independently confirms the *qualitative direction* of the RU-processing-vs-fronthaul-cost trade-off this mapping assumes (in cost percentages, not power or bandwidth). A same-day follow-up check (a HUBER+SUHNER/CubeOptics infographic reproducing 3GPP TR 38.801's real per-split bandwidth table) went further, giving *quantitative* fronthaul-bandwidth figures for exactly the three mapped options (Option 2 = 3/4 Gbps, Option 6 = 7.1/5.6 Gbps, Option 8 = 157.3/157.3 Gbps) — a real numeric confirmation of the monotonic direction, though the 3-level abstraction itself is still a tractability simplification, not a literature-validated mapping in the sense of matching this model's own power-array ratios (see §10.5's note on the resulting bandwidth-vs-power ratio mismatch). Obtaining 3GPP TR 38.801 itself afterward gave an **exact** cross-validation of these pixel-verified figures from its own Annex A Table A-1 (Option 2 = 4016/3024 Mb/s, Option 6 = 5626.7/7140 Mb/s, Option 8 = 157.3/157.3 Gb/s) plus a full latency table not previously available (§10.2's own note has the details) — the option definitions and bandwidth figures now rest on the primary document itself, not only a secondary reproduction
- Default scenario scale (`n_ru=4, n_ue=8`, §10.3) — **partially
  informed** after a 2026-08-30 check of the 8 already-supplied O-RAN
  sources for scenario-scale content (see §10.3's own note): two directly
  comparable RAN-DRL papers give concrete scales (DQRL: 12 RUs, 12-16
  UEs; OREO: 42 RUs, 100 UEs) whose UE:RU ratio (~1.0-2.4) brackets this
  repo's own ratio (8/4=2), and OREO's own discussion independently flags
  scalability challenges for single-agent centralized RL as RU count
  grows — corroborating the tractability rationale, not the exact counts.
  `n_ru=4`/`n_ue=8` themselves remain an unvalidated tractability choice,
  just one now shown to sit within precedented ranges. A 2025 MASc thesis
  on CF-mMIMO under O-RAN (`K=16` users, `L=20`-`50` APs) adds a third
  reference point, though its UE:AP ratio (0.32-0.8) sits *below* this
  repo's own ratio (2), unlike DQRL/OREO's ratios which bracketed it —
  disclosed as a genuine difference, not cherry-picked
