# Daily Log

> Filled instances of `docs/daily_log_template.md`. Newest entry first.

## Date: 2026-08-30

### What I Did Today
- [x] The candidate supplied 3 more O-RAN-context sources (Open RAN Handbook 2nd Edition, Vodafone + Keysight, Feb 2025; a Hoffmann/Dryjanski/Kliks Rimedo Labs/i4y Lab E2E energy-testing-framework presentation; Rony et al. 2021's IEEE Access PHY-layer fronthaul functional-split cost analysis), continuing the same literature-verification pass as 2026-08-29 for the O-RAN track's still-open needs-validation flags. Read all three in full (58, 38, and 18 pages respectively).
- [x] Still no source gives a split-level RU/DU/CU/fronthaul wattage table matching `oran_env/power_model.py`'s parameterization -- that flag stays open -- but two genuinely new results came out of it: one independent qualitative corroboration, and one important disclosable limitation (see Decisions Made).

### Time Spent
| Activity | Hours |
|----------|-------|
| Coding | 0 |
| Writing | 0.4 |
| Reading | 0.9 (3 uploaded PDFs: Open RAN Handbook 2nd Ed., Hoffmann et al. presentation, Rony et al. 2021) |
| Debugging | 0 |
| Running experiments | 0 |
| **Total** | ~1.3 |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Left every O-RAN power-model numeric constant unchanged again | None of the 3 new sources gives a matching wattage table either -- same Ethical AI Rule reasoning as 2026-08-29. |
| Documented Rony et al. 2021 as independent qualitative (not numeric) support for Concept Note §10.2's monotonic centralization mapping | Their own PHY-split taxonomy (Split-A..D) shows the same RU-processing-vs-fronthaul-capacity trade-off direction as this repo's Option 2→Option 8 c-level mapping -- their most-centralized split needs the most fronthaul capacity and least RU processing, and vice versa for their least-centralized split. Their evidence is CAPEX/OPEX cost-percentage weights, not bandwidth or power, so it corroborates the trade-off's *direction* only. |
| Disclosed a genuine scale mismatch rather than rescaling any constant | The Open RAN Handbook's real measured Fujitsu macro-O-RU static power (~200-550 W) and the Hoffmann presentation's real Dell R750 enterprise-server power draw (~625-780 W, used there as an O-DU/O-CU compute-host proxy) are both 20-100x above this model's own RU/DU/CU placeholder scale. Neither source states what power scale a small `n_ru=4` simulation/testbed scenario like this one should use, so nothing was rescaled from this alone -- guessing a scaling factor would be exactly the kind of unsupported numeric claim the Ethical AI Rule forbids. Documented as a limitation to disclose in the thesis instead, in `oran_env/power_model.py`'s docstring, Concept Note §10.5, and `docs/oran_thesis_guide.md`'s Needs-Validation Flags. |
| Reconfirmed (independently of the 2026-08-29 OREO footnote) that no O-RAN Alliance normative power-measurement framework yet exists | The Hoffmann presentation quotes an O-RAN SuFG technical report stating this directly; it and the Handbook both point to real standardized test methodologies this model's own linear/step form only loosely resembles (ETSI ES 202 706 static, ETSI TS 103 786 dynamic, 3GPP TR 38.864 power-consumption model) -- added as citable context, not as a source of new numeric constants. |

### Blockers
| Blocker | Severity | Plan |
|---------|----------|------|
| None | -- | The power-model wattage flag remains open; no further supplied source has resolved it after 8 sources checked across two days. |

### Tomorrow's Plan
- [ ] The O-RAN power-model wattage flag likely needs an O-RAN Alliance/vendor hardware measurement source at the *specific small-cell/testbed scale* this environment models, or an explicit supervisor-approved decision to state the constants as a deliberately unvalidated tractability placeholder in the thesis text (mirroring how §10.2's 3-level split abstraction is already framed) rather than continuing to search for an exact match that may not exist in the literature at this scale.
- [ ] Traffic-model breakpoints/Poisson rate and default scenario scale (`docs/oran_thesis_guide.md`) remain fully open; no source checked so far (13 total across both literature-verification passes) has addressed either.

### Notes
No code or config changes this round -- purely a documentation/citation pass, mirroring the honest "partial support documented, nothing invented" outcome of 2026-08-29's O-RAN check. Full source list check so far for the power-model flag: Qazzaz et al. 2026 (OREO), Barker/Seyfi/Afghah 2025, Lassoued & Boujnah 2026, Eskandarinia et al.'s DQRL paper, the O-RAN Alliance's 2021 MVP white paper (2026-08-29); Open RAN Handbook 2nd Ed., Hoffmann et al.'s presentation, Rony et al. 2021 (2026-08-30).

---

## Date: 2026-08-29

### What I Did Today
- [x] Closed the blocker open since 2026-08-13/2026-08-15: actually ran `training/hyperparam_search.py::run_proxy_sensitivity_sweep()` against the real, current config values. `run_proxy_sensitivity_sweep()`'s own default `base_config_path` (`config/small_network.yaml`) now has `lr_discrete=1e-4`/`lr_actor=3e-4` (matching `config/default.yaml`), so no override was needed -- confirmed this first by reading the file directly rather than assuming.
- [x] Logged the resulting keep/change decision (below) per Section 12.11 item 3.
- [x] Ran this via the just-fixed decision logic (round-15 audit fix, PR #56) that genuinely compares the default variant's tail critic loss against the swept alternatives, not just whether the default itself crashed -- so this run's `default_kept=True` verdicts are a real comparison, not the old rubber-stamp.
- [x] The candidate supplied the actual primary-source PDFs for the C-RAN power model's citations (Al-Zubaedi 2019's PhD thesis, Auer et al. 2011's EARTH paper) -- previously unverifiable in this environment (no internet egress to the hosting domains). Read them directly and checked every `power:` constant in `config/default.yaml` against them for the first time (see Decisions Made).
- [x] The candidate then supplied 5 O-RAN-specific sources (Qazzaz et al. 2026 OREO; Barker/Seyfi/Afghah 2025 MEC/Open RAN survey; Lassoued & Boujnah 2026 Computers 5G energy-efficiency review; Eskandarinia et al.'s DQRL clustered-RAN paper; the O-RAN Alliance's 2021 MVP white paper) to attempt the same check for the O-RAN track's own needs-validation flags. None gives a split-level RU/DU/CU/fronthaul wattage table matching `oran_env/power_model.py`'s parameterization -- that flag stays genuinely open -- but real, useful things came out of it (see Decisions Made): partial order-of-magnitude support for the RU power model, the O-RAN Alliance's actual specified split (Option 7-2x, not literally Option 2/6/8), and a real bug: three per-split power arrays that encode the whole Section 10.2 monotonicity trade-off were never wired from config to `ORANPowerModel` at all -- fixed.

### Time Spent
| Activity | Hours |
|----------|-------|
| Coding | 0.2 |
| Writing | 0.5 |
| Reading | 1.1 (11 uploaded PDFs total: Al-Zubaedi 2019 thesis x3 parts, EARTH paper, EARTH book chapter, a telco-cloud power-modeling survey, OREO, an MEC/Open RAN survey, a 5G energy-efficiency review, a DQRL clustered-RAN paper, the O-RAN Alliance MVP white paper) |
| Debugging | 0 |
| Running experiments | ~0.3 (background wall-clock on a CPU-only sandbox; a GPU host would be faster) |
| **Total** | ~2.1 |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep the default branch/continuous-net learning rates (lr_discrete=1e-4, lr_actor=3e-4) unchanged | Sweeping ~half an order of magnitude down (lr_discrete~3.16e-5, lr_actor~9.49e-5) and up (lr_discrete~3.16e-4, lr_actor~9.49e-4) produced no crash and no non-finite reward at either extreme, and no tail-critic-loss divergence relative to the alternatives (down: 6865.8, default: 6446.1, up: 6419.3 -- all the same order of magnitude). Per Section 12.11 item 2, a not-visibly-unstable default is kept without further search. This is the sweep the 2026-08-05 entry should have been but wasn't (that one tested a since-superseded 1e-3/1e-4 pair). |
| Keep the default tau (0.005) unchanged | Same protocol: down (~1.58e-3) and up (~1.58e-2) both ran without crashing, and tau_default's tail critic loss (7962.8) is not more than 3x either alternative's (down: 6298.0, up: 5504.9) -- no visible divergence. Kept unchanged. |
| Confirmed `power.bbu.{p_stat_w,p_dyn_w,delta_p}` (175 W / 250 W / 0.44) exactly match Al-Zubaedi (2019) Table 3.1 "Simulation parameters" | Direct verification against the primary-source PDF (Chapter 3, page 63) rather than a citation claim -- exact match on all three values, no change needed |
| Fixed `power.fronthaul.{p_lc_w,p_onu_active_w}`: were 10.0/5.0, corrected to 5.0/10.5 | Table 3.1 gives "Power consumption of LC" = 5 W and "Power consumption of ONU" = 10.5 W -- the two values in `config/default.yaml` were transposed (a transcription error only catchable by actually checking the source, not previously verifiable in this environment). `tests/test_env.py::test_power_model_fronthaul_line_card_term` constructs `PowerModel` with its own explicit kwargs rather than reading `config/default.yaml`, so this fix doesn't touch that test's assertions -- confirmed by re-running it. |
| Corrected the `p_stat_w`/`p_dyn_w`/`delta_p` comment to stop implying Auer et al. (2011) directly reports these three values | Auer et al. 2011's own Table II (per-BS-type power model) has no row matching 175/250/0.44 for any BS type -- Auer et al. 2011 is the source of the underlying linear power-model *form* (`P_in = N_TRX*(P_0 + delta_p*P_out)`) that Al-Zubaedi's thesis adapts for a BBU-pool context, not of these specific numbers. Citing both papers as if they independently reported the same three values was inaccurate. |
| Left `power.fronthaul.{p_olt_w,p_onu_sleep_w}` unchanged, flagged unverified | Table 3.1 has no row for OLT power or ONU sleep-mode power -- no evidence either way in the supplied sources, so left as-is rather than guessing, per the Ethical AI Rule (`docs/rules.md` §10) |
| Left every O-RAN power-model numeric constant in `oran_env/power_model.py`/`config/oran_default.yaml` unchanged | None of the 5 O-RAN sources checked gives a split-level RU/DU/CU/fronthaul wattage table matching this model's parameterization -- OREO's own energy model explicitly excludes DU/CU/fronthaul from its scope (their footnote 1), so a source directly validating those specific numbers likely doesn't yet exist in the O-RAN RL literature. Guessing numbers to "resolve" the flag would violate the same Ethical AI Rule as above. Added a literature-grounding note to `oran_env/power_model.py`'s docstring and Concept Note §10.5 documenting what partial (order-of-magnitude, structural) support does exist, without claiming full validation. |
| Fixed a real bug found while doing this check: `oran_env/oran_env.py` never passed `power.ru.p_proc_by_split_w`/`power.du.p_per_ru_by_split_w`/`power.fronthaul.p_per_ru_by_split_w` to `ORANPowerModel` | These three arrays are the actual mechanism behind Section 10.2's monotonic centralization trade-off -- arguably the most important constants in the whole power model -- yet they weren't even exposed in `config/oran_default.yaml` and always silently used `ORANPowerModel`'s Python-side hardcoded defaults. Exposed them in config (same numeric values, so no behavior change yet) and wired them through, matching every other constructor argument's existing convention, so a future citation can actually be applied via config once found. Added `tests/test_oran_env.py::test_power_model_per_split_arrays_read_from_config`. |
| Added a real-world data point to Concept Note §10.2: the O-RAN Alliance's 2021 MVP white paper states the actual specified O-DU/O-RU fronthaul split is Option 7-2x, not literally Option 2/6/8 | This doesn't change the modeling choice (the 3-level Option 2/6/8 abstraction is still needed for tractability), but the thesis text should acknowledge this rather than imply Option 2/6/8 is how real O-RAN deployments work |

### Blockers
| Blocker | Severity | Plan |
|---------|----------|------|
| None | -- | The 2026-08-13/2026-08-15 blocker is closed: the sweep now covers the real config values. |

### Tomorrow's Plan
- [ ] Proceed to the full 10-seed x 11-method experiment matrix (Phase 4, `docs/workflow.md`) now that this gate is genuinely satisfied -- ideally on a GPU host given how slow this sweep was on a CPU-only sandbox (5 network, 100 episodes x 2 seeds x 6 variants)
- [ ] The O-RAN track's needs-validation flags are now partially informed but not resolved (`docs/oran_thesis_guide.md`): RU/DU/CU/fronthaul power constants and traffic breakpoints/Poisson rate/default scenario scale remain fully open (no source yet gives numbers for these); the split-centralization mapping now has one concrete real-world data point (O-RAN Alliance's actual Option 7-2x) but the 3-level abstraction itself is still an unvalidated tractability simplification. A source giving actual O-RAN Alliance/vendor RU-DU-CU hardware power measurements, or a traffic trace/standard for 5G small-cell deployments, would close the remaining gaps.

### Notes
Full per-variant results (`data/results/proxy_sweep_2026-08-29/proxy_sweep_summary.json`, session-local, not committed per `data/results/*`'s `.gitignore` entry -- same convention as the 2026-08-05 entry's now-gone `data/results/proxy_sweep/`):

| Variant | Mean final eval reward (2 seeds) | Mean tail critic loss | Crashed? |
|---|---|---|---|
| lr_pair_down | -10282.326 | 6865.779 | No |
| lr_pair_default | **-10403.361** | 6446.147 | No |
| lr_pair_up | -10196.111 | 6419.297 | No |
| tau_down | -10366.821 | 6297.988 | No |
| tau_default | **-10349.359** | 7962.808 | No |
| tau_up | -10259.337 | 5504.877 | No |

Unlike the 2026-08-05 sweep, the default wasn't the best-performing reward here (lr_pair_up and tau_up both scored marginally better) -- but Section 12.11 item 2's bar is "not visibly unstable relative to the alternatives," not "best of the three," and none of these differences are large relative to the run-to-run variance visible across just 2 seeds. Same caveat as 2026-08-05 applies: at only 100 episodes on R=5/U=2, none of these runs show a converged policy (QoS satisfaction 15-72%, 1-2.8 of 5 RRHs active) -- expected and fine for a sensitivity check, not a preview of final results.

---

## Date: 2026-08-15

### What I Did Today
- [x] Found the actual root cause of the 2026-08-13 entry's lr-pair discrepancy: it wasn't just that `config/default.yaml` drifted from the 2026-08-05 sweep's tested values — three source files (`agents/branching_mp_dqn.py`, `agents/pdqn_agent.py` (inherited by `agents/mpdqn_agent.py`), and `training/hyperparam_search.py::run_proxy_sensitivity_sweep`) still hardcoded the old 1e-3/1e-4 pair as their Python-side fallback default for `get_val("lr_discrete"/"lr_actor", ...)`. Every actual config file (`default.yaml`, `small_network.yaml`, `large_network.yaml`) already explicitly overrides both keys with the real 1e-4/3e-4 values, so this was masked in normal use — but it meant the fallback itself, and any future config that omitted these keys, would silently train at the wrong, unvalidated rates.
- [x] Corrected all three fallback defaults to 1e-4/3e-4, matching the real config values and Concept Note v4.0 §12.2/§12.11.
- [ ] The underlying blocker from 2026-08-13 is still open: nothing in this fix constitutes actually running Section 12.11's proxy sweep against 1e-4/3e-4. That requires an actual `training/hyperparam_search.py::run_proxy_sensitivity_sweep()` execution, which was not run today — fixing the code's fallback defaults is a prerequisite for that sweep being meaningful, not a substitute for running it.

### Time Spent
| Activity | Hours |
|----------|-------|
| Coding | 0.1 |
| Writing | 0.05 |
| Reading | 0.1 |
| Debugging | 0 |
| Running experiments | 0 |
| **Total** | ~0.25 |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Fix the three hardcoded fallback defaults to 1e-4/3e-4 without also running the proxy sweep | The code fix is a correctness bug (a fallback default that would silently activate for any config omitting these keys) independent of whether the sweep has been re-run; fabricating a sweep result to close out the 2026-08-13 blocker would violate the Ethical AI Rule (`docs/rules.md` §10) the same way an unverified citation claim would. |

### Blockers
| Blocker | Severity | Plan |
|---------|----------|------|
| Section 12.11's proxy sweep still has not actually been run against the real lr_discrete/lr_actor (1e-4/3e-4) | Medium — unchanged from 2026-08-13 | Re-run `training/hyperparam_search.py::run_proxy_sensitivity_sweep()` centered on 1e-4/3e-4 before the full 10-seed matrix, and log a fresh keep/change decision here |

### Tomorrow's Plan
- [ ] Re-run the proxy sweep centered on the actual config defaults (1e-4/3e-4), per the blocker above (carried over from 2026-08-13, still not done)

### Notes
This closes the code-level half of the 2026-08-13 gap (the fallback defaults now match reality) but not the empirical half (the sweep itself). Left the 2026-08-13 and 2026-08-05 entries below unedited as a historical record.

---

## Date: 2026-08-13

### What I Did Today
- [x] Found that `config/default.yaml`'s actual `lr_discrete`/`lr_actor` (1e-4/3e-4) do not match the lr-pair the 2026-08-05 proxy sweep tested and validated (1e-3/1e-4) — the two entries were never the same values, meaning the sweep's "kept unchanged" conclusion below doesn't actually cover today's real defaults.
- [x] Corrected Concept Note v4.0 §12.2's hyperparameter table to describe the actual `config/default.yaml` values instead of the untested 1e-3/1e-4 pair.
- [ ] Re-run the Section 12.11 proxy sweep centered on the real defaults (1e-4/3e-4) before treating them as validated

### Time Spent
| Activity | Hours |
|----------|-------|
| Coding | 0 |
| Writing | 0.1 |
| Reading | 0.1 |
| Debugging | 0 |
| Running experiments | 0 |
| **Total** | ~0.2 |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Leave `config/default.yaml`'s lr_discrete/lr_actor at 1e-4/3e-4 (don't retroactively change config to match the old sweep) | The 2026-08-05 sweep's "kept unchanged" decision was about the pair it actually tested (1e-3/1e-4), not today's config values — changing config to match the sweep would be retrofitting the config to a decision that was never really about it. Documenting the real values and flagging the gap is more honest than either silently leaving the mismatch or quietly rewriting one side to match the other. |

### Blockers
| Blocker | Severity | Plan |
|---------|----------|------|
| Section 12.11's proxy sweep has not actually been run against the config's real lr_discrete/lr_actor (1e-4/3e-4) | Medium — the current defaults are unvalidated by any sweep | Re-run `training/hyperparam_search.py::run_proxy_sensitivity_sweep()` centered on 1e-4/3e-4 before the full 10-seed matrix, and log a fresh keep/change decision here |

### Tomorrow's Plan
- [ ] Re-run the proxy sweep centered on the actual config defaults (1e-4/3e-4), per the blocker above

### Notes
This doesn't invalidate the 2026-08-05 entry below — that sweep genuinely ran and genuinely validated the pair it tested. The gap is that `config/default.yaml` was never updated to match afterward (or was edited independently later), so the two drifted apart. Left the 2026-08-05 entry unedited as a historical record; this entry documents the discrepancy and the follow-up needed.

---

## Date: 2026-08-05

### What I Did Today
- [x] Ran `training/hyperparam_search.py::run_proxy_sensitivity_sweep()` at full scale — Concept Note v4.0 Section 12.11's hyperparameter proxy sweep (R=5, U=2, 100 episodes, 2 seeds, 6 variants: lr-pair down/default/up, τ down/default/up), the gate this section requires before committing to the full 10-seed × 11-method experiment matrix.
- [x] Logged the resulting keep/change decision (below) per Section 12.11 item 3.
- [ ] Begin the full 10-seed × 11-method experiment matrix (Phase 4)

### Time Spent
| Activity | Hours |
|----------|-------|
| Coding | 0 |
| Writing | 0.25 |
| Reading | 0 |
| Debugging | 0 |
| Running experiments | ~0.7 (40 min automated wall-clock) |
| **Total** | ~1 |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep the default branch/continuous-net learning rates (lr_discrete=1e-3, lr_actor=1e-4) unchanged | Sweeping ±half an order of magnitude (down: ~3.16e-4/3.16e-5; up: ~3.16e-3/3.16e-4) produced no crash and no non-finite reward at either extreme, and the default's mean final eval reward (-2672.17 over 2 seeds) was in fact the least-negative (best) of the three lr-pair variants tested (down: -3426.28, up: -3023.72). Per Section 12.11 item 2, a not-visibly-unstable default is kept without further search. |
| Keep the default τ (0.005) unchanged | Same protocol: down (~1.58e-3) and up (~1.58e-2) both ran without crashing, and the default's mean final eval reward (-1838.70) was again the best of the three τ variants (down: -3141.78, up: -3810.16). Kept unchanged. |

### Blockers
| Blocker | Severity | Plan |
|---------|----------|------|
| None | — | — |

### Tomorrow's Plan
- [ ] Begin the full 10-seed × 11-method experiment matrix (Phase 4), now that Section 12.11's gate has run and kept the defaults
- [x] Reconcile `evaluation/scalability.py`'s RRH-size set (6/12/24) with Section 12.2's table (5/12/20/35/50) — already done: `SCALABILITY_SWEEP_N_RRH`/`scales` in `evaluation/scalability.py` and `evaluation/latency_benchmark.py` both use {5,12,20,35,50}

### Notes
Full per-variant results (`data/results/proxy_sweep/proxy_sweep_summary.json`, raw log in `data/results/proxy_sweep/run_log.txt`):

| Variant | Mean final eval reward (2 seeds) | Mean tail critic loss | Crashed? |
|---|---|---|---|
| lr_pair_down | -3426.284 | 11277.614 | No |
| lr_pair_default | **-2672.172** | 10830.119 | No |
| lr_pair_up | -3023.716 | 14252.331 | No |
| tau_down | -3141.779 | 11482.052 | No |
| tau_default | **-1838.700** | 18126.815 | No |
| tau_up | -3810.164 | 16713.043 | No |

Caveat worth flagging for whoever runs the full matrix: at only 100 episodes on a 5-RRH/2-UE network, none of these six runs show a converged policy — QoS satisfaction ranged 59-85% and only 1-2 of 5 RRHs were active on average across variants, both well short of the thesis's eventual targets. That's expected and fine for a Section 12.11 *sensitivity* check (its job is only to catch outright instability before the full matrix, not to reach a good policy), but the absolute reward/QoS numbers above should not be read as a preview of final results.
