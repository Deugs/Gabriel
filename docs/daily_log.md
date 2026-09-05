# Daily Log

> Filled instances of `docs/daily_log_template.md`. Newest entry first.

## Date: 2026-08-30 (CF-mMIMO thesis, remaining pages)

### What I Did Today
- [x] The candidate supplied the remaining pages (75-117) of the same MASc thesis flagged as missing in the prior entry today -- a genuine follow-through on the specific gap disclosed there, rather than a new source.
- [x] Read Chapter 4 (Simulation Setup, Results, Discussion) and both Appendices in full. Found two genuinely usable results: (1) an ETSI standard (TR 103 737, via the thesis's own citation) for 24-hour power averaging with three weighted load periods -- Busy=6h, Medium=10h, Low=8h -- that **exactly** matches this repo's own traffic model's implied floor duration (8h) and active duration (16h), a real confirmation of the aggregate day-fraction split; (2) Appendix A's formula-derived fronthaul-rate worked example (Split 7.2≈2.764 Gbps, Split 8≈5.898 Gbps at N=8 antennas), which -- honestly disclosed -- disagrees both with the same thesis's own Chapter 4 simulation assumptions (10/20 Gbps) and with 3GPP TR 38.801's real bandwidth ratio, reinforcing rather than closing the already-known bandwidth-vs-power-ratio gap.
- [x] Confirmed the Appendix A table does not decompose power by RU/DU/CU component either (it's fronthaul bandwidth only) -- the RU/DU/CU wattage flag remains the one fully "still open" flag after 6 literature-check passes across two days, honestly reported as such rather than stretched to claim partial resolution it doesn't have.

### Time Spent
| Activity | Hours |
|----------|-------|
| Coding | 0 |
| Writing | 0.3 |
| Reading | 0.4 (thesis pages 75-117, ~43 pages: Chapter 4, Chapter 5, both appendices, bibliography) |
| Debugging | 0 |
| Running experiments | 0 |
| **Total** | ~0.7 |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Documented the ETSI 24h-weighting match as a genuine confirmation, precisely scoped | The match (8h floor / 16h active, both exact) is real and worth crediting -- but three aggregate durations don't uniquely determine four breakpoint values, so I was explicit that `t1`-`t4` individually (and `floor_ratio`) remain open, only the aggregate split is now grounded. Overstating this as "breakpoints validated" would be exactly the kind of imprecision the Ethical AI Rule warns against. |
| Did not pick a "winning" fronthaul-rate figure among the thesis's own two internally-disagreeing numbers (10/20 Gbps vs. 2.764/5.898 Gbps) or against 3GPP TR 38.801's ratio | All three are legitimate in their own context (simulation assumption, formula-derived example, real standard) but disagree with each other -- disclosing the spread honestly is more useful than silently picking one to cite as "the" number. |
| No numeric constant changed | Neither new finding gives a clean RU/DU/CU/fronthaul Watt decomposition; the ETSI finding is duration-only (already matches, nothing to change), and the fronthaul-rate figures are yet more bandwidth data disagreeing with each other, not power. |

### Blockers
| Blocker | Severity | Plan |
|---------|----------|------|
| None | -- | The RU/DU/CU/fronthaul wattage flag remains open; per the last several entries, only a source that actually decomposes measured or assumed power by component (not bandwidth, not GOPS, not a percentage of a differently-scoped total) would close it. |

### Tomorrow's Plan
- [ ] This closes out the currently-supplied literature; ready for whatever the candidate directs next
- [ ] If more literature is wanted, a component-level power breakdown (RU vs. DU vs. CU vs. fronthaul, in Watts) remains the single most valuable missing document type

### Notes
No code or config changes this round. This is the sixth O-RAN literature-check pass in two days and the first to fully resolve a previously-disclosed "missing pages" gap by the candidate directly supplying exactly what was flagged as missing -- a good sign the disclosure practice (naming specific missing pages rather than a vague "partial read") is actionable.

---

## Date: 2026-08-30 (O-RAN EE survey + CF-mMIMO thesis)

### What I Did Today
- [x] The candidate supplied 3 more sources without comment: Abubakar et al. 2023 ("Energy Efficiency of Open Radio Access Network: A Survey," IEEE VTC2023-Spring), and two PDF parts of a 2025 MASc thesis (SK Razib Ahmed, UBC, "Cell-Free Massive MIMO under the Open Radio Access Network Flexible Functional Splits towards Efficient Cellular Network").
- [x] Fixed an environment issue first: `pdftoppm`/`pdftotext` (poppler-utils) had gone missing from this sandbox since the last literature-check round (likely a container/session reset since the base image only ships the `libpoppler134` library, not the CLI tools) -- reinstalled via `apt-get install poppler-utils` before it was needed again.
- [x] Read Abubakar et al. 2023 in full (8 pages). Found a genuinely new, useful data point: a real fronthaul *power* percentage (not bandwidth) cited from Lopez-Perez et al., split-dependent (2%/30%/60% for split options 6/7/8) -- the first source in either literature-check round giving fronthaul's power *share*, as opposed to bandwidth or an absolute Watt figure.
- [x] Read the two-part MASc thesis as far as it was supplied (thesis pages 1-22 and 51-74 of what appears to be a ~115+ page document) -- discovered a genuine gap (pages ~23-50 not included) and that both parts stop before Chapter 4's likely numeric parameter table and the thesis's own referenced Appendix A ("Table A.1: Maximum Supported APs per DU under 20 Gbps Fronthaul Budget for Split 7.2 and Split 8," page 115) -- exactly the kind of table this session has been hoping to find for the RU/fronthaul power flag. Documented what was actually supplied honestly rather than guessing at what the missing pages might contain.

### Time Spent
| Activity | Hours |
|----------|-------|
| Coding | 0 |
| Writing | 0.4 |
| Reading | 0.6 (Abubakar et al., 8 pages; MASc thesis, ~53 pages across the two supplied parts) |
| Debugging | 0.1 (poppler-utils reinstall) |
| Running experiments | 0 |
| **Total** | ~1.1 |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| No numeric constant changed this round | Neither source gives a clean, absolute-Watt, per-component decomposition matching this model's own parameterization. Abubakar et al.'s 2%/30%/60% figure is a *percentage of total power*, not directly convertible to this model's Watt-valued `p_fh_per_ru_by_split` without an assumption about what "total power" means in this model's own terms -- documented as further quantification of an already-disclosed gap, not used to rescale anything. |
| Documented Abubakar et al.'s own survey conclusion that RU/fronthaul-specific O-RAN power modeling is an open research gap field-wide | This is valuable context distinct from a numeric finding: it confirms this repo's own "still open" flag status reflects a genuine, literature-wide gap as of a comprehensive 2023 survey, not a shortcoming of this session's own search effort. |
| Documented the MASc thesis's structural power-model similarity (static + load-dependent) and closed-form fronthaul-rate formulas as further corroboration, without extracting any numbers | The thesis's own equations are general/symbolic (no numeric instantiation appears in the pages supplied) -- useful as independent structural/directional confirmation, not as a numeric source. |
| Explicitly disclosed the gap in the supplied thesis PDFs (missing pages, stops before Ch.4/Appendix A) rather than silently working around it | The Ethical AI Rule requires disclosing what is and isn't actually available, not just what would be convenient. Telling the candidate exactly what's missing (and that the missing Appendix A table is likely the single most useful remaining lead) is more useful than quietly noting "partial thesis read" without specifics. |
| Documented the thesis's own CF-mMIMO scenario scale (K=16, L=20-50 APs) for the default-scenario-scale flag, disclosing its ratio differs from this repo's own | Consistent with the existing DQRL/OREO treatment: report the comparison honestly, including where it doesn't line up as neatly (this thesis's UE:AP ratio is *below* ours, unlike DQRL/OREO which bracketed it). |

### Blockers
| Blocker | Severity | Plan |
|---------|----------|------|
| None | -- | If the candidate can supply the missing pages of the MASc thesis (specifically pages ~75-115+, covering Chapter 4's simulation parameters and Appendix A's AP-per-DU-under-fronthaul-budget table), that is now the single most promising remaining lead for the RU/fronthaul power flag. |

### Tomorrow's Plan
- [ ] If the candidate wants to keep pursuing the power-model flag, ask for the MASc thesis's remaining pages (75-115+) specifically, rather than a new source
- [ ] Otherwise, ready to move to whatever the candidate directs next

### Notes
No code or config changes this round -- documentation-only, same discipline as every other 2026-08-30 entry. This is now the fifth O-RAN literature-check pass in two days; the RU/DU/CU/fronthaul power flag remains the only one of the four O-RAN needs-validation flags still fully "still open" rather than "partially resolved/informed." The sandbox's Python dependency stack (numpy, pytest, etc.) had also reset alongside poppler-utils; rather than a full reinstall for a docstring-only change, verified safety directly via `git diff` (confirmed the edit touches only the module docstring) and `ast.parse()` (confirmed the file still parses as valid Python) -- black/flake8 (installed separately) passed clean.

---

## Date: 2026-08-30 (3GPP TR 38.801 primary source + real O-RAN power measurements)

### What I Did Today
- [x] The candidate supplied two more sources without further comment: Al-Tahmeesschi et al. 2025 (arXiv:2507.00928, "Enhancing Open RAN Digital Twin Through Power Consumption Measurement") and 3GPP TR 38.801 itself (V0.4.0, 2016-08, Release 14 -- the actual primary document recommended earlier today, previously only seen via a secondary HUBER+SUHNER infographic reproducing its bandwidth table).
- [x] TR 38.801's Annex A Table A-1 gave exact, non-rounded bandwidth figures for all 8 split options (plus 7a/7b/7c sub-variants and per-option latency) -- and these figures **exactly** cross-validated the pixel-verified HUBER+SUHNER infographic reading from earlier today (Option 8 = 157.3/157.3 Gb/s, matching to the decimal). A genuine, welcome confirmation that the pixel-verification methodology used earlier was correct.
- [x] Al-Tahmeesschi et al. 2025 is the first source in either literature-check round to give real, RU/DU/CU-*decomposed* O-RAN power measurements (not whole-BS, not macro-cell, not a single vendor total). Their Split 8 testbed is an exact match to this model's `c=2` (both are literally Option 8/PHY-RF split), giving the closest RU power anchor found yet (~44 W measured vs. ~11 W predicted by this model's own constants, a ~4x gap).

### Time Spent
| Activity | Hours |
|----------|-------|
| Coding | 0 |
| Writing | 0.5 |
| Reading | 0.5 (TR 38.801, 36 pages, focused on §6.1.2 and Annex A; Al-Tahmeesschi et al., 6 pages, full read) |
| Debugging | 0 |
| Running experiments | 0 |
| **Total** | ~1.0 |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| No numeric constant changed this round | Neither source gives a clean decomposition of measured RU/DU/CU power into this model's separate processing/RF/static-power terms. TR 38.801's bandwidth table confirms (exactly) figures already cited from a secondary source -- a citation-quality upgrade, not a new numeric fact. |
| Documented TR 38.801's exact bandwidth table as superseding/confirming the HUBER+SUHNER infographic's pixel-verified reading | The primary document is now the citable source (Option 2 = 4016/3024 Mb/s, Option 6 = 5626.7/7140 Mb/s, Option 8 = 157.3/157.3 Gb/s, exactly matching the earlier pixel-verified reading) -- the infographic remains a valid secondary corroboration, but the thesis should cite the primary TR. |
| Documented Al-Tahmeesschi et al.'s Split 8 measurements (RU ~43-45 W, DU+CU ~119.5-141.6 W) as the closest real anchor for `c=2`, without rescaling any constant | The ~4x gap between this model's own composite RU estimate and the real measurement is the closest found in either round, but the paper gives only combined DU+CU (not separate) for Split 8, and RU power isn't decomposed into processing-vs-RF -- setting any specific array element would still require guessing a split of the total. |
| Explicitly flagged a hardware-choice confound between the paper's two testbeds | Split 8 uses one shared server for DU+CU; Split 7.2b uses two separate dedicated servers, one per component. Comparing Split 7.2b's DU (~187-194 W) and CU (~189.6-192.7 W) figures against Split 8's combined DU+CU (~119.5-141.6 W) to conclude "Split 7.2b needs more DU/CU power than Split 8" would be misleading -- most of that gap is which server class was used, not the split option. Stating this caveat explicitly avoids a plausible-looking but wrong inference. |
| Cited the paper's "power doesn't scale with load" finding as validation of an existing design choice, not a needed change | This model's `compute_du_power()`/`compute_cu_power()` already depend on active-RU-count and split choice, not on instantaneous PRB/throughput -- exactly the structural choice this real measurement independently supports. |
| Noted the RAN550's measured Split-7.2b RU power (~28.3-30.1 W) is lower than its own datasheet's "typical power consumption: 40 W" claim | A real-vs-nominal discrepancy worth disclosing for completeness, though it's a different physical quantity from the max-TX-power figure already used for the `p_max_dbm` fix, so that fix is unaffected. |

### Blockers
| Blocker | Severity | Plan |
|---------|----------|------|
| None | -- | The RU/DU/CU/fronthaul wattage table remains the one still-open needs-validation flag; a source that decomposes a single O-RU's or O-DU's power into processing vs. RF vs. static-baseline shares (rather than giving only a bundled total) would be needed to close it fully. |

### Tomorrow's Plan
- [ ] If the candidate wants to keep pursuing this, the natural next ask is a source that decomposes total measured O-RU/O-DU/O-CU power into sub-component shares, since every source checked so far (RAN550 datasheet, Open RAN Handbook, Hoffmann presentation, now Al-Tahmeesschi et al.) gives only bundled totals
- [ ] Otherwise, ready to move to whatever the candidate directs next

### Notes
No code or config changes this round -- a documentation-only pass, same discipline as the other 2026-08-30 entries. This closes out the literature thread the candidate opened by asking "what is still pending" / "what is needed in the literature": both of the two specific documents recommended then (3GPP TR 38.801, and a vendor/measurement source with real O-RAN component-level power data) have now been supplied and incorporated.

---

## Date: 2026-08-30 (vendor datasheet + fronthaul bandwidth table)

### What I Did Today
- [x] The candidate supplied 3 more sources without further comment: Benetel's RAN550 datasheet (a real, small-cell-class, Split-7.2x indoor O-RU product) and two identical copies (confirmed via `md5sum`) of a HUBER+SUHNER/CubeOptics infographic reproducing 3GPP TR 38.801's own functional-split taxonomy and per-split fronthaul bandwidth table.
- [x] The infographic's bandwidth table is dense and multi-column; the linearized PDF text extraction scrambled the option-to-value mapping (naive reading order suggested Option 1 had the *highest* bandwidth, contradicting the infographic's own prose stating Option 8 has "the highest bandwidth requirements of all functional split options" -- a red flag). Rendered the page at 300 DPI, located the exact red vertical guide-line x-coordinates separating each split's column programmatically, annotated them, and cropped/re-read the header-number row and the bandwidth row against those same coordinates to get an unambiguous, pixel-verified mapping -- avoiding a transposition-style error like the one caught in the C-RAN power model on 2026-08-29.
- [x] Result: a real, quantitative confirmation of the O-RAN split-centralization mapping's (§10.2) monotonic direction for the three mapped options, plus one genuine numeric constant fix (`power.ru.p_max_dbm`, 30 -> 33 dBm) from the RAN550 datasheet's real max-TX-power spec.

### Time Spent
| Activity | Hours |
|----------|-------|
| Coding | 0.3 |
| Writing | 0.4 |
| Reading | 0.3 (RAN550 datasheet, 4 pages; HUBER+SUHNER infographic, 1 dense page) |
| Debugging | 0.4 (pixel-level re-analysis of the infographic to avoid a transposition error from scrambled text extraction) |
| Running experiments | 0 |
| **Total** | ~1.4 |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Changed `power.ru.p_max_dbm` 30 -> 33 dBm (1 W -> 2 W) | Benetel RAN550's datasheet states "Maximum TX output power (total EIRP): 2 W" for a real, commercially available, Split-7.2x indoor small-cell O-RU -- the same physical quantity, same units, as this model's `p_max_dbm` action-space ceiling. A clean primary-source match, not a guess, following the same principle as the FTP Model 3 traffic-model fix earlier today. |
| Did NOT decompose RAN550's "typical power consumption: 40 W" into this model's RU processing/DU/fronthaul arrays | The datasheet gives only a single total-power figure with no breakdown into processing vs. RF vs. fronthaul-interface shares -- assigning it to specific array elements would require guessing that decomposition. Documented as a scale-mismatch-narrowing data point (5-14x this model's own composite RU estimate, vs. 20-100x for the earlier macro-cell/enterprise-server figures) instead. |
| Precisely pixel-verified the HUBER+SUHNER infographic's bandwidth table before citing any number from it | The naive linearized-text reading order was backwards relative to well-known 5G fronthaul facts (it implied Option 1 has the highest bandwidth, when Option 8/CPRI is well known to have the highest) -- a clear signal the text extraction order didn't match the visual column layout. Rendered at 300 DPI, found the red guide-line x-coordinates programmatically, and cross-checked against the unambiguous "3GPP TR 38.801 / 1 2 3 4 5 6 7-3 7-2 7-1 8" header row and the Small Cell Forum naming row (PDCP-RLC=2, RLC-MAC=3, etc.) before trusting any bandwidth figure. Final verified table: Option 2=3/4 Gbps, Option 6=7.1/5.6 Gbps, Option 8=157.3/157.3 Gbps (full 10-option table in the Concept Note). |
| Did not rescale `p_fh_per_ru_by_split` to match the real bandwidth ratio | The verified bandwidth figures show Option 8 requires ~39-52x Option 2's bandwidth, far steeper than this model's own 1:2:5 power ratio -- but no source states that fronthaul *power* scales linearly with fronthaul *bandwidth* requirement, so inventing that proportionality to "fix" the ratio would be exactly the kind of unsupported claim the Ethical AI Rule forbids. Documented as a disclosable, now-quantified version of the existing "shape confirmed, magnitude unconfirmed" gap instead. |
| Confirmed the two HUBER+SUHNER PDFs are byte-identical (`md5sum`) before treating them as one source | Avoided double-counting or wasting effort reading the same document twice. |
| Added `tests/test_oran_env.py::test_p_max_dbm_matches_ran550_datasheet` | Locks in both the config value and the derived `env.p_max_w`, mirroring the traffic-model regression test added earlier today. |

### Blockers
| Blocker | Severity | Plan |
|---------|----------|------|
| None | -- | The RU/DU/CU/fronthaul wattage table remains the one still-fully-open needs-validation flag; would need a vendor datasheet that actually decomposes total O-RU/O-DU/O-CU power by subsystem (not just a single "typical power consumption" figure) to close it fully. |

### Tomorrow's Plan
- [ ] If the candidate wants to keep pursuing this, a vendor datasheet with a component-level power breakdown (not just a single total-power figure) would be the natural next ask, along with 3GPP TR 38.801 itself (recommended earlier, not yet supplied) for the split's own text describing these bandwidth figures' derivation
- [ ] Otherwise, ready to move to whatever the candidate directs next

### Notes
Full test/lint verification before commit: `black --check`, `flake8 --max-line-length=100`, `mypy --ignore-missing-imports`, and the full O-RAN test suite all clean. No C-RAN files touched (grep-confirmed). This continues the same-day pattern of the 3GPP TR 38.864 entry above: literature can sometimes give a genuine primary-source match (p_max_dbm, lambda_peak, packet_size_bits) worth acting on, and sometimes only narrows/contextualizes a gap without resolving it (the RU/DU/CU/fronthaul wattage table, the bandwidth-vs-power ratio mismatch) -- both outcomes are documented with equal rigor rather than the latter being quietly dropped.

---

## Date: 2026-08-30 (3GPP TR 38.864 primary source)

### What I Did Today
- [x] The candidate asked what specific literature was still needed to close the remaining O-RAN needs-validation flags; I recommended obtaining 3GPP TR 38.864 ("Study on network energy savings for NR") and 3GPP TR 38.801 directly, rather than more secondary citations of them.
- [x] The candidate supplied 3GPP TR 38.864 itself (a `.docx`, V18.1.0). Extracted its text (`unzip` + XML strip, since `pandoc` was unavailable in this sandbox) and read §5.1 (Energy consumption model for BS) and Annex A (Evaluation scenarios, traffic models and loads) in full.
- [x] This is the first source in either literature-check round that gives a genuine, primary-source, right-units numeric match rather than order-of-magnitude/qualitative context: Annex A's FTP Model 3 (0.5 MB packet size, 200 ms mean inter-arrival time) is a real, standard 3GPP Poisson traffic model. Updated `lambda_peak` and `packet_size_bits` to derive directly from it -- the first *numeric constant change* driven by literature in either O-RAN literature-check round (the 2026-08-29 C-RAN fix was also a real value change, but that was a same-day correction against Al-Zubaedi's own table, not part of this O-RAN check series).

### Time Spent
| Activity | Hours |
|----------|-------|
| Coding | 0.2 |
| Writing | 0.4 |
| Reading | 0.5 (3GPP TR 38.864, 72 pages, focused on §5.1 and Annex A/B) |
| Debugging | 0.1 (pandoc unavailable in this sandbox; worked around via unzip + XML text extraction) |
| Running experiments | 0 |
| **Total** | ~1.2 |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Changed `lambda_peak` 5.0 → 0.5 and `packet_size_bits` 1.0e6 → 4.0e6 (both in `config/oran_default.yaml` and `oran_env/traffic_model.py`'s class defaults) | TR 38.864 Annex A's FTP Model 3 gives 200 ms mean inter-arrival time (= 5 arrivals/s = 0.5 per this model's 0.1s step) and 0.5 MB (=4e6 bits) packet/file size -- a real 3GPP standard traffic model in exactly the units this model needs, not an approximate/order-of-magnitude match like every other source checked so far. This is a genuine primary-source derivation, following the same principle as the 2026-08-29 C-RAN fix (Al-Zubaedi's Table 3.1): when a primary source gives an exact match in the right units, use it. |
| Left `floor_ratio` and `t1`-`t4` unchanged | TR 38.864 Annex A's own "load (L)%" scenarios (Table A-1) are instantaneous PRB-utilization snapshots with no time-of-day association, and the TR's own stated scope ("prioritizes idle/empty and low/medium load scenarios") stops at 50% load with no busy-hour/full-load reference point -- it gives no floor:peak ratio or diurnal timing to derive these from. Extending the fix to these would require guessing, which the Ethical AI Rule forbids. |
| Adopted FTP Model 3 specifically (not FTP Model 3 IM or VoIP) | 3GPP leaves the traffic-model choice to the evaluating party; FTP Model 3 is the most commonly used baseline across 3GPP energy-saving evaluations, a defensible choice, documented as such rather than presented as the only possible one. Noted (not implemented) that FTP3-IM's lighter traffic (0.1 MB, 2s inter-arrival) could plausibly inform off-peak/floor behavior better than a flat `floor_ratio` scaling, but this module's structure only supports one packet size for the whole day -- changing that is a design change, left for a future round if wanted. |
| Documented TR 38.864 §5.1's real 3GPP power-consumption model as additional power-model context, without changing any power-model constant | §5.1's `P_static + P_dynamic` structure (scaled by active-TRX fraction, RF-bandwidth ratio, PSD ratio) independently confirms, from the actual governing 3GPP source, that this family of model is right for the C-RAN/O-RAN power models already in use -- but its Table 5.1-3 values are relative units with no absolute-Watt anchor, and the model is whole-BS, not disaggregated into O-RAN's RU/DU/CU/fronthaul components. Converting relative units to Watts would require inventing a scale factor, so no power-model constant was changed. |
| Added `tests/test_oran_env.py::test_traffic_model_defaults_match_3gpp_ftp_model_3` | Locks in the new literature-derived defaults (both the `ORANTrafficModel` class default and `config/oran_default.yaml`'s value), mirroring how the C-RAN power-model fix was paired with a re-run of its own regression test. |

### Blockers
| Blocker | Severity | Plan |
|---------|----------|------|
| None | -- | `floor_ratio`/`t1`-`t4` remain open; would need a source giving diurnal timing or a busy-hour:off-peak traffic ratio specifically, which TR 38.864 does not provide. |

### Tomorrow's Plan
- [ ] If the candidate wants to pursue 3GPP TR 38.801 (the actual source of the Option 2/6/7/8 split numbering, recommended alongside TR 38.864) next, it could similarly upgrade §10.2's split-mapping flag from qualitative to numeric
- [ ] Otherwise, ready to move to whatever the candidate directs next (e.g. running real experiments)

### Notes
Full test/lint verification before commit: `black --check`, `flake8 --max-line-length=100`, and `pytest tests/test_oran_env.py -v` all clean; confirmed no other O-RAN test/training/evaluation file hardcodes the old `lambda_peak`/`packet_size_bits` defaults before changing them (`grep` across `tests/test_oran_*.py`, `oran_training/`, `oran_evaluation/`). This is the first literature-driven numeric change in the O-RAN track since the 2026-08-29 config-wiring bug fix, and the first one driven by an exact primary-source match rather than a bug.

---

## Date: 2026-08-30 (default scenario scale)

### What I Did Today
- [x] Moved on to the last untouched O-RAN needs-validation flag: default scenario scale (`n_ru=4, n_du=1, n_cu=1, n_ue=8, n_splits=3`, Concept Note §10.3). No new sources supplied, so re-mined the same 8 already-held O-RAN PDFs (keyword search via `pdftotext`) for scenario-scale content this time.
- [x] Found two directly comparable RAN-DRL papers' own scenario scales (DQRL: 12 RUs; OREO: 42 RUs, 100 UEs) and one qualitative O-RAN Alliance deployment-scenario description. Documented as context/corroboration for the tractability rationale, without treating any of it as validating the exact `n_ru=4`/`n_ue=8` counts.

### Time Spent
| Activity | Hours |
|----------|-------|
| Coding | 0 |
| Writing | 0.3 |
| Reading | 0.3 (keyword search + targeted re-reads of DQRL's simulation-setup section, OREO's discussion section, the MVP white paper's Deployment Scenario E.1) |
| Debugging | 0 |
| Running experiments | 0 |
| **Total** | ~0.6 |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Left `n_ru=4, n_du=1, n_cu=1, n_ue=8, n_splits=3` unchanged | No source specifies these exact counts for a 5G/O-RAN small-cell scenario; changing them to match another paper's own scale (12 or 42 RUs) would defeat the tractability reason they were chosen (`2^n_ru * n_splits^n_ru` must stay enumerable for the flat MP-DQN baseline) and would not itself be "validation" -- those papers picked their own scales for their own reasons, not ours. |
| Documented DQRL (12 RUs, 12-16 UEs) and OREO (42 RUs, 100 UEs) as context, not validation | Both are directly comparable (RAN-DRL, energy-focused) recent papers with their own concrete scenario scales. This repo's UE:RU ratio (2) sits inside the range both papers use (~1.0-2.4) -- a genuine, if coincidental, consistency worth noting -- but the absolute RU count (4) is markedly smaller than either paper's, which is disclosed as a real scale difference rather than glossed over. |
| Highlighted OREO's own admitted scalability concern as independent corroboration | OREO's discussion section states its single-agent centralised RL formulation "may... face scalability challenges as the number of RUs grows substantially" -- this is a literature-stated version of exactly the tractability concern already motivating this repo's small `n_ru`, from a paper that is not this repo and had no reason to support this repo's design choice. Worth citing as corroboration of the *rationale*, explicitly not as validation of the *number*. |
| Noted the O-RAN Alliance's own MVP white paper's "Deployment Scenario E.1" (single cloudified O-DU serving "several" non-virtualized O-RUs) as qualitative-only support | "Several" is not a specific count, so this only supports the single-DU/multi-RU *structure* (`n_du=1` serving `n_ru=4`), not any specific number |

### Blockers
| Blocker | Severity | Plan |
|---------|----------|------|
| None | -- | This flag is now "partially informed" rather than "still open" (`docs/oran_thesis_guide.md`); closing it fully would need a source that specifically justifies a small-cell/testbed-scale RU/UE count, which none of the 8 sources on hand provides. |

### Tomorrow's Plan
- [ ] All four O-RAN needs-validation flags in `docs/oran_thesis_guide.md` have now been checked against the 8 supplied sources at least once; further progress on any of them needs new sources with more specific numeric content (a 5G/O-RAN-specific traffic trace, an O-RAN Alliance/vendor RU-DU-CU hardware power measurement at small-cell scale, or a paper justifying a specific small `n_ru`/`n_ue` count) rather than re-mining the same 8 PDFs again
- [ ] Otherwise, ready to move to whatever the candidate directs next (e.g. actually running experiments per `docs/oran_experiment_guide.md`/`docs/cran_experiment_guide.md`)

### Notes
No code or config numeric changes this round -- only a docstring/config-comment/doc update, same "no fabrication" discipline as the three prior 2026-08-30 entries. This is the fourth and last of the O-RAN needs-validation flags addressed via the 8-source pool gathered from the 2026-08-29/2026-08-30 supplied-PDF rounds.

---

## Date: 2026-08-30 (traffic model)

### What I Did Today
- [x] Moved on to the O-RAN track's traffic-model needs-validation flag (`oran_env/traffic_model.py`'s trapezoid breakpoints/Poisson rate). No new sources were supplied for this round, so re-examined all 8 O-RAN-context PDFs already on hand from the two power-model literature-check rounds, this time searching specifically for traffic-shape content (keyword search via `pdftotext` across all 8, then targeted re-reads of the promising hits).
- [x] Found one genuinely useful result (Lassoued & Boujnah 2026's Figure 7, a real diurnal traffic-load curve) and one non-result worth recording (no source directly precedents this module's specific temporal-Poisson-arrival design). Documented both without changing any numeric constant, since neither gives a matching lambda_peak/floor_ratio/packet_size_bits/t1-t4 for a 5G/O-RAN scenario.

### Time Spent
| Activity | Hours |
|----------|-------|
| Coding | 0 |
| Writing | 0.3 |
| Reading | 0.4 (keyword search across 8 already-held PDFs via `pdftotext`, then a targeted re-read of Lassoued & Boujnah 2026's Figure 7 and the OREO/MEC-survey Poisson-model passages) |
| Debugging | 0 |
| Running experiments | 0 |
| **Total** | ~0.7 |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Left every traffic-model numeric constant in `oran_env/traffic_model.py`/`config/oran_default.yaml` unchanged | No source gives a 5G/O-RAN-specific Poisson arrival rate, floor ratio, packet size, or exact trapezoid breakpoints. Lassoued & Boujnah 2026's Figure 7 is a generic macro-cellular *relative occupation rate* (%) curve, not a bps/Poisson-rate source, and its timing is only a rough (not exact) match to this module's `t1=7`/`t4=23` -- its decline looks closer to ~18:00 than this module's `t3=20`. Adjusting `t3` to fit a hand-read bar chart pixel height would be exactly the kind of unsupported precision the Ethical AI Rule forbids, so nothing was changed. |
| Documented Figure 7 as qualitative/order-of-magnitude shape support, added new Concept Note §10.6 | No dedicated needs-validation subsection existed for the traffic model in `manuscript/ORAN_BMPP_DQN_Concept_Note_v1.md` before now (only a one-line mention in §5.1); added §10.6 mirroring §10.5's structure, and corrected `docs/oran_thesis_guide.md`'s stale cross-reference (previously pointed traffic-model flag at §10.3, which is actually the unrelated "Default scenario scale" flag). |
| Documented that the temporal-Poisson-arrival design itself isn't directly precedented in the sources checked | OREO uses a Poisson *point process* for UE spatial positions (not per-step arrival counts), and a cited work in the MEC/Open RAN survey table explicitly describes its own traffic model as "Poisson point process (without temporal variability)" -- i.e., spatial-only, the opposite structural choice from this module's diurnal/temporal design. Poisson-based traffic modeling is precedented in this literature broadly; this specific temporal form is not directly precedented by any source checked so far. |

### Blockers
| Blocker | Severity | Plan |
|---------|----------|------|
| None | -- | The traffic-model flag remains open; would need a 5G/O-RAN-specific traffic trace or standard (e.g. a 3GPP traffic model TR, or ETSI's dynamic-load UE-emulator profile referenced in the Open RAN Handbook) to close. |

### Tomorrow's Plan
- [ ] If more sources are supplied, prioritize anything with an explicit 5G/O-RAN small-cell traffic trace or arrival-rate model over another generic macro-cellular curve
- [ ] Default scenario scale (`n_ru=4, n_ue=8`, Concept Note §10.3) remains the one fully-untouched needs-validation flag left in `docs/oran_thesis_guide.md`

### Notes
No code or config changes this round -- purely a documentation/citation pass, same honest "partial support documented, nothing invented" outcome as the two power-model rounds. Unlike those rounds, no new PDF was supplied this time; the finding came from re-mining sources already on hand for content relevant to a different flag, using `pdftotext` keyword search to avoid a costly blind re-read of all 8 PDFs.

---

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
