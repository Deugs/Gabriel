# AI Assistance Log

## Disclosure Statement for Thesis

This research utilized AI-assisted tools for the following purposes:
- [x] Literature search and summarization
- [x] Code generation and debugging
- [ ] LaTeX formatting and figure generation
- [ ] Grammar and style checking
- [x] Statistical analysis suggestions

All AI-generated content has been independently verified for accuracy — including cases where AI-assisted verification itself failed and had to be corrected (see the OREO citation entry below), which is disclosed here rather than omitted.

---

## Log Entries

| Date | Tool | Purpose | Content Summary | Verified By |
|------|------|---------|-----------------|-------------|
| 2026-08-05 | Claude Code | Statistical analysis | Ran and logged `training/hyperparam_search.py::run_proxy_sensitivity_sweep()`'s Section 12.11 proxy sweep; recorded the keep/change decision for lr_discrete/lr_actor/tau | Candidate (see `docs/daily_log.md`) |
| 2026-08-13 | Claude Code | Code review / consistency check | Identified that `config/default.yaml`'s lr_discrete/lr_actor had drifted from the values the 2026-08-05 sweep actually tested; corrected Concept Note v4.0 §12.2 to describe the real config values | Candidate (see `docs/daily_log.md`) |
| 2026-08-29 | Claude Code | Literature verification | Read the candidate-supplied primary-source PDFs (Al-Zubaedi 2019 PhD thesis, Auer et al. 2011 EARTH paper) directly and checked every `power:` constant in `config/default.yaml` against them for the first time -- previously an unverified citation claim, since neither source was reachable from this environment's network sandbox before being supplied as files. Confirmed `p_stat_w`/`p_dyn_w`/`delta_p` (175/250/0.44) exactly match Al-Zubaedi's Table 3.1; found and fixed a transposition error in `p_lc_w`/`p_onu_active_w` (were 10.0/5.0, should be 5.0/10.5 per the same table); corrected an inaccurate implication that Auer et al. 2011's own per-BS-type table reports these same numbers (it doesn't -- Auer et al. 2011 is the source of the linear power-model form Al-Zubaedi's thesis adapts, not of these specific values) | Candidate (see `docs/daily_log.md` 2026-08-29 entry; `tests/test_env.py` re-run clean after the fix) |
| 2026-08-29 | Claude Code | Literature verification | Read 5 candidate-supplied O-RAN-context PDFs directly, attempting the same primary-source check for the O-RAN track's power-model needs-validation flags. Explicitly did NOT claim these constants are now validated -- none of the 5 sources gives a split-level RU/DU/CU/fronthaul wattage table matching `oran_env/power_model.py`'s parameterization, and one of them (OREO) explicitly scopes its own O-RAN RL energy model to exclude exactly those components. Documented what partial, order-of-magnitude support does exist versus what remains open, per the Ethical AI Rule, rather than either fabricating a validation or silently doing nothing. Did find and fix one genuine bug while checking: three per-split power arrays central to the Section 10.2 monotonicity design were never wired from config to `ORANPowerModel` at all | Candidate (see `docs/daily_log.md` 2026-08-29 entry; `tests/test_oran_env.py` re-run clean after the fix) |
| Multiple (session-based) | Claude Code | Code generation and debugging | Implemented `cran_env/`, all baselines, and `agents/branching_mp_dqn.py`; found and fixed several silent-bug classes (a dict-config-lookup bug that discarded most of `algorithm:`'s hyperparameters across four agent files; a duplicated, non-shared encoder; per-step instead of per-episode exploration decay) via recurring audit passes comparing the concept note against the actual code | Candidate (all changes covered by `pytest tests/`, `black`, `flake8`, `mypy` before merge; see PR history) |
| Multiple (session-based) | Claude Code | Literature search and summarization | Verified/corrected citation details against primary sources where accessible (e.g. the Qazzaz et al. OREO citation, retracted then restored once the actual paper was obtained and read); explicitly flagged citations that could NOT be verified in this environment (e.g. HySoft, Eskandarinia et al.) rather than asserting them as confirmed | Candidate — per the Ethical AI Rule (`docs/rules.md` §10), unverified claims are marked as such, not silently accepted |

---

## Verification Checklist

- [x] All AI-suggested equations manually derived — cross-checked against `docs/equation_code_mapping.md`; known coverage gap noted there (only Eq. 3.5 has a dedicated test)
- [ ] All AI-suggested references independently confirmed — several remain explicitly unconfirmed (HySoft, Eskandarinia et al.; see Concept Note v4.0 §4.2/§17), by design rather than oversight
- [x] All AI-generated code tested and validated — `pytest tests/`, `black --check`, `flake8`, `mypy` gate every change
- [ ] All AI-suggested claims backed by primary sources — the Iqbal et al. "reported power savings" figure is deliberately NOT restated anywhere in this repo for exactly this reason (no primary-source access available in this environment)
- [ ] Supervisor aware of AI usage extent — confirm before thesis submission; not something this repo's own audit process can attest to
