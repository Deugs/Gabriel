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
| Multiple (session-based) | Claude Code | Code generation and debugging | Implemented `cran_env/`, all baselines, and `agents/branching_mp_dqn.py`; found and fixed several silent-bug classes (a dict-config-lookup bug that discarded most of `algorithm:`'s hyperparameters across four agent files; a duplicated, non-shared encoder; per-step instead of per-episode exploration decay) via recurring audit passes comparing the concept note against the actual code | Candidate (all changes covered by `pytest tests/`, `black`, `flake8`, `mypy` before merge; see PR history) |
| Multiple (session-based) | Claude Code | Literature search and summarization | Verified/corrected citation details against primary sources where accessible (e.g. the Qazzaz et al. OREO citation, retracted then restored once the actual paper was obtained and read); explicitly flagged citations that could NOT be verified in this environment (e.g. HySoft, Eskandarinia et al.) rather than asserting them as confirmed | Candidate — per the Ethical AI Rule (`docs/rules.md` §10), unverified claims are marked as such, not silently accepted |

---

## Verification Checklist

- [x] All AI-suggested equations manually derived — cross-checked against `docs/equation_code_mapping.md`; known coverage gap noted there (only Eq. 3.5 has a dedicated test)
- [ ] All AI-suggested references independently confirmed — several remain explicitly unconfirmed (HySoft, Eskandarinia et al.; see Concept Note v4.0 §4.2/§17), by design rather than oversight
- [x] All AI-generated code tested and validated — `pytest tests/`, `black --check`, `flake8`, `mypy` gate every change
- [ ] All AI-suggested claims backed by primary sources — the Iqbal et al. "reported power savings" figure is deliberately NOT restated anywhere in this repo for exactly this reason (no primary-source access available in this environment)
- [ ] Supervisor aware of AI usage extent — confirm before thesis submission; not something this repo's own audit process can attest to
