# Daily Log

> Filled instances of `docs/daily_log_template.md`. Newest entry first.

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
