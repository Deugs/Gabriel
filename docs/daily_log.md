# Daily Log

> Filled instances of `docs/daily_log_template.md`. Newest entry first.

## Date: 2026-08-05

### What I Did Today
- [x] Ran `training/hyperparam_search.py::run_proxy_sensitivity_sweep()` at full scale — Concept Note v4.0 Section 12.11's hyperparameter proxy sweep (R=5, U=2, 100 episodes, 2 seeds, 6 variants: lr-pair down/default/up, τ down/default/up), the gate this section requires before committing to the full 10-seed × 9-method experiment matrix.
- [x] Logged the resulting keep/change decision (below) per Section 12.11 item 3.
- [ ] Begin the full 10-seed × 9-method experiment matrix (Phase 4)

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
- [ ] Begin the full 10-seed × 9-method experiment matrix (Phase 4), now that Section 12.11's gate has run and kept the defaults
- [ ] Reconcile `evaluation/scalability.py`'s RRH-size set (6/12/24) with Section 12.2's table (5/12/20/35/50)

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
