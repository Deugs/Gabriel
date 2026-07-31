---
name: build-environment
description: Implement or extend the C-RAN Gymnasium environment (cran_env/) — channel model, traffic model, power model, and the main Env class. Use when building or modifying anything under cran_env/, or when asked to build the C-RAN simulation environment.
---

Follow the full component spec in [docs/skills/skill_environment.md](../../../docs/skills/skill_environment.md) — it defines `ChannelModel`, `TrafficModel`, `PowerModel`, and `CRANEnv`, plus the rules (Markov state, hybrid discrete/continuous action space matching, differentiable reward, deterministic-given-seed) and the validation checklist.

Steps:
1. Read `docs/skills/skill_environment.md` in full before writing any code.
2. Implement in this order (matches `docs/dev_guide.md` Phase 1): `channel_model.py` → `traffic_model.py` → `power_model.py` → `cran_env.py` → `tests/test_env.py`.
3. Use the EARTH-model power parameters from `config/default.yaml` (`p_stat_w: 175.0`, `p_dyn_w: 250.0`, etc.) — do not invent different values.
4. For every equation you implement, add a row to `docs/equation_code_mapping.md` linking the thesis equation number to the function that implements it.
5. Run through the spec's Validation Checklist (reset/step determinism, power model matches Al-Zubaedi reference values, Rayleigh statistics, Gymnasium API compliance) before considering the component done.
6. Run `pytest tests/test_env.py -v` and confirm it passes.
