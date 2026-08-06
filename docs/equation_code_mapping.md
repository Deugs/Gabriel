# Equation-Code Mapping

This document maintains traceability between thesis equations and their code implementations.

## Format
| Thesis Eq. | Description | Code File | Function | Status |
|------------|-------------|-----------|----------|--------|

---

## Chapter 3: System Model

| Thesis Eq. | Description | Code File | Function | Status |
|------------|-------------|-----------|----------|--------|
| (3.1) | RB allocation fraction | `cran_env/resource_model.py` | `compute_rb_fraction()` | Pending |
| (3.2) | SNR/SINR model | `cran_env/cran_env.py` | `CRANEnv._compute_sinr()` | Implemented |
| (3.3) | Shannon capacity | `cran_env/cran_env.py` | `CRANEnv.step()` | Implemented |
| (3.4) | RRH computing resource load | `cran_env/resource_model.py` | `compute_rrh_load()` | Pending |
| (3.5) | Total C-RAN power | `cran_env/power_model.py` | `PowerModel.compute_total_power()` | Implemented |
| (3.6) | RRH power consumption | `cran_env/power_model.py` | `PowerModel.compute_rrh_power()` | Implemented |
| (3.7) | Fronthaul power (PON) | `cran_env/power_model.py` | `PowerModel.compute_fronthaul_power()` | Implemented |
| (3.8) | Fronthaul power (detailed) | `cran_env/power_model.py` | `PowerModel.compute_fronthaul_power()` | Implemented |
| (3.9) | BBU power consumption | `cran_env/power_model.py` | `PowerModel.compute_bbu_power()` | Implemented |
| (3.10) | BBU dynamic power components | `cran_env/power_model.py` | `PowerModel.compute_bbu_power()` | Implemented |
| (3.11) | BBU total power (substituted) | `cran_env/power_model.py` | `PowerModel.compute_bbu_power()` | Implemented |
| (3.12) | BBU pool processing power | `cran_env/power_model.py` | `PowerModel.compute_bbu_power()` | Implemented |

---

## Chapter 3: MDP Formulation (NEW — Currently Missing from Draft)

| Thesis Eq. | Description | Code File | Function | Status |
|------------|-------------|-----------|----------|--------|
| (3.13) | State space definition | `cran_env/cran_env.py` | `CRANEnv._get_obs()` | Implemented |
| (3.14) | Action space definition | `cran_env/cran_env.py` | `CRANEnv.action_space` | Implemented |
| (3.15) | Reward function | `cran_env/cran_env.py` | `CRANEnv.step()` | Implemented |
| (3.16) | Transition dynamics | `cran_env/cran_env.py` | `CRANEnv.step()` | Implemented |
| (3.17) | Optimization objective | `training/train_hybrid.py` | `train_hybrid_agent()` | Pending |

---

## Chapter 3: Proposed Algorithm

Rows below reflect the current proposed method, `agents/branching_mp_dqn.py::BranchingMPDQN`
(Branching MP-DQN + TD3) — the architecture pivoted twice since these rows were
first written (vanilla DDPG → Hybrid SAC-DDQN → Branching MP-DQN + TD3); the
superseded Hybrid SAC-DDQN implementation is kept at `agents/hybrid_sac_dqn.py`
for reference only and is no longer what these equations should map to.

| Thesis Eq. | Description | Code File | Function | Status |
|------------|-------------|-----------|----------|--------|
| (3.18) | Bellman expectation equation | `agents/branching_mp_dqn.py` | `BranchingMPDQN.update()` | Implemented |
| (3.19) | Optimal policy | `agents/branching_mp_dqn.py` | `BranchingMPDQN.select_action()` | Implemented |
| (3.20) | Bellman optimality equation | `agents/branching_mp_dqn.py` | `BranchingMPDQN.update()` | Implemented |
| (3.21) | Q-function update | `agents/branching_mp_dqn.py` | `BranchingMPDQN.update()` | Implemented |
| (3.22) | Critic loss (MSE) | `agents/branching_mp_dqn.py` | `BranchingMPDQN.update()` | Implemented |
| (3.23) | Policy gradient (actor) | `agents/branching_mp_dqn.py` | `BranchingMPDQN.update()` | Implemented |
| (3.24) | Soft update target networks | `agents/branching_mp_dqn.py` | `BranchingMPDQN._soft_update()` | Implemented |
| (3.25) | SAC entropy regularization | `agents/hybrid_sac_dqn.py` | `HybridSACDDQN.update()` | N/A (superseded) |

---

## Validation Tests

Each mapping must have a corresponding test in `tests/test_equation_consistency.py`,
e.g. for Eq. (3.5):

```python
class TestEquationConsistency:
    def test_power_model_matches_equation_3_5(self, default_config):
        """Verify total power matches Eq. (3.5): P_total = P_RRH + P_BBU + P_FH"""
        env = CRANEnv(default_config)
        env.reset(seed=42)

        action = {
            "rrh_on": np.ones(env.n_rrh, dtype=int),
            "power": np.ones(env.n_rrh, dtype=np.float32) * (env.p_max_w / 2.0),
        }
        _, _, _, _, info = env.step(action)

        expected_total = (
            info["rrh_power_w"] + info["bbu_power_w"] + info["fronthaul_power_w"]
        )
        assert info["total_power_w"] == pytest.approx(expected_total, rel=1e-5)
```

(Note: `env.power`, not `env.power_model`; `info["total_power_w"]`, not
`info["total_power"]` — matching the actual `CRANEnv`/`PowerModel` API.)

---

## Maintenance

**Update Protocol**:
1. When adding a new equation to the thesis, add a row to this table
2. When implementing a new function, verify it matches the corresponding equation
3. When modifying an equation, update the code and this table simultaneously
4. Run `pytest tests/test_equation_consistency.py` before each commit
