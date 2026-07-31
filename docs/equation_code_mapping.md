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
| (3.17) | Optimization objective | `training/train_hybrid.py` | `train()` | Pending |

---

## Chapter 3: Proposed Algorithm

| Thesis Eq. | Description | Code File | Function | Status |
|------------|-------------|-----------|----------|--------|
| (3.18) | Bellman expectation equation | `agents/hybrid_sac_dqn.py` | `HybridSACDDQN.update()` | Implemented |
| (3.19) | Optimal policy | `agents/hybrid_sac_dqn.py` | `HybridSACDDQN.select_action()` | Implemented |
| (3.20) | Bellman optimality equation | `agents/hybrid_sac_dqn.py` | `HybridSACDDQN.update()` | Implemented |
| (3.21) | Q-function update | `agents/hybrid_sac_dqn.py` | `HybridSACDDQN.update()` | Implemented |
| (3.22) | Critic loss (MSE) | `agents/hybrid_sac_dqn.py` | `HybridSACDDQN.update()` | Implemented |
| (3.23) | Policy gradient (actor) | `agents/hybrid_sac_dqn.py` | `HybridSACDDQN.update()` | Implemented |
| (3.24) | Soft update target networks | `agents/hybrid_sac_dqn.py` | `HybridSACDDQN._soft_update()` | Implemented |
| (3.25) | SAC entropy regularization | `agents/hybrid_sac_dqn.py` | `HybridSACDDQN.update()` | Implemented |

---

## Validation Tests

Each mapping must have a corresponding test:

```python
# tests/test_equation_consistency.py

class TestEquationConsistency:
    def test_power_model_matches_equation_3_5(self):
        """Verify total power matches Eq. (3.5): P_total = P_RRH + P_BBU + P_FH"""
        env = CRANEnv(config)
        env.reset(seed=42)

        action = {"rrh_on": np.ones(env.n_rrh), "power": np.ones(env.n_rrh) * 10}
        _, _, _, _, info = env.step(action)

        p_rrh = env.power_model.compute_rrh_power(action["rrh_on"], action["power"])
        p_bbu = env.power_model.compute_bbu_power(np.ones(env.n_bbu))
        p_fh = env.power_model.compute_fronthaul_power(action["rrh_on"])

        expected_total = p_rrh + p_bbu + p_fh
        assert info["total_power"] == pytest.approx(expected_total, rel=1e-5)

    def test_reward_matches_equation_3_15(self):
        """Verify reward matches Eq. (3.15) with given weights."""
        # Implementation pending
        pass
```

---

## Maintenance

**Update Protocol**:
1. When adding a new equation to the thesis, add a row to this table
2. When implementing a new function, verify it matches the corresponding equation
3. When modifying an equation, update the code and this table simultaneously
4. Run `pytest tests/test_equation_consistency.py` before each commit
