# Rules: C-RAN DRL Thesis Development

## 1. Code-Text Consistency Rule
**Rule**: Every equation in the thesis MUST have a corresponding implementation in code, and vice versa.

**Enforcement**:
- Before writing any equation in LaTeX, verify the corresponding Python function exists and produces identical output for test cases
- Before implementing any new feature, check if it requires a thesis equation update
- Maintain a mapping file: `docs/equation_code_mapping.md`

**Example**:
```
Thesis Eq. (3.5): P_total(t) = P_RRH(t) + P_BBU(t) + P_FH(t)
Code: power_model.py::PowerModel.compute_total_power()
Test: tests/test_equation_consistency.py::TestEquationConsistency::test_power_model_matches_equation_3_5
```

## 2. Reference Validation Rule
**Rule**: Every parameter value, algorithm choice, and performance claim MUST be traceable to a cited reference or an ablation study.

**Enforcement**:
- Parameter table in thesis must include source column
- Algorithm choices require justification paragraph with citations
- Performance claims require comparison against cited baselines

**Penalty**: Unreferenced parameters default to Al-Zubaedi (2019) EARTH model values.

## 3. Baseline Fairness Rule
**Rule**: All algorithms must be evaluated under identical conditions: same environment, same random seeds, same evaluation protocol.

**Enforcement**:
- Single evaluation script: `evaluation/compare_all.py` (planned — not yet
  implemented; `training/train_baselines.py` currently runs the shared
  10-method benchmark suite under one config: all_on, greedy, nmbs, convex,
  ddqn, ann_gsbf, ddqn_socp, ddpg, pdqn, mpdqn — plus the proposed hybrid
  agent trained separately via `training/train_hybrid.py`, 11 methods total)
- Shared config file for all experiments
- Fixed random seed list (10 seeds, revised from 5 per supervisor review, Concept Note v3.0 §12.4): [42, 123, 456, 789, 1011, 1337, 2024, 2718, 3141, 4242]

**Forbidden**: Training proposed method with different hyperparameters than baselines; using different traffic traces; evaluating on different network sizes.

## 4. Reproducibility Rule
**Rule**: Any experiment must be reproducible from a single command within 24 hours on standard hardware.

**Enforcement**:
- All experiments defined in `experiments/` YAML files (planned — not yet
  implemented; `config/*.yaml` currently serves this role)
- Docker container with pinned dependency versions — done (`Dockerfile`,
  `docker-compose.yml`, pinned `requirements.txt`)
- Random seeds fixed; deterministic operations where possible
- Results saved with full config hash

**Required** (planned — `run_experiment.py`/`experiments/*.yaml` not yet
implemented; today's equivalent is):
```bash
# Reproduce the proposed method's training run
python training/train_hybrid.py --config config/default.yaml --seed 42
# or, via the Docker setup:
docker compose run --rm train hybrid --config config/default.yaml --seed 42
```

## 5. Novelty Defense Rule
**Rule**: Every claim of novelty must be accompanied by a specific comparison showing how prior work differs.

**Enforcement**:
- Novelty claims table in thesis (Section 2.5)
- For each claim: "Unlike [Author] who [did X], we [do Y] because [Z]"
- Peer review checklist before submission

**Template**:
> Unlike Iqbal et al. [6], who decouple RRH selection and power allocation into a two-stage process (DDQN for discrete + convex optimization for continuous), we propose a unified hybrid actor-critic that jointly optimizes both action types through a shared critic, eliminating the need for an analytical sub-problem solver and enabling end-to-end learning of the joint policy.

## 6. Scope Boundary Rule
**Rule**: The thesis scope is strictly bounded. Deviations require explicit justification and supervisor approval.

**In Scope**:
- Downlink transmission only
- Single BBU pool
- Single DRL agent (centralized)
- Perfect CSI (acknowledged limitation)
- Simulation-based evaluation

**Out of Scope** (without approval):
- Uplink traffic
- Multi-pool scenarios
- Multi-agent RL
- Imperfect CSI
- Hardware testbed validation
- 6G-specific features

## 7. Quality Gate Rule
**Rule**: No chapter may be marked complete without passing all quality gates.

**Gates**:
1. Mathematical consistency: All equations numbered, variables defined
2. Reference alignment: Every claim cited
3. Code-text consistency: Equations match implementation
4. Reproducibility: Experiment runnable from config
5. Baseline comparison: Results compared against ≥2 baselines
6. Statistical significance: n≥10 seeds (revised from 5 per supervisor review, Concept Note v3.0/v4.0 §12.4 — see Rule 3's seed list above), confidence intervals reported
7. Peer review: Signed off by supervisor or designated reviewer

## 8. Version Control Rule
**Rule**: All code changes must be committed with descriptive messages; thesis text changes tracked in Git.

**Commit Message Format**:
```
<type>(<scope>): <subject>

<body>

Refs: <issue/ticket number>
```

**Types**: feat, fix, docs, refactor, test, experiment, thesis
**Scopes**: env, agent, baseline, training, eval, thesis/chN

## 9. Documentation Rule
**Rule**: Every public function/class must have a docstring; every design decision must be documented.

**Docstring Format** (Google style):
```python
def compute_sinr(channel_gains, power, noise_power):
    """Compute Signal-to-Interference-plus-Noise Ratio.

    Args:
        channel_gains (np.ndarray): Complex channel gains, shape (n_rrh, n_ue).
        power (np.ndarray): Transmit power per RRH, shape (n_rrh,).
        noise_power (float): Background noise power in Watts.

    Returns:
        np.ndarray: SINR per UE, shape (n_ue,).

    Raises:
        ValueError: If channel_gains and power dimensions mismatch.

    References:
        Eq. (3.2) in thesis; corresponds to Iqbal et al. [6] Eq. (4).
    """
```

## 10. Ethical AI Rule
**Rule**: All AI-assisted writing must be disclosed; generated content must be verified for accuracy.

**Enforcement**:
- AI assistance log in `docs/ai_assistance_log.md`
- All AI-generated equations manually verified
- All AI-suggested references independently confirmed
- Thesis declaration includes AI usage statement
