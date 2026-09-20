# Skill: Evaluation and Analysis

> **Status**: Invokable as the Antigravity `run-evaluation` skill (`.agents/skills/run-evaluation/`), which points back at this file as the spec of record.
>
> **Correction (this audit round)**: unlike its siblings `docs/skills/skill_environment.md` and `docs/skills/skill_hybrid_agent.md`, this file had never been updated since it was first written — it still described the superseded `HybridSACDDQN` agent, a `fronthaul_weight` reward key that was never implemented (the real key is `gamma_fronthaul`), function names (`evaluate_convergence`, `evaluate_energy_efficiency`, `ablation_study`, `scalability_analysis`, `compare_algorithms`) that don't match any function actually exported by `evaluation/__init__.py`, and a 9-color scheme covering only a subset of the real 11-method roster. It also never mentioned 5 of the 10 real evaluation modules (`csi_robustness.py`, `demand_response.py`, `generalization.py`, `power_time_profile.py`, `reward_sensitivity.py`). Everything below now describes the actual `evaluation/` package; read the module files directly (they're short) rather than treating this as a second source of truth.

## Purpose
Systematically evaluate DRL agents, compare against baselines, perform ablation studies, and generate publication-quality figures and tables for the thesis.

## The Real Module Roster

`evaluation/__init__.py` exports:
`analyze_convergence, compute_cohens_d, run_ablation_study, analyze_scalability, run_csi_robustness_evaluation, run_demand_response_evaluation, run_generalization_evaluation, run_latency_benchmark, run_power_time_profile_evaluation, run_reward_sensitivity_sweep, compute_confidence_interval, plot_learning_curves, plot_energy_efficiency_bar, plot_scalability_analysis, plot_ablation_comparison, plot_degradation_curve`.

Ten modules, one per evaluation concern:

### 1. `convergence.py` — `analyze_convergence(results_dir, save_dir, table_save_dir, ...)`

Aggregates every `summary.json` under `results_dir` — recognizing both the proposed method's one-file-per-seed layout and the baselines' one-list-per-algorithm layout — computes 95% CIs (`compute_confidence_interval`, t-distribution-based) per algorithm, and runs paired t-tests + Cohen's d (`compute_cohens_d`) between the proposed method (matched by the exact string `"Branching_MP_DQN"`, **not** the superseded `"Hybrid_SAC_DDQN"`) and every baseline found. Comparisons are paired **by seed**, not by list/filesystem-discovery order — an earlier version of this function paired by position, which could silently mismatch seeds between the proposed run and a baseline run; this is now a regression-tested fix (`tests/test_evaluation.py`). Exports a LaTeX table, `convergence_summary.tex`.

### 2. `ablation.py` — `run_ablation_study(config_path, save_dir, ...)`

Trains the proposed agent under 4 variants via `training.train_hybrid_agent`'s `config_overrides` argument — **Full**, **No-Switching-Cost** (`gamma_switch=0`), **No-Fronthaul-Term** (`gamma_fronthaul=0` — not `fronthaul_weight`, which was never a real key), **No-QoS-Penalty** (`beta_qos=0`) — comparing final held-out evaluation reward. Produces `ablation_study.pdf`. An earlier version silently dropped these overrides before they reached `train_hybrid_agent`; this is now regression-tested.

### 3. `scalability.py` — `analyze_scalability(config_path, save_dir, ...)`

Trains the proposed agent at 5 network scales — `R=5/U=2, R=12/U=10, R=20/U=20, R=35/U=25, R=50/U=30` (the last a stretch goal, per `docs/workflow.md`'s committed Experiment Matrix) — measuring final power, per-step execution time, QoS rate, and switching frequency at each scale.

### 4. `csi_robustness.py` — `run_csi_robustness_evaluation(config_path, save_dir, ...)`

The thesis's flagship robustness experiment (Concept Note §12.5, S3), addressing the perfect-CSI training assumption. Trains (or loads a pre-trained checkpoint, proposed-method only) `branching_mp_dqn`/`ddqn`/`ddpg` under perfect CSI, then at evaluation time only perturbs the *observed* channel-gain magnitude with additive Gaussian noise at `sigma ∈ {0, 0.01, 0.05, 0.1}` — the environment's true physics/reward still use the real channel. Reports EE and QoS-violation-rate degradation curves. Exposes the shared `_TRAINERS` dict and `_evaluate_under_csi_noise`, reused by `generalization.py` below.

### 5. `generalization.py` — `run_generalization_evaluation(config_path, save_dir, ...)`

Concept Note §12.3 (A5). Trains on the `weekday_urban` traffic profile, evaluates without retraining on both `weekday_urban` (matched) and `weekend_suburban` (generalization), reporting the EE change. Reuses `csi_robustness.py`'s `_TRAINERS`/`_evaluate_under_csi_noise` at `sigma=0`.

### 6. `latency_benchmark.py` — `run_latency_benchmark(config_path, save_dir, ...)`

Concept Note §12.3 (A3/G14). Measures pure forward-pass (`select_action`) latency in ms, isolated from training/env-step cost, at `R ∈ {5,12,20,35,50}` paired with `n_ue ∈ {2,10,20,25,30}` (the same R↔U pairing `scalability.py` uses). P-DQN/MP-DQN are included only at R≤12 and skipped gracefully (returns `None`, no crash) above that.

### 7. `demand_response.py` — `run_demand_response_evaluation(config_path, save_dir, ...)`

Comparable to Iqbal et al.'s Figs. 3 and 5. Trains each method once at default demand, then sweeps a **frozen** policy across demand multipliers `{0.5, 1.0, 1.5, 2.0, 2.5}` (scaling `traffic.base_rate_mbps`), reporting EE and mean power vs. demand.

### 8. `power_time_profile.py` — `run_power_time_profile_evaluation(config_path, save_dir, ...)`

Comparable to Iqbal et al.'s Fig. 4. Trains each method once, rolls out under the frozen policy, buckets per-step power by hour-of-day (0–23), producing a diurnal power profile per method.

### 9. `reward_sensitivity.py` — `run_reward_sensitivity_sweep(config_path, save_dir, ...)`

Concept Note §12.6 (S5). Grid-sweeps the *training-time* reward weight `gamma_switch` over `{0.01, 0.05, 0.1, 0.5, 1.0}` at fixed `beta_qos` — unlike the CSI-robustness/generalization/demand-response/power-time-profile evaluations above, each grid point here trains a **fresh** `BranchingMPDQN` from scratch (a reward weight, unlike CSI noise or demand scale, changes what a policy is trained to do, not just what it's evaluated against). Reports EE, QoS-violation rate, and switching frequency per `gamma_switch` value.

### 10. `plot_utils.py` — shared plotting helpers, no evaluation logic

`setup_matplotlib_style()` (IEEE/Nature-style rcParams), `compute_confidence_interval(data, confidence=0.95)`, `plot_learning_curves`, `plot_energy_efficiency_bar`, `plot_scalability_analysis` (2×2 grid; gracefully drops QoS/switching subplots if the data lacks those keys), `plot_degradation_curve` (generic metric-vs-swept-x curve, reused by CSI-robustness/generalization/demand-response/power-time-profile/reward-sensitivity/latency), `plot_ablation_comparison` (horizontal bar chart).

## Statistical Significance Testing

`analyze_convergence` (not a separate `compare_algorithms` function — that name never existed in this codebase) computes, for the proposed method vs. every baseline, paired **by seed**:

- Paired t-test (`scipy.stats.ttest_rel`)
- Cohen's d (`compute_cohens_d`, pooled standard deviation)

over 10 seeds (`[42, 123, 456, 789, 1011, 1337, 2024, 2718, 3141, 4242]`, per `docs/rules.md` Rule 3 — not the 5 an earlier draft of this file assumed).

## Figure Standards

### Color Scheme

The real 11-method roster (`config/default.yaml`'s `algorithm.name`, `baselines/__init__.py`, `agents/__init__.py`) is: All-ON/Uniform, Greedy, NMBS, Convex, DDQN, DDQN+SOCP, ANN+GSBF, DDPG (pure), P-DQN, MP-DQN, and the proposed Branching MP-DQN+TD3. The superseded Hybrid SAC-DDQN (`agents/hybrid_sac_dqn.py`) is not part of the comparison suite and should not appear in a results figure's legend. There is no single hardcoded `COLORS` dict in `evaluation/plot_utils.py` today — colors are assigned per-call via `matplotlib`'s `tab10`/`tab20` colormap indexed by algorithm order, not a fixed name→hex mapping. If a fixed palette is wanted for consistency across figures (e.g. for the final thesis submission), define one covering all 11 real method names above, not the 9-name placeholder set this file previously suggested.

### LaTeX Figure Export

```python
def setup_matplotlib_for_latex():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
    })
```

(Illustrative — `plot_utils.py::setup_matplotlib_style()` is the actual, currently-used equivalent; check it directly if you need the exact rcParams in force today.)

## Required Outputs for Thesis

| Output | File | Section |
|--------|------|---------|
| Convergence curves (11 methods, 10 seeds) | `thesis/figures/convergence_*.pdf` | 4.2 |
| Convergence statistics (CIs, paired t-test, Cohen's d) | `thesis/tables/convergence_summary.tex` | 4.2 |
| 24-hour energy profile | `power_time_profile.py` output | 4.3 / §12.3 |
| SINR CDF | (not yet a dedicated module — derive from `CRANEnv.step()`'s `mean_sinr_db`/per-user SINR if needed) | 4.4 |
| Ablation bar chart | `thesis/figures/ablation_study.pdf` | 4.5 |
| Scalability sweep (R=5..50) | `analyze_scalability` output | 4.6 |
| Inference-latency benchmark (R=5..50) | `latency_benchmark.py` output | 4.6 / §12.3 |
| CSI-robustness degradation curve | `csi_robustness.py` output | 4.7 / §12.5 |
| Cross-profile generalization | `generalization.py` output | 4.8 / §12.3 |
| Demand-response curve | `demand_response.py` output | cf. Iqbal Figs. 3/5 |
| Reward-weight (gamma_switch) sensitivity | `reward_sensitivity.py` output | §12.6 |

## Validation Checklist
- [ ] `analyze_convergence` pairs the proposed method against each baseline strictly by seed, not list position
- [ ] Comparisons use `"Branching_MP_DQN"` as the proposed-method name, not the superseded `"Hybrid_SAC_DDQN"`
- [ ] All error bars/CIs use 95% confidence intervals (`compute_confidence_interval`)
- [ ] Statistical significance (paired t-test) and effect size (Cohen's d) both reported for every head-to-head comparison
- [ ] Figures are vector graphics (PDF)
- [ ] `run_ablation_study`'s config overrides (`gamma_switch`, `gamma_fronthaul`, `beta_qos`) genuinely reach `train_hybrid_agent` — regression-tested, don't silently reintroduce the drop
- [ ] P-DQN/MP-DQN are skipped gracefully (not crashed) above their `n_rrh` tractability cap in `latency_benchmark.py`/`scalability.py`
- [ ] Captions are self-contained (explain what is shown and the key takeaway)
