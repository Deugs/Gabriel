# C-RAN Track — Setup and Experiment Runbook

> Governs: `manuscript/MPhil_Thesis_Concept_Note_v4.md` (publications track).
> For the O-RAN/thesis track's equivalent guide, see
> `docs/oran_experiment_guide.md`. The two tracks share no code — running
> one never requires or affects the other.

This is a step-by-step runbook for setting up the environment and running
every experiment `docs/workflow.md`'s Phase 4 ("Experiments") requires for
this track, either by hand (this document) or in one shot via
`scripts/run_cran_experiments.sh`, which automates every step below.

## 1. Setup

**Option A: local virtualenv**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pre-commit install
export PYTHONPATH="$(pwd):$PYTHONPATH"   # no setup.py/pyproject.toml -- needed for every command below
```

**Option B: Docker (GPU, CUDA 12.1)**

```bash
docker compose build
docker compose run --rm train hybrid --config config/default.yaml --seed 42
```

See the `Dockerfile`/`docker-compose.yml`/`entrypoint.sh` at the repo root —
`entrypoint.sh` dispatches `hybrid`, `baselines`, `hpsearch`, and `sweeps` to
the matching `training/*.py` script. Falls back to CPU automatically if run
without `--gpus`/without an `nvidia-container-toolkit` host.

**Verify the setup**:

```bash
pytest tests/ -q
```

All C-RAN tests (and the separate O-RAN ones) should pass before running
anything at scale.

## 2. Run everything in one command

```bash
bash scripts/run_cran_experiments.sh
```

Defaults to `config/default.yaml`, 3000 episodes/seed (the convergence
target), and writes raw results to `data/results/`, figures to
`thesis/figures/`, tables to `thesis/tables/`. Every default is overridable
via environment variables — see the script's own header comment. For a fast
sanity check before committing to a full run:

```bash
EPISODES=5 bash scripts/run_cran_experiments.sh
```

The sections below are what that script does, broken out so you can re-run
any single piece (e.g. after fixing one baseline) without re-running
everything.

## 3. Step by step

### 3.1 (Optional, already done) Hyperparameter proxy sweep

Concept Note v4.0 Section 12.11's pre-training sensitivity gate. Already run
at full scale and logged (`docs/daily_log.md`, 2026-08-05 entry;
`data/results/proxy_sweep/`). Re-run only if a hyperparameter default
changes:

```bash
python -m training.hyperparam_search
```

### 3.2 Core matrix: all 10 baselines + the proposed method, 10 seeds

`training/run_extended_sweeps.py` runs this whole step in one call — all 10
baselines across all 10 seeds, the proposed Branching MP-DQN + TD3 agent
across the same 10 seeds, optionally the CSI-robustness and cross-profile
generalization evaluations per seed (reusing that seed's just-trained
checkpoint rather than training fresh), and the final statistical
aggregation (`evaluation/convergence.py`: 95% CIs, paired t-tests, Cohen's
d).

```bash
python training/run_extended_sweeps.py \
  --config config/default.yaml \
  --episodes 3000 \
  --save-dir data/results \
  --run-csi-and-generalization
```

Output: `data/results/<algo>_seed<N>/summary.json` per method/seed,
`thesis/figures/convergence_*.pdf`, `thesis/tables/convergence_summary.tex`.

If you'd rather run the pieces individually instead:

```bash
# All 10 baselines, all 10 seeds
python training/train_baselines.py --config config/default.yaml --episodes 3000

# Proposed method, one seed at a time (repeat per seed)
python training/train_hybrid.py --config config/default.yaml --seed 42 --episodes 3000
```

### 3.3 Scalability sweep (5 network sizes)

```bash
python -c "
from evaluation import analyze_scalability
analyze_scalability(config_path='config/default.yaml', save_dir='thesis/figures')
"
```

R=5,12,20,35,50 (R=50 is a stretch goal, not committed per Concept Note v4.0
§8/§15).

### 3.4 Inference-latency benchmark

```bash
python -c "
from evaluation import run_latency_benchmark
run_latency_benchmark(config_path='config/default.yaml', save_dir='thesis/figures')
"
```

Same R=5,12,20,35,50 sizes; P-DQN/MP-DQN are skipped gracefully above R=12
(their tractability cap).

### 3.5 Ablation study (RQ3/RQ4)

```bash
python -c "
from evaluation import run_ablation_study
run_ablation_study(config_path='config/default.yaml', save_dir='thesis/figures')
"
```

Hybrid vs. pure-DDPG (continuous relaxation, RQ3) and hybrid vs. P-DQN/MP-DQN
(RQ4).

### 3.6 CSI-robustness curve (if not already run via 3.2's flag)

```bash
python -c "
from evaluation import run_csi_robustness_evaluation
run_csi_robustness_evaluation(config_path='config/default.yaml', save_dir='thesis/figures')
"
```

σ ∈ {0, 0.01, 0.05, 0.1}, evaluation-only, no retraining (Concept Note v4.0
§12.5).

### 3.7 Cross-profile generalization (if not already run via 3.2's flag)

```bash
python -c "
from evaluation import run_generalization_evaluation
run_generalization_evaluation(config_path='config/default.yaml', save_dir='thesis/figures')
"
```

Weekday/urban-trained policy evaluated, without retraining, on a
weekend/suburban profile.

### 3.8 Demand-response curve

```bash
python -c "
from evaluation import run_demand_response_evaluation
run_demand_response_evaluation(config_path='config/default.yaml', save_dir='thesis/figures')
"
```

### 3.9 Power-vs-time-of-day profile

```bash
python -c "
from evaluation import run_power_time_profile_evaluation
run_power_time_profile_evaluation(config_path='config/default.yaml', save_dir='thesis/figures')
"
```

### 3.10 Reward-weight sensitivity sweep

```bash
python -c "
from evaluation import run_reward_sensitivity_sweep
run_reward_sensitivity_sweep(config_path='config/default.yaml', save_dir='thesis/figures')
"
```

## 4. After the runs

- `thesis/tables/convergence_summary.tex` is the headline statistical table
  (CIs, paired t-tests, Cohen's d for every head-to-head comparison).
- Cross-check every produced figure/table against
  `docs/equation_code_mapping.md` and the Validation Checkpoints in
  `docs/workflow.md`'s Phase 4 before citing a number in the thesis text.
- Needs-validation numeric constants (power model, traffic model) are
  flagged directly in `config/default.yaml`'s own comments — resolve those
  before stating any of them as fact in the thesis.
