# O-RAN / BMPP-DQN Track — Setup and Experiment Runbook

> Governs: `manuscript/ORAN_BMPP_DQN_Concept_Note_v1.md` (the actual MPhil
> thesis submission). For the C-RAN/publications track's equivalent guide,
> see `docs/cran_experiment_guide.md`. The two tracks share no code —
> running one never requires or affects the other.

This is a step-by-step runbook for setting up the environment and running
every experiment this track's scope (Concept Note §5.3, §6.1) requires,
either by hand (this document) or in one shot via
`scripts/run_oran_experiments.sh`, which automates every step below.

## 1. Setup

**Option A: local virtualenv** — identical to the C-RAN track's, since both
tracks share one `requirements.txt`/one virtualenv:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pre-commit install
export PYTHONPATH="$(pwd):$PYTHONPATH"   # needed for every command below
```

**Option B: Docker (CPU by default; GPU opt-in)**

```bash
docker compose build
docker compose run --rm train-oran oran_hybrid --config config/oran_default.yaml --seed 42

# Full 3-seed/4-method experiment matrix in the container:
docker compose run --rm full-oran
```

`entrypoint.sh` also dispatches `oran_hybrid`/`oran_bmpp_dqn`,
`oran_baselines`, `full_oran` (the entire matrix above), and
`oran_power_sensitivity` to the matching script. `docker-compose.yml`
builds a small CPU-only image (`Dockerfile.cpu`) and requests no GPU, so
this works on any Docker host. To use an NVIDIA GPU instead (requires the
NVIDIA Container Toolkit), layer the GPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  run --rm full-oran
```

See README.md's "Running the Experiments" section for the full command
reference across both tracks.

**Verify the setup**:

```bash
pytest tests/test_oran_env.py tests/test_oran_agents.py tests/test_oran_training.py tests/test_oran_evaluation.py -v
```

## 2. Run everything in one command

```bash
bash scripts/run_oran_experiments.sh
```

Defaults to `config/oran_default.yaml`, 3 seeds (`[42, 123, 456]`, per
Concept Note §5.3), 500 episodes/seed (`config/oran_default.yaml`'s
`algorithm.max_episodes` cap), and writes raw results to
`data/results_oran/`, figures to `thesis/figures_oran/`, tables to
`thesis/tables_oran/`. Every default is overridable via environment
variables — see the script's own header comment. For a fast sanity check
before committing to a full run:

```bash
EPISODES=5 bash scripts/run_oran_experiments.sh
```

## 3. Step by step

### 3.1 All 3 baselines (DQN, DDPG, MP-DQN), all 3 seeds

One call runs the full baseline matrix:

```bash
python -m oran_training.train_oran_baselines \
  --config config/oran_default.yaml \
  --episodes 500 \
  --save-dir data/results_oran
```

Output: `data/results_oran/oran_benchmark_<algo>/summary.json` per
algorithm (each containing all 3 seeds' results).

### 3.2 Proposed BMPP-DQN agent, one seed at a time

Unlike the baseline runner, this trains one seed per invocation — repeat for
each of the 3 seeds:

```bash
python -m oran_training.train_bmpp_dqn \
  --config config/oran_default.yaml \
  --seed 42 \
  --episodes 500 \
  --save-dir data/results_oran

python -m oran_training.train_bmpp_dqn --config config/oran_default.yaml --seed 123 --episodes 500 --save-dir data/results_oran
python -m oran_training.train_bmpp_dqn --config config/oran_default.yaml --seed 456 --episodes 500 --save-dir data/results_oran
```

Output: `data/results_oran/bmpp_dqn_seed<N>/summary.json` and a saved model
checkpoint, per seed.

### 3.3 Statistical aggregation + inference-latency benchmark

```bash
python -c "
from oran_evaluation import analyze_convergence, run_latency_benchmark
analyze_convergence(results_dir='data/results_oran', save_dir='thesis/figures_oran', table_save_dir='thesis/tables_oran')
run_latency_benchmark(config_path='config/oran_default.yaml', save_dir='thesis/figures_oran')
"
```

`analyze_convergence` produces the 95% CI / paired t-test / Cohen's d table
(`thesis/tables_oran/convergence_summary_oran.tex`) and convergence-curve
figures for BMPP-DQN vs. all 3 baselines. `run_latency_benchmark` measures a
single scenario (this track's focused single-gNB scope, Concept Note
§6.1/7.1) — not a scalability sweep like the C-RAN track's.

### 3.4 Power-model constant sensitivity analysis (optional, recommended before citing any power-model-derived number)

`oran_env/power_model.py`'s RU/DU/CU/fronthaul constants are literature-style
placeholders (Concept Note §10.5) — nine literature-verification passes
found no source that validates them directly (see that module's own
docstring). `oran_evaluation/power_sensitivity.py` instead checks whether
the 4-method comparison's ranking is an artifact of any single unvalidated
constant, by perturbing each constant group within the ranges the
literature review brackets and retraining all 4 methods from scratch at
each point:

```bash
python -m oran_evaluation.power_sensitivity \
  --train-episodes 500 \
  --seeds 42 123 456
```

For a fast sanity check before committing to the full sweep (7 scenarios ×
4 methods × N seeds, each a from-scratch training run):

```bash
python -m oran_evaluation.power_sensitivity --quick
```

Output: `data/results_oran/power_sensitivity/summary.json` (per-scenario
metrics and a `robustness` verdict against the unperturbed baseline's
ranking) and one bar chart per scenario in `thesis/figures_oran/`. This
does not validate the power model's absolute values — nothing can, per the
disclosure above — only whether the comparative finding reported in
Chapter 4 depends on any single placeholder constant.

## 4. After the runs

- `thesis/tables_oran/convergence_summary_oran.tex` is the headline
  statistical table.
- Before citing any number in the thesis text, resolve the needs-validation
  placeholders listed in `docs/oran_thesis_guide.md`'s "Needs-Validation
  Flags to Resolve Before Submission" section (power-model constants,
  traffic breakpoints, split→centralization mapping, default scenario
  scale) — they were chosen for internal consistency (e.g. monotonicity),
  not verified physical constants.
- This track's evaluation scope is deliberately narrow (Concept Note §6.1) —
  there is no ablation/scalability/CSI-robustness/generalization sweep
  equivalent to the C-RAN track's; don't add one without updating the
  concept note first.
