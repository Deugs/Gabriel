# Deployment: Running Experiments in Docker / on a Cloud Host

This project satisfies `docs/rules.md` Rule 4 (Reproducibility) via three
pieces: a pinned, containerized environment (`Dockerfile`,
`requirements-runtime.txt`), a single-command experiment runner
(`run_experiment.py`), and named experiment definitions (`experiments/*.yaml`).
This doc covers building and running that image locally and on a cloud host.

There is no dependency on any specific "Docker Cloud" product (Docker Inc.'s
own hosted CI service by that name was discontinued years ago) — everything
below is a plain Docker image that runs unmodified on any host with Docker
installed: a laptop, a bare cloud VM (AWS EC2, GCP Compute Engine, Azure VM),
a managed container service (ECS/Fargate, Cloud Run, Azure Container
Instances), or a Kubernetes Job.

## 1. What's in the image

The `Dockerfile` installs only `requirements-runtime.txt` — the subset of
packages actually imported by `agents/`, `baselines/`, `cran_env/`,
`evaluation/`, `training/` (verified by grep, not by inspection). It does
**not** include the full `requirements.txt` dev environment (notebooks,
lint, `stable-baselines3`, `ray[rllib]`) — none of that is used by any
training or evaluation code path, and including it would roughly double the
image size and build time for no benefit at experiment-run time.

## 2. Build

CPU (default — works everywhere, including free-tier cloud VMs):

```bash
docker build -t cran-drl:latest .
```

GPU (CUDA) — override the torch install with the wheel matching your
driver's CUDA version. Check `docker run --rm cran-drl:latest python -c
"import torch; print(torch.__version__)"` afterward to confirm the CUDA
build actually loaded:

```bash
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t cran-drl:cuda .
```

The image runs `python -c "import agents, baselines, cran_env, evaluation, training"`
as its last build step — the build itself fails fast if the environment is
broken, rather than failing on first `docker run`.

## 3. Run an experiment

Every experiment is one command, per Rule 4:

```bash
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/thesis:/app/thesis" \
    cran-drl:latest \
    python run_experiment.py --config experiments/hybrid_medium.yaml --seed 42
```

- `-v "$(pwd)/data:/app/data"` — mounts the host's `data/` directory into the
  container, so training results/checkpoints survive after the container
  exits (without it, everything written to `data/results/` is lost when the
  container is removed).
- `-v "$(pwd)/thesis:/app/thesis"` — same, for the evaluation modules that
  save figures/tables under `thesis/figures/`, `thesis/tables/`.
- `--seed 42` overrides the experiment file's own seed list for this one
  run — see Section 5 below for why that matters on a cloud host.

Or via Compose, which wires the same mounts up once:

```bash
docker compose run --rm experiment
# or, to run a different experiment file:
docker compose run --rm experiment python run_experiment.py --config experiments/csi_robustness.yaml
```

## 4. Available experiments

| File | Type | What it runs |
|---|---|---|
| `experiments/hybrid_small.yaml` | `train_hybrid` | Fast smoke test (R=5, 20 episodes) — run this first on any new image/host |
| `experiments/hybrid_medium.yaml` | `train_hybrid` | Proposed agent, R=12, 1000 episodes, 10 seeds |
| `experiments/hybrid_large.yaml` | `train_hybrid` | Proposed agent, R=50 stretch goal |
| `experiments/baseline_matrix_small.yaml` | `baseline_matrix` | All 9 baselines, R=5 |
| `experiments/baseline_matrix_medium.yaml` | `baseline_matrix` | All 9 baselines, R=12 |
| `experiments/csi_robustness.yaml` | `csi_robustness` | CSI-noise degradation curve (§12.5) |
| `experiments/generalization.yaml` | `generalization` | Cross-profile zero-shot evaluation (§12.3, A5) |
| `experiments/latency_benchmark.yaml` | `latency_benchmark` | Forward-pass latency at R=5,12,20,35,50 (§12.3, A3) |
| `experiments/proxy_sweep.yaml` | `proxy_sweep` | §12.11 hyperparameter sensitivity sweep |

Add a new one by copying an existing file with a similar `type` and
changing `config`/`episodes`/`seeds` — `run_experiment.py` doesn't need
changes unless you're adding a genuinely new experiment *type*.

## 5. Fanning the full 10-seed matrix out on a cloud host

`experiments/hybrid_medium.yaml` lists all 10 seeds, but `run_experiment.py`
runs *one* seed per invocation (`--seed` overrides the file's list). This is
deliberate: it's the natural unit of parallelism for a cloud job array —
one container per seed, all running concurrently instead of sequentially.
On any platform that can run "the same image, N times, with a different
argument each time" (a Kubernetes Job with a matrix, an ECS/Fargate task
array, a GCP Cloud Run Job with `--tasks`, or just N `docker run` processes
on one big VM), fan out like this:

```bash
for seed in 42 123 456 789 1011 1337 2024 2718 3141 4242; do
  docker run -d --rm -v "$(pwd)/data:/app/data" cran-drl:latest \
      python run_experiment.py --config experiments/hybrid_medium.yaml --seed "$seed"
done
```

Each run writes to its own `data/results/branching_mp_dqn_seed<N>/` (and a
`run_manifest.json` recording the exact config hash used), so results from
concurrent runs never collide.

## 6. Traceability

Every run writes `run_manifest.json` alongside its results, containing:

- the experiment YAML's path and SHA-256 hash
- the referenced `config/*.yaml`'s path and SHA-256 hash
- any `--seed` override used
- the full parsed experiment spec

This is what satisfies Rule 4's "results saved with full config hash" — any
result can be traced back to the exact experiment and config file content
that produced it, even if those files are later edited.
