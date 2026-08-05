# C-RAN DRL Thesis — experiment runtime image.
#
# Lean, CPU-by-default image containing only what agents/, baselines/,
# cran_env/, evaluation/, training/ actually import (requirements-runtime.txt),
# not the full local dev environment in requirements.txt (notebooks, lint,
# experiment tracking, stable-baselines3, ray — none of which the codebase
# uses). Satisfies docs/rules.md Rule 4 (Reproducibility): a pinned,
# containerized environment plus experiments/*.yaml + run_experiment.py.
#
# Build (CPU):
#   docker build -t cran-drl:latest .
#
# Build for a CUDA-enabled cloud instance instead, override the torch
# install with the wheel matching your driver, e.g.:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t cran-drl:cuda .
#
# Run an experiment (see docs/deployment.md for the full cloud workflow):
#   docker run --rm -v "$(pwd)/data:/app/data" cran-drl:latest \
#       python run_experiment.py --config experiments/hybrid_medium.yaml --seed 42

FROM python:3.11-slim

ARG TORCH_INDEX_URL=""

WORKDIR /app

# Keep Python output unbuffered so logs stream immediately in `docker logs`
# / cloud log viewers, rather than only appearing when the process exits.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements-runtime.txt .

RUN pip install --upgrade pip \
    && if [ -n "$TORCH_INDEX_URL" ]; then \
         pip install torch==2.13.0 --index-url "$TORCH_INDEX_URL"; \
       fi \
    && pip install -r requirements-runtime.txt

# Only what's needed to run/reproduce experiments and tests — thesis/,
# references/, manuscript/, .git/ etc. are excluded via .dockerignore.
COPY agents/ agents/
COPY baselines/ baselines/
COPY cran_env/ cran_env/
COPY evaluation/ evaluation/
COPY training/ training/
COPY tests/ tests/
COPY config/ config/
COPY experiments/ experiments/
COPY run_experiment.py .

RUN mkdir -p data/results data/traces thesis/figures thesis/tables

# Smoke-test the image at build time: fail the build if the environment
# can't even import the core packages, rather than failing on first use.
RUN python -c "import agents, baselines, cran_env, evaluation, training"

ENTRYPOINT ["python"]
CMD ["run_experiment.py", "--help"]
