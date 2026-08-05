#!/usr/bin/env bash
# Run this on the GPU server itself (over SSH) to build and smoke-test the
# training image from the Deugs/Gabriel repo's Docker setup.
set -euo pipefail

REPO_URL="https://github.com/Deugs/Gabriel.git"
REPO_DIR="Gabriel"

if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull

echo "== Checking NVIDIA driver + container toolkit =="
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

echo "== Building image =="
docker compose build

echo "== Sanity: entrypoint dispatch for all four scripts =="
docker compose run --rm train hybrid --help
docker compose run --rm train baselines --help
docker compose run --rm train hpsearch --help
docker compose run --rm train sweeps --help

echo "== Running existing test suite inside the container =="
docker compose run --rm train pytest tests/test_env.py -v

echo "== Short GPU training smoke test (2 episodes) =="
docker compose run --rm train hybrid --config config/default.yaml --seed 42 --episodes 2

echo "== Checking results landed on host =="
ls -la data/results/

echo "All checks passed."
