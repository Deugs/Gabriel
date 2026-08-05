# CUDA-enabled image for training the C-RAN DRL agents (see docs/dev_guide.md).
# Falls back to CPU automatically if run without --gpus (agents check torch.cuda.is_available()).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

# torch/torchvision/torchaudio pinned to the CUDA 12.1 wheel index, matching docs/dev_guide.md.
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["hybrid", "--config", "config/default.yaml"]
