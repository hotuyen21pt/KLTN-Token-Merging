# Backend: FastAPI + PyTorch (CPU-only) pipeline inference
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Build tools for any packages without prebuilt wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first from the PyTorch CPU index so the huge CUDA
# wheels are never pulled, then the rest of the requirements from PyPI.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

# Application code. Heavy/generated dirs (checkpoints, runs_joint, venv, node
# modules, ...) are excluded via .dockerignore and mounted at runtime instead.
COPY . .

EXPOSE 5000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "5000"]
