#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.1}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu121}"
DGL_VERSION="${DGL_VERSION:-2.4.0}"

python -m pip install --quiet --upgrade pip setuptools wheel
python -m pip uninstall -y torch torchvision torchaudio dgl dglgo > /dev/null 2>&1 || true
python -m pip install --quiet \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
python -m pip install --quiet \
  "dgl==${DGL_VERSION}+${TORCH_CUDA_TAG}" \
  -f "https://data.dgl.ai/wheels/torch-2.4/${TORCH_CUDA_TAG}/repo.html"
python -m pip install --quiet -r requirements-colab.txt

python - <<'PY'
import os
import torch
import dgl
import pandas
import sklearn
import networkx

print("Environment ready")
print(f"torch={torch.__version__}")
print(f"dgl={dgl.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
print(f"pandas={pandas.__version__}")
print(f"sklearn={sklearn.__version__}")
print(f"networkx={networkx.__version__}")
print(f"cwd={os.getcwd()}")
PY
