#!/bin/bash
# Install GR00T N1.7 natively on the RAPID Jetson Thor (host, not the ROS container).
#
# The Thor runs JetPack 7.1 (L4T R38.4, driver 580 / CUDA 13.0). NVIDIA's Thor installer
# targets JetPack 7.2 (R39) and refuses R38. The cu132 PyTorch stack it installs was
# checked to work on this driver (2026-11-23: torch 2.13.0+cu132, bf16 matmul and
# flash-attn on sm_110), so the installer's platform check is skipped with
# DOCKER_CONTAINER=1. Everything else runs as NVIDIA wrote it.
#
# Afterwards, in each shell:
#   cd ~/Isaac-GR00T && source .venv/bin/activate && source scripts/activate_thor.sh
# and use plain `python`, not `uv run` (uv run re-syncs against the dGPU pyproject).
set -euo pipefail

GROOT_DIR="${GROOT_DIR:-$HOME/Isaac-GR00T}"
GROOT_COMMIT="${GROOT_COMMIT:-708617f}"   # the commit installed and tested on 2026-11-23

if [ ! -d "$GROOT_DIR" ]; then
    # Skip LFS smudging: the demo data is large and unused here.
    GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T "$GROOT_DIR"
fi
cd "$GROOT_DIR"
git checkout -q "$GROOT_COMMIT"

# The prebuilt flash-attn wheel is in LFS. The object is registered under jetson/, and
# thor/wheels holds the same file.
git lfs pull --include="scripts/deployment/jetson/wheels/*,scripts/deployment/thor/wheels/*"

DOCKER_CONTAINER=1 bash scripts/deployment/thor/install_deps.sh

source .venv/bin/activate
source scripts/activate_thor.sh >/dev/null
python - <<'EOF'
import torch
from flash_attn import flash_attn_func
import gr00t, torchcodec, tensorrt
q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
flash_attn_func(q, q, q)
print("OK: torch", torch.__version__, "| flash-attn on", torch.cuda.get_device_name(0),
      "| tensorrt", tensorrt.__version__)
EOF

echo "Gated backbone: HF_TOKEN must belong to an account granted access to"
echo "https://huggingface.co/nvidia/Cosmos-Reason2-2B, or model loading fails with 403."
