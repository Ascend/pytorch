#!/usr/bin/bash
# Main CI build entry point for torch-npu.
# Modeled after PyTorch upstream .ci/pytorch/build.sh
#
# Orchestrates:
#   1. PyTorch build (from source or pre-built wheel)
#   2. torch_npu build
#   3. Integration verification
#
# Environment variables:
#   PYTORCH_SRC          - Path to PyTorch source (default: ../pytorch)
#   TORCH_NPU_SRC        - Path to torch_npu source (default: repo root)
#   BUILD_MODE           - "develop" (PR) or "wheel" (nightly)
#   PYTORCH_BUILD_MODE   - "source" (build from source) or "wheel" (install pre-built)
#   PYTORCH_WHEEL_URL    - URL/path to pre-built PyTorch wheel (when PYTORCH_BUILD_MODE=wheel)
#   PYTORCH_VERSION      - PyTorch version for pre-built wheel (default: 2.7.1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PYTORCH_SRC="${PYTORCH_SRC:-${REPO_ROOT}/../pytorch}"
TORCH_NPU_SRC="${TORCH_NPU_SRC:-${REPO_ROOT}}"
BUILD_MODE="${BUILD_MODE:-develop}"
PYTORCH_BUILD_MODE="${PYTORCH_BUILD_MODE:-source}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.7.1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

echo "============================================"
echo "  torch-npu CI Build"
echo "============================================"
echo "  Build mode:       ${BUILD_MODE}"
echo "  PyTorch mode:     ${PYTORCH_BUILD_MODE}"
echo "  PyTorch src:      ${PYTORCH_SRC}"
echo "  torch_npu src:    ${TORCH_NPU_SRC}"
echo "============================================"

# Step 1: Install or build PyTorch
case "${PYTORCH_BUILD_MODE}" in
  source)
    echo ">>> Step 1/3: Building PyTorch from source..."
    PYTORCH_SRC="${PYTORCH_SRC}" BUILD_MODE="${BUILD_MODE}" bash "${SCRIPT_DIR}/build_pytorch.sh"
    ;;
  wheel)
    echo ">>> Step 1/3: Installing pre-built PyTorch ${PYTORCH_VERSION}..."
    pip_install --index-url "${PYTORCH_INDEX_URL}" "torch==${PYTORCH_VERSION}"
    conda_run python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
    ;;
  *)
    echo "ERROR: Unknown PYTORCH_BUILD_MODE: ${PYTORCH_BUILD_MODE}"
    exit 1
    ;;
esac

# Step 2: Build torch_npu
echo ">>> Step 2/4: Building torch_npu..."
TORCH_NPU_SRC="${TORCH_NPU_SRC}" BUILD_MODE="${BUILD_MODE}" bash "${SCRIPT_DIR}/build_torch_npu.sh"

# Step 3: Install post-build dependencies (packages that require PyTorch)
echo ">>> Step 3/4: Installing post-build dependencies..."
REQUIREMENTS_POST="${TORCH_NPU_SRC}/.ci/docker/requirements-post.txt"
if [ -f "${REQUIREMENTS_POST}" ]; then
  pip_install -r "${REQUIREMENTS_POST}"
  echo "Post-build dependencies installed."
else
  echo "WARNING: requirements-post.txt not found at ${REQUIREMENTS_POST}, skipping."
fi

# Step 4: Integration verification
echo ">>> Step 4/4: Verifying integration..."
bash "${SCRIPT_DIR}/integration_verify.sh"

echo "============================================"
echo "  Build Summary: SUCCESS"
echo "============================================"
