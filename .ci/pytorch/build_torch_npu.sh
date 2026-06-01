#!/usr/bin/bash
# Build torch_npu from source for NPU CI.
# Must be run AFTER PyTorch is installed (via build_pytorch.sh or pre-built wheel).
#
# Usage:
#   ./build_torch_npu.sh [--python=3.10] [--mode=develop|wheel]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TORCH_NPU_SRC="${TORCH_NPU_SRC:-${REPO_ROOT}}"
PYTHON_VERSION="${ANACONDA_PYTHON_VERSION:-3.10}"
BUILD_MODE="${BUILD_MODE:-wheel}"

setup_build_env

echo "=== Building torch_npu from source ==="
echo "  Source dir:      ${TORCH_NPU_SRC}"
echo "  Python version:  ${PYTHON_VERSION}"
echo "  Build mode:      ${BUILD_MODE}"

cd "${TORCH_NPU_SRC}"

# Install torch_npu build requirements
if [[ -f requirements.txt ]]; then
  echo "=== Installing torch_npu build requirements ==="
  pip_install -r requirements.txt
fi

# Verify PyTorch is installed before building torch_npu
echo "=== Checking PyTorch installation ==="
if ! conda_run python -c "import torch; print(torch.__version__)" 2>/dev/null; then
  echo "ERROR: PyTorch is not installed. Please run build_pytorch.sh first."
  exit 1
fi

case "${BUILD_MODE}" in
  develop)
    echo "=== Building torch_npu (develop mode) ==="
    conda_run python setup.py develop
    echo "=== torch_npu develop install complete ==="
    ;;
  wheel)
    echo "=== Building torch_npu (wheel mode) ==="
    bash ci/build.sh --python="${PYTHON_VERSION}"
    pip_install_whl "$(echo dist/torch_npu*.whl)"
    echo "=== torch_npu wheel build and install complete ==="
    ;;
  *)
    echo "ERROR: Unknown build mode: ${BUILD_MODE}"
    exit 1
    ;;
esac

echo "=== torch_npu build successful ==="
