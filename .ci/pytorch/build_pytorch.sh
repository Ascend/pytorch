#!/usr/bin/bash
# Build PyTorch from source for NPU CI.
# Modeled after PyTorch upstream .ci/pytorch/build.sh
#
# This mirrors the upstream CPU-only build path:
#   USE_CUDA=0 USE_XNNPACK=0 WERROR=1 python -m build --wheel --no-isolation
#
# For CI PR verification, we use setup.py develop (faster, no packaging).
# For nightly/release, we use python -m build --wheel (full verification).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PYTORCH_SRC="${PYTORCH_SRC:-${REPO_ROOT}/../pytorch}"
BUILD_MODE="${BUILD_MODE:-develop}"  # "develop" for PR, "wheel" for nightly

setup_build_env

echo "=== Building PyTorch from source ==="
echo "  Source dir:  ${PYTORCH_SRC}"
echo "  Build mode:  ${BUILD_MODE}"

cd "${PYTORCH_SRC}"

# Install PyTorch build requirements
echo "=== Installing PyTorch build requirements ==="
pip_install -r requirements.txt

# Install numpy for build compatibility
pip_install "numpy>=1.23,<3.0"

case "${BUILD_MODE}" in
  develop)
    echo "=== Building PyTorch (develop mode) ==="
    conda_run python setup.py clean
    conda_run python setup.py develop
    echo "=== PyTorch develop install complete ==="
    ;;
  wheel)
    echo "=== Building PyTorch (wheel mode) ==="
    conda_run python setup.py clean
    conda_run python -m build --wheel --no-isolation
    pip_install_whl "$(echo dist/*.whl)"
    echo "=== PyTorch wheel build and install complete ==="
    ;;
  *)
    echo "ERROR: Unknown build mode: ${BUILD_MODE}"
    exit 1
    ;;
esac

# Verify PyTorch installation
echo "=== Verifying PyTorch installation ==="
conda_run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Build passed: OK')
"

echo "=== PyTorch build successful ==="
