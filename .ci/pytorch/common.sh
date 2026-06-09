#!/usr/bin/bash
# Shared utilities for torch-npu CI build scripts.
# Modeled after PyTorch upstream .ci/pytorch/common.sh and common-build.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- Conda helpers (aligned with upstream common_utils.sh) ---

conda_run() {
  conda run -n "py_${ANACONDA_PYTHON_VERSION:-3.10}" --no-capture-output "$@"
}

pip_install() {
  conda_run pip install --progress-bar off "$@"
}

pip_install_whl() {
  conda_run pip install --progress-bar off "$@"
}

# --- Build environment setup ---

setup_build_env() {
  # Default Python version if not set
  export ANACONDA_PYTHON_VERSION="${ANACONDA_PYTHON_VERSION:-3.10}"

  # NPU build: no CUDA, no XNNPACK (aligned with upstream CPU-only builds)
  export USE_CUDA=0
  export USE_XNNPACK=0

  # Use C++11 ABI (required for torch_npu compatibility)
  export _GLIBCXX_USE_CXX11_ABI="${_GLIBCXX_USE_CXX11_ABI:-1}"

  # Set MAX_JOBS to avoid OOM
  if [[ -z "${MAX_JOBS:-}" ]]; then
    export MAX_JOBS=$(($(nproc) - 2))
    # Use at least 1 job
    if [[ "$MAX_JOBS" -lt 1 ]]; then
      export MAX_JOBS=1
    fi
  fi

  echo "=== Build Environment ==="
  echo "  ANACONDA_PYTHON_VERSION: ${ANACONDA_PYTHON_VERSION}"
  echo "  USE_CUDA:                ${USE_CUDA}"
  echo "  USE_XNNPACK:             ${USE_XNNPACK}"
  echo "  _GLIBCXX_USE_CXX11_ABI:  ${_GLIBCXX_USE_CXX11_ABI}"
  echo "  MAX_JOBS:                ${MAX_JOBS}"
  echo "  Python:                  $(conda_run python --version 2>&1)"
  echo "  Python path:             $(conda_run which python)"
}

# --- Error handling ---

trap_add() {
  local trap_add_cmd="$1"
  shift || true
  local trap_add_signal="$*"

  if [[ -z "$(trap -p "${trap_add_signal}")" ]]; then
    trap "${trap_add_cmd}" "${trap_add_signal}"
  else
    trap "${trap_add_cmd};$(trap -p "${trap_add_signal}" | sed "s/^trap -- '\(.*\)' \([A-Z0-9 ]*\)$/\1/")" "${trap_add_signal}"
  fi
}
