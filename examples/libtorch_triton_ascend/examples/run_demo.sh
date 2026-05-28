#!/bin/bash
set -e

export TORCH_DEVICE_BACKEND_AUTOLOAD=0

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=${SCRIPT_DIR}/build
DEMO_BIN=${BUILD_DIR}/demo

# ── Environment ──

PYTHON3="${PYTHON3:-python3}"
PYTHONHOME=$($PYTHON3 -c 'import sys; print(sys.prefix)')
TORCH_PATH=$($PYTHON3 -c 'import torch; print(torch.__path__[0])')
TORCH_NPU_PATH=$($PYTHON3 -c 'import torch_npu; print(torch_npu.__path__[0])')
CANN_ROOT="${ASCEND_HOME_PATH:-${ASCEND_HOME:-}}"

if [ -z "$CANN_ROOT" ]; then
    echo "[ERROR] Set ASCEND_HOME_PATH or ASCEND_HOME first"
    exit 1
fi

TORCH_CMAKE_PREFIX=$($PYTHON3 -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null)
PYBIND11_CMAKE_DIR=$($PYTHON3 -m pybind11 --cmakedir 2>/dev/null)

# ── Build ──

echo "=== Build demo ==="
echo "PYTHON3: $(which $PYTHON3)"
echo "PYTHONHOME: ${PYTHONHOME}"
echo "TORCH_CMAKE_PREFIX: ${TORCH_CMAKE_PREFIX}"
echo "TORCH_PATH: ${TORCH_PATH}"
echo "TORCH_NPU_PATH: ${TORCH_NPU_PATH}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
    -DPYTORCH_PYTHON_PACKAGES="${TORCH_PATH}" \
    -DPYTORCH_NPU_PACKAGES="${TORCH_NPU_PATH}" \
    -DCANN_PATH="${CANN_ROOT}" \
    -DPYTHON_EXECUTABLE="$(which $PYTHON3)" \
    -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX};${PYBIND11_CMAKE_DIR}"
cmake --build "${BUILD_DIR}" -j$(nproc)

# ── Run ──

echo ""
echo "=== Run demo ==="
echo ""

TORCH_LIB="${TORCH_PATH}/lib"
TORCH_NPU_LIB="${TORCH_NPU_PATH}/lib"
CANN_LIB="${CANN_ROOT}/lib64"
RT_BUILD="${SCRIPT_DIR}/../build"

PROJECT_ROOT="${SCRIPT_DIR}/../.."

PYTHONHOME="${PYTHONHOME}" \
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="${RT_BUILD}:${TORCH_LIB}:${TORCH_NPU_LIB}:${CANN_LIB}:${LD_LIBRARY_PATH}" \
    "$DEMO_BIN"
