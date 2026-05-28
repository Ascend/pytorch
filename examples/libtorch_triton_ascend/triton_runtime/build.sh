#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=${SCRIPT_DIR}/build

# ── Environment ──
if [ -z "$ASCEND_HOME_PATH" ] && [ -z "$ASCEND_HOME" ]; then
    echo "[ERROR] Set ASCEND_HOME_PATH or ASCEND_HOME first"
    exit 1
fi
CANN_ROOT="${ASCEND_HOME_PATH:-$ASCEND_HOME}"

PYTHON3="${PYTHON3:-python3}"
if ! command -v $PYTHON3 &>/dev/null; then
    echo "[ERROR] python3 not found"
    exit 1
fi

TORCH_PATH=$($PYTHON3 -c 'import torch; print(torch.__path__[0])')
TORCH_NPU_PATH=$($PYTHON3 -c 'import torch_npu; print(torch_npu.__path__[0])')

TORCH_CMAKE_PREFIX=$($PYTHON3 -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null)
if [ -z "$TORCH_CMAKE_PREFIX" ]; then
    echo "[ERROR] torch not installed or not importable"
    exit 1
fi

PYBIND11_CMAKE_DIR=$($PYTHON3 -m pybind11 --cmakedir 2>/dev/null)
if [ -z "$PYBIND11_CMAKE_DIR" ]; then
    echo "[ERROR] pybind11 not installed or not importable"
    exit 1
fi

echo "=== Build triton_runtime ==="
echo "  CANN_ROOT:         $CANN_ROOT"
echo "  Python3:           $PYTHON3"
echo "  Torch cmake prefix: $TORCH_CMAKE_PREFIX"
echo "  pybind11 cmake dir: $PYBIND11_CMAKE_DIR"

# ── CMake Configure ──
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPYTHON_EXECUTABLE="$(which $PYTHON3)" \
    -DPYTORCH_PYTHON_PACKAGES="${TORCH_PATH}" \
    -DPYTORCH_NPU_PACKAGES="${TORCH_NPU_PATH}" \
    -DCANN_PATH="${CANN_ROOT}" \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX;$PYBIND11_CMAKE_DIR"

# ── Build ──
make -j$(nproc)

echo ""
echo "=== Build succeeded ==="
echo "  SO: $BUILD_DIR/libtriton_runtime.so"
