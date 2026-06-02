#!/usr/bin/bash
# Install triton-ascend for NPU.
# Usage: ./install_triton.sh <PYTHON_VERSION>
#   PYTHON_VERSION: e.g. 3.10, 3.11, 3.12, 3.13

set -e

TRITON_VERSION="${TRITON_VERSION:-3.2.1}"
PYTHON_VERSION="${1:?Usage: $0 <PYTHON_VERSION> (e.g. 3.10)}"

ARCH=$(uname -m)
PY_SHORT=$(echo "${PYTHON_VERSION}" | tr -d '.')

TRITON_WHL="triton_ascend-${TRITON_VERSION}-cp${PY_SHORT}-cp${PY_SHORT}-manylinux_2_27_${ARCH}.manylinux_2_28_${ARCH}.whl"
TRITON_URL="https://gitcode.com/Ascend/triton-ascend/releases/download/v${TRITON_VERSION}/${TRITON_WHL}"

echo "Installing triton-ascend ${TRITON_VERSION} for Python ${PYTHON_VERSION} (${ARCH})..."
pip3 install --no-cache-dir "${TRITON_URL}"
echo "triton-ascend installed."
