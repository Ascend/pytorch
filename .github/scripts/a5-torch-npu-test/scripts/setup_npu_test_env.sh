#!/usr/bin/env bash
set -eo pipefail

WHEEL_ARTIFACT_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wheel-artifact-dir)  WHEEL_ARTIFACT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --wheel-artifact-dir DIR"
            exit 0 ;;
        *) echo "[ERROR] Unknown: $1"; exit 1 ;;
    esac
done

PYTHON=python3.10
WORKSPACE="${ATOMGIT_WORKSPACE:-/__w/ComputingActionTest/pytorch}"

echo "============================================"
echo "NPU Test Environment Setup"
echo "============================================"

WHEEL_ARTIFACT_DIR="$(cd "${WHEEL_ARTIFACT_DIR}" 2>/dev/null && pwd || echo "${WORKSPACE}/${WHEEL_ARTIFACT_DIR}")"

# ── 1. torch_npu wheel ──
echo ">>> Step 1: Install torch_npu wheel"
if [ ! -d "${WHEEL_ARTIFACT_DIR}" ]; then
    echo "[ERROR] Directory not found: ${WHEEL_ARTIFACT_DIR}"
    exit 1
fi

if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    source /usr/local/Ascend/cann/set_env.sh || echo "WARNING: cann set_env.sh failed"
else
    echo "WARNING: /usr/local/Ascend/cann/set_env.sh not found"
fi

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh || echo "WARNING: atb set_env.sh failed"
else
    echo "WARNING: /usr/local/Ascend/nnal/atb/set_env.sh not found"
fi

if ! ls "${WHEEL_ARTIFACT_DIR}"/*.whl 1> /dev/null 2>&1; then
    echo "[ERROR] No .whl files found in ${WHEEL_ARTIFACT_DIR}"
    exit 1
fi

${PYTHON} -m pip install --no-deps "${WHEEL_ARTIFACT_DIR}"/*.whl || {
    echo "[ERROR] Failed to install torch_npu wheel"
    exit 1
}

# ── 2. Verify NPU ──
echo ">>> Step 2: Verify NPU device and availability"
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    source /usr/local/Ascend/cann/set_env.sh || echo "WARNING: cann set_env.sh failed"
else
    echo "WARNING: /usr/local/Ascend/cann/set_env.sh not found"
fi

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh || echo "WARNING: atb set_env.sh failed"
else
    echo "WARNING: /usr/local/Ascend/nnal/atb/set_env.sh not found"
fi
cd /tmp && ${PYTHON} -c "
import torch
import torch_npu
print(f'torch_npu: {torch_npu.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
print(f'NPU count: {torch.npu.device_count()}')
" || {
    echo "[ERROR] NPU verification failed"
    exit 1
}

echo "============================================"
echo "NPU Test Environment Setup — DONE"
echo "============================================"
