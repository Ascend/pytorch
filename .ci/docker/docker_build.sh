#!/usr/bin/bash
# Build torch-npu CI Docker images.
#
# Usage:
#   ./docker_build.sh <TAG>
#
# Builder (2.13 only): torch-npu-builder-<ARCH>-torch<PYTORCH_VERSION>
# Test:                torch-npu-test-<ARCH>-cann<CHIP>-py<PYTHON_VERSION>-torch<PYTORCH_VERSION>
#
# Examples:
#   ./docker_build.sh torch-npu-builder-x86_64-torch2.13.0
#   ./docker_build.sh torch-npu-test-aarch64-cann-a2-py3.10-torch2.13.0
#   ./docker_build.sh torch-npu-test-x86_64-cann-a1-py3.10-torch-master
#   ./docker_build.sh torch-npu-test-aarch64-cann-a3-py3.10-torch-master

set -euo pipefail

BASE_TAG="${1:?Usage: $0 <TAG>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT="${SCRIPT_DIR}"

case "$BASE_TAG" in
  # --- v2.13.0 builder ---
  torch-npu-builder-x86_64-torch2.13.0)
    IMAGE_TYPE=builder
    ARCH=x86_64
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  torch-npu-builder-aarch64-torch2.13.0)
    IMAGE_TYPE=builder
    ARCH=aarch64
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  # --- v2.13.0 test ---
  torch-npu-test-x86_64-cann-a1-py3.10-torch2.13.0)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A1
    PYTHON_VERSION=3.10
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  torch-npu-test-x86_64-cann-a2-py3.10-torch2.13.0)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A2
    PYTHON_VERSION=3.10
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  torch-npu-test-x86_64-cann-a3-py3.10-torch2.13.0)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A3
    PYTHON_VERSION=3.10
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  torch-npu-test-aarch64-cann-a1-py3.10-torch2.13.0)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A1
    PYTHON_VERSION=3.10
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  torch-npu-test-aarch64-cann-a2-py3.10-torch2.13.0)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A2
    PYTHON_VERSION=3.10
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  torch-npu-test-aarch64-cann-a3-py3.10-torch2.13.0)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A3
    PYTHON_VERSION=3.10
    PYTORCH_VERSION=2.13.0
    VERSION_DIR=2.13
    ;;
  # --- master (nightly) builder ---
  torch-npu-builder-x86_64-torch-master)
    IMAGE_TYPE=builder
    ARCH=x86_64
    PYTORCH_VERSION=2.14.0.dev20260708
    VERSION_DIR=master
    ;;
  torch-npu-builder-aarch64-torch-master)
    IMAGE_TYPE=builder
    ARCH=aarch64
    PYTORCH_VERSION=2.14.0.dev20260708
    VERSION_DIR=master
    ;;
  # --- master (nightly) test ---
  torch-npu-test-x86_64-cann-a1-py3.10-torch-master)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A1
    PYTHON_VERSION=3.10
    VERSION_DIR=master
    ;;
  torch-npu-test-x86_64-cann-a2-py3.10-torch-master)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A2
    PYTHON_VERSION=3.10
    VERSION_DIR=master
    ;;
  torch-npu-test-x86_64-cann-a3-py3.10-torch-master)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A3
    PYTHON_VERSION=3.10
    VERSION_DIR=master
    ;;
  torch-npu-test-aarch64-cann-a1-py3.10-torch-master)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A1
    PYTHON_VERSION=3.10
    VERSION_DIR=master
    ;;
  torch-npu-test-aarch64-cann-a2-py3.10-torch-master)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A2
    PYTHON_VERSION=3.10
    VERSION_DIR=master
    ;;
  torch-npu-test-aarch64-cann-a3-py3.10-torch-master)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A3
    PYTHON_VERSION=3.10
    VERSION_DIR=master
    ;;
  *)
    echo "ERROR: Unknown image tag: ${BASE_TAG}"
    echo ""
    echo "Supported tags:"
    echo "  Builder: torch-npu-builder-<x86_64|aarch64>-torch2.13.0"
    echo "  Test:    torch-npu-test-<x86_64|aarch64>-cann-<a1|a2|a3>-py3.10-torch<2.13.0|master>"
    exit 1
    ;;
esac

TIMESTAMP="${TIMESTAMP:-$(TZ=Asia/Shanghai date +%Y%m%d%H%M)}"
TAG="${BASE_TAG}-${TIMESTAMP}"

DOCKERFILE="${SCRIPT_DIR}/${VERSION_DIR}/${IMAGE_TYPE}/Dockerfile.${ARCH}"

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "ERROR: Dockerfile not found: ${DOCKERFILE}"
  exit 1
fi

BUILD_ARGS=()
if [[ -n "${CANN_CHIP:-}" ]]; then
  BUILD_ARGS+=(--build-arg CANN_CHIP="${CANN_CHIP}")
fi
if [[ -n "${PYTHON_VERSION:-}" ]]; then
  BUILD_ARGS+=(--build-arg PYTHON_VERSION="${PYTHON_VERSION}")
fi
if [[ -n "${PYTORCH_VERSION:-}" ]]; then
  BUILD_ARGS+=(--build-arg PYTORCH_VERSION="${PYTORCH_VERSION}")
fi

echo "=== Image Configuration ==="
echo "  Image Type:   ${IMAGE_TYPE}"
echo "  Architecture: ${ARCH}"
echo "  Version Dir:  ${VERSION_DIR}"
echo "  CANN Chip:    ${CANN_CHIP:-}"
echo "  Python:       ${PYTHON_VERSION:-}"
echo "  PyTorch:      ${PYTORCH_VERSION:-nightly (built in CI)}"
echo "  Full Tag:     ${TAG}"
echo "  Dockerfile:   ${DOCKERFILE}"

echo "=== Building ${IMAGE_TYPE} image: ${TAG} ==="
docker build \
  "${BUILD_ARGS[@]}" \
  --tag "${TAG}" \
  --file "${DOCKERFILE}" \
  "${BUILD_CONTEXT}"

echo "=== Image built successfully: ${TAG} ==="
