#!/usr/bin/bash
# Docker image build script for torch-npu CI test images (aarch64 only).
#
# Usage:
#   ./docker_build.sh <IMAGE_TAG>
#
# Examples:
#   ./docker_build.sh torch-npu-test-aarch64-cann-a2-py3.10-torch-nightly
#   ./docker_build.sh torch-npu-test-aarch64-cann-a3-py3.10-torch-nightly

set -euo pipefail

TAG="${1:?Usage: $0 <IMAGE_TAG>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT="${SCRIPT_DIR}"

case "$TAG" in
  torch-npu-test-aarch64-cann-a2-py3.10-torch-nightly)
    CANN_CHIP=A2
    PYTHON_VERSION=3.10
    ;;
  torch-npu-test-aarch64-cann-a3-py3.10-torch-nightly)
    CANN_CHIP=A3
    PYTHON_VERSION=3.10
    ;;
  *)
    echo "ERROR: Unknown image tag: $TAG"
    echo ""
    echo "Supported tags:"
    echo "  torch-npu-test-aarch64-cann-a2-py3.10-torch-nightly"
    echo "  torch-npu-test-aarch64-cann-a3-py3.10-torch-nightly"
    exit 1
    ;;
esac

DOCKERFILE="${SCRIPT_DIR}/test/Dockerfile.aarch64"

echo "=== Image Configuration ==="
echo "  Architecture:   aarch64"
echo "  CANN Chip:      ${CANN_CHIP}"
echo "  Python:         ${PYTHON_VERSION}"
echo "  PyTorch:        nightly"
echo "  Full Tag:       ${TAG}"

echo "=== Building test image: ${TAG} ==="
docker build \
  --build-arg CANN_CHIP="${CANN_CHIP}" \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
  --tag "${TAG}" \
  --file "${DOCKERFILE}" \
  "${BUILD_CONTEXT}"

echo "=== Image built successfully: ${TAG} ==="
