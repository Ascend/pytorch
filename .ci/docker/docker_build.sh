#!/usr/bin/bash
# Docker image build script for torch-npu CI images.
#
# Usage:
#   ./docker_build.sh <IMAGE_TAG>
#
# Examples:
#   ./docker_build.sh torch-npu-builder-x86_64-torch-nightly
#   ./docker_build.sh torch-npu-builder-aarch64-torch-nightly
#   ./docker_build.sh torch-npu-test-x86_64-cann-a1-py3.10-torch-nightly
#   ./docker_build.sh torch-npu-test-aarch64-cann-a2-py3.10-torch-nightly

set -euo pipefail

TAG="${1:?Usage: $0 <IMAGE_TAG>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Build context is .ci/docker/ — Dockerfile COPY paths (common/, requirements-*.txt)
# are relative to this directory.
BUILD_CONTEXT="${SCRIPT_DIR}"

case "$TAG" in
  # --- Builder images ---
  torch-npu-builder-x86_64-torch-nightly)
    IMAGE_TYPE=builder
    ARCH=x86_64
    ;;
  torch-npu-builder-aarch64-torch-nightly)
    IMAGE_TYPE=builder
    ARCH=aarch64
    ;;

  # --- Test images (x86_64) ---
  torch-npu-test-x86_64-cann-a1-py3.10-torch-nightly)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A1
    PYTHON_VERSION=3.10
    ;;
  torch-npu-test-x86_64-cann-a2-py3.10-torch-nightly)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A2
    PYTHON_VERSION=3.10
    ;;
  torch-npu-test-x86_64-cann-a3-py3.10-torch-nightly)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A3
    PYTHON_VERSION=3.10
    ;;

  # --- Test images (aarch64) ---
  torch-npu-test-aarch64-cann-a1-py3.10-torch-nightly)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A1
    PYTHON_VERSION=3.10
    ;;
  torch-npu-test-aarch64-cann-a2-py3.10-torch-nightly)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A2
    PYTHON_VERSION=3.10
    ;;
  torch-npu-test-aarch64-cann-a3-py3.10-torch-nightly)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A3
    PYTHON_VERSION=3.10
    ;;

  *)
    echo "ERROR: Unknown image tag: $TAG"
    echo ""
    echo "Supported tags:"
    echo "  Builder:"
    echo "    torch-npu-builder-x86_64-torch-nightly"
    echo "    torch-npu-builder-aarch64-torch-nightly"
    echo "  Test:"
    echo "    torch-npu-test-x86_64-cann-a1-py3.10-torch-nightly"
    echo "    torch-npu-test-x86_64-cann-a2-py3.10-torch-nightly"
    echo "    torch-npu-test-x86_64-cann-a3-py3.10-torch-nightly"
    echo "    torch-npu-test-aarch64-cann-a1-py3.10-torch-nightly"
    echo "    torch-npu-test-aarch64-cann-a2-py3.10-torch-nightly"
    echo "    torch-npu-test-aarch64-cann-a3-py3.10-torch-nightly"
    exit 1
    ;;
esac

echo "=== Image Configuration ==="
echo "  Type:           ${IMAGE_TYPE}"
echo "  Architecture:   ${ARCH}"
if [[ "$IMAGE_TYPE" == "test" ]]; then
  echo "  CANN Chip:      ${CANN_CHIP}"
  echo "  Python:         ${PYTHON_VERSION}"
  echo "  PyTorch:        nightly"
fi
echo "  Full Tag:       ${TAG}"

case "${IMAGE_TYPE}" in
  builder)
    DOCKERFILE="${SCRIPT_DIR}/builder/Dockerfile.${ARCH}"
    echo "=== Building builder image: ${TAG} ==="
    docker build \
      --tag "${TAG}" \
      --file "${DOCKERFILE}" \
      "${BUILD_CONTEXT}"
    ;;
  test)
    DOCKERFILE="${SCRIPT_DIR}/test/Dockerfile.${ARCH}"
    echo "=== Building test image: ${TAG} ==="
    docker build \
      --build-arg CANN_CHIP="${CANN_CHIP}" \
      --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
      --tag "${TAG}" \
      --file "${DOCKERFILE}" \
      "${BUILD_CONTEXT}"
    ;;
esac

echo "=== Image built successfully: ${TAG} ==="
