#!/usr/bin/bash
# Build torch-npu CI Docker images.
#
# Usage:
#   ./docker_build.sh <TAG>
#
# Builder: torch-npu-builder-<ARCH>-py<PYTORCH_VERSION>
# Test:    torch-npu-test-<ARCH>-cann<CHIP>-py<PYTORCH_VERSION>
#
# Examples:
#   ./docker_build.sh torch-npu-builder-x86_64-py2.7.1
#   ./docker_build.sh torch-npu-test-aarch64-cannA2-py2.7.1
#
# Reference: pytorch/pytorch .ci/docker/build.sh

set -ex

tag="${1:?Usage: $0 <TAG>}"
shift

case "$tag" in
  torch-npu-builder-x86_64-py2.7.1)
    IMAGE_TYPE=builder
    ARCH=x86_64
    PYTORCH_VERSION=2.7.1
    ;;
  torch-npu-builder-aarch64-py2.7.1)
    IMAGE_TYPE=builder
    ARCH=aarch64
    PYTORCH_VERSION=2.7.1
    ;;
  torch-npu-test-x86_64-cannA1-py2.7.1)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A1
    PYTORCH_VERSION=2.7.1
    ;;
  torch-npu-test-x86_64-cannA2-py2.7.1)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A2
    PYTORCH_VERSION=2.7.1
    ;;
  torch-npu-test-x86_64-cannA3-py2.7.1)
    IMAGE_TYPE=test
    ARCH=x86_64
    CANN_CHIP=A3
    PYTORCH_VERSION=2.7.1
    ;;
  torch-npu-test-aarch64-cannA1-py2.7.1)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A1
    PYTORCH_VERSION=2.7.1
    ;;
  torch-npu-test-aarch64-cannA2-py2.7.1)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A2
    PYTORCH_VERSION=2.7.1
    ;;
  torch-npu-test-aarch64-cannA3-py2.7.1)
    IMAGE_TYPE=test
    ARCH=aarch64
    CANN_CHIP=A3
    PYTORCH_VERSION=2.7.1
    ;;
  *)
    echo "Unknown tag: ${tag}"
    echo "  Builder: torch-npu-builder-<x86_64|aarch64>-py2.7.1"
    echo "  Test:    torch-npu-test-<x86_64|aarch64>-cann<A1|A2|A3>-py2.7.1"
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/${IMAGE_TYPE}/Dockerfile.${ARCH}"

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "Dockerfile not found: ${DOCKERFILE}"
  exit 1
fi

BUILD_ARGS=(
  --build-arg PYTORCH_VERSION="${PYTORCH_VERSION}"
)
if [[ -n "${CANN_CHIP:-}" ]]; then
  BUILD_ARGS+=(--build-arg CANN_CHIP="${CANN_CHIP}")
fi

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d%H%M)}"
IMAGE_TAG="${tag}-${TIMESTAMP}"

echo "Building ${IMAGE_TAG} ..."
echo "  Dockerfile: ${DOCKERFILE}"
echo "  PyTorch:    ${PYTORCH_VERSION}"
[[ -n "${CANN_CHIP:-}" ]] && echo "  CANN chip:  ${CANN_CHIP}"

docker build \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_TAG}" \
  "${BUILD_ARGS[@]}" \
  "${SCRIPT_DIR}"

echo "Image built: ${IMAGE_TAG}"
