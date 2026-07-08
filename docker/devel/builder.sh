#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
IMAGE_NAME="manylinux-builder"
IMAGE_TAG="v1"
CONTAINER_NAME="torch-npu-builder"

PY_VERSION="3.10"
INSTALL_CANN=0
CANN_VERSION="9.1.0_beta.1"
CANN_PRODUCT="910b"
INSTALL_NNAL=0
NO_CACHE=0
CANN_RELEASE_TRAIN=""
IMAGE_TARGET="builder"
TORCH_VERSION="2.12.0"

function usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  -p, --python VERSION      Python version: 3.10 3.11 3.12 3.13 (default: 3.10)
  --torch-version VER       PyTorch version: x.x.x (e.g. 2.12.0)
  --no-cache                Build docker image without cache
  -h, --help                Show this help message

CANN Options:
  --cann                    Build dev image with CANN (default: builder image without CANN)
  --cann-version VER        CANN version (default: 9.1.0_beta.1)
  --cann-product PROD       CANN product: 950, A3, 910b, 910, 310p, 310b (default: 910b)
  --cann-release-train VER  CANN release train, e.g. CANN%209.1.T1 (required when --cann-version != default)
  --nnal                    Install CANN NNAL (requires --cann)

Examples:
  $0                                Build image & launch container with default python 3.10
  $0 -p 3.11                       Use python 3.11
  $0 --cann                         Build with CANN (default: 910b ops)
  $0 --cann --cann-product A3       Build with CANN for Atlas A3
  $0 --cann --nnal                  Build with CANN + NNAL
  $0 --cann --cann-version 9.0.0 --cann-release-train CANN%209.0.0
                          Build with CANN 9.0.0 (release train is version-specific)

EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    -p | --python)
        PY_VERSION="$2"
        shift 2
        ;;
    --no-cache)
        NO_CACHE=1
        shift
        ;;
    --torch-version)
        TORCH_VERSION="$2"
        shift 2
        ;;
    --cann)
        INSTALL_CANN=1
        shift
        ;;
    --cann-version)
        CANN_VERSION="$2"
        shift 2
        ;;
    --cann-product)
        CANN_PRODUCT="$2"
        shift 2
        ;;
    --cann-release-train)
        CANN_RELEASE_TRAIN="$2"
        shift 2
        ;;
    --nnal)
        INSTALL_NNAL=1
        shift
        ;;
    -h | --help)
        usage 0
        ;;
    *)
        echo "Unknown option: $1"
        usage 1
        ;;
    esac
done

if [ "${INSTALL_NNAL}" -eq 1 ] && [ "${INSTALL_CANN}" -ne 1 ]; then
    echo "ERROR: --nnal requires --cann"
    exit 1
fi

DEFAULT_CANN_VERSION="9.1.0_beta.1"
DEFAULT_CANN_RELEASE_TRAIN="CANN%209.1.T1"
if [ "${INSTALL_CANN}" -eq 1 ] && [ "${CANN_VERSION}" != "${DEFAULT_CANN_VERSION}" ] && [ -z "${CANN_RELEASE_TRAIN}" ]; then
    echo "ERROR: --cann-version '${CANN_VERSION}' requires --cann-release-train"
    echo "       The OBS directory name is version-specific and cannot be derived from the version string."
    echo "       Look up the correct release train at https://www.hiascend.com/ and pass it via --cann-release-train."
    echo "       Example: --cann-release-train ${DEFAULT_CANN_RELEASE_TRAIN}"
    exit 1
fi

VALID_CANN_PRODUCTS="950 A3 910b 910 310p 310b"
if [ "${INSTALL_CANN}" -eq 1 ]; then
    found=0
    for p in ${VALID_CANN_PRODUCTS}; do
        if [ "${CANN_PRODUCT}" = "${p}" ]; then
            found=1
            break
        fi
    done
    if [ "${found}" -eq 0 ]; then
        echo "ERROR: invalid --cann-product '${CANN_PRODUCT}', must be one of: ${VALID_CANN_PRODUCTS}"
        exit 1
    fi
fi

ARCH=$(uname -m)
DOCKERFILE_DIR="${SCRIPT_DIR}"

if [ ! -f "${DOCKERFILE_DIR}/Dockerfile" ]; then
    echo "Dockerfile not found: ${DOCKERFILE_DIR}/Dockerfile"
    exit 1
fi

SOURCE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
if [ ! -d "${SOURCE_DIR}" ]; then
    echo "Source directory not found: ${SOURCE_DIR}"
    exit 1
fi

BUILD_ARGS=""
BUILD_ARGS="${BUILD_ARGS} --build-arg PY_VERSION=${PY_VERSION}"
BUILD_ARGS="${BUILD_ARGS} --build-arg TORCH_VERSION=${TORCH_VERSION}"
if [ "${INSTALL_CANN}" -eq 1 ]; then
    IMAGE_TARGET="dev"
    BUILD_ARGS="${BUILD_ARGS} --build-arg CANN_VERSION=${CANN_VERSION}"
    BUILD_ARGS="${BUILD_ARGS} --build-arg CANN_PRODUCT=${CANN_PRODUCT}"
    if [ -n "${CANN_RELEASE_TRAIN}" ]; then
        BUILD_ARGS="${BUILD_ARGS} --build-arg CANN_RELEASE_TRAIN=${CANN_RELEASE_TRAIN}"
    fi
fi
if [ "${INSTALL_NNAL}" -eq 1 ]; then
    BUILD_ARGS="${BUILD_ARGS} --build-arg INSTALL_NNAL=1"
fi
if [ "${NO_CACHE}" -eq 1 ]; then
    BUILD_ARGS="${BUILD_ARGS} --no-cache"
fi

echo "=========================================="
echo " Building docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo " Architecture: ${ARCH}"
echo " Dockerfile:   ${DOCKERFILE_DIR}/Dockerfile"
echo " Target:       ${IMAGE_TARGET}"
echo " Python:       ${PY_VERSION}"
echo " Torch:        ${TORCH_VERSION}"
if [ "${INSTALL_CANN}" -eq 1 ]; then
    echo " CANN:         ${CANN_VERSION} (${CANN_PRODUCT})"
    if [ -n "${CANN_RELEASE_TRAIN}" ]; then
        echo " CANN Release: ${CANN_RELEASE_TRAIN}"
    fi
    if [ "${INSTALL_NNAL}" -eq 1 ]; then
        echo " NNAL:         enabled"
    fi
fi
echo "=========================================="
docker build ${BUILD_ARGS} --target ${IMAGE_TARGET} -t "${IMAGE_NAME}:${IMAGE_TAG}" "${DOCKERFILE_DIR}" || {
    echo "Failed to build docker image."
    exit 1
}
echo "Docker image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"

echo "=========================================="
echo " Launching container: ${CONTAINER_NAME}"
echo " Source:  ${SOURCE_DIR} -> /home/pytorch"
echo " Python:  ${PY_VERSION}"
if [ "${INSTALL_CANN}" -eq 1 ]; then
    echo " Mode:    runtime (with NPU driver passthrough)"
else
    echo " Mode:    compile-only (no NPU driver)"
fi
echo "=========================================="

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
if [ "${INSTALL_CANN}" -eq 1 ]; then
    docker run -d --rm \
        --name "${CONTAINER_NAME}" \
        --privileged \
        -v /dev:/dev \
        -v "${SOURCE_DIR}:/home/pytorch" \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
        -v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi \
        -v /var/log/npu:/usr/slog \
        -e PY_VERSION="${PY_VERSION}" \
        -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/base:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        tail -f /dev/null
else
    docker run -d --rm \
        --name "${CONTAINER_NAME}" \
        -v "${SOURCE_DIR}:/home/pytorch" \
        -e PY_VERSION="${PY_VERSION}" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        tail -f /dev/null
fi

echo ""
echo "Container started. To enter:"
echo "  docker exec -it ${CONTAINER_NAME} bash"
echo ""
if [ "${INSTALL_CANN}" -eq 1 ]; then
    echo "Verify NPU visible:"
    echo "  docker exec ${CONTAINER_NAME} npu-smi info"
    echo ""
fi
echo "Then run (inside container):"
echo "  bash ci/build.sh --python=${PY_VERSION}"
