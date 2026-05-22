#!/usr/bin/bash
# Install CANN toolkit for Ascend NPU.
# Usage: CANN_CHIP=A1 ./install_cann.sh
#   CANN_CHIP: A1 (Ascend 910), A2 (Ascend 910b), A3 (Ascend A3)
# Automatically detects architecture (x86_64 / aarch64).

set -e

CANN_CHIP="${CANN_CHIP:-A1}"
ARCH=$(uname -m)

BASE_URL="https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package"

case "${ARCH}_${CANN_CHIP}" in
  # x86_64
  x86_64_A1)
    TOOLKIT_URL="${BASE_URL}/20260513/Ascend-cann-toolkit_9.1.0_linux-x86_64.run"
    OPS_URL="${BASE_URL}/20260513/Ascend-cann-910-ops_9.1.0_linux-x86_64.run"
    NNAL_URL="${BASE_URL}/20260513/Ascend-cann-nnal_9.1.0_linux-x86_64.run"
    OPS_GLOB="Ascend-cann-910*"
    SET_ENV_PATH="/usr/local/Ascend/cann/set_env.sh"
    ;;
  x86_64_A2)
    TOOLKIT_URL="${BASE_URL}/20260116/Ascend-cann-toolkit_8.5.0_linux-x86_64.run"
    OPS_URL="${BASE_URL}/20260116/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run"
    NNAL_URL="${BASE_URL}/20260116/Ascend-cann-nnal_8.5.0_linux-x86_64.run"
    OPS_GLOB="Ascend-cann-910b*"
    SET_ENV_PATH="/usr/local/Ascend/ascend-toolkit/set_env.sh"
    ;;
  x86_64_A3)
    TOOLKIT_URL="${BASE_URL}/20260302/Ascend-cann-toolkit_9.0.0-beta.1_linux-x86_64.run"
    OPS_URL="${BASE_URL}/20260302/Ascend-cann-A3-ops_9.0.0-beta.1_linux-x86_64.run"
    NNAL_URL="${BASE_URL}/20260302/Ascend-cann-nnal_9.0.0-beta.1_linux-x86_64.run"
    OPS_GLOB="Ascend-cann-A3*"
    SET_ENV_PATH="/usr/local/Ascend/cann/set_env.sh"
    ;;
  # aarch64
  aarch64_A1)
    TOOLKIT_URL="${BASE_URL}/20260302/Ascend-cann-toolkit_9.0.0-beta.1_linux-aarch64.run"
    OPS_URL="${BASE_URL}/20260302/Ascend-cann-910b-ops_9.0.0-beta.1_linux-aarch64.run"
    NNAL_URL="${BASE_URL}/20260302/Ascend-cann-nnal_9.0.0-beta.1_linux-aarch64.run"
    OPS_GLOB="Ascend-cann-910b*"
    SET_ENV_PATH="/usr/local/Ascend/cann/set_env.sh"
    ;;
  aarch64_A2)
    TOOLKIT_URL="${BASE_URL}/20260513/Ascend-cann-toolkit_9.1.0_linux-aarch64.run"
    OPS_URL="${BASE_URL}/20260513/Ascend-cann-910b-ops_9.1.0_linux-aarch64.run"
    NNAL_URL="${BASE_URL}/20260513/Ascend-cann-nnal_9.1.0_linux-aarch64.run"
    OPS_GLOB="Ascend-cann-910b*"
    SET_ENV_PATH="/usr/local/Ascend/ascend-toolkit/set_env.sh"
    ;;
  aarch64_A3)
    TOOLKIT_URL="${BASE_URL}/20260330/Ascend-cann-toolkit_9.0.0-beta.2_linux-aarch64.run"
    OPS_URL="${BASE_URL}/20260330/Ascend-cann-A3-ops_9.0.0-beta.2_linux-aarch64.run"
    NNAL_URL="${BASE_URL}/20260330/Ascend-cann-nnal_9.0.0-beta.2_linux-aarch64.run"
    OPS_GLOB="Ascend-cann-A3*"
    SET_ENV_PATH="/usr/local/Ascend/cann/set_env.sh"
    ;;
  *)
    echo "Unsupported combination: ${ARCH} + ${CANN_CHIP}"
    exit 1
    ;;
esac

echo "Installing CANN ${CANN_CHIP} for ${ARCH}..."

rm -rf cann
mkdir -p cann && cd cann

echo "=== Downloading CANN packages ==="
curl -O "${TOOLKIT_URL}"
curl -O "${OPS_URL}"
curl -O "${NNAL_URL}"
echo "Download complete."

chmod +x Ascend-cann*.run

echo "=== Installing CANN toolkit ==="
./Ascend-cann-toolkit*.run --full --quiet --install-path=/usr/local/Ascend
source "${SET_ENV_PATH}"
echo "toolkit install success"

echo "=== Installing CANN ops ==="
./${OPS_GLOB}.run --install --quiet --install-path=/usr/local/Ascend
echo "ops install success"

echo "=== Installing CANN nnal ==="
./Ascend-cann-nnal*.run --install --quiet --install-path=/usr/local/Ascend
source /usr/local/Ascend/nnal/atb/set_env.sh
echo "nnal install success"

# Some CANN versions install to versioned paths (e.g. cann-9.0.0-beta.2)
# instead of /usr/local/Ascend/cann/. Fix broken symlinks so runtime
# sourcing of set_env.sh works.
if [ ! -f /usr/local/Ascend/cann/set_env.sh ]; then
  CANN_REAL_DIR=$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | head -1)
  if [ -n "${CANN_REAL_DIR}" ]; then
    ln -sf "${CANN_REAL_DIR}" /usr/local/Ascend/cann
    echo "Fixed: linked ${CANN_REAL_DIR} -> /usr/local/Ascend/cann"
  fi
fi

rm -rf *
echo "CANN ${CANN_CHIP} installation complete."
