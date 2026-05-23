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
  x86_64_A1)
    TOOLKIT_URL="${BASE_URL}/20260513/Ascend-cann-toolkit_9.1.0_linux-x86_64.run"
    OPS_URL="${BASE_URL}/20260513/Ascend-cann-910-ops_9.1.0_linux-x86_64.run"
    NNAL_URL="${BASE_URL}/20260513/Ascend-cann-nnal_9.1.0_linux-x86_64.run"
    OPS_GLOB="Ascend-cann-910*"
    SET_ENV_PATH="/usr/local/Ascend/cann/set_env.sh"
    INSTALL_METHOD="run"
    ;;
  x86_64_A2)
    CANN_VERSION="9.1.0-beta.1"
    OPS_PACKAGE="ascend-cann-910b-ops"
    SET_ENV_PATH="/usr/local/Ascend/cann/set_env.sh"
    INSTALL_METHOD="apt"
    ;;
  x86_64_A3)
    CANN_BASE_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.T1"
    TOOLKIT_URL="${CANN_BASE_URL}/Ascend-cann_9.1.0-beta.1_linux-x86_64.run"
    OPS_URL="${CANN_BASE_URL}/Ascend-cann-A3-ops_9.1.0-beta.1_linux-x86_64.run"
    NNAL_URL="${CANN_BASE_URL}/Ascend-cann-nnal_9.1.0-beta.1_linux-x86_64.run"
    TOOLKIT_GLOB="Ascend-cann_9.1*"
    OPS_GLOB="Ascend-cann-A3-ops*"
    NNAL_GLOB="Ascend-cann-nnal*"
    SET_ENV_PATH="/usr/local/Ascend/cann/set_env.sh"
    INSTALL_METHOD="run_combined"
    ;;
  aarch64_A1)
    TOOLKIT_URL="${BASE_URL}/20260302/Ascend-cann-toolkit_9.0.0-beta.1_linux-aarch64.run"
    OPS_URL="${BASE_URL}/20260302/Ascend-cann-910b-ops_9.0.0-beta.1_linux-aarch64.run"
    NNAL_URL="${BASE_URL}/20260302/Ascend-cann-nnal_9.0.0-beta.1_linux-aarch64.run"
    OPS_GLOB="Ascend-cann-910b*"
    SET_ENV_PATH="/usr/local/Ascend/cann/set_env.sh"
    INSTALL_METHOD="run"
    ;;
  aarch64_A2)
    CANN_VERSION="9.1.0-beta.1"
    OPS_PACKAGE="ascend-cann-910b-ops"
    SET_ENV_PATH="/usr/local/Ascend/cann/set_env.sh"
    INSTALL_METHOD="apt"
    ;;
  aarch64_A3)
    CANN_BASE_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.T1"
    TOOLKIT_URL="${CANN_BASE_URL}/Ascend-cann_9.1.0-beta.1_linux-aarch64.run"
    OPS_URL="${CANN_BASE_URL}/Ascend-cann-A3-ops_9.1.0-beta.1_linux-aarch64.run"
    NNAL_URL="${CANN_BASE_URL}/Ascend-cann-nnal_9.1.0-beta.1_linux-aarch64.run"
    TOOLKIT_GLOB="Ascend-cann_9.1*"
    OPS_GLOB="Ascend-cann-A3-ops*"
    NNAL_GLOB="Ascend-cann-nnal*"
    SET_ENV_PATH="/usr/local/Ascend/cann/set_env.sh"
    INSTALL_METHOD="run_combined"
    ;;
  *)
    echo "Unsupported combination: ${ARCH} + ${CANN_CHIP}"
    exit 1
    ;;
esac

echo "Installing CANN ${CANN_CHIP} for ${ARCH}..."

if [ "${INSTALL_METHOD}" = "apt" ]; then
  echo "=== Configuring Ascend apt repository ==="
  wget -q https://ascend.devcloud.huaweicloud.com/cann/debian/cann-keyring_1.0.0_all.deb
  dpkg -i cann-keyring_1.0.0_all.deb
  apt-get update

  echo "=== Installing CANN toolkit ==="
  apt-get install -y ascend-cann-toolkit=${CANN_VERSION}
  source "${SET_ENV_PATH}"
  echo "toolkit install success"

  echo "=== Installing CANN ops ==="
  apt-get install -y ${OPS_PACKAGE}=${CANN_VERSION}
  echo "ops install success"

  echo "=== Installing CANN nnal ==="
  apt-get install -y ascend-cann-nnal=${CANN_VERSION}
  source /usr/local/Ascend/nnal/atb/set_env.sh
  echo "nnal install success"

  rm -f cann-keyring_1.0.0_all.deb
  rm -rf /var/lib/apt/lists/*
  echo "CANN ${CANN_CHIP} installation complete."
elif [ "${INSTALL_METHOD}" = "run_combined" ]; then
  echo "=== Creating HwHiAiUser user and group ==="
  groupadd -f HwHiAiUser
  id -u HwHiAiUser >/dev/null 2>&1 || useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

  echo "=== Installing dependencies ==="
  apt-get update
  apt-get install -y make gcc python3 python3-pip
  apt-get install -y dkms "linux-headers-$(uname -r)" || echo "Warning: dkms/linux-headers not available, skipping"
  rm -rf /var/lib/apt/lists/*

  rm -rf cann
  mkdir -p cann && cd cann

  echo "=== Downloading CANN packages ==="
  wget -q "${TOOLKIT_URL}"
  wget -q "${OPS_URL}"
  wget -q "${NNAL_URL}"
  echo "Download complete."

  chmod +x Ascend-cann*.run

  echo "=== Installing CANN driver & toolkit (combined package) ==="
  bash ./${TOOLKIT_GLOB}.run --install
  source "${SET_ENV_PATH}"
  echo "toolkit install success"

  echo "=== Installing CANN ops ==="
  bash ./${OPS_GLOB}.run --install
  echo "ops install success"

  echo "=== Installing CANN nnal ==="
  bash ./${NNAL_GLOB}.run --install
  source /usr/local/Ascend/nnal/atb/set_env.sh
  echo "nnal install success"

  rm -rf *
  echo "CANN ${CANN_CHIP} installation complete."
else
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

  if [ ! -f /usr/local/Ascend/cann/set_env.sh ]; then
    CANN_REAL_DIR=$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | head -1)
    if [ -n "${CANN_REAL_DIR}" ]; then
      ln -sf "${CANN_REAL_DIR}" /usr/local/Ascend/cann
      echo "Fixed: linked ${CANN_REAL_DIR} -> /usr/local/Ascend/cann"
    fi
  fi

  rm -rf *
  echo "CANN ${CANN_CHIP} installation complete."
fi
