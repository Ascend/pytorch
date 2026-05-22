#!/usr/bin/bash
# Install CANN toolkit for Ascend NPU.
# Usage: CANN_CHIP=A1 ./install_cann.sh
#   CANN_CHIP: A1 (Ascend 910), A2 (Ascend 910b), A3 (Ascend A3)
# Automatically detects architecture (x86_64 / aarch64).

set -e

CANN_CHIP="${CANN_CHIP:-A1}"
ARCH=$(uname -m)

# CANN package definitions: date, version, ops_suffix, toolkit_set_env
# Format: "date|version|ops_suffix|set_env_path"
declare -A CANN_MAP

case "${ARCH}" in
  x86_64)
    ARCH_SUFFIX="x86_64"
    CANN_MAP=(
      [A1]="20260513|9.1.0|910|cann/set_env.sh"
      [A2]="20260116|8.5.0|910b|ascend-toolkit/set_env.sh"
      [A3]="20260302|9.0.0-beta.1|A3|cann/set_env.sh"
    )
    ;;
  aarch64)
    ARCH_SUFFIX="aarch64"
    CANN_MAP=(
      [A1]="20260302|9.0.0-beta.1|910b|cann/set_env.sh"
      [A2]="20260513|9.1.0|910b|ascend-toolkit/set_env.sh"
      [A3]="20260330|9.0.0-beta.2|A3|cann/set_env.sh"
    )
    ;;
  *)
    echo "Unsupported architecture: ${ARCH}"
    exit 1
    ;;
esac

if [[ -z "${CANN_MAP[$CANN_CHIP]}" ]]; then
  echo "Unknown CANN_CHIP: ${CANN_CHIP}. Supported: A1, A2, A3"
  exit 1
fi

IFS='|' read -r CANN_DATE CANN_VERSION OPS_SUFFIX SET_ENV_PATH <<< "${CANN_MAP[$CANN_CHIP]}"

CANN_BASE="https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/${CANN_DATE}"

TOOLKIT_PKG="Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH_SUFFIX}.run"
OPS_PKG="Ascend-cann-${OPS_SUFFIX}-ops_${CANN_VERSION}_linux-${ARCH_SUFFIX}.run"
NNAL_PKG="Ascend-cann-nnal_${CANN_VERSION}_linux-${ARCH_SUFFIX}.run"

echo "Installing CANN ${CANN_CHIP} (${CANN_VERSION}) for ${ARCH}..."

rm -rf cann
mkdir -p cann && cd cann

curl -O "${CANN_BASE}/${TOOLKIT_PKG}"
curl -O "${CANN_BASE}/${OPS_PKG}"
curl -O "${CANN_BASE}/${NNAL_PKG}"

if [[ $? -ne 0 ]]; then
  echo "Failed to download CANN packages"
  exit 1
fi

chmod +x Ascend-cann*.run
./Ascend-cann-toolkit*.run --full --quiet --install-path=/usr/local/Ascend \
  && source "/usr/local/Ascend/${SET_ENV_PATH}" \
  && echo "toolkit install success"

./Ascend-cann-${OPS_SUFFIX}*.run --install --quiet --install-path=/usr/local/Ascend \
  && echo "ops install success"

./Ascend-cann-nnal*.run --install --quiet --install-path=/usr/local/Ascend \
  && source /usr/local/Ascend/nnal/atb/set_env.sh \
  && echo "nnal install success"

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
