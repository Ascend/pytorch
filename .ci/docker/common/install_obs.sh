#!/usr/bin/bash
# Install Huawei OBS util for object storage access.

set -e

ARCH=$(uname -m)
case "${ARCH}" in
  x86_64)  OBS_ARCH="amd64" ;;
  aarch64) OBS_ARCH="arm64" ;;
  *)       echo "Unsupported architecture: ${ARCH}"; exit 1 ;;
esac

OBS_URL="https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_${OBS_ARCH}.tar.gz"

wget -q "${OBS_URL}"
mkdir -p /usr/local/obsutil
tar -zxf "obsutil_linux_${OBS_ARCH}.tar.gz" -C /usr/local/obsutil/
rm -f "obsutil_linux_${OBS_ARCH}.tar.gz"
ln -sf /usr/local/obsutil/obsutil_linux_${OBS_ARCH}_*/obsutil /usr/local/bin/obsutil

echo "OBS util installed."
