# 基于 PyTorch manylinux builder 镜像
FROM ghcr.io/pytorch/manylinux-builder:aarch64

# 设置工作目录
WORKDIR /root

# 安装 CANN 9.0.0-beta.2
RUN mkdir -p cann && cd cann && \
    curl -O https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/20260330/Ascend-cann-toolkit_9.0.0-beta.2_linux-aarch64.run && \
    curl -O https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/20260330/Ascend-cann-A3-ops_9.0.0-beta.2_linux-aarch64.run && \
    curl -O https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/20260330/Ascend-cann-nnal_9.0.0-beta.2_linux-aarch64.run && \
    chmod +x Ascend-cann*.run && \
    ./Ascend-cann-toolkit*.run --full --quiet --install-path=/usr/local/Ascend && \
    ./Ascend-cann-A3*.run --install --quiet --install-path=/usr/local/Ascend && \
    ./Ascend-cann-nnal*.run --install --quiet --install-path=/usr/local/Ascend && \
    rm -rf cann

# 设置环境变量
ENV CANN_PATH=/usr/local/Ascend/cann
ENV NNAL_PATH=/usr/local/Ascend/nnal
ENV ASCEND_HOME=/usr/local/Ascend

# 添加 CANN 环境初始化脚本
RUN printf '#!/bin/bash\nsource /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true\nsource /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true\n' > /etc/profile.d/cann_env.sh && \
    chmod +x /etc/profile.d/cann_env.sh

# 预安装 pytest 等测试依赖
RUN pip3.11 install pytest pytest-timeout pytest-xdist hypothesis pyyaml zstandard