# 基于 PyPA manylinux 2_28 aarch64 镜像 (与 PyTorch 主干一致)
FROM quay.io/pypa/manylinux_2_28_aarch64

# 安装必要的 OS 包
RUN yum -y install epel-release && \
    yum -y update && \
    yum install -y \
        autoconf \
        automake \
        bison \
        bzip2 \
        curl \
        diffutils \
        file \
        git \
        less \
        libffi-devel \
        libgomp \
        make \
        openssl-devel \
        patch \
        perl \
        unzip \
        util-linux \
        wget \
        which \
        xz \
        yasm \
        zstd \
        sudo && \
    yum install -y --enablerepo=powertools ninja-build && \
    rm -rf /var/cache/yum

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

# 设置 Python 3.11 为默认版本
ENV PYTHON_VERSION=3.11
ENV PATH=/opt/python/cp311-cp311/bin:$PATH

# 预安装 pytest 等测试依赖
RUN pip install pytest pytest-timeout pytest-xdist hypothesis pyyaml zstandard