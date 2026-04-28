# 基于 PyPA manylinux 2_28 aarch64 镜像 (与 PyTorch 主干一致)
FROM quay.io/pypa/manylinux_2_28_aarch64

ARG GCCTOOLSET_VERSION=13

# Language variables
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# 安装必要的 OS 包 (与 PyTorch 官方 Dockerfile 一致)
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
        sudo \
        gcc-toolset-${GCCTOOLSET_VERSION}-gcc \
        gcc-toolset-${GCCTOOLSET_VERSION}-gcc-c++ \
        gcc-toolset-${GCCTOOLSET_VERSION}-gcc-gfortran \
        gcc-toolset-${GCCTOOLSET_VERSION}-gdb && \
    yum install -y --enablerepo=powertools ninja-build && \
    rm -rf /var/cache/yum

# 确保使用正确的 devtoolset
ENV PATH=/opt/rh/gcc-toolset-${GCCTOOLSET_VERSION}/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-${GCCTOOLSET_VERSION}/root/usr/lib64:/opt/rh/gcc-toolset-${GCCTOOLSET_VERSION}/root/usr/lib:$LD_LIBRARY_PATH

# git 2.36+ 需要配置 safe.directory
RUN git config --global --add safe.directory "*"

# 设置 Python 3.11 为默认版本 (CANN 安装需要 Python 环境)
ENV PYTHON_VERSION=3.11
ENV PATH=/opt/python/cp311-cp311/bin:$PATH

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