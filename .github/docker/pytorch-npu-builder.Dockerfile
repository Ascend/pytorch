# 基于 PyPA manylinux 2_28 aarch64 镜像 (与 PyTorch 主干一致)
FROM quay.io/pypa/manylinux_2_28_aarch64

ARG GCCTOOLSET_VERSION=13

# CANN 包下载 URL（通过 build-arg 传入）
ARG CANN_TOOLKIT_URL
ARG CANN_A3OPS_URL
ARG CANN_NNAL_URL
ARG CANN_VERSION

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

# ============================================================
# 预装所有 Python 版本（镜像支持多 Python 版本）
# ============================================================
# manylinux 镜像已包含 cp310-cp310, cp311-cp311, cp312-cp312, cp313-cp313
# 默认使用 Python 3.11（可通过环境变量切换）

ENV DEFAULT_PYTHON_VERSION=3.11
ENV PATH=/opt/python/cp311-cp311/bin:$PATH

# 创建 Python 版本切换脚本
RUN printf '#!/bin/bash\n\
# Python 版本切换辅助脚本\n\
# 使用方法: source /usr/local/bin/switch_python.sh 3.11\n\
\n\
PYTHON_VERSION="${1:-3.11}"\n\
\n\
case "$PYTHON_VERSION" in\n\
    3.10) PYTHON_DIR="cp310-cp310" ;;\n\
    3.11) PYTHON_DIR="cp311-cp311" ;;\n\
    3.12) PYTHON_DIR="cp312-cp312" ;;\n\
    3.13) PYTHON_DIR="cp313-cp313" ;;\n\
    *) echo "Unsupported Python version: $PYTHON_VERSION"; return 1 ;;\n\
esac\n\
\n\
export PATH=/opt/python/$PYTHON_DIR/bin:$PATH\n\
echo "Switched to Python $PYTHON_VERSION ($(python --version))"\n\
' > /usr/local/bin/switch_python.sh && \
    chmod +x /usr/local/bin/switch_python.sh

# 为每个 Python 版本安装常用包
RUN for py_dir in cp310-cp310 cp311-cp311 cp312-cp312 cp313-cp313; do \
        /opt/python/$py_dir/bin/pip install --upgrade pip setuptools wheel; \
    done

# ============================================================
# 安装 CANN（使用传入的 URL）
# ============================================================

WORKDIR /root

RUN mkdir -p cann && cd cann && \
    curl -O "${CANN_TOOLKIT_URL}" && \
    curl -O "${CANN_A3OPS_URL}" && \
    curl -O "${CANN_NNAL_URL}" && \
    chmod +x Ascend-cann*.run && \
    ./Ascend-cann-toolkit*.run --full --quiet --install-path=/usr/local/Ascend && \
    ./Ascend-cann-A3*.run --install --quiet --install-path=/usr/local/Ascend && \
    source /usr/local/Ascend/cann/set_env.sh && \
    ./Ascend-cann-nnal*.run --install --quiet --install-path=/usr/local/Ascend && \
    rm -rf cann

# 设置环境变量
ENV CANN_PATH=/usr/local/Ascend/cann
ENV NNAL_PATH=/usr/local/Ascend/nnal
ENV ASCEND_HOME=/usr/local/Ascend
ENV CANN_VERSION=${CANN_VERSION}

# 添加 CANN 环境初始化脚本
RUN printf '#!/bin/bash\n\
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true\n\
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true\n\
' > /etc/profile.d/cann_env.sh && \
    chmod +x /etc/profile.d/cann_env.sh

# ============================================================
# 预安装 pytest 等测试依赖（为所有 Python 版本）
# ============================================================

RUN for py_dir in cp310-cp310 cp311-cp311 cp312-cp312 cp313-cp313; do \
        /opt/python/$py_dir/bin/pip install pytest pytest-timeout pytest-xdist hypothesis pyyaml zstandard cmake ninja; \
    done

# ============================================================
# 设置工作目录和默认命令
# ============================================================

WORKDIR /workspace

# 创建 welcome 消息
RUN printf '\n\
========================================\n\
PyTorch NPU Builder Image\n\
========================================\n\
CANN Version: %s\n\
Python Versions: 3.10, 3.11, 3.12, 3.13 (default: 3.11)\n\
\n\
To switch Python version:\n\
  source /usr/local/bin/switch_python.sh 3.12\n\
\n\
To setup CANN environment:\n\
  source /etc/profile.d/cann_env.sh\n\
========================================\n\
\n' "${CANN_VERSION}" > /etc/motd

CMD ["bash"]