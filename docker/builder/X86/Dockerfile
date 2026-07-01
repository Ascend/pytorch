FROM quay.io/pypa/manylinux_2_28_x86_64 AS base

ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV PATH=/usr/local/bin:$PATH

RUN echo "alias ll='ls -l --color=auto'" >> /root/.bashrc

ARG PY_VERSION=3.9

RUN cd /usr/local/bin \
    && PY_TAG="cp$(echo ${PY_VERSION} | tr -d .)" \
    && if [ ! -d "/opt/python/${PY_TAG}-${PY_TAG}" ]; then echo "Unsupported Python version: ${PY_VERSION}" && exit 1; fi \
    && ln -sf /opt/python/${PY_TAG}-${PY_TAG}/bin/pip${PY_VERSION} pip${PY_VERSION} \
    && ln -sf /opt/python/${PY_TAG}-${PY_TAG}/bin/pip${PY_VERSION} pip3 \
    && ln -sf /opt/python/${PY_TAG}-${PY_TAG}/bin/pip${PY_VERSION} pip \
    && ln -sf /opt/python/${PY_TAG}-${PY_TAG}/bin/python${PY_VERSION} python${PY_VERSION} \
    && ln -sf /opt/python/${PY_TAG}-${PY_TAG}/bin/python${PY_VERSION} python3 \
    && ln -sf /opt/python/${PY_TAG}-${PY_TAG}/bin/python${PY_VERSION} python

RUN mkdir /root/.pip \
    && echo "[global]" > /root/.pip/pip.conf \
    && echo "index-url=https://mirrors.huaweicloud.com/repository/pypi/simple" >> /root/.pip/pip.conf \
    && echo "trusted-host=repo.huaweicloud.com" >> /root/.pip/pip.conf \
    && echo "timeout=120" >> /root/.pip/pip.conf

RUN yum install -y \
    bzip2 \
    curl \
    file \
    git \
    sudo \
    unzip \
    util-linux \
    wget \
    which && \
    yum clean all && \
    rm -rf /var/cache/yum && \
    rm -rf /tmp/*

RUN git config --global --add safe.directory "*"

FROM base AS builder

ARG PY_VERSION
ARG DEVTOOLSET_VERSION=14

RUN yum install -y \
    autoconf \
    automake \
    bison \
    diffutils \
    make \
    mold \
    ninja-build \
    patch \
    perl \
    yum-utils \
    zlib-devel \
    gcc-toolset-${DEVTOOLSET_VERSION}-gcc \
    gcc-toolset-${DEVTOOLSET_VERSION}-gcc-c++ \
    gcc-toolset-${DEVTOOLSET_VERSION}-gcc-gfortran \
    gcc-toolset-${DEVTOOLSET_VERSION}-gdb && \
    yum clean all && \
    rm -rf /var/cache/yum && \
    rm -rf /tmp/*

ENV PATH=/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/lib64:/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/lib:$LD_LIBRARY_PATH

RUN python3 -mpip install cmake==3.18.4.post1 && \
    ln -sf /usr/local/bin/cmake /usr/bin/cmake3

RUN case "${PY_VERSION}" in \
        3.9|3.10) NUMPY_VER=1.22.4 ;; \
        3.11|3.12) NUMPY_VER=1.26.2 ;; \
        3.13) NUMPY_VER=2.1.2 ;; \
        *) echo "Unsupported Python version: ${PY_VERSION}" && exit 1 ;; \
    esac && \
    pip${PY_VERSION} install pyyaml && \
    pip${PY_VERSION} install setuptools wheel && \
    pip${PY_VERSION} install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip${PY_VERSION} install numpy==${NUMPY_VER}

FROM builder AS dev

ARG CANN_VERSION=9.1.0_beta.1
ARG CANN_PRODUCT=910b
ARG INSTALL_NNAL=0
ARG CANN_RELEASE_TRAIN=CANN%209.1.T1

RUN CANN_FILE_VER=$(echo ${CANN_VERSION} | sed 's/_beta/-beta/') && \
    ARCH_NAME=$(uname -m) && \
    CANN_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/${CANN_RELEASE_TRAIN} && \
    ln -sf /dev/stdout /dev/tty && \
    wget ${CANN_URL}/Ascend-cann-toolkit_${CANN_FILE_VER}_linux-${ARCH_NAME}.run && \
    bash Ascend-cann-toolkit_${CANN_FILE_VER}_linux-${ARCH_NAME}.run --install --install-path=/usr/local/Ascend --quiet && \
    rm -f Ascend-cann-toolkit_${CANN_FILE_VER}_linux-${ARCH_NAME}.run && \
    wget ${CANN_URL}/Ascend-cann-${CANN_PRODUCT}-ops_${CANN_FILE_VER}_linux-${ARCH_NAME}.run && \
    bash Ascend-cann-${CANN_PRODUCT}-ops_${CANN_FILE_VER}_linux-${ARCH_NAME}.run --install --type=toolkit --install-path=/usr/local/Ascend --quiet && \
    rm -f Ascend-cann-${CANN_PRODUCT}-ops_${CANN_FILE_VER}_linux-${ARCH_NAME}.run && \
    if [ "$INSTALL_NNAL" = "1" ]; then \
        CANN_NNAL_VER=$(echo ${CANN_VERSION} | sed 's/_beta\.[0-9][0-9]*//') && \
        wget ${CANN_URL}/Ascend-cann-nnal_${CANN_NNAL_VER}_linux-${ARCH_NAME}.run && \
        bash Ascend-cann-nnal_${CANN_NNAL_VER}_linux-${ARCH_NAME}.run --install --install-path=/usr/local/Ascend --quiet && \
        rm -f Ascend-cann-nnal_${CANN_NNAL_VER}_linux-${ARCH_NAME}.run; \
    fi && \
    rm -f /dev/tty && \
    echo "source /usr/local/Ascend/cann/set_env.sh" > /etc/profile.d/cann.sh && \
    if [ "$INSTALL_NNAL" = "1" ]; then \
        echo "source /usr/local/Ascend/nnal/atb/set_env.sh" >> /etc/profile.d/cann.sh; \
    fi && \
    chmod +x /etc/profile.d/cann.sh && \
    echo "source /usr/local/Ascend/cann/set_env.sh" >> /root/.bashrc && \
    echo "export ASCEND_GLOBAL_LOG_LEVEL=3" >> /root/.bashrc && \
    echo "export ASCEND_SLOG_PRINT_TO_STDOUT=0" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/Ascend/cann/lib64:/usr/local/Ascend/cann/opp/op_proto/built-in:\$LD_LIBRARY_PATH" >> /root/.bashrc

WORKDIR /home
