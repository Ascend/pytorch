ARG CANN_VERSION
ARG CHIP_ARCH
ARG OS
ARG OS_VERSION
ARG PY_VERSION

FROM quay.io/ascend/cann:${CANN_VERSION}-${CHIP_ARCH}-${OS}${OS_VERSION}-py${PY_VERSION} AS base

ARG ARCH
ARG PY_TAG
ARG TORCH_NPU_PATCH_TAG
ARG TORCH_NPU_RELEASE_TAG
ARG MANYLINUX_VER=manylinux_2_28

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 config set global.index-url ${PIP_INDEX_URL}

RUN TORCH_VERSION=$(echo "${TORCH_NPU_RELEASE_TAG}" | sed -E 's/.*pytorch([0-9.]+).*/\1/') \
    && if [ "${ARCH}" = "arm" ]; then \
        wget https://download.pytorch.org/whl/cpu/torch-${TORCH_VERSION}%2Bcpu-${PY_TAG}-${PY_TAG}-${MANYLINUX_VER}_aarch64.whl \
        && pip3 install --no-cache-dir --progress-bar off torch-${TORCH_VERSION}+cpu-${PY_TAG}-${PY_TAG}-${MANYLINUX_VER}_aarch64.whl \
        && rm -f torch-${TORCH_VERSION}+cpu-${PY_TAG}-${PY_TAG}-${MANYLINUX_VER}_aarch64.whl \
        && wget https://gitcode.com/Ascend/pytorch/releases/download/${TORCH_NPU_RELEASE_TAG}/torch_npu-${TORCH_NPU_PATCH_TAG}-${PY_TAG}-${PY_TAG}-${MANYLINUX_VER}_aarch64.whl \
        && pip3 install --no-cache-dir --progress-bar off torch_npu-${TORCH_NPU_PATCH_TAG}-${PY_TAG}-${PY_TAG}-${MANYLINUX_VER}_aarch64.whl \
        && rm -f torch_npu-${TORCH_NPU_PATCH_TAG}-${PY_TAG}-${PY_TAG}-${MANYLINUX_VER}_aarch64.whl; \
    else \
        wget https://download.pytorch.org/whl/cpu/torch-${TORCH_VERSION}%2Bcpu-${PY_TAG}-${PY_TAG}-${MANYLINUX_VER}_x86_64.whl \
        && pip3 install --no-cache-dir --progress-bar off torch-${TORCH_VERSION}+cpu-${PY_TAG}-${PY_TAG}-${MANYLINUX_VER}_x86_64.whl \
        && rm -f torch-${TORCH_VERSION}+cpu-${PY_TAG}-${PY_TAG}-${MANYLINUX_VER}_x86_64.whl \
        && wget https://gitcode.com/Ascend/pytorch/releases/download/${TORCH_NPU_RELEASE_TAG}/torch_npu-${TORCH_NPU_PATCH_TAG}-${PY_TAG}-${PY_TAG}-${MANYLINUX_VER}_x86_64.whl \
        && pip3 install --no-cache-dir --progress-bar off torch_npu-${TORCH_NPU_PATCH_TAG}-${PY_TAG}-${PY_TAG}-${MANYLINUX_VER}_x86_64.whl \
        && rm -f torch_npu-${TORCH_NPU_PATCH_TAG}-${PY_TAG}-${PY_TAG}-${MANYLINUX_VER}_x86_64.whl; \
    fi \
    && pip3 install --no-cache-dir --progress-bar off pyyaml numpy attrs>=23.0.0 decorator>=5.1.0 psutil>=5.9 scipy>=1.7.3 jinja2 MarkupSafe