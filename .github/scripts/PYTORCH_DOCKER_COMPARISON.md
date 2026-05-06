# PyTorch 上游镜像构建逻辑对比分析

## 对比概述

对比上游 PyTorch 仓库和当前 torch-npu 项目的镜像构建策略，分析相似点和差异。

**对应关系**：
- PyTorch CUDA → torch-npu CANN
- PyTorch cuDNN → torch-npu NNAL/A3-ops
- PyTorch Python 版本 → torch-npu Python 版本

---

## 一、镜像命名策略对比

### PyTorch 上游命名

**CI 镜像命名**（用于内部测试）：
```
pytorch-linux-jammy-cuda13.0-cudnn9-py3.10-clang18
pytorch-linux-jammy-cuda13.0-cudnn9-py3-gcc11
pytorch-linux-jammy-py3.11-clang18          # CPU 版本
```

特点：
- 格式：`pytorch-linux-{OS}-{CUDA}-{cuDNN}-py{Python}-{Compiler}`
- CUDA 版本格式：`cuda13.0`（去掉小版本号）
- cuDNN 版本格式：`cudnn9`（只保留大版本）
- Python 版本格式：`py3.10` 或 `py3`（默认最新）

**官方发布镜像命名**（用户使用）：
```
ghcr.io/pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
ghcr.io/pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
ghcr.io/pytorch/pytorch:2.5.1-runtime          # CPU 版本
```

特点：
- 格式：`{PyTorch版本}-cuda{CUDA简版}-cudnn{cuDNN}-{类型}`
- 镜像类型：`runtime`（运行时）vs `devel`（开发）
- CUDA 版本简化：`cuda12.1`（去掉补丁版本）

**Nightly 镜像额外标签**：
```
ghcr.io/pytorch/pytorch-nightly:2.5.0.dev20250101-cuda12.1-cudnn9-runtime
ghcr.io/pytorch/pytorch-nightly:{git_commit}-cu121
ghcr.io/pytorch/pytorch-nightly:latest          # Stable CUDA 的 latest
```

---

### torch-npu 当前命名

**当前标签格式**：
```
quay.io/kerer/pytorch:py3.11-cann9.0.0-beta.2-20260506   # 完整版
quay.io/kerer/pytorch:py3.11-cann9.0.0-beta.2            # 标准版
quay.io/kerer/pytorch:py3.11-cann9.0.0                   # 大版本简化
quay.io/kerer/pytorch:py3.11-latest                      # Python latest
quay.io/kerer/pytorch:latest                            # 全局 latest
```

---

### 命名策略对比表

| 维度 | PyTorch 上游 | torch-npu 当前 | 建议 |
|------|-------------|---------------|------|
| **前缀** | 无前缀 | `py` 前缀 | ✅ torch-npu 更直观 |
| **Python 版本** | `py3.10` 或 `py3` | `py3.11` | ✅ 相似，都用 py 前缀 |
| **CUDA/CANN** | `cuda13.0` | `cann9.0.0-beta.2` | ⚠️ PyTorch 更简化 |
| **cuDNN/NNAL** | `cudnn9` | 未包含 | ⚠️ torch-npu 可添加 |
| **镜像类型** | `runtime`/`devel` | 无区分 | ⚠️ 可考虑添加 |
| **时间戳** | nightly 包含 | 完整版包含 | ✅ 相似 |
| **latest 位置** | `latest` 无后缀 | `latest` 或 `py3.11-latest` | ✅ torch-npu 更细致 |

---

## 二、版本管理策略对比

### PyTorch CUDA 版本矩阵

**支持的 CUDA 版本**：
```python
CUDA_ARCHES = ["12.6", "13.0", "13.2"]
CUDA_STABLE = "13.0"    # 稳定版本

CUDA_ARCHES_FULL_VERSION = {
    "12.6": "12.6.3",
    "13.0": "13.0.2",
    "13.2": "13.2.1",
}

CUDA_ARCHES_CUDNN_VERSION = {
    "12.6": "9",
    "13.0": "9",
    "13.2": "9",
}
```

**特点**：
- 维护简化版本 → 完整版本映射表
- 每个 CUDA 版本对应固定的 cuDNN 版本
- 明确标记稳定版本（用于 latest 标签）

---

### torch-npu CANN 版本管理

**当前实现**：
```bash
DEFAULT_CANN_VERSION="9.0.0-beta.2"
DEFAULT_CANN_DATE="20260330"

# 提取大版本号
cann_major=$(echo "$CANN_VERSION" | sed 's/-beta.*//' | sed 's/-rc.*//')
```

**差异**：
- ❌ 没有版本映射表（简化版本 → 完整版本）
- ❌ 没有 stable 版本标记
- ❌ CANN 日期硬编码在参数中

---

### 建议改进：版本映射表

创建类似 PyTorch 的版本映射配置：

```bash
# 在 build_image.sh 中添加版本映射
declare -A CANN_ARCHES_FULL_VERSION=(
    ["9.0"]="9.0.0"
    ["8.0"]="8.0.RC3"
)

declare -A CANN_ARCHES_DATE=(
    ["9.0"]="20260330"
    ["8.0"]="20250101"
)

declare -A CANN_ARCHES_NNAL_VERSION=(
    ["9.0"]="9.0.0"   # 对应 cuDNN
    ["8.0"]="8.0"
)

CANN_STABLE="9.0"  # 稳定版本
```

---

## 三、构建组织方式对比

### PyTorch 构建架构

**文件组织**：
```
pytorch/pytorch/
├── Dockerfile                           # 用户发布镜像
├── docker.Makefile                      # 构建脚本
├── .ci/docker/                          # CI 镜像目录
│   ├── ubuntu/Dockerfile                # Ubuntu 基础镜像
│   ├── common/                          # 公共安装脚本
│   │   ├── install_cuda.sh              # CUDA 安装脚本
│   │   ├── install_conda.sh             # Conda 安装脚本
│   │   └── install_gcc.sh               # GCC 安装脚本
│   ├── requirements-ci.txt              # CI 依赖
│   └── ci_commit_pins/                  # 版本锁定
│       ├── triton.txt                   # Triton 版本
│       ├── nccl*                        # NCCL 版本
│       └── jax.txt                      # JAX 版本
├── .github/workflows/
│   ├── docker-builds.yml                # CI 镜像构建
│   ├── docker-release.yml               # 发布镜像构建
│   └── docker-cache-rocm.yml            # ROCm 缓存
└── .github/scripts/
    ├── generate_docker_release_matrix.py  # 矩阵生成
    └── generate_binary_build_matrix.py    # 二进制矩阵
```

**关键特点**：
1. **分层组织**：CI 镜像和发布镜像分离
2. **公共脚本**：`common/` 目录下有各种安装脚本
3. **版本锁定**：`ci_commit_pins/` 目录锁定所有依赖版本
4. **矩阵生成**：Python 脚本动态生成构建矩阵

---

### torch-npu 构建架构

**当前文件组织**：
```
ascend-pytorch/
├── .github/
│   ├── docker/
│   │   └── pytorch-npu-builder.Dockerfile   # 单一 Dockerfile
│   ├── scripts/
│   │   ├── build_image.sh                   # 构建脚本
│   │   └── BUILD_IMAGE_README.md            # 文档
│   └── workflows/
│       └── build-docker-image.yml           # Workflow
```

**对比差异**：

| 组织方式 | PyTorch 上游 | torch-npu 当前 | 建议 |
|---------|-------------|---------------|------|
| **CI vs 发布分离** | ✅ 分离 | ❌ 单一 Dockerfile | ⚠️ 可考虑分离 |
| **公共安装脚本** | ✅ `common/` 目录 | ❌ 直接在 Dockerfile | ⚠️ 建议拆分 |
| **版本锁定文件** | ✅ `ci_commit_pins/` | ❌ 硬编码参数 | ⚠️ 强烈建议 |
| **矩阵生成脚本** | ✅ Python 脚本 | ✅ Shell 脚本 | ✅ 相似 |
| **文档完整性** | ❌ 较少 | ✅ README 文档 | ✅ torch-npu 更好 |

---

## 四、Workflow 设计对比

### PyTorch docker-builds.yml

**触发条件**：
```yaml
on:
  workflow_dispatch:
  pull_request:
    paths:
      - .ci/docker/**
      - .github/workflows/docker-builds.yml
  push:
    branches: [main, release/*]
    paths:
      - .ci/docker/**
      - .github/workflows/docker-builds.yml
  schedule:
    - cron: 1 3 * * 3   # 每周三 UTC 03:01
```

**Matrix 策略**：
```yaml
matrix:
  docker-image-name: [
    pytorch-linux-jammy-cuda13.0-cudnn9-py3-gcc11,
    pytorch-linux-jammy-cuda13.0-cudnn9-py3.12-gcc11-vllm,
    pytorch-linux-jammy-py3.10-clang18,      # CPU 版本
    # ... 30+ 种镜像配置
  ]
  include:
    - docker-image-name: pytorch-linux-jammy-aarch64-py3.10-gcc13
      runner: linux.arm64.m7g.4xlarge       # ARM64 特定 runner
```

**镜像推送**：
```yaml
# 推送到 ECR（AWS）
- name: Build docker image
  uses: pytorch/test-infra/.github/actions/calculate-docker-image@main
  with:
    docker-image-name: ci-image:${{ matrix.docker-image-name }}
    always-rebuild: true
    push: true

# 推送到 ghcr.io（公共）
- name: Push to https://ghcr.io/
  if: ${{ github.event_name == 'push' }}
  run: |
    ghcr_image="ghcr.io/pytorch/ci-image"
    tag=${ECR_DOCKER_IMAGE##*:}
    docker tag "${ECR_DOCKER_IMAGE}" "${ghcr_image}:${tag}"
    docker push "${ghcr_image}:${tag}"
    # Also push a tag without the hash
    docker tag "${ECR_DOCKER_IMAGE}" "${ghcr_image}:${{ matrix.docker-image-name }}"
    docker push "${ghcr_image}:${{ matrix.docker-image-name }}"
```

---

### torch-npu build-docker-image.yml

**当前实现**：
```yaml
on:
  push:
    branches: [dev_master]
    paths:
      - '.github/docker/**'
      - '.github/workflows/**'
  schedule:
    - cron: '0 2 * * 0'   # 每周日 UTC 02:00
  workflow_dispatch:
    inputs:
      python_version: ['all', '3.10', '3.11', '3.12', '3.13']
      cann_version: '9.0.0-beta.2'
      push_image: true

matrix:
  python: ['3.10', '3.11', '3.12', '3.13']
```

---

### Workflow 对比表

| 设计要点 | PyTorch 上游 | torch-npu 当前 | 建议 |
|---------|-------------|---------------|------|
| **触发路径** | `.ci/docker/**` | `.github/docker/**` | ✅ 相似 |
| **定时构建** | 每周三 | 每周日 | ✅ 合理 |
| **Matrix 配置** | 硬编码镜像名列表 | Python 版本列表 | ⚠️ PyTorch 更详细 |
| **多 Registry** | ECR + ghcr.io | 单一 quay.io | ⚠️ 可考虑多 Registry |
| **推送策略** | Hash tag + Name tag | 多层级标签 | ✅ torch-npu 更细致 |
| **手动触发** | 无参数 | 多参数输入 | ✅ torch-npu 更灵活 |

---

## 五、CUDA/CANN 安装方式对比

### PyTorch CUDA 安装

**install_cuda.sh 脚本**（分离式）：
```bash
# 调用方式
ARG CUDA_VERSION
COPY ./common/install_cuda.sh install_cuda.sh
RUN bash ./install_cuda.sh ${CUDA_VERSION}

# Dockerfile 中的环境变量
ENV DESIRED_CUDA ${CUDA_VERSION}
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
```

**特点**：
- CUDA 安装逻辑独立在 `install_cuda.sh` 中
- Dockerfile 只负责调用脚本
- 版本通过 ARG 参数传递

---

### torch-npu CANN 安装

**当前实现**（嵌入式）：
```dockerfile
ARG CANN_VERSION
ARG CANN_DATE

RUN mkdir -p cann && cd cann && \
    curl -O https://.../Ascend-cann-toolkit_${CANN_VERSION}_linux-aarch64.run && \
    curl -O https://.../Ascend-cann-A3-ops_${CANN_VERSION}_linux-aarch64.run && \
    curl -O https://.../Ascend-cann-nnal_${CANN_VERSION}_linux-aarch64.run && \
    chmod +x Ascend-cann*.run && \
    ./Ascend-cann-toolkit*.run --full --quiet --install-path=/usr/local/Ascend && \
    # ...
```

**对比差异**：

| 安装方式 | PyTorch 上游 | torch-npu 当前 | 建议 |
|---------|-------------|---------------|------|
| **脚本分离** | ✅ `install_cuda.sh` | ❌ 嵌入 Dockerfile | ⚠️ 建议拆分 |
| **依赖安装** | ✅ NCCL/cuSPARSE 等独立脚本 | ❌ 混在一起 | ⚠️ 建议拆分 |
| **版本管理** | ✅ 参数传递 + 环境变量 | ✅ 参数传递 | ✅ 相似 |
| **安装路径** | `/usr/local/cuda` | `/usr/local/Ascend` | ✅ 合理 |

---

## 六、镜像类型对比

### PyTorch 镜像类型

**两种镜像类型**：
```dockerfile
# runtime 镜像（精简）
FROM official as runtime
# 只包含运行时必需的组件

# devel 镜像（完整）
FROM official as dev
# 包含开发工具、编译器等
```

**docker.Makefile 定义**：
```makefile
runtime-image: DOCKER_TAG := $(PYTORCH_VERSION)-cuda$(CUDA_VERSION_SHORT)-cudnn$(CUDNN_VERSION)-runtime
devel-image: DOCKER_TAG := $(PYTORCH_VERSION)-cuda$(CUDA_VERSION_SHORT)-cudnn$(CUDNN_VERSION)-devel
```

---

### torch-npu 当前状态

**单一镜像类型**：
- 当前只有一种镜像，包含构建和运行时所有工具
- 没有区分 runtime 和 devel

**建议**：
```dockerfile
# 可以添加多阶段构建
FROM base as runtime
# 只包含 CANN runtime + Python

FROM runtime as devel
# 添加编译工具、调试工具等
```

---

## 七、关键差异总结

### 相似点 ✅

1. **使用 ARG 参数化版本**
   - 都通过 `ARG CUDA_VERSION` / `ARG CANN_VERSION` 传递版本
   - 都支持多 Python 版本

2. **Matrix 策略构建**
   - 都使用 GitHub Actions matrix 并行构建
   - 都支持定时构建和手动触发

3. **Registry 推送**
   - 都推送到公共 Registry
   - 都生成多层级标签

4. **版本简化处理**
   - 都从完整版本提取简化版本
   - 都有 latest 标签策略

---

### 差异点 ⚠️

| 差异 | PyTorch 上游优势 | torch-npu 待改进 |
|------|-----------------|----------------|
| **文件组织** | CI/发布分离，公共脚本目录 | 单一 Dockerfile，建议拆分 |
| **版本管理** | 版本映射表，stable 标记 | 硬编码日期，建议映射表 |
| **版本锁定** | `ci_commit_pins/` 目录锁定所有依赖 | 无版本锁定文件 |
| **镜像类型** | runtime/devel 分离 | 单一镜像，可考虑分离 |
| **多 Registry** | ECR（私有）+ ghcr.io（公共） | 单一 quay.io |
| **依赖分离** | CUDA/NCCL/cuSPARSE 独立脚本 | CANN 组件混在一起 |

---

## 八、改进建议优先级

### P0（必须改进）

1. **创建版本锁定文件**
   ```
   .github/docker/cann_versions.txt
   .github/docker/nnal_versions.txt
   .github/docker/a3_ops_versions.txt
   ```

2. **创建版本映射表**
   ```bash
   # 在 build_image.sh 中
   declare -A CANN_VERSIONS=(
       ["9.0"]="9.0.0|20260330"
       ["8.0"]="8.0.RC3|20250101"
   )
   ```

---

### P1（建议改进）

1. **拆分安装脚本**
   ```
   .github/scripts/docker/
   ├── install_cann.sh
   ├── install_nnal.sh
   ├── install_a3_ops.sh
   └── common_utils.sh
   ```

2. **添加镜像类型区分**
   ```dockerfile
   FROM base as runtime   # 精简镜像
   FROM runtime as devel  # 完整镜像
   ```

3. **添加 stable 版本标记**
   ```bash
   CANN_STABLE="9.0"  # 用于生成 latest 标签
   ```

---

### P2（可选改进）

1. **多 Registry 支持**
   - AWS ECR（私有缓存）
   - ghcr.io（公共发布）

2. **CI/发布镜像分离**
   - CI 镜像：包含测试工具
   - 发布镜像：精简运行时

---

## 九、标签命名建议调整

### 当前 torch-npu 标签（保持）

```
✅ py3.11-cann9.0.0-beta.2          # 标准版
✅ py3.11-latest                     # Python latest
✅ latest                            # 全局 latest
```

### 建议新增标签

```
新增：py3.11-cann9.0-runtime        # 镜像类型标记
新增：py3.11-cann9.0-devel          # 开发镜像
新增：cann9.0-stable                # Stable 版本标记
新增：2.5.1-py3.11-cann9.0-runtime  # 包含 PyTorch 版本（可选）
```

---

## 十、代码示例：版本映射实现

### 建议在 build_image.sh 中添加

```bash
#!/bin/bash

# CANN 版本映射表（类似 PyTorch）
declare -A CANN_ARCHES=(
    ["9.0"]="9.0.0"
    ["8.0"]="8.0.RC3"
)

declare -A CANN_ARCHES_DATE=(
    ["9.0"]="20260330"
    ["8.0"]="20250101"
)

declare -A CANN_ARCHES_NNAL=(
    ["9.0"]="9.0.0"
    ["8.0"]="8.0"
)

CANN_STABLE="9.0"  # Stable 版本（用于 latest）

# 解析版本参数
parse_cann_version() {
    local input="$1"

    # 如果输入是简化版本（如 "9.0"），查找完整版本
    if [[ -v CANN_ARCHES[$input] ]]; then
        CANN_VERSION="${CANN_ARCHES[$input]}"
        CANN_DATE="${CANN_ARCHES_DATE[$input]}"
        NNAL_VERSION="${CANN_ARCHES_NNAL[$input]}"
        CANN_MAJOR="$input"
    else
        # 如果输入是完整版本（如 "9.0.0-beta.2"），提取简化版本
        CANN_VERSION="$input"
        CANN_MAJOR=$(echo "$input" | sed 's/-beta.*//' | sed 's/-rc.*//' | sed 's/\.[0-9]*$//')
        CANN_DATE="${CANN_ARCHES_DATE[$CANN_MAJOR]:-DEFAULT_CANN_DATE}"
        NNAL_VERSION="${CANN_ARCHES_NNAL[$CANN_MAJOR]:-$CANN_VERSION}"
    fi

    # 判断是否为 stable 版本
    IS_STABLE=$([[ "$CANN_MAJOR" == "$CANN_STABLE" ]] && echo "true" || echo "false")
}
```

---

## 结论

### 总体评价

✅ **相似度高**：torch-npu 的设计思路与 PyTorch 上游基本一致，都采用了参数化构建、Matrix 策略、多版本支持等现代 CI/CD 最佳实践。

⚠️ **待改进点**：
1. 版本管理缺乏映射表和锁定文件
2. 安装脚本未拆分，维护性较弱
3. 镜像类型未区分 runtime/devel
4. 缺少 stable 版本标记

### 下一步行动

**建议按照优先级顺序改进**：
1. P0：创建版本锁定文件和映射表
2. P1：拆分安装脚本，添加镜像类型
3. P2：考虑多 Registry 和 CI/发布分离

**标签命名**：
- 当前命名策略已经很好，符合 PyTorch 风格
- 可以考虑添加 runtime/devel 类型标记
- 建议添加 stable 版本的 latest 标签

---

**生成时间**: 2026-05-06
**对比版本**: PyTorch upstream main branch (2026-05-06)