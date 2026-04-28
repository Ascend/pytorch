# Torch-NPU 编译和上游 PyTorch 测试用例拉起流程设计

## Context

设计一个完整的 CI/CD 流程，用于：
1. 拉取上游 PyTorch main 分支代码并编译
2. 安装上游 PyTorch wheel 包
3. 编译 torch-npu
4. 扫描 PyTorch test 目录下的所有测试用例
5. 每个用例独立进程执行全量测试
6. 记录测试结果并上传日志制品

参考 PR #117 (https://github.com/Ascend/pytorch/pull/117) 的实现方案，简化 patch 和白名单逻辑。

**关键改进**：采用上游 PyTorch 社区的最佳实践 —— **构建和测试使用同一个 Docker 镜像 URL**，通过 Job 依赖传递机制保证环境绝对一致。

---

## 方案确认

| 项目 | 选择 |
|------|------|
| 基础镜像 | `pytorch/manylinux-builder:aarch64` |
| 镜像托管 | GitHub Container Registry (ghcr.io) |
| CANN安装 | 单独镜像构建流程，定期构建推送 |
| Runner规格 | 统一使用 `linux-aarch64-a3-16` |
| 镜像标签 | 带时间戳版本号，如 `manylinux-cann9.0-20260428` |
| Dockerfile位置 | 项目仓库 `.github/docker/` 目录 |

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GitHub Actions Workflow                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │           阶段1: 镜像构建 (单独 workflow，定期运行)                    │   │
│  │  Workflow: build-docker-image.yml                                    │   │
│  │  Runner: linux-aarch64-a3-16                                         │   │
│  │  基础镜像: pytorch/manylinux-builder:aarch64                         │   │
│  │  安装 CANN 9.0.0-beta.2                                               │   │
│  │  推送: ghcr.io/${{ github.repository_owner }}/pytorch-npu-builder    │   │
│  │  标签: manylinux-cann9.0-${{ timestamp }}                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              │ 镜像已预构建                                  │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │           阶段2: 统一构建 (PyTorch + torch_npu)                       │   │
│  │  Workflow: _build.yml                                                │   │
│  │  Runner: linux-aarch64-a3-16                                         │   │
│  │  Container: ghcr.io/xxx/pytorch-npu-builder:manylinux-cann9.0-xxx    │   │
│  │  - Clone 上游 PyTorch main                                           │   │
│  │  - 编译 PyTorch wheel                                                │   │
│  │  - Checkout torch_npu                                                │   │
│  │  - 编译 torch_npu wheel                                              │   │
│  │  - 打包测试源码                                                      │   │
│  │  Outputs: docker-image, torch-wheel, torch-npu-wheel, test-src       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              │ docker-image URL 传递                        │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │           阶段3: 用例收集                                             │   │
│  │  Job: collect_cases                                                  │   │
│  │  Runner: linux-aarch64-a3-16                                         │   │
│  │  Container: SAME Docker 镜像                                         │   │
│  │  - 安装 torch + torch_npu                                            │   │
│  │  - pytest --collect-only 收集所有用例                                │   │
│  │  - 按用例分片 (distributed/regular)                                  │   │
│  │  Outputs: distributed_matrix, regular_matrix, total_cases            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│              ┌───────────────┴───────────────┐                              │
│              │ docker-image URL 传递         │                              │
│              ▼                               ▼                              │
│  ┌─────────────────────────┐   ┌─────────────────────────────────┐         │
│  │  阶段4: Distributed测试 │   │      阶段4: Regular测试          │         │
│  │  Runner: a3-16          │   │      Runner: a3-16               │         │
│  │  Container: SAME镜像    │   │      Container: SAME镜像         │         │
│  │  串行执行               │   │      并发执行 (32 workers)       │         │
│  │  2 shards               │   │      5 shards                    │         │
│  └─────────────────────────┘   └─────────────────────────────────┘         │
│              │                               │                              │
│              └───────────────┬───────────────┘                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │           阶段5: 报告生成                                             │   │
│  │  - 汇总所有 shard 结果                                               │   │
│  │  - 生成 Markdown + JSON 报告                                        │   │
│  │  - 压缩上传制品                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心设计：Docker 镜像传递机制

### 上游 PyTorch 社区的做法

在 PyTorch 的 CI 中，Build Job 和 Test Job 使用**同一个 Docker 镜像**：

```yaml
# Build Job 计算并输出完整的镜像 URL
linux-build:
  outputs:
    docker-image: ${{ steps.calculate-docker-image.outputs.docker-image }}

# Test Job 通过 needs 获取同一个镜像
linux-test:
  needs: linux-build
  container:
    image: ${{ needs.linux-build.outputs.docker-image }}
```

这种做法的优势：
1. **环境绝对一致** - 构建和测试使用同一镜像
2. **避免重复安装** - 镜像中预装依赖
3. **减少 artifact 传递** - 直接挂载构建产物
4. **可追溯性** - 镜像 URL 带 SHA256 哈希，确保版本锁定

---

## 文件结构

```
.github/
├── docker/
│   └── pytorch-npu-builder.Dockerfile   # 基于 manylinux + CANN 的构建镜像
│
├── workflows/
│   ├── build-docker-image.yml           # 镜像构建 workflow (定期运行)
│   ├── npu-full-test.yml                # 主 workflow
│   ├── _build.yml                       # 统一构建 workflow (可复用)
│   └── _test.yml                        # 测试执行 workflow (可复用)
│
├── scripts/
│   ├── collect_all_cases.py             # 收集所有测试用例
│   ├── run_npu_test_shard.py            # 执行测试分片
│   ├── parse_test_results.py            # 解析测试结果
│   └── generate_report.py               # 生成汇总报告
```

---

## 阶段1: Docker 镜像构建

### Dockerfile: pytorch-npu-builder.Dockerfile

```dockerfile
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
RUN echo '#!/bin/bash\n\
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true\n\
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true' > /etc/profile.d/cann_env.sh && \
    chmod +x /etc/profile.d/cann_env.sh

# 预安装 pytest 等测试依赖
RUN pip3.11 install pytest pytest-timeout pytest-xdist hypothesis pyyaml zstandard
```

### Workflow: build-docker-image.yml

```yaml
name: Build Docker Image

on:
  schedule:
    - cron: '0 2 * * 0'  # UTC 02:00, Beijing 10:00, every Sunday
  workflow_dispatch:
    inputs:
      cann_version:
        description: 'CANN version'
        default: '9.0.0-beta.2'
      force_build:
        description: 'Force rebuild even if image exists'
        default: false
        type: boolean

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: pytorch-npu-builder

jobs:
  build:
    runs-on: linux-aarch64-a3-16
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Generate timestamp tag
        id: tag
        run: |
          TIMESTAMP=$(date +%Y%m%d)
          echo "tag=manylinux-cann${{ inputs.cann_version || '9.0.0-beta.2' }}-${TIMESTAMP}" >> $GITHUB_OUTPUT

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .github/docker
          file: .github/docker/pytorch-npu-builder.Dockerfile
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:${{ steps.tag.outputs.tag }}
            ${{ env.REGISTRY }}/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Output image URL
        run: |
          echo "Built image: ${{ env.REGISTRY }}/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:${{ steps.tag.outputs.tag }}"
```

---

## 阶段2: 统一构建 (_build.yml)

```yaml
name: Build PyTorch and torch_npu

on:
  workflow_call:
    inputs:
      pytorch_branch:
        required: true
        type: string
        default: 'main'
      python_version:
        required: true
        type: string
        default: '3.11'
      docker_image_tag:
        required: true
        type: string
        description: 'Docker image tag with timestamp'
    outputs:
      docker-image:
        description: 'Full Docker image URL'
        value: ${{ jobs.build.outputs.docker-image }}
      torch-wheel:
        description: 'PyTorch wheel artifact name'
        value: 'torch-wheel-main'
      torch-npu-wheel:
        description: 'torch_npu wheel artifact name'
        value: 'torch-npu-wheel-main'
      test-src:
        description: 'Test source artifact name'
        value: 'test-src-main'

env:
  REGISTRY: ghcr.io

jobs:
  build:
    runs-on: linux-aarch64-a3-16
    outputs:
      docker-image: ${{ steps.set_image.outputs.docker-image }}
    
    container:
      image: ${{ env.REGISTRY }}/${{ github.repository_owner }}/pytorch-npu-builder:${{ inputs.docker_image_tag }}
      options: --user root
    
    steps:
      - name: Set Docker image URL
        id: set_image
        run: |
          DOCKER_IMAGE="${{ env.REGISTRY }}/${{ github.repository_owner }}/pytorch-npu-builder:${{ inputs.docker_image_tag }}"
          echo "docker-image=${DOCKER_IMAGE}" >> $GITHUB_OUTPUT
          echo "Using Docker image: ${DOCKER_IMAGE}"

      - name: Setup CANN environment
        run: |
          source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
          source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

      - name: Clone upstream PyTorch main
        run: |
          git clone --depth=1 --branch ${{ inputs.pytorch_branch }} \
            https://github.com/pytorch/pytorch.git pytorch-src
          PYTORCH_SHA=$(cd pytorch-src && git rev-parse HEAD)
          echo "pytorch_sha=${PYTORCH_SHA}"

      - name: Checkout torch_npu
        uses: actions/checkout@v4
        with:
          path: torch_npu-src
          submodules: recursive

      - name: Build PyTorch wheel
        run: |
          source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
          
          cd pytorch-src
          pip${{ inputs.python_version }} install pyyaml setuptools wheel cmake ninja
          
          export MAX_JOBS=40
          export USE_CUDA=0
          export USE_CUDNN=0
          export CMAKE_BUILD_TYPE=Release
          
          python${{ inputs.python_version }} setup.py build bdist_wheel
          
          echo "PyTorch wheel built:"
          ls -la dist/

      - name: Build torch_npu wheel
        run: |
          source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
          source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
          
          # 安装刚编译的 PyTorch
          pip${{ inputs.python_version }} install pytorch-src/dist/*.whl
          
          cd torch_npu-src
          pip${{ inputs.python_version }} install pyyaml setuptools wheel cmake ninja
          
          export MAX_JOBS=40
          bash ci/build.sh --python=${{ inputs.python_version }}
          
          echo "torch_npu wheel built:"
          ls -la dist/

      - name: Package test source
        run: |
          tar -czf test-src.tar.gz pytorch-src
          ls -la test-src.tar.gz

      - name: Upload PyTorch wheel
        uses: actions/upload-artifact@v4
        with:
          name: torch-wheel-main
          path: pytorch-src/dist/*.whl
          retention-days: 7

      - name: Upload torch_npu wheel
        uses: actions/upload-artifact@v4
        with:
          name: torch-npu-wheel-main
          path: torch_npu-src/dist/*.whl
          retention-days: 7

      - name: Upload test source
        uses: actions/upload-artifact@v4
        with:
          name: test-src-main
          path: test-src.tar.gz
          retention-days: 7
```

---

## 阶段3: 用例收集 (collect_cases Job)

```yaml
collect_cases:
  needs: build
  runs-on: linux-aarch64-a3-16
  timeout-minutes: 60
  
  # 使用 Build Job 输出的同一个 Docker 镜像
  container:
    image: ${{ needs.build.outputs.docker-image }}
    options: --user root
  
  outputs:
    distributed_matrix: ${{ steps.collect.outputs.distributed_matrix }}
    regular_matrix: ${{ steps.collect.outputs.regular_matrix }}
    total_cases: ${{ steps.collect.outputs.total_cases }}

  steps:
    - name: Checkout scripts
      uses: actions/checkout@v4
      with:
        sparse-checkout: .github/scripts

    - name: Download wheels
      uses: actions/download-artifact@v4
      with:
        name: torch-wheel-main
        path: wheels

    - name: Download torch_npu wheel
      uses: actions/download-artifact@v4
      with:
        name: torch-npu-wheel-main
        path: wheels

    - name: Download test source
      uses: actions/download-artifact@v4
      with:
        name: test-src-main

    - name: Extract test source
      run: tar -xzf test-src.tar.gz

    - name: Install wheels
      run: |
        source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
        pip3.11 install wheels/*.whl

    - name: Collect all test cases
      id: collect
      run: |
        source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
        cd pytorch-src
        
        python3.11 ../.github/scripts/collect_all_cases.py \
          --test-dir test \
          --distributed-shards 2 \
          --regular-shards 5 \
          --output-dir cases_shards \
          --parallel 16
        
        TOTAL=$(python3 -c "import json; d=json.load(open('cases_shards/cases_collection_summary.json')); print(d['total_cases'])")
        echo "total_cases=${TOTAL}" >> $GITHUB_OUTPUT
        
        echo "distributed_matrix=[1,2]" >> $GITHUB_OUTPUT
        echo "regular_matrix=[1,2,3,4,5]" >> $GITHUB_OUTPUT

    - name: Upload cases shard JSONs
      uses: actions/upload-artifact@v4
      with:
        name: cases-shards
        path: pytorch-src/cases_shards/
```

---

## 阶段4: 测试执行 (_test.yml)

```yaml
name: Run NPU Tests

on:
  workflow_call:
    inputs:
      docker-image:
        required: true
        type: string
      test-type:
        required: true
        type: string
      shard-index:
        required: true
        type: number
      max-workers:
        required: false
        type: number
        default: 1
      timeout:
        required: false
        type: number
        default: 1200

jobs:
  test:
    runs-on: linux-aarch64-a3-16
    timeout-minutes: 1200
    
    # 使用 Build Job 输出的同一个 Docker 镜像
    container:
      image: ${{ inputs.docker-image }}
      options: --user root

    steps:
      - name: Checkout scripts
        uses: actions/checkout@v4
        with:
          sparse-checkout: .github/scripts

      - name: Download torch wheel
        uses: actions/download-artifact@v4
        with:
          name: torch-wheel-main
          path: wheels

      - name: Download torch_npu wheel
        uses: actions/download-artifact@v4
        with:
          name: torch-npu-wheel-main
          path: wheels

      - name: Download test source
        uses: actions/download-artifact@v4
        with:
          name: test-src-main

      - name: Download cases shard
        uses: actions/download-artifact@v4
        with:
          name: cases-shards
          path: cases-shards

      - name: Extract test source
        run: tar -xzf test-src.tar.gz

      - name: Install wheels
        run: |
          source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
          pip3.11 install wheels/*.whl

      - name: Run ${{ inputs.test-type }} shard ${{ inputs.shard-index }}
        run: |
          source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
          source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
          
          python .github/scripts/run_npu_test_shard.py \
            --cases-json cases-shards/${{ inputs.test-type }}_cases_shard_${{ inputs.shard-index }}.json \
            --test-dir pytorch-src/test \
            --report-dir test-reports \
            --timeout ${{ inputs.timeout }} \
            --max-workers ${{ inputs.max-workers }} \
            --verbose

      - name: Upload test reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-reports-${{ inputs.test-type }}-${{ inputs.shard-index }}
          path: test-reports/
          retention-days: 30
```

---

## 主 Workflow: npu-full-test.yml

```yaml
name: PyTorch NPU Full Test

on:
  push:
    branches: [main, master]
    paths:
      - '.github/workflows/**'
      - '.github/scripts/**'
      - '.github/docker/**'
  schedule:
    - cron: '0 22 * * 1'  # UTC 22:00, Beijing 06:00, every Monday
  workflow_dispatch:
    inputs:
      docker_image_tag:
        description: 'Docker image tag (e.g., manylinux-cann9.0-20260428)'
        default: 'latest'
      pytorch_branch:
        description: 'PyTorch branch to build'
        default: 'main'
      distributed_shards:
        description: 'Distributed test shards'
        default: '2'
      regular_shards:
        description: 'Regular test shards'
        default: '5'

env:
  REGISTRY: ghcr.io
  PYTHON_VERSION: '3.11'

jobs:
  build:
    uses: ./.github/workflows/_build.yml
    with:
      pytorch_branch: ${{ inputs.pytorch_branch || 'main' }}
      python_version: '3.11'
      docker_image_tag: ${{ inputs.docker_image_tag || 'latest' }}

  collect_cases:
    needs: build
    uses: ./.github/workflows/_collect.yml
    with:
      docker-image: ${{ needs.build.outputs.docker-image }}

  test_distributed:
    needs: 
      - build
      - collect_cases
    strategy:
      matrix:
        shard: ${{ fromJson(needs.collect_cases.outputs.distributed_matrix) }}
      fail-fast: false
      max-parallel: 2
    uses: ./.github/workflows/_test.yml
    with:
      docker-image: ${{ needs.build.outputs.docker-image }}
      test-type: distributed
      shard-index: ${{ matrix.shard }}
      max-workers: 1  # Distributed 串行执行
      timeout: 1200

  test_regular:
    needs: 
      - build
      - collect_cases
    strategy:
      matrix:
        shard: ${{ fromJson(needs.collect_cases.outputs.regular_matrix) }}
      fail-fast: false
      max-parallel: 5
    uses: ./.github/workflows/_test.yml
    with:
      docker-image: ${{ needs.build.outputs.docker-image }}
      test-type: regular
      shard-index: ${{ matrix.shard }}
      max-workers: 32  # Regular 并发执行
      timeout: 1200

  report:
    needs: 
      - test_distributed
      - test_regular
    runs-on: ubuntu-latest
    if: always()
    
    steps:
      - name: Checkout scripts
        uses: actions/checkout@v4
        with:
          sparse-checkout: .github/scripts

      - name: Download all test reports
        uses: actions/download-artifact@v4
        with:
          pattern: test-reports-*
          path: all-reports
          merge-multiple: false

      - name: Generate consolidated report
        run: |
          python .github/scripts/generate_report.py \
            --reports-root all-reports \
            --output-markdown report.md \
            --output-json report.json

      - name: Upload final report
        uses: actions/upload-artifact@v4
        with:
          name: npu-full-test-report
          path: |
            report.md
            report.json
          retention-days: 30

      - name: Package all logs
        run: tar -czf all-test-logs.tar.gz all-reports/

      - name: Upload logs artifact
        uses: actions/upload-artifact@v4
        with:
          name: all-test-logs
          path: all-test-logs.tar.gz
          retention-days: 30
```

---

## 关键脚本设计

### collect_all_cases.py (用例收集)

核心功能：
1. 扫描 test 目录下所有 test_*.py 文件
2. 分类为 distributed/regular 类型
3. 通过 pytest --collect-only 收集每个文件的用例
4. 将用例均分到 shards
5. 保存 shard JSON 文件

关键实现参考 PR #117 的 collect_all_cases.py。

### run_npu_test_shard.py (测试执行)

核心功能：
1. 加载 shard JSON 中的用例列表
2. 每个用例启动独立 pytest subprocess (崩溃隔离)
3. 并发执行 (ThreadPoolExecutor，max_workers 参数控制)
4. 收集结果：passed/failed/error/crashed/timeout
5. 保存每个用例的结果 JSON

关键实现参考 PR #117 的 run_npu_test_shard.py。

### generate_report.py (报告生成)

核心功能：
1. 读取所有 shard 的 cases JSON
2. 统计汇总：总用例数、passed/failed/crashed/timeout
3. 生成 Markdown 报告表格
4. 生成 JSON 详细报告

---

## 关键设计要点

### 1. Docker 镜像传递机制

与上游 PyTorch 一致的做法：

```yaml
# Build Job 输出镜像 URL
outputs:
  docker-image: ${{ steps.set_image.outputs.docker-image }}

# Test Job 使用同一个镜像
container:
  image: ${{ needs.build.outputs.docker-image }}
```

优势：
- 构建和测试环境完全一致
- 避免 artifact 传递导致的环境差异
- 镜像标签带时间戳，便于追溯

### 2. CANN 环境初始化

在 Dockerfile 中预装 CANN，并在镜像中添加环境初始化脚本：

```bash
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
```

每个 job 开始时调用此脚本确保环境正确。

### 3. 用例级分片

- pytest --collect-only 预收集所有用例
- 按用例数量均分到 shards，实现负载均衡
- 遏制某些 shard 因包含大文件而过载

### 4. 每用例独立进程执行

- 每个 pytest case 在独立 subprocess 中运行
- 崩溃隔离：一个 case crash 不影响其他 case
- 支持超时控制
- 结果独立记录

### 5. Distributed vs Regular 区分

- Distributed: 需要 HCCL 分布式环境，串行执行 (max-workers=1)
- Regular: 单机测试，并发执行 (max-workers=32)

### 6. PYTHONPATH 处理

某些测试文件导入同级模块：
- 执行前动态添加测试文件父目录到 PYTHONPATH

### 7. Runner 统一使用 a3-16

- Build、Collect、Test 都使用 linux-aarch64-a3-16
- 简化配置，统一环境
- 16-card NPU 支持分布式测试

---

## 验证方案

### 步骤1: 验证 Docker 镜像构建

```bash
# 本地构建测试
cd .github/docker
docker build -t pytorch-npu-builder:test -f pytorch-npu-builder.Dockerfile .

# 验证 CANN 安装
docker run --rm pytorch-npu-builder:test \
  bash -c "source /usr/local/Ascend/cann/set_env.sh && echo CANN installed"
```

### 步骤2: 验证构建流程

触发 workflow_dispatch，使用少量配置：
- docker_image_tag: latest (或手动构建的标签)
- pytorch_branch: main
- distributed_shards: 1
- regular_shards: 1

### 步骤3: 验证用例收集和执行

检查 collect_cases job 输出：
- cases_collection_summary.json 内容
- shard JSON 文件数量和内容

检查 test job 输出：
- cases JSON 结果统计
- 失败用例日志

### 步骤4: 验证报告生成

检查 report job 输出：
- report.md Markdown 报告
- report.json 详细数据
- all-test-logs.tar.gz 日志制品

---

## 预估工作量

| 任务 | 预估时间 |
|------|----------|
| 创建 Dockerfile | 1 小时 |
| 创建 build-docker-image.yml | 1 小时 |
| 创建 _build.yml | 2 小时 |
| 创建 _test.yml | 2 小时 |
| 创建 npu-full-test.yml | 1 小时 |
| 开发 collect_all_cases.py | 3 小时 |
| 开发 run_npu_test_shard.py | 4 小时 |
| 开发 generate_report.py | 2 小时 |
| 本地镜像构建验证 | 2 小时 |
| CI 调试和优化 | 4 小时 |
| **总计** | **~20 小时** |

---

## 待确认事项

1. **镜像仓库权限**: ghcr.io 推送需要 `packages: write` 权限，确认仓库已启用
2. **Runner 访问**: linux-aarch64-a3-16 runner 确认可用
3. **CANN 版本**: 当前使用 9.0.0-beta.2，是否需要支持多版本
4. **镜像更新频率**: 每周构建一次是否合适，或需要手动触发机制