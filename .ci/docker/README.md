# torch-npu CI Docker Images

本目录管理 torch-npu 项目的 CI Docker 镜像，包括**构建镜像 (builder)** 和**测试镜像 (test)** 两类，每类分别支持 x86_64 和 aarch64 架构。

当前支持两个版本：

| 版本目录 | PyTorch 版本 | 镜像基座 (x86_64) | 镜像基座 (aarch64) |
|---------|------------|-------------------|---------------------|
| `2.13/` | 2.13.0 | `pytorch/manylinux2_28-builder:cpu-v2.13.0-rc15` | `cpu-aarch64-v2.13.0-rc15` |
| `master/` | 2.14.0.dev20260708 (nightly) | `pytorch/manylinux2_28-builder:cpu` | `cpu-aarch64` |

## 镜像类型

| 类型 | 基座 | 用途 |
|------|------|------|
| **builder (x86_64)** | manylinux2_28-builder | 编译构建 torch-npu wheel 包，包含完整编译工具链 |
| **builder (aarch64)** | manylinux2_28_aarch64-builder | 编译构建 torch-npu wheel 包，包含完整编译工具链 |
| **test** | `ubuntu:22.04` | CI 单元测试运行环境，包含 PyTorch CPU、CANN runtime、triton-ascend 和测试框架 |

## 目录结构

```text
.ci/docker/
├── README.md                      # 本文档
├── docker_build.sh                # 构建入口脚本（区分 2.13/master）
├── requirements-builder.txt       # Builder 镜像公共 pip 依赖
├── common/                        # 公共共享脚本
│   ├── install_cann.sh            # 安装 CANN toolkit (支持 A1/A2/A3)
│   ├── install_triton.sh          # 安装 triton-ascend (需传 Python 版本)
│   └── install_obs.sh             # 安装华为 OBS util
├── 2.13/                          # v2.13.0 版本特定
│   ├── requirements-test.txt      # Test 镜像依赖 (torch 2.13.0)
│   ├── builder/
│   │   ├── Dockerfile.x86_64
│   │   └── Dockerfile.aarch64
│   └── test/
│       ├── Dockerfile.x86_64
│       └── Dockerfile.aarch64
└── master/                        # master (nightly) 版本特定
    ├── requirements-test.txt      # Test 镜像依赖 (torch nightly)
    ├── builder/
    │   ├── Dockerfile.x86_64
    │   └── Dockerfile.aarch64
    └── test/
        ├── Dockerfile.x86_64
        └── Dockerfile.aarch64
```

## 快速构建

```bash
# Builder 镜像 (不含 CANN)
./docker_build.sh torch-npu-builder-x86_64-torch2.13.0
./docker_build.sh torch-npu-builder-aarch64-torch2.13.0
./docker_build.sh torch-npu-builder-x86_64-torch-master
./docker_build.sh torch-npu-builder-aarch64-torch-master

# Test 镜像 (含 CANN)
./docker_build.sh torch-npu-test-x86_64-cann-a1-py3.10-torch2.13.0
./docker_build.sh torch-npu-test-aarch64-cann-a2-py3.10-torch2.13.0
./docker_build.sh torch-npu-test-x86_64-cann-a1-py3.10-torch-master
./docker_build.sh torch-npu-test-aarch64-cann-a2-py3.10-torch-master
```

## Tag 命名规范

参考上游 PyTorch `pytorch-linux-jammy-cuda12.4-cudnn9-py3-gcc11` 模式，tag 即为最终镜像名：

**Builder**（不含 CANN）：

```text
torch-npu-builder-<ARCH>-torch<PYTORCH_VERSION>
```

其中 `<PYTORCH_VERSION>` 可以是 `2.13.0` 或 `master`（nightly）。

**Test**（含 CANN runtime）：

```text
torch-npu-test-<ARCH>-cann<CHIP>-py<PYTHON_VERSION>-torch<PYTORCH_VERSION>
```

| 字段 | 可选值 |
|------|--------|
| IMAGE_TYPE | builder, test |
| ARCH | x86_64, aarch64 |
| CHIP | A1 (Ascend 910), A2 (Ascend 910b), A3 (仅 test) |
| PYTHON_VERSION | 3.10 (仅 test) |
| PYTORCH_VERSION | 2.13.0, master |

## Python 版本支持

Builder 镜像支持以下 Python 版本（由基座镜像提供）：

- Python 3.10 (cpython-3.10.20)
- Python 3.11 (cpython-3.11.15)
- Python 3.12 (cpython-3.12.13)
- Python 3.13 (cpython-3.13.14)
- Python 3.14 (cpython-3.14.6)

Test 镜像仅使用系统 Python 3.10。

## CANN 芯片映射

| CANN_CHIP | 芯片 | CANN 版本 |
|-----------|------|----------|
| A1 | Ascend 910 | 9.1.0-beta.3 |
| A2 | Ascend 910b | 9.1.0-beta.3 |
| A3 | Ascend A3 | 9.1.0-beta.3 |
