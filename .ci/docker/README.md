# torch-npu CI Docker Images

本目录管理 torch-npu 项目的 CI Docker 镜像，包括**构建镜像 (builder)** 和**测试镜像 (test)** 两类，每类分别支持 x86_64 和 aarch64 架构。

## 镜像类型

| 类型 | 基座 | 用途 |
|------|------|------|
| **builder** | manylinux2_28-builder | 编译构建 torch-npu wheel 包，包含完整编译工具链 |
| **test** | ubuntu:22.04 | CI 单元测试运行环境，包含 PyTorch CPU、CANN runtime、triton-ascend 和测试框架 |

## 目录结构

```
.ci/docker/
├── README.md
├── requirements-builder.txt      # Builder 镜像 pip 依赖
├── requirements-test.txt         # Test 镜像 pip 依赖
├── docker_build.sh               # 构建入口脚本
├── common/                       # 共享安装脚本
│   ├── install_cann.sh           # 安装 CANN toolkit (支持 A1/A2/A3)
│   ├── install_triton.sh         # 安装 triton-ascend (需传 Python 版本)
│   ├── install_obs.sh            # 安装华为 OBS util
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
./docker_build.sh torch-npu-builder-x86_64-py2.7.1
./docker_build.sh torch-npu-builder-aarch64-py2.7.1

# Test 镜像 (含 CANN)
./docker_build.sh torch-npu-test-x86_64-cann-a1-py2.7.1
./docker_build.sh torch-npu-test-aarch64-cann-a2-py2.7.1
```

## Tag 命名规范

参考上游 PyTorch `pytorch-linux-jammy-cuda12.4-cudnn9-py3-gcc11` 模式，tag 即为最终镜像名：

**Builder**（不含 CANN）：
```
torch-npu-builder-<ARCH>-py<PYTORCH_VERSION>
```
```
./docker_build.sh torch-npu-builder-x86_64-py2.7.1
#                   ^          ^       ^    ^
#                   |          |       |    └── PyTorch 版本 (py2.7.1)
#                   |          |       └── 架构
#                   |          └── 镜像类型
#                   └── 固定前缀
```

**Test**（含 CANN runtime）：
```
torch-npu-test-<ARCH>-cann<CHIP>-py<PYTORCH_VERSION>
```
```
./docker_build.sh torch-npu-test-x86_64-cann-a1-py2.7.1
#                   ^         ^       ^    ^ ^    ^
#                   |         |       |    | |    └── PyTorch 版本
#                   |         |       |    | └── py 前缀
#                   |         |       |    └── CANN 芯片 (A1/A2/A3)
#                   |         |       └── cann 前缀
#                   |         └── 架构
#                   └── 镜像类型
```

| 字段 | 可选值 |
|------|--------|
| IMAGE_TYPE | builder, test |
| ARCH | x86_64, aarch64 |
| CHIP | A1 (Ascend 910), A2 (Ascend 910b), A3 (仅 test) |
| PYTORCH_VERSION | 2.7.1 |

## CANN 芯片映射

| CANN_CHIP | 芯片 | CANN 版本 |
|-----------|------|----------|
| A1 | Ascend 910 | 9.1.0 |
| A2 | Ascend 910b | 8.5.0 (x86_64) / 9.1.0 (aarch64) |
| A3 | Ascend A3 | 9.0.0-beta.1 (x86_64) / 9.0.0-beta.2 (aarch64) |
