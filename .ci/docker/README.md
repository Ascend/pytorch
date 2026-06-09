# torch-npu CI Docker Images

本目录管理 torch-npu 项目的测试镜像 (test)，当前仅支持 aarch64 架构。

## 镜像类型

| 类型 | 基座 | 用途 |
|------|------|------|
| **test** | ubuntu:22.04 | CI 单元测试运行环境，包含 PyTorch nightly、CANN runtime、triton-ascend 和测试框架 |

## 目录结构

```
.ci/docker/
├── README.md
├── requirements-test.txt         # Test 镜像 pip 依赖
├── docker_build.sh               # 构建入口脚本
├── common/                       # 共享安装脚本
│   ├── install_cann.sh           # 安装 CANN toolkit (支持 A2/A3)
│   ├── install_triton.sh         # 安装 triton-ascend
│   └── install_obs.sh            # 安装华为 OBS util
└── test/
    └── Dockerfile.aarch64
```

## 快速构建

```bash
./docker_build.sh torch-npu-test-aarch64-cann-a2-py3.10-torch-nightly
./docker_build.sh torch-npu-test-aarch64-cann-a3-py3.10-torch-nightly
```

## Tag 命名规范

```
torch-npu-test-aarch64-cann<CHIP>-py<PYTHON_VERSION>-torch-nightly
```

| 字段 | 可选值 |
|------|--------|
| CHIP | A2 (Ascend 910b), A3 |
| PYTHON_VERSION | 3.10 |

## CANN 芯片映射

| CANN_CHIP | 芯片 | CANN 版本 |
|-----------|------|----------|
| A2 | Ascend 910b | 9.1.0-beta.1 |
| A3 | Ascend A3 | 9.0.0-beta.2 |
