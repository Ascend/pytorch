# Patch 批量应用脚本使用说明

## 目录结构

1. 核心仓库地址
   - [官方 PyTorch 仓库（v2.7.1 版本）](https://github.com/pytorch/pytorch/tree/v2.7.1)，需拉取该仓库并切换至 tags/v2.7.1 标签。
   - [补丁仓库（Ascend/pytorch）](https://gitcode.com/Ascend/pytorch)，仅需提取该仓库中的 patch 目录。

2. 核心目录结构

```text
    pytorch/                  # PyTorch 源码根目录
    ├─ ...（其他 PyTorch 原生文件/目录）
    └─ test_upstream/                  # 补丁目录
       ├─ apply_test_patch.sh          # 源码测试 patch 批量应用脚本
       ├─ torch_env_patch.sh           # 环境 torch 包 patch 应用脚本
       ├─ test/                        # 源码测试用例 patch 文件
       ├─ torch/                       # 环境 torch 安装包 patch 文件
       └─ ...（其他补丁子目录）
```

## 脚本说明

本目录包含两类 patch，分别由不同脚本负责应用：

| 脚本 | patch 目录 | 应用目标 |
|------|-----------|---------|
| `apply_test_patch.sh` | `test/` | PyTorch 源码中的测试文件 |
| `torch_env_patch.sh` | `torch/` | Python 环境中安装的 torch 包 |

两个脚本相互独立，互不干扰。

## 环境要求

- `apply_test_patch.sh`：仅需安装 git
- `torch_env_patch.sh`：需要 Python 环境已安装 torch 包

## 使用方法

### apply_test_patch.sh —— 源码测试 patch

将 `test/` 目录下的 patch 应用到 PyTorch 源码中的测试文件。

1. 将本仓库的 test_upstream 文件夹整体复制到 PyTorch 官方仓库中
2. 运行脚本：

```bash
cd test_upstream
./apply_test_patch.sh
```

脚本自动定位 PyTorch 根目录，递归扫描 `test/` 目录下所有 .patch/.diff 文件，按文件名排序强制应用，冲突部分生成 .rej 文件。

### torch_env_patch.sh —— 环境 torch 包 patch

将 `torch/` 目录下的 patch 应用到 Python 环境中已安装的 torch 包（如 site-packages/torch）。

1. 确保 Python 环境已安装 torch 包
2. 运行脚本：

```bash
cd test_upstream
./torch_env_patch.sh [--python=<version>]
```

脚本自动定位 torch 包的安装路径，将 `torch/` 目录下的 patch 应用到对应文件。

## 注意事项

- 所有补丁仅适配 PyTorch tags/v2.7.1，其他版本将导致应用失败，务必提前校验版本。
- test_upstream 目录需整体复制至 PyTorch 根目录。
- `apply_test_patch.sh` 仅应用 `test/` 下的 patch 到源码，不会触碰 `torch/` 目录。
- `torch_env_patch.sh` 仅应用 `torch/` 下的 patch 到安装后的 torch 包，不会触碰 `test/` 目录。
- 生成 .rej 冲突文件时，需手动解决冲突后重新执行脚本。
