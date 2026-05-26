# Patch 批量应用脚本使用说明

## 目录结构

### 核心仓库

- PyTorch 源码仓库：需拉取官方 PyTorch 源码，并切换至 `tags/v2.12.0` 标签。
- 补丁目录：从 Ascend/pytorch 仓库中提取 `test_upstream` 目录。

### 核心目录结构

```text
pytorch/                    # PyTorch 源码根目录
├─ ...                      # 其他 PyTorch 原生文件/目录
└─ test_upstream/           # 补丁目录
   ├─ apply_patch.sh        # 批量应用脚本
   ├─ *.patch               # 补丁文件，支持子目录嵌套
   └─ ...                   # 其他补丁子目录
```

## 环境要求

仅需安装 Git。

## 使用方法

1. 将本仓库的 `test_upstream` 文件夹整体复制到本地 PyTorch 源码根目录中。
2. 运行脚本文件。

```bash
cd test_upstream
./apply_patch.sh
```

脚本会自动定位 PyTorch 根目录，递归扫描所有 `.patch` 文件，按文件名排序并强制应用；冲突部分会生成 `.rej` 文件。

## 注意事项

- 所有补丁仅适配 PyTorch `tags/v2.12.0`，其他版本可能导致应用失败，务必提前校验版本。
- `test_upstream` 目录需整体复制至 PyTorch 源码根目录。
- 生成 `.rej` 冲突文件时，需手动解决冲突后重新执行脚本。
