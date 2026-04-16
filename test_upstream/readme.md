# Patch 批量应用脚本使用说明

## 目录结构

1. 核心仓库地址

- 官方 PyTorch 仓库（v2.7.1 版本）：https://github.com/pytorch/pytorch/tree/v2.7.1，需拉取该仓库并切换至 tags/v2.7.1 标签。
- 补丁仓库（Ascend/pytorch）：https://gitcode.com/Ascend/pytorch，仅需提取该仓库中的 patch 目录。

2. 核心目录结构

```coldFusion
    pytorch/                  # PyTorch 源码根目录
    ├─ ...（其他 PyTorch 原生文件/目录）
    └─ test_upstream/                 # 补丁目录
       ├─ apply_patches.sh            # 批量应用脚本
       ├─ *.patch                     # 补丁文件（支持子目录嵌套）
       ├─ ...（其他补丁子目录）
```

## 环境要求

仅需安装git即可

## 使用方法

1. 将本仓库的test_upstream文件夹整体复制到本地的PyTorch官方仓库中

2. 运行脚本文件

```bash
cd test_upstream
./apply_patches.sh
```

脚本执行说明：自动定位 PyTorch 根目录，递归扫描所有 .patch文件，按文件名排序强制应用，冲突部分生成 .rej 文件.

## 注意事项

- 所有补丁仅适配 PyTorch tags/v2.7.1，其他版本将导致应用失败，务必提前校验版本。
- test_upstream 目录需整体复制至 PyTorch 根目录。
- 生成 .rej 冲突文件时，需手动解决冲突后重新执行脚本。