# Docker 镜像构建脚本使用指南

## 概述

`build_image.sh` 脚本用于构建支持多 Python 版本的 PyTorch NPU Docker 镜像。

## 核心设计理念

### CANN 与 Python 版本关系

**重要说明**：
- CANN 包安装只需要 Python 3 环境，**不与特定 Python 版本绑定**
- 编译和运行 PyTorch 时，切换 Python 版本不会影响 CANN
- 因此镜像只需按 CANN 版本构建，无需按 Python 版本重复构建

### 镜像特性

1. **一个镜像支持所有 Python 版本**
   - 预装 Python 3.10/3.11/3.12/3.13
   - 通过环境变量或脚本切换 Python 版本

2. **按 CANN 版本构建**
   - 一个 CANN 版本对应一个镜像
   - 维护版本映射表，自动获取下载地址

3. **镜像标签简化**
   - 标签只显示 CANN 版本
   - 不再包含 Python 版本信息

---

## 支持的配置

### CANN 版本映射表

脚本维护以下版本映射（三个包的下载 URL）：

| 版本号 | Toolkit | A3-ops | NNAL |
|--------|---------|--------|------|
| `9.0` | toolkit_9.0.0 | A3-ops_9.0.0 | nnal_9.0.0 |
| `9.0.0-beta.2` | toolkit_9.0.0-beta.2 | A3-ops_9.0.0-beta.2 | nnal_9.0.0-beta.2 |
| `8.0` | toolkit_8.0.RC3 | A3-ops_8.0.RC3 | nnal_8.0.RC3 |

**Stable 版本标记**：
- `CANN_STABLE="9.0"` - 用于生成 `latest` 标签

---

## 使用方式

### 查看支持的 CANN 版本

```bash
./build_image.sh --list-versions
```

输出：
```
支持的 CANN 版本：

  - 9.0
  - 9.0.0-beta.2
  - 8.0

Stable 版本（用于 latest 标签）: 9.0
```

### 本地构建

```bash
# 使用简化版本号（推荐）
./build_image.sh --cann-version 9.0

# 使用完整版本号
./build_image.sh --cann-version 9.0.0-beta.2

# 查看详细日志
./build_image.sh --cann-version 9.0 --verbose
```

### 推送镜像

```bash
# 需要设置环境变量
export QUAY_USERNAME="your_username"
export QUAY_PASSWORD="your_password"

# 构建并推送
./build_image.sh --cann-version 9.0 --push
```

---

## 镜像使用指南

### 拉取镜像

```bash
# 拉取指定 CANN 版本
docker pull quay.io/kerer/pytorch:cann9.0

# 拉取 latest（stable 版本）
docker pull quay.io/kerer/pytorch:latest
```

### 运行容器

```bash
# 启动容器（默认 Python 3.11）
docker run -it quay.io/kerer/pytorch:cann9.0 bash

# 启动容器并挂载工作目录
docker run -it -v $(pwd):/workspace quay.io/kerer/pytorch:cann9.0 bash
```

### 切换 Python 版本

**方法 1：使用切换脚本**
```bash
# 在容器内执行
source /usr/local/bin/switch_python.sh 3.11
source /usr/local/bin/switch_python.sh 3.12
source /usr/local/bin/switch_python.sh 3.13
```

**方法 2：修改环境变量**
```bash
# Python 3.11
export PATH=/opt/python/cp311-cp311/bin:$PATH

# Python 3.12
export PATH=/opt/python/cp312-cp312/bin:$PATH

# Python 3.13
export PATH=/opt/python/cp313-cp313/bin:$PATH
```

**验证 Python 版本**
```bash
python --version
pip --version
```

### 初始化 CANN 环境

```bash
# 在容器内执行
source /etc/profile.d/cann_env.sh
```

---

## 镜像标签说明

### 标签层级

每个 CANN 版本生成以下标签：

| 标签类型 | 格式 | 示例 | 用途 |
|---------|------|------|------|
| **完整版**（带时间戳） | `cann${VERSION}-${TIMESTAMP}` | `cann9.0-20260506` | 版本追溯 |
| **标准版** | `cann${VERSION}` | `cann9.0.0-beta.2` | 日常使用 ⭐ |
| **简化版** | `cann${MAJOR}` | `cann9.0` | 快速识别 ⭐ |
| **latest**（仅 stable） | `latest` | `latest` | 使用最新 |

### Stable 版本额外标签

CANN stable 版本（当前为 9.0）额外生成：
- `latest` - 全局最新
- `cann-latest` - CANN 最新
- `cann9.0-latest` - 该 CANN 版本最新

---

## Workflow 使用

### 手动触发构建

1. 进入 GitHub Actions 页面
2. 选择 "Build Docker Image" workflow
3. 点击 "Run workflow"
4. 选择参数：
   - `cann_version`: 输入 CANN 版本（如 `9.0` 或 `9.0.0-beta.2`）
   - `push_image`: 是否推送镜像
   - `force_build`: 是否强制构建

### 自动触发

- **Push 触发**: 当修改相关文件时自动触发（默认构建 stable 版本）
- **定时触发**: 每周日凌晨 2:00 UTC 自动构建 stable 版本

---

## 添加新的 CANN 版本

### 步骤 1：更新版本映射表

在 `build_image.sh` 中添加新版本：

```bash
declare -A CANN_VERSIONS=(
    # 已有版本...

    # 新增版本
    ["9.1"]="https://...toolkit_9.1.0_linux-aarch64.run|https://...A3-ops_9.1.0_linux-aarch64.run|https://...nnal_9.1.0_linux-aarch64.run"
)
```

格式：`"版本号"="toolkit_url|a3_ops_url|nnal_url"`

### 步骤 2：更新 Stable 版本（可选）

如果新版本成为 stable，更新：

```bash
CANN_STABLE="9.1"
```

---

## 与 PyTorch 上游对比

### 关键差异

| 维度 | PyTorch CUDA | torch-npu CANN |
|------|-------------|---------------|
| **构建策略** | 按 CUDA + Python 版本矩阵 | 只按 CANN 版本 ⭐ |
| **镜像数量** | 多个（每种组合一个） | 少量（每个 CANN 一个） |
| **Python 切换** | 不同镜像 | 同一镜像切换环境变量 ⭐ |
| **版本映射** | 简化版 → 完整版 | URL 映射表 ⭐ |

### 优势

1. **镜像数量减少**：1 个 CANN 版本 = 1 个镜像（而非 4 个）
2. **灵活性更高**：无需预判 Python 版本需求
3. **维护更简单**：只需维护 CANN 版本映射表

---

## 常见问题

### Q1: 为什么一个镜像支持多个 Python 版本？

A: CANN 安装只需要 Python 3 环境，不绑定特定版本。切换 Python 版本不影响 CANN 功能。

### Q2: 如何在 CI 中使用特定 Python 版本？

A: 在容器内执行切换脚本：
```bash
source /usr/local/bin/switch_python.sh 3.12
```

或在 Dockerfile/脚本中修改 PATH：
```bash
export PATH=/opt/python/cp312-cp312/bin:$PATH
```

### Q3: 如何验证 CANN 是否正常工作？

A:
```bash
source /etc/profile.d/cann_env.sh
python -c "import torch; import torch_npu; print(torch_npu.npu.is_available())"
```

### Q4: 不同 CANN 版本有什么区别？

A:
- **9.0**: Stable 版本，推荐用于生产环境
- **9.0.0-beta.2**: Beta 版本，包含最新特性
- **8.0**: 旧版本，用于兼容性测试

---

## 脚本参数详解

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cann-version` | CANN 版本号 | 必需 |
| `--registry` | Docker registry | `quay.io` |
| `--quay-org` | Quay.io 组织 | `kerer` |
| `--image-name` | 镜像名称 | `pytorch` |
| `--push` | 推送镜像 | 不推送 |
| `--force` | 强制构建 | 不强制 |
| `--verbose` | 详细日志 | 不显示 |
| `--list-versions` | 显示版本列表 | - |

---

## 更新日志

### 2026-05-06 重构

**主要变更**：
1. ❌ 移除 Python 版本参数（不再按 Python 构建镜像）
2. ✅ 预装所有 Python 版本（3.10/3.11/3.12/3.13）
3. ✅ 添加 Python 版本切换脚本
4. ✅ 只按 CANN 版本构建镜像
5. ✅ 维护 CANN 包 URL 映射表

**镜像标签变化**：
- 原：`py3.11-cann9.0`
- 新：`cann9.0`

---

**生成时间**: 2026-05-06