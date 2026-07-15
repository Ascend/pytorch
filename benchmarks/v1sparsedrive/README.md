# SimpleV1SparseDrive (TorchNPU / Ascend)

---

## 1. 项目概述

本项目实现了名为 **SimpleV1SparseDrive** 的模型结构，整体由以下模块组成：

| 模块 | 说明 |
|------|------|
| `data_preprocessor` | `BaseDataPreprocessor()` |
| `img_backbone` | ResNet（包含大量 `BatchNorm2d`） |
| `img_neck` | FPN |
| `head` | `V1SparseDriveHead` |
| ├─ `det_head` | Sparse4DHeadLike（含 DeformableFeatureAggregation / FlashAttention / Refinement） |
| ├─ `map_head` | Sparse4DHeadLike |
| └─ `motion_plan_head` | Motion & Planning（含多层 FlashAttention / FFN / refinement） |
| `depth_branch` | DenseDepthNetLike（3 个 depth layer） |
| `grid_mask` | GridMask |

---

## 2. 环境与依赖

### 2.1 硬件/系统要求

- Ascend NPU 环境已正确安装（驱动 / CANN 等）

### 2.2 Pip 环境包（已验证可复现）

- 基础依赖包

```bash
pip install -r requirement.txt
```

- 源码安装mmcv

```bash
git clone -b 1.x https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
cd ..
```

> 建议固定上述版本以保证结果可复现。

---

## 3. 运行训练和验证

本工程提供两种运行模式：

- **Eager（默认）**：不启用 `torch.compile`，直接动态图执行。
- **Graph/Compile**：通过 `--compile` 启用 `torch.compile`（Inductor 图模式），首步通常包含编译开销。

> 默认运行 **200 step**（如需修改步数，请按工程内 `config.py` 的参数/配置为准）。

### 3.1 训练（生成日志）

#### 3.1.1 Eager 模式（默认）

```bash
python train.py 2>&1 | tee eager.log
```

#### 3.1.2 Graph / torch.compile 模式

```bash
python train.py --npu-backend dvm --compile 2>&1 | tee compile.log
```
