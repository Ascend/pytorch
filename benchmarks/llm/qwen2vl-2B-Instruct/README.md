# qwen2vl-2B-Instruct 微调训练

本 README 说明如何使用 **qwen2vl-2B-Instruct** 模型权重，结合 **pixparse/cc12m-wds** 提供的 `cc12images` 示例数据，完成数据下载、预处理与训练启动（含 eager / torch.compile 两种模式）。

---

## 目录

- [1. 模型权重](#1-模型权重)
- [2. 数据获取和预处理](#2-数据获取和预处理)
- [3. 模型训练](#3-模型训练)
  - [3.1 eager mode（默认）](#31-eager-mode默认)
  - [3.2 启用 torchcompile（可选）](#32-启用-torchcompile可选)
  - [3.3 采集profile文件（可选）](#33-采集profile文件可选)

---

## 1. 模型权重

- Hugging Face 模型：**mistralai/Mamba-Codestral-7B-v0.1**  
  https://huggingface.co/mistralai/Mamba-Codestral-7B-v0.1

- 可使用如下的自定义脚本下载

```bash
python ../utils/download_hf.py --model Qwen/Qwen2-VL-2B-Instruct --save_path ./Qwen2-VL-2B-Instruct
```

### 环境提示

本项目依赖已整理到 `requirements.txt`，可直接安装：

```bash
pip install -r ../utils/requirements.txt
```

## 2. 数据获取和预处理

```bash
pip install -U huggingface_hub webdataset pillow
python build_dataset.py \
  --pattern "cc12m-train-0000.tar" \
  --out-json train_data_0.json \
  --img-dir ./cc12m_dataset
```

## 3. 模型训练

训练脚本：`run_train.sh`

开始训练前，请修改脚本中的路径参数：

- 模型权重路径
- 训练数据路径

### 3.1 eager mode（默认）

```bash
bash run_train.sh
```

### 3.2 启用 `torch.compile`（可选）

可通过添加 `--enable_compile` 选项运行图模式
当在 GPU 上训练时，默认后端使用triton；当在 NPU 上训练时，可进一步指定后端为 mlir 或 dvm，默认使用 mlir。

**默认后端（mlir，可不写 --npu-backend）：**

```bash
bash run_train.sh \
  --enable_compile
```

**显式指定后端为 mlir：**

```bash
bash run_train.sh \
  --enable_compile \
  --npu-backend mlir
```

**切换后端为 dvm：**

```bash
bash run_train.sh \
  --enable_compile \
  --npu-backend dvm
```

当在 NPU 上训练时，可通过 `--mfusion` 参数开启 MFusion 图算融合优化功能, 配合不同的NPU图模式后端, 进一步提升模型的性能，使用示例如下

```bash
bash run_train.sh \
  --enable_compile \
  --npu-backend dvm \
  --mfusion
```

### 3.3 采集profile文件（可选）

脚本已支持 `--enable_profiler` 这样的开关，开启方式为：

```bash
bash run_train.sh \
  --enable_profiler \
  --profiler_start_step 5 \
  --profiler_end_step 6 \
```

可以通过 `--profiler_start_step` 和 `--profiler_end_step` 分别设置profile开始和结束步数。
