# SDXL 微调训练

本 README 说明如何使用 **SDXL** 模型权重，完成数据下载、预处理与训练启动（含 eager / torch.compile 两种模式）。

---

## 目录

- [SDXL 微调训练](#sdxl-微调训练)
  - [目录](#目录)
  - [1. 模型权重](#1-模型权重)
    - [环境提示](#环境提示)
  - [2. 数据获取](#2-数据获取)
  - [3. 模型训练](#3-模型训练)
    - [3.1 eager mode（默认）](#31-eager-mode默认)
    - [3.2 启用 `torch.compile`（可选）](#32-启用-torchcompile可选)

---

### 环境提示

本项目依赖已整理到 `requirements.txt`，可直接安装：

```bash
pip install -r ../utils/requirements.txt
```

## 1. 模型权重

- Hugging Face 模型：**stabilityai/stable-diffusion-xl-base-1.0**  
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

- 可使用如下的自定义脚本下载

```bash
python ../utils/download_hf.py --model stabilityai/stable-diffusion-xl-base-1.0 --save_path ./stable-diffusion-xl-base-1.0
```

## 2. 数据获取

训练数据集链接如下：

https://huggingface.co/datasets/AdamLucek/oldbookillustrations-small/tree/main/data

- 可使用以下命令获取

```bash
wget https://huggingface.co/datasets/AdamLucek/oldbookillustrations-small/resolve/main/data/train-00000-of-00001.parquet
```

## 3. 模型训练

训练脚本：`run_sdxl.sh`，脚本支持在 GPU 和 NPU 训练

开始训练前，请修改脚本中的路径参数：

- 模型权重路径
- 训练数据路径

### 3.1 eager mode（默认）

```bash
bash run_sdxl.sh
```

### 3.2 启用 `torch.compile`（可选）

可通过添加 `--enable_compile` 选项运行图模式
当在 GPU 上训练时，默认后端使用triton；当在 NPU 上训练时，可进一步指定后端为 mlir 或 dvm，默认使用 mlir。

**默认后端（mlir，可不写 --npu-backend）：**
```bash
bash run_sdxl.sh \
  --enable_compile
```

**显式指定后端为 mlir：**
```bash
bash run_sdxl.sh \
  --enable_compile \
  --npu-backend mlir
```

**切换后端为 dvm：**
```bash
bash run_sdxl.sh \
  --enable_compile \
  --npu-backend dvm
```

### 3.3 采集profile文件（可选）

脚本已支持 `--enable_profiler` 这样的开关，开启方式为：

```bash
bash run_sdxl.sh \
  --enable_profiler \
  --profiler_start_step 5 \
  --profiler_end_step 6 \
```
可以通过 `--profiler_start_step ` 和 `--profiler_end_step` 分别设置profile开始和结束步数。