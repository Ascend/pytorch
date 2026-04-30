# Baichuan2-7B-Chat 微调训练

本 README 说明如何使用 **Baichuan2-7B-Chat** 模型权重，结合 **LlamaFactory** 提供的 `alpaca_zh_demo.json` 示例数据，完成数据下载、预处理与训练启动（含 eager / torch.compile 两种模式）。

---

## 目录

- [1. 模型权重](#1-模型权重)
- [2. 数据获取](#2-数据获取)
- [3. 数据预处理](#3-数据预处理)
- [4. 模型训练](#4-模型训练)
  - [4.1 eager mode（默认）](#41-eager-mode默认)
  - [4.2 启用 torchcompile（可选）](#42-启用-torchcompile可选)

---

### 环境配置

本项目依赖已整理到 `requirements.txt`，可直接安装：

```bash
pip install -r ../utils/requirements.txt
```

## 1. 模型权重

- Hugging Face 模型：**baichuan-inc/Baichuan2-7B-Chat**  
  https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat

- 可使用如下的自定义脚本下载

```bash
python ../utils/download_hf.py --model baichuan-inc/Baichuan2-7B-Chat --save_path ./Baichuan2-7B-Chat
```


## 2. 数据获取

训练数据集来自 LlamaFactory 仓库示例数据：

- `alpaca_zh_demo.json`  
  https://github.com/hiyouga/LlamaFactory/blob/main/data/alpaca_zh_demo.json

建议直接下载 raw 文件到本地：

```bash
wget -O alpaca_zh_demo.json \
https://raw.githubusercontent.com/hiyouga/LlamaFactory/main/data/alpaca_zh_demo.json
```

## 3. 数据预处理

本项目提供 `preprocess.py`，用于把 Alpaca 格式（`instruction`/`input`/`output`）转换为 Baichuan 常用的 `conversations` 格式。

### 命令

```bash
python preprocess.py -i alpaca_zh_demo.json -o train.json
```

## 4. 模型训练

训练脚本：`run_baichuan2.sh`，脚本支持在 GPU 和 NPU 训练

开始训练前，请修改脚本中的路径参数：

- 模型权重路径（本地目录）
- 训练数据路径（预处理后的 `train.json`）

### 4.1 eager mode（默认）

```bash
bash run_baichuan2.sh
```

### 4.2 启用 `torch.compile`（可选）

可通过添加 `--enable_compile` 选项运行图模式
当在 GPU 上训练时，默认后端使用triton；当在 NPU 上训练时，可进一步指定后端为 mlir 或 dvm，默认使用 mlir。

**默认后端（mlir，可不写 --npu-backend）：**
```bash
bash run_baichuan2.sh \
  --enable_compile
```

**显式指定后端为 mlir：**
```bash
bash run_baichuan2.sh \
  --enable_compile \
  --npu-backend mlir
```

**切换后端为 dvm：**
```bash
bash run_baichuan2.sh \
  --enable_compile \
  --npu-backend dvm
```

### 4.3 采集profile文件（可选）

脚本已支持 `--enable_profiler` 这样的开关，开启方式为：

```bash
bash run_baichuan2.sh \
  --enable_profiler \
  --profiler_start_step 5 \
  --profiler_end_step 6 \
```
可以通过 `--profiler_start_step ` 和 `--profiler_end_step` 分别设置profile开始和结束步数。