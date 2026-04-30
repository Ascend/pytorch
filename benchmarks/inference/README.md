# 推理说明
 	 
本 README 说明如何使用 **DCNv2**，**DIN**，**MMOE**和**ETA**模型进行推理（含 eager / torch.compile 两种模式）。
在运行完成后输出
---

## 目录

- [1. 数据集处理](#1-数据集处理)
  - [1.1 DIN & DCNv2](#11-DIN-&-DCNv2)
  - [1.2 MMOE & ETA](#12-MMOE-&-ETA)
- [2. 模型推理](#2-模型推理)
    - [2.1 eager mode（默认）](#21-eager-mode默认)
    - [2.2 启用 torch.compile（可选）](#22-启用-torch.compile可选)
    - [2.3 采集 profile文件（可选）](#23-采集profile文件可选)
---

## 1. 数据集处理

### 1.1 DIN & DCNv2

**DIN** 和 **DCNv2** 的推理数据集来自 CTR_Algorithm 仓库示例数据，该数据集无需手动下载

https://github.com/Prayforhanluo/CTR_Algorithm/tree/main/data

在DIN或DCNv2目录下执行如下命令：
```bash
git clone https://github.com/Prayforhanluo/CTR_Algorithm.git
```
- 数据集目录：`CTR_Algorithm/data/data.csv`  


### 1.2 MMOE & ETA

进入**MMOE**或**ETA**文件夹下，进行如下数据预处理

推理数据集来自 https://tianchi.aliyun.com/dataset/408 Ali-CPP数据集，需做如下处理：
1. 下载数据集，存放至当前目录下alicpp文件夹中
2. 后续流程参考：https://gitee.com/ascend/RecSDK/blob/develop_torch_benchmark/torch_examples_benchmark/model_zoo/README.md

执行完成后数据集会默认生成至aliccp_out目录下。

网络依赖已整理到 `requirements.txt`，可直接安装：

```bash
pip install -r ./requirements.txt
```

在模型执行时，默认加载数据集路径为: `./aliccp/aliccp_out/`；模型脚本同时支持运行时使用如下命令动态指定数据集目录：

```bash
python eta.py --data_dir path/to/your/data/
```


## 2. 模型推理(DCNv2网络为例)

进入**DCNv2**文件夹，直接执行推理脚本`dcnv2.py`即可

- 注：运行该网络前需要先将patch文件加上，具体方式如下：
  ```bash
  cd DCNv2
  unix2dos dcnv2.patch        #如果在arm机器上需要执行
  git apply dcnv2.patch
  ```

### 2.1 eager mode（默认）

```bash
python dcnv2.py
```

### 2.2 启用 `torch.compile`（可选）

脚本已支持 `--enable_compile` 这样的开关，开启方式为：

```bash
python dcnv2.py \
--enable_compile
```

### 2.3 采集profile文件（可选）

脚本已支持 `--enable_profiler` 这样的开关，开启方式为：

```bash
python dcnv2.py \
  --enable_profiler \
  --profiler_start_step 5 \
  --profiler_end_step 6 \
```
可以通过 `--profiler_start_step ` 和 `--profiler_end_step` 分别设置profile开始和结束步数。