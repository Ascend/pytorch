# NSLB\_CP

## 功能描述

通过此环境变量可配置HCCL操作的NSLB（Non-Stop Load Balancing）采样记录目录路径。当设置为非空路径时，每个collective操作会记录操作类型、数据量和rank信息到该目录下的文件中。

- 未设置：关闭NSLB采样。
- 设置路径：开启NSLB采样，记录文件创建在指定路径下。

默认值：未设置（关闭）。

> [!NOTE]
>
> - 此环境变量在`ProcessGroupHCCL.cpp`文件级全局初始化时读取，**应在加载`torch_npu`扩展前设置**。
> - NSLB采样依赖由`torchrun/elastic`启动器注入的`RANK`和`MASTER_ADDR`两个PyTorch的环境变量。`RANK`用于标记记录来源，`MASTER_ADDR`参与文件名构造。
> - 如果环境中存在`HCCL_ALGO`，新建记录文件时会写入一行`HCCL_ALGO=<value>`作为说明。
> - NSLB采样不读取Flight Recorder ring buffer，与FR dump是独立的采样路径。
> - 记录文件名格式为：`<master_addr>_<comm_name>_<rank>.log`。
> - 每个PG的最大采样记录数量由`NSLB_MAX_RECORD_NUM`控制。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export NSLB_CP=/data/nslb_records
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
