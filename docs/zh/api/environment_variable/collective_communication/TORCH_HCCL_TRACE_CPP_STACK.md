# TORCH\_HCCL\_TRACE\_CPP\_STACK

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制Flight Recorder在记录事件时是否同时采集C++调用栈。

- 0：不采集C++调用栈，仅记录Python和TorchScript调用栈。
- 1：采集C++调用栈，在事件记录时保存C++ instruction pointer，导出dump时可通过符号化转换为函数名和行号。

默认值：0。

> [!NOTE]
>
> - 此环境变量在`HCCLTraceBuffer`单例首次构造时读取并缓存，运行中修改不会生效。
> - 此环境变量控制的是**记录阶段**是否采集栈，不等同于dump阶段是否输出栈。如果记录时未采集，dump时无法还原C++栈。
> - 开启C++栈采集会略微增加记录阶段的性能开销。

该变量对应PyTorch的`TORCH_FR_CPP_STACK`。PyTorch同时兼容`TORCH_NCCL_TRACE_CPP_STACK`名称。配置方式一致，默认均关闭。PyTorch使用`TORCH_FR_*`作为主名称。

## 配置示例

```bash
export TORCH_HCCL_TRACE_CPP_STACK=1
```

## 使用约束

- 需在加载`torch_npu`扩展前设置。
- 需同时设置`TORCH_HCCL_TRACE_BUFFER_SIZE > 0`，否则无事件可记录。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
