# ASCEND\_LAUNCH\_BLOCKING

## 功能描述

通过此环境变量可控制算子执行时是否启用同步模式。

由于在昇腾NPU上进行模型训练时默认算子异步执行，导致算子执行过程中出现报错时，打印的报错堆栈信息并不是实际的调用栈信息。当设置为“1”时，强制算子采用同步模式运行，这样能够打印正确的调用栈信息，从而更容易地调试和定位代码中的问题。设置为“0”时则会采用异步方式执行。

默认配置为“0”。

该环境变量对应PyTorch的[CUDA_LAUNCH_BLOCKING](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)，功能一致，均用于控制算子是否采用同步执行模式。

## 配置示例

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

## 使用约束

- ASCEND\_LAUNCH\_BLOCKING设置为“1”时，强制算子采用同步模式运行会导致性能下降。
- ASCEND\_LAUNCH\_BLOCKING设置为“1”时，task\_queue算子队列关闭，[TASK\_QUEUE\_ENABLE](TASK_QUEUE_ENABLE.md)设置不生效。
- ASCEND\_LAUNCH\_BLOCKING设置为“0”时，会增加内存消耗，有导致OOM的风险。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id5 -->
