# TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST

## 功能描述

控制FlexAttention反向预生成掩码路径是否允许使用任务列表调度dK/dV，默认值为1。

| 值 | 说明 |
| --- | --- |
| 1 | 满足条件时允许使用任务列表路径。 |
| 0 | 关闭任务列表路径，保留默认dK/dV调度。 |

## 配置示例

```shell
export TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST=1
python flex_attention_example.py
```

其中`flex_attention_example.py`见[快速入门](./overview.md)。

## 使用约束

- 仅控制dK/dV的任务列表调度，不控制前向或dQ，也不改变自动求导接口。
- 需要预生成掩码路径、静态形状、可支持的掩码布局及int32连续元数据等条件。当前路径还要求Query和Key/Value的batch均为1、掩码head维度为1，并满足FP32累加及片上存储资源限制。
- 开启后仍会检查适用条件；条件不满足时使用默认dK/dV路径。是否更快应以实际输入的稳态耗时为准。
- 设置环境变量后使用新进程。

## 支持的型号

- <term>Ascend 950PR&950DT系列产品</term>
