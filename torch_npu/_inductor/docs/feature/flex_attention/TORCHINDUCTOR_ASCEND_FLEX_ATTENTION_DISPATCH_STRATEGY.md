# TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_DISPATCH_STRATEGY

## 功能描述

选择FlexAttention的可选运行时调度策略。默认不设置，即不启用额外的运行时分支。

| 值 | 说明 |
| --- | --- |
| 未设置 | 使用默认编译调度路径。 |
| `fwd` | 前向根据实际完整块计数，在满足条件时选择只处理部分块的特化路径。 |
| `bwd_dkdv` | 反向在满足条件时使用任务列表调度dK/dV。 |

两种策略不能通过逗号等方式组合；该变量一次选择一种策略。

## 配置示例

```shell
export TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_DISPATCH_STRATEGY=fwd
python flex_attention_example.py
```

恢复默认设置：

```shell
unset TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_DISPATCH_STRATEGY
```

其中`flex_attention_example.py`见[快速入门](./overview.md)。

## 使用约束

- 本配置适用于Python wrapper下满足条件的预生成掩码路径。C++ wrapper、AOT编译及不支持的子图使用默认路径。
- `fwd`需要在运行时检查掩码元数据；设备到主机的同步会带来开销，应实测收益。
- `bwd_dkdv`受静态形状、掩码布局、数据类型及硬件资源限制；不满足条件时保留默认dK/dV路径。它不控制dQ调度。
- 设置环境变量后使用新进程，分别核对前向输出、梯度和性能。

## 支持的型号

- <term>Ascend 950PR&950DT系列产品</term>
