# TORCH\_HCCL\_RANKS\_PER\_ROOT

## 功能描述

开启Scalable RootInfo分级建链后，通过此环境变量可配置每个root期望管理的最大rank数量，并据此计算当前通信域使用的root数量。

该环境变量默认值为“128”，取值范围为1~4294967295的十进制整数。

设当前ProcessGroup包含N个rank，本环境变量的取值为R：

- 当N不大于R时：使用原单RootInfo路径。
- 当N大于R时：root数量K为`ceil(N / R)`。N个rank会被连续、均衡地划分到K个group，各group的rank数量最多相差1，每个group的首个`local rank`作为root。

## 配置示例

开启Scalable RootInfo，并将每个root期望管理的rank数量设置为128：

```bash
export ROOTINFO_SCALABLE_ENABLE=1
export TORCH_HCCL_RANKS_PER_ROOT=128
```

仅配置功能开关时，本环境变量使用默认值“128”：

```bash
export ROOTINFO_SCALABLE_ENABLE=1
unset TORCH_HCCL_RANKS_PER_ROOT
```

## 使用约束

- 仅在`ROOTINFO_SCALABLE_ENABLE=1`且设备支持Scalable RootInfo时参与通信域路径选择。
- 配置为空字符串、0、负数、非数字、包含多余字符或超过4294967295时，会返回参数错误。
- 环境变量应在启动分布式worker前完成配置，详细使用方法请参见[Scalable RootInfo分级建链](../../../developer_notes/distributed/communication_strategy/scalable_rootinfo_link_setup.md)。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
