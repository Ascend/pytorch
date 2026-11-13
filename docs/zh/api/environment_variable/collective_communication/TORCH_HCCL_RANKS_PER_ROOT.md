# TORCH\_HCCL\_RANKS\_PER\_ROOT

## 功能描述

通过此环境变量可控制Scalable RootInfo分级建链，并配置每个root期望管理的最大rank数量。

该环境变量默认值为“-1”，表示关闭Scalable RootInfo，使用原RootInfo建链路径。设置为1~4294967295的十进制整数时，表示每个root期望管理的最大rank数量。

设置为正整数时，设当前ProcessGroup包含N个rank，本环境变量的取值为R：

- 当N小于等于R时：使用原单RootInfo路径。
- 当N大于R时：root数量K为`ceil(N / R)`。N个rank会被连续、均衡地划分到K个group，各group的rank数量最多相差1，每个group的首个`local rank`作为root。

## 配置示例

开启Scalable RootInfo，并将每个root期望管理的rank数量设置为128：

```bash
export TORCH_HCCL_RANKS_PER_ROOT=128
```

关闭Scalable RootInfo：

```bash
export TORCH_HCCL_RANKS_PER_ROOT=-1
```

不设置或执行`unset TORCH_HCCL_RANKS_PER_ROOT`时，同样使用默认值“-1”，关闭该功能。

## 使用约束

- 设置为正整数且设备支持Scalable RootInfo时参与通信域路径选择。
- 配置为空字符串、0、小于-1的负数、非数字、包含多余字符或超过4294967295时，会返回参数错误。
- 环境变量应在启动分布式worker前完成配置，详细使用方法请参见[Scalable RootInfo分级建链](../../../developer_notes/distributed/communication_strategy/scalable_rootinfo_link_setup.md)。

## 支持的型号

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id2 -->
