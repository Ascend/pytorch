# ROOTINFO\_SCALABLE\_ENABLE

## 功能描述

当使用HCCL作为通信后端且采用RootInfo方式建立通信域时，通过此环境变量可控制是否开启Scalable RootInfo分级建链。

- 配置为“0”（默认值）：关闭Scalable RootInfo，使用原单RootInfo路径建立通信域。
- 配置为“1”：开启Scalable RootInfo。满足使用条件时，将当前通信域内的rank连续、均衡地划分到多个group，每个group选择一个root，通过分级方式建立通信域。

开启功能后，每个root期望管理的rank数量由`TORCH_HCCL_RANKS_PER_ROOT`配置。当前通信域的rank数量不大于该配置值时，仍使用原单RootInfo路径。

该功能需配套使用支持Scalable RootInfo接口的CANN/HCCL版本，详细使用方法请参见[Scalable RootInfo分级建链](../../../developer_notes/distributed/communication_strategy/scalable_rootinfo_link_setup.md)。

## 配置示例

开启Scalable RootInfo分级建链：

```bash
export ROOTINFO_SCALABLE_ENABLE=1
```

关闭Scalable RootInfo分级建链：

```bash
unset ROOTINFO_SCALABLE_ENABLE
```

## 使用约束

- 该环境变量仅作用于RootInfo建链路径，不改变rank table建链行为。
- 仅对DEFAULT类型的HCCL通信域生效，P2P通信域继续使用原RootInfo路径。
- HCCL通信域采用懒加载机制，首次执行需要通信域的集合通信操作时才会触发建链。
- 该环境变量在进程内首次读取后会被缓存，应在启动分布式worker前完成配置，不支持在同一进程内动态切换。

## 支持的型号

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id2 -->
