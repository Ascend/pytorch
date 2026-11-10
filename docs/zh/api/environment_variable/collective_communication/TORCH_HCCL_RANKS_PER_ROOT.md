# TORCH\_HCCL\_RANKS\_PER\_ROOT

## 功能描述

通过此环境变量可控制是否开启Scalable RootInfo分级建链，配置每个root期望管理的最大rank数量，并据此计算当前通信域使用的root数量。

- 配置为“-1”时：关闭分级建链，使用原单RootInfo路径。
- 配置为1~4294967295的十进制整数时：设置每个root期望管理的最大rank数量，满足使用条件时采用分级建链。

该环境变量默认值为“128”，未设置时使用默认值。

在支持Scalable RootInfo的设备上，HCCL分级接口齐全且配置值为正数时，设当前ProcessGroup包含N个rank，本环境变量的取值为R：

- 当N小于等于R时：使用原单RootInfo路径。
- 当N大于R时：root数量K为`ceil(N / R)`。N个rank会被连续、均衡地划分到K个group，各group的rank数量最多相差1，每个group的首个`local rank`作为root。

## 配置示例

将每个root期望管理的rank数量设置为128，满足使用条件且当前ProcessGroup的rank数量大于128时采用Scalable RootInfo：

```bash
export TORCH_HCCL_RANKS_PER_ROOT=128
```

恢复默认值“128”（满足使用条件时自动采用分级建链）：

```bash
unset TORCH_HCCL_RANKS_PER_ROOT
```

关闭Scalable RootInfo分级建链：

```bash
export TORCH_HCCL_RANKS_PER_ROOT=-1
```

## 使用约束

- 其他型号使用原RootInfo路径；未设置本变量时按默认值128处理，与显式设置正整数时一样，每个进程仅打印一次机型不支持Warning，且不以N大于R为前提。设置为“-1”时不打印机型不支持告警。
- 仅作用于RootInfo建链路径中的DEFAULT类型HCCL通信域，不改变ranktable和P2P通信域的建链行为。DEFAULT表示通信域类型，全局ProcessGroup和通过`torch.distributed.new_group`创建的非全局ProcessGroup均可使用。
- HCCL库可正常加载但缺少`HcclGetRootInfoScalable`或`HcclCommInitRootInfoScalable`接口时，使用原RootInfo路径。HCCL库加载失败或分级接口调用失败时会报错，不会自动回退。
- 配置为空字符串、0、小于-1的负数、非数字、包含尾随非法字符或超过4294967295时，会返回参数错误。
- 旧开关`ROOTINFO_SCALABLE_ENABLE`不再读取。升级后需要关闭分级建链的作业，应显式设置`TORCH_HCCL_RANKS_PER_ROOT=-1`；`unset TORCH_HCCL_RANKS_PER_ROOT`仅恢复默认值，不能用于关闭该功能。
- 环境变量应在启动分布式worker前完成配置，详细使用方法请参见[Scalable RootInfo分级建链](../../../developer_notes/distributed/communication_strategy/scalable_rootinfo_link_setup.md)。

## 支持的型号

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id2 -->
