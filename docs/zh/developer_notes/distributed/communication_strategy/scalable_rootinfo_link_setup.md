# Scalable RootInfo分级建链

## 简介

随着分布式集群规模增大，HCCL通信域中的rank数量随之增加。传统RootInfo建链方式由单个root承担通信域初始化所需的信息协商，单root的处理压力会随通信域规模增大，可能成为大规模集群的建链性能瓶颈。

Scalable RootInfo采用分级建链方式，将通信域内的rank连续、均衡地划分为多个group，并由多个root共同完成信息协商，降低单个root处理的rank数量，提升大规模通信域的建链效率。

Scalable RootInfo功能默认在满足使用条件时开启，每个root期望管理的最大rank数量默认为128。设置`TORCH_HCCL_RANKS_PER_ROOT=-1`可关闭该功能，保持原RootInfo建链行为。

## 使用场景

当大规模集群中单root协商成为通信域初始化的性能瓶颈时，可以使用Scalable RootInfo分级建链，降低单个root处理的rank数量。

该功能适用于默认全局 ProcessGroup 以及通过 `torch.distributed.new_group` 创建的非全局 ProcessGroup。分组规模基于当前 ProcessGroup 内的 rank 数量计算，而非全局 world size。

## 使用指导

通过环境变量`TORCH_HCCL_RANKS_PER_ROOT`控制Scalable RootInfo，并配置每个root期望管理的最大rank数量。该环境变量默认值为“128”，取值为“-1”或1~4294967295的十进制整数。未设置时使用默认值，设置为“-1”时关闭分级建链。

环境变量的完整说明请参见“[TORCH\_HCCL\_RANKS\_PER\_ROOT](../../../api/environment_variable/collective_communication/TORCH_HCCL_RANKS_PER_ROOT.md)”章节。

在支持Scalable RootInfo的设备上，HCCL分级接口齐全且配置值为正数时，设当前ProcessGroup包含N个rank，`TORCH_HCCL_RANKS_PER_ROOT`为R：

- 当N不大于R时，继续使用原单RootInfo路径。
- 当N大于R时，root数量K为`ceil(N / R)`，进入Scalable RootInfo路径。
- N个rank会被连续、均衡地分到K个group，各group的rank数量最多相差1，每个group的首个local rank作为root。

## 使用样例

将每个root期望管理的rank数量设置为128，满足使用条件且当前ProcessGroup的rank数量大于128时采用Scalable RootInfo：

```bash
export TORCH_HCCL_RANKS_PER_ROOT=128
```

例如，通信域包含8个rank且R为2时，K为4，root local rank分别为0、2、4、6，各root管理2个rank。通信域包含8个rank且R为3时，K为3，三个group的rank数量分别为3、3、2，root local rank分别为0、3、6。

关闭Scalable RootInfo时，执行：

```bash
export TORCH_HCCL_RANKS_PER_ROOT=-1
```

旧开关`ROOTINFO_SCALABLE_ENABLE`不再读取。升级后需要保持原RootInfo建链行为的作业，应显式设置`TORCH_HCCL_RANKS_PER_ROOT=-1`。`unset TORCH_HCCL_RANKS_PER_ROOT`仅恢复默认值128，不能用于关闭该功能。

## 约束说明

<!-- npu="A3,910b" id1 -->
- 仅<term>Atlas A2训练系列产品</term>和<term>Atlas A3训练系列产品</term>支持Scalable RootInfo。其他型号使用原RootInfo路径；未设置`TORCH_HCCL_RANKS_PER_ROOT`时按默认值128处理，与显式设置正整数时一样，每个进程仅打印一次机型不支持Warning，且不以N大于R为前提。设置为“-1”时不打印机型不支持告警。
<!-- end id1 -->
- 需要配套使用包含`HcclGetRootInfoScalable`和`HcclCommInitRootInfoScalable`接口的CANN/HCCL版本。HCCL库可正常加载但缺少任一接口时，使用原RootInfo路径；HCCL库加载失败或分级接口调用失败时会报错，不会自动回退。
- 该功能仅作用于RootInfo建链路径，不改变ranktable建链行为。配置有效的`RANK_TABLE_FILE`并选择ranktable路径时，不会进入Scalable RootInfo路径。
- 仅对DEFAULT类型的HCCL通信域生效，P2P通信域继续使用原RootInfo路径。DEFAULT表示通信域类型，不表示仅对全局ProcessGroup生效。
- 环境变量应在启动分布式worker前完成配置，不支持在同一进程内动态切换功能开关。
- Scalable RootInfo只改变通信域初始化方式，不改变通信算子的调用方法以及通信域的缓存和销毁方式。
