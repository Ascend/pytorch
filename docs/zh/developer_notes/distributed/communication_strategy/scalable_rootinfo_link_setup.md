# Scalable RootInfo分级建链

## 简介

随着分布式集群规模增大，HCCL通信域中的rank数量随之增加。传统RootInfo建链方式由单个root承担通信域初始化所需的信息协商，单root的处理压力会随通信域规模增大，可能成为大规模集群的建链性能瓶颈。

Scalable RootInfo采用分级建链方式，将通信域内的rank连续、均衡地划分为多个group，并由多个root共同完成信息协商，降低单个root处理的rank数量，提升大规模通信域的建链效率。

Scalable RootInfo功能默认关闭。关闭时保持原RootInfo建链行为不变。

## 使用场景

当大规模集群中单root协商成为通信域初始化的性能瓶颈时，可以使用Scalable RootInfo分级建链，降低单个root处理的rank数量。

该功能适用于默认全局 ProcessGroup 以及通过 `torch.distributed.new_group` 创建的非全局 ProcessGroup。分组规模基于当前 ProcessGroup 内的 rank 数量计算，而非全局 world size。

## 使用指导

通过环境变量开启Scalable RootInfo，并配置每个root期望管理的rank数量：

- `ROOTINFO_SCALABLE_ENABLE`：Scalable RootInfo功能开关。默认值为“0”，设置为“1”时开启。
- `TORCH_HCCL_RANKS_PER_ROOT`：每个root期望管理的最大rank数量，默认值为“128”，取值必须为1~4294967295的十进制整数。

环境变量的完整说明请参见环境变量中的“[ROOTINFO\_SCALABLE\_ENABLE](../../../api/environment_variable/collective_communication/ROOTINFO_SCALABLE_ENABLE.md)”和“[TORCH\_HCCL\_RANKS\_PER\_ROOT](../../../api/environment_variable/collective_communication/TORCH_HCCL_RANKS_PER_ROOT.md)”章节。

设当前ProcessGroup包含N个rank，`TORCH_HCCL_RANKS_PER_ROOT`为R：

- 当N不大于R时，继续使用原单RootInfo路径。
- 当N大于R时，root数量K为`ceil(N / R)`，进入Scalable RootInfo路径。
- N个rank会被连续、均衡地分到K个group，各group的rank数量最多相差1，每个group的首个local rank作为root。

## 使用样例

开启Scalable RootInfo，并将每个root期望管理的rank数量设置为128：

```bash
export ROOTINFO_SCALABLE_ENABLE=1
export TORCH_HCCL_RANKS_PER_ROOT=128
```

例如，通信域包含8个rank且R为2时，K为4，root local rank分别为0、2、4、6，各root管理2个rank。通信域包含8个rank且R为3时，K为3，三个group的rank数量分别为3、3、2，root local rank分别为0、3、6。

关闭Scalable RootInfo时，执行：

```bash
unset ROOTINFO_SCALABLE_ENABLE
unset TORCH_HCCL_RANKS_PER_ROOT
```

## 约束说明

- 仅<term>Atlas A2 训练系列产品</term>和<term>Atlas A3 训练系列产品</term>支持Scalable RootInfo。其他型号即使开启功能，也会打印Warning并回退原RootInfo路径。
- 需要配套使用包含`HcclGetRootInfoScalable`和`HcclCommInitRootInfoScalable`接口的CANN/HCCL版本。在支持的设备上已经选中Scalable路径但运行库缺少接口时，会返回不支持错误。
- 该功能仅作用于RootInfo建链路径，不改变rank table建链行为。配置有效的`RANK_TABLE_FILE`并选择rank table路径时，不会进入Scalable RootInfo路径。
- P2P通信域继续使用原RootInfo路径。
- 环境变量应在启动分布式worker前完成配置，不支持在同一进程内动态切换功能开关。
- Scalable RootInfo只改变通信域初始化方式，不改变通信算子的调用方法以及通信域的缓存和销毁方式。
