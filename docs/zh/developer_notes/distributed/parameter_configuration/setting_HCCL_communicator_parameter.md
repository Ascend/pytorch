# 通过pg\_options配置HCCL通信域参数

## 简介

本特性可以针对不同的通信域配置不同的HCCL参数。通过pg\_options添加hccl\_config配置，将HCCL配置参数从Python层通过TorchNPU传递到HCCL供使用。

## 使用场景

当模型使用多个通信域，且不同通信域对buffer大小、通信算法或执行超时时间有不同需求时，可以在模型脚本中通过本特性分别配置。

## 使用指导

通过 `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options` 的 `hccl_config` 字典，可以为通信域配置HCCL参数。调用 `torch.distributed.init_process_group` 或 `torch.distributed.new_group` 时，通过 `pg_options` 传入该配置。

TorchNPU将支持的配置项转换为 `HcclCommConfig` 的对应字段，并在创建HCCL通信域时传入。各字段的功能、取值范围、默认值及硬件约束，统一参见 [HcclCommConfig](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/commlib/hcclug/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclCommConfig.md)。请在该文档中选择与实际安装的CANN版本对应的说明；可用配置项还需以所使用的TorchNPU版本为准。

### 参数对应关系

`hccl_config` 配置项请使用下表左列所示的参数名，不可直接使用 `HcclCommConfig` 结构体字段名。

| hccl_config参数 | HcclCommConfig字段 |
| --- | --- |
| `hccl_buffer_size` | `hcclBufferSize` |
| `group_name` | `hcclUdi` |
| `qos_traffic_class` | `hcclRdmaTrafficClass` |
| `qos_service_level` | `hcclRdmaServiceLevel` |
| `hccl_sdma_qos` | `hcclQos` |
| `hccl_op_expansion_mode` | `hcclOpExpansionMode` |
| `hccl_exec_timeout` | `hcclExecTimeOut` |
| `hccl_algo` | `hcclAlgo` |
| `hccl_retry_enable` | `hcclRetryEnable` |
| `hccl_retry_params` | `hcclRetryParams` |
| `hccl_buffer_name` | `hcclBufferName` |
| `hccl_sym_win_max_mem_size_per_rank` | `hcclSymWinMaxMemSizePerRank` |

### 配置优先级

对于支持对应环境变量的配置项，优先级为：**`hccl_config` 中的有效配置（通信域粒度） > 对应环境变量（全局配置） > 默认配置**。

- 未指定通信域配置时：使用对应环境变量；
- 环境变量未设置时：使用默认配置。表示“未配置”的特殊取值及各字段的具体规则，以HCCL的配置优先级为准。

例如，环境变量配置为 `HCCL_BUFFSIZE=200`，同时在某个通信域的 `hccl_config` 中设置 `"hccl_buffer_size": 100`，则该通信域使用100 MB的buffer；未单独设置该参数的通信域使用环境变量指定的200 MB。

默认通信域与通过 `new_group` 创建的通信域分别配置。新通信域不会自动继承默认通信域的 `hccl_config`，需要通过其自身的 `pg_options` 传入。

## 使用样例

该示例基于 `torchrun` 启动，分别对默认通信域和新创建通信域配置了buffer大小、通信域标识、执行超时时间及通信算法。示例中的参数值仅用于演示配置方法，实际使用时请根据CANN版本、硬件环境及业务需求进行调整。

```python
import os

import torch
import torch.distributed as dist
import torch_npu

torch.npu.set_device(int(os.environ["LOCAL_RANK"]))

options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config = {
    "hccl_buffer_size": 200,  # 单位：MB
    "group_name": "default_comm",  # 对应hcclUdi
    "hccl_exec_timeout": 500,  # 单位：秒
    "hccl_algo": "allreduce=level0:NA;level1:ring",
}
dist.init_process_group(backend="hccl", pg_options=options)

# 所有进程按相同顺序调用new_group。
group_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
group_options.hccl_config = {
    "hccl_buffer_size": 100,
    "group_name": "custom_comm",
    "hccl_exec_timeout": 600,
    "hccl_algo": "allreduce=level0:NA;level1:ring",
}
group = dist.new_group(
    ranks=list(range(dist.get_world_size())),
    backend="hccl",
    pg_options=group_options,
)

dist.destroy_process_group(group)
dist.destroy_process_group()
```

## 约束说明

- 配置应在创建通信域前传入；修改字典不会重新配置已经创建的HCCL通信域。
- TorchNPU会校验配置值的类型。字符串超出对应字段的缓冲区长度时，会截断并告警；请按HCCL文档中的长度约束设置。
- `group_name` 写入的是 `hcclUdi`。`hcclCommName` 和 `hcclDeterministic` 由框架处理，不通过该字典配置。
- 使用 `hccl_buffer_name` 时，TorchNPU还会在同一设备上为相同buffer名称复用通信流。
- `hccl_world_rank_id` 和 `hccl_job_id` 分别对应 `hcclWorldRankID` 和 `hcclJobID`，用于NSLB场景，由框架内部设置，无需用户手动配置。
