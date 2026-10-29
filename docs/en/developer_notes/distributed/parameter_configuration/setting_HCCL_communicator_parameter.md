# Configuring HCCL Communicator Parameters Through pg_options

## Introduction

This feature allows different HCCL parameters to be configured for different communicators. Add the hccl\_config configuration through pg\_options to pass HCCL configuration parameters from the Python layer to HCCL via TorchNPU for use.

## Application Scenarios

When a model uses multiple communicators and different communicators have different requirements for buffer size, communication algorithms, or execution timeout, you can configure them separately in the model script through this feature.

## Usage Guide

Through `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options` and its `hccl_config` dictionary, HCCL parameters can be configured for a communicator. When calling `torch.distributed.init_process_group` or `torch.distributed.new_group`, pass this configuration through `pg_options`.

TorchNPU converts the supported configuration items into the corresponding fields of `HcclCommConfig` and passes them when creating an HCCL communicator. For the functions, value ranges, default values, and hardware constraints of each field, see [HcclCommConfig](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclCommConfig.md). In that document, select the description corresponding to the actually installed CANN version; the available configuration items are also subject to the TorchNPU version in use.

### Parameter Mapping

For `hccl_config` configuration items, use the parameter names shown in the left column of the following table. The field names of the `HcclCommConfig` structure cannot be used directly.

| hccl_config Parameter | HcclCommConfig Field |
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

### Configuration Priority

For configuration items that support corresponding environment variables, the priority is: **valid configuration in `hccl_config` (communicator granularity) > corresponding environment variable (global configuration) > default configuration**.

- When no communicator configuration is specified: use the corresponding environment variable;
- When the environment variable is not set: use the default configuration. The special values indicating "not configured" and the specific rules of each field are subject to the HCCL configuration priority.

For example, if the environment variable is configured as `HCCL_BUFFSIZE=200`, and in the `hccl_config` of a communicator, `"hccl_buffer_size": 100` is set, that communicator uses a 100 MB buffer; communicators without this parameter configured separately use the 200 MB specified by the environment variable.

The default communicator and communicators created via `new_group` are configured separately. A new communicator does not automatically inherit the `hccl_config` of the default communicator; it needs to be passed through its own `pg_options`.

## Usage Examples

This example is started based on `torchrun`, and configures the buffer size, communicator identifier, execution timeout, and communication algorithm for the default communicator and the newly created communicator respectively. The parameter values in the example are only used to demonstrate the configuration method. In actual use, adjust them according to the CANN version, hardware environment, and business requirements.

```python
import os

import torch
import torch.distributed as dist
import torch_npu

torch.npu.set_device(int(os.environ["LOCAL_RANK"]))

options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config = {
    "hccl_buffer_size": 200,  # Unit: MB.
    "group_name": "default_comm",  # Corresponds to hcclUdi.
    "hccl_exec_timeout": 500,  # Unit: second.
    "hccl_algo": "allreduce=level0:NA;level1:ring",
}
dist.init_process_group(backend="hccl", pg_options=options)

# All processes call new_group in the same order.
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

## Constraints

- The configuration must be passed before the communicator is created; modifying the dictionary does not reconfigure an already created HCCL communicator.
- TorchNPU validates the types of configuration values. When a string exceeds the buffer length of the corresponding field, it is truncated with a warning; set values according to the length constraints in the HCCL documentation.
- `group_name` writes `hcclUdi`. `hcclCommName` and `hcclDeterministic` are handled by the framework and are not configured through this dictionary.
- When `hccl_buffer_name` is used, TorchNPU also reuses communication streams for the same buffer name on the same device.
- `hccl_world_rank_id` and `hccl_job_id` correspond to `hcclWorldRankID` and `hcclJobID` respectively. They are used in NSLB scenarios and are set internally by the framework, so no manual configuration is required.
