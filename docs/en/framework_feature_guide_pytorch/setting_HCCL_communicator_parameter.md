# Configuring HCCL Communication Domain Parameters Through pg_options

## Introduction

This feature allows different HCCL parameters to be configured for different communication domains. Add the `hccl_config` configuration through `pg_options` to pass HCCL configuration parameters from the Python layer through TorchNPU to HCCL for use.

The communication domain parameters currently supported are as follows:

- `hccl_buffer_size`
- `group_name`
- `qos_service_level`, `qos_traffic_class`
- `hccl_op_expansion_mode`

## Use Scenarios

Configure HCCL parameters at the communication domain granularity in the model script.

## Usage Guide

> [!NOTE]
>
> If both the environment variable and `pg_options` are set, the parameter values configured in `pg_options` in the code take precedence.

The following HCCL communication domain parameters can be configured:

- `hccl_buffer_size`: Sets the `hccl_buffer_size` of the communication domain. The default value is the value of the environment variable `HCCL_BUFFSIZE`. If the environment variable `HCCL_BUFFSIZE` is not set, the default value of this parameter is 200. For details about the environment variable `HCCL_BUFFSIZE`, see the "HCCL_BUFFSIZE" section in *CANN HCCL Communication Library*.
- `group_name`: Sets a custom name for the communication group of the HCCL communication domain. The value is a string with a maximum length of 32 characters.
- `qos_service_level`, `qos_traffic_class`: Sets the service level and traffic class of the RDMA NIC.
    - `qos_service_level`: The value range of this parameter is 0\~7. The default value is 0xffffffff. In this case, HCCL reads the value of the environment variable `HCCL_RDMA_SL`. For details about the environment variable `HCCL_RDMA_SL`, see the "HCCL_RDMA_SL" section in *CANN HCCL Communication Library*.
    - `qos_traffic_class`: The value range of this parameter is 0\~255. The default value is 0xffffffff. In this case, HCCL reads the value of the environment variable `HCCL_RDMA_TC`. For details about the environment variable `HCCL_RDMA_TC`, see the "HCCL_RDMA_TC" section in *CANN HCCL Communication Library*.
    <!-- see the "[HCCL_BUFFSIZE](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/commlib/hcclug/docs/zh/user_guide/hccl_env/HCCL_BUFFSIZE.md)" 26行-->
    <!-- see the "[HCCL_RDMA_SL](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/commlib/hcclug/docs/zh/user_guide/hccl_env/HCCL_RDMA_SL.md)" 29行-->
    <!-- see the "[HCCL_RDMA_TC](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/commlib/hcclug/docs/zh/user_guide/hccl_env/HCCL_RDMA_TC.md)" 30行-->

- `hccl_op_expansion_mode`: Sets the expansion position of the communication algorithm. The default value is the value of the environment variable `HCCL_OP_EXPANSION_MODE`. If the environment variable `HCCL_OP_EXPANSION_MODE` is not set, the default value of this parameter is 0. For the parameter values and descriptions supported by different AI processor models, see the hcclOpExpansionMode parameter in the "HcclCommConfig" section in *CANN HCCL Communication Library*.

    - 0: Represents the default expansion position of the communication algorithm.
    - 1: Represents that the expansion position of the communication algorithm is the CPU on the host side.
    - 2: Represents that the expansion position of the communication algorithm is the AI CPU compute unit on the device side.
    - 3: Represents that the expansion position of the communication algorithm is the AI Vector Core compute unit on the device side.

    For details about the environment variable `HCCL_OP_EXPANSION_MODE`, see the "HCCL_OP_EXPANSION_MODE" section in *CANN HCCL Communication Library*.
<!-- see the "[HCCL_OP_EXPANSION_MODE](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/commlib/hcclug/docs/zh/user_guide/hccl_env/HCCL_OP_EXPANSION_MODE.md)" section in *CANN HCCL Communication Library*. -->
<!-- see the hcclOpExpansionMode parameter in the "[HcclCommConfig](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/commlib/hcclug/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclCommConfig.md)" section in *CANN HCCL Communication Library*. 35行-->
## Usage Examples

Example of configuring `hccl_buffer_size`:

```Python
options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config = {"hccl_buffer_size": 200}
torch.distributed.init_process_group(backend="hccl", pg_options=options)
```

Example of configuring `group_name`:

```Python
options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config = {"group_name": "group0"}
torch.distributed.init_process_group(backend="hccl", pg_options=options)
```

Example of configuring `qos_service_level` and `qos_traffic_class`:

```Python
options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config = {"qos_service_level":7, "qos_traffic_class":224}
torch.distributed.init_process_group(backend="hccl", pg_options=options)
```

Example of configuring `hccl_op_expansion_mode`:

```Python
options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config ={"hccl_op_expansion_mode":3}
torch.distributed.init_process_group(backend="hccl", pg_options=options)
```

## Constraints

None
