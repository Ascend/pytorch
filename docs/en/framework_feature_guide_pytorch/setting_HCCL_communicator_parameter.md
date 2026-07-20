# Configuring HCCL Communication Domain Parameters Through pg_options

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T07:52:23.475Z pushedAt=2026-06-15T12:00:44.112Z -->

## Introduction

This feature allows different HCCL configurations for different communication domains. By adding hccl_config configuration through pg_options, HCCL configuration parameters are passed from the Python layer through Ascend Extension for PyTorch to HCCL for use.

Currently supported communication domain parameter configurations are as follows:

- hccl_buffer_size
- group_name
- qos_service_level, qos_traffic_class
- hccl_op_expansion_mode

## Use Scenario

Configure HCCL parameters at the communication domain granularity in the model script.

## Usage Guide

> [!NOTE]
>
> If both environment variables and pg_options are set, the parameter values specified in pg_options within the code take precedence.

The following HCCL communication domain parameters are supported for configuration:

- hccl_buffer_size: Sets the hccl_buffer_size for the communication domain. The default value is the value of the environment variable **HCCL_BUFFSIZE**. If the environment variable HCCL_BUFFSIZE is not set, the default value of this parameter is 200. For details about the environment variable **HCCL_BUFFSIZE**, see the [HCCL_BUFFSIZE](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0080.html) section in the *CANN Description of Environment Variables*.
- group_name: Sets a custom name for the communication group of the HCCL communication domain. The value is a string with a maximum length of 32 characters.
- qos_service_level, qos_traffic_class: Sets the service level and traffic class of the RDMA NIC.
  - qos\_service\_level: The value range of this parameter is 0\~7. The default value is 0xffffffff, in which case HCCL reads the value of the environment variable **HCCL\_RDMA\_SL**. For details about the environment variable **HCCL\_RDMA\_SL**, see the [HCCL\_RDMA\_SL](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0090.html) section in the *CANN Description of Environment Variables*.
  - qos\_traffic\_class: The value range of this parameter is 0\~255. The default value is 0xffffffff, in which case HCCL reads the value of the environment variable **HCCL\_RDMA\_TC**. For details about the environment variable **HCCL\_RDMA\_TC**, see the [HCCL\_RDMA\_TC](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0089.html) section in the *CANN Description of Environment Variables*.

- hccl\_op\_expansion\_mode: Sets the arrangement expansion position of the communication algorithm. The default value is the value of the environment variable **HCCL\_OP\_EXPANSION\_MODE**. If the environment variable **HCCL\_OP\_EXPANSION\_MODE** is not set, the default value of this parameter is 0. For the parameter values supported by different AI processor models and their descriptions, see the hcclOpExpansionMode parameter in the [HcclCommConfig](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0047.html) section in the *CANN Huawei Collective Communication Library (HCCL)*.

  - 0: Represents the default arrangement expansion position of the communication algorithm.
  - 1: Represents the arrangement expansion position of the communication algorithm on the host-side CPU.
  - 2: Represents that the arrangement expansion position of the communication algorithm encoding is on the AI CPU compute unit on the device side.
  - 3: Represents that the arrangement expansion position of the communication algorithm encoding is on the AI Vector Core compute unit on the device side.

    For details about the environment variable **HCCL\_OP\_EXPANSION\_MODE**, see the "[HCCL\_OP\_EXPANSION\_MODE](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0096.html)" section in the *CANN Description of Environment Variables*.

## Usage Example

Example of configuring hccl\_buffer\_size:

```Python
options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config = {"hccl_buffer_size": 200}
torch.distributed.init_process_group(backend="hccl", pg_options=options)
```

Example of configuring group_name:

```Python
options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config = {"group_name": "group0"}
torch.distributed.init_process_group(backend="hccl", pg_options=options)
```

Example of configuring qos_service_level and qos_traffic_class:

```Python
options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config = {"qos_service_level":7, "qos_traffic_class":224}
torch.distributed.init_process_group(backend="hccl", pg_options=options)
```

Example of configuring hccl_op_expansion_mode:

```Python
options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config ={"hccl_op_expansion_mode":3}
torch.distributed.init_process_group(backend="hccl", pg_options=options)
```

## Constraints

None
