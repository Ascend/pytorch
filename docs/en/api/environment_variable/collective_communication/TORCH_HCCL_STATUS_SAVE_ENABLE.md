# TORCH\_HCCL\_STATUS\_SAVE\_ENABLE

## Function Description

When HCCL is used as the communication backend, this environment variable controls the periodic saving of HCCL process group status information. The status file contains information such as the `seq`, `op_type`, `pg_id`, `comm_ids`, and `status` of collective operations, as well as error information at abnormal exit.

- When set to "0": Status information is not saved.
- When set to "1": Status saving is enabled.

The default value of this environment variable is "0".

> [!NOTE]
>
> - This environment variable is read during the global initialization phase of `ProcessGroupHCCL.cpp` and must be set before loading the `torch_npu` extension.
> - The status file is saved in the directory specified by `TORCH_HCCL_STATUS_SAVE_PATH`, and the saving interval is controlled by `TORCH_HCCL_STATUS_SAVE_INTERVAL`.
> - Status saving is performed in the WatchDog thread. When the WatchDog exits abnormally, the error status is automatically appended.

This environment variable is provided by TorchNPU. PyTorch has no directly corresponding environment variable.

## Configuration Example

```bash
export TORCH_HCCL_STATUS_SAVE_ENABLE=1
```

## Usage Constraints

None

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT products</term>
<!-- end id4 -->
